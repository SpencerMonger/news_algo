import feedparser
import pandas as pd
import os
import time
import re
import logging
import csv
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import sys
import io
import asyncio
from price_checker import check_price_on_news

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set up logging
log_file = "rss_news_monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\data_files"
LOGS_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\logs"
TICKERS_FILE = os.path.join(DATA_DIR, "FV_master_float_u10.csv")
NEWS_LOG_FILE = os.path.join(LOGS_DIR, "ticker_news_alerts.csv")
CHECK_INTERVAL = 30  # seconds between checks
MAX_AGE_SECONDS = 90  # only return articles published in the last 90 seconds

# RSS Feed URLs
RSS_FEEDS = {
    "BusinessWire": [
        "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEF9YXA==",  # Company News
        "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeGFpfWw=="   # Financial News
    ],
    "GlobeNewswire": [
        "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20Public%20Companies",
        "https://www.globenewswire.com/RssFeed/subjectcode/22/feedTitle/Earnings%20Releases%20and%20Operating%20Results",
        "https://www.globenewswire.com/RssFeed/subjectcode/65/feedTitle/Mergers%20And%20Acquisitions"
    ],
    "PRNewswire": [
        "https://www.prnewswire.com/rss/news-releases-list.rss",
        "https://www.prnewswire.com/rss/earnings-list.rss",
        "https://www.prnewswire.com/rss/financial-services-list.rss"
    ]
}

def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def load_tickers():
    """Load ticker symbols from the master list."""
    try:
        df = pd.read_csv(TICKERS_FILE)
        # Assuming the ticker column is named 'Symbol'
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].tolist()
            logging.info(f"Loaded {len(tickers)} tickers from {TICKERS_FILE}")
            return tickers
        else:
            # Try to find a column that might contain ticker symbols
            potential_columns = ['Ticker', 'ticker', 'SYMBOL', 'symbol']
            for col in potential_columns:
                if col in df.columns:
                    tickers = df[col].dropna().tolist()
                    logging.info(f"Found tickers in column '{col}', loaded {len(tickers)} tickers")
                    return tickers
            
            # If no obvious column found, try to identify by pattern
            for col in df.columns:
                if any(re.match(r'^[A-Z]{1,5}$', str(val)) for val in df[col].dropna().head(10)):
                    tickers = df[col].dropna().tolist()
                    logging.info(f"Found tickers in column '{col}', loaded {len(tickers)} tickers")
                    return tickers
            
            logging.error(f"Could not identify ticker column in {TICKERS_FILE}")
            return []
    except Exception as e:
        logging.error(f"Error loading tickers: {str(e)}")
        return []

def initialize_news_log():
    """Initialize the news log file if it doesn't exist."""
    if not os.path.exists(NEWS_LOG_FILE):
        ensure_directory_exists(LOGS_DIR)
        with open(NEWS_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Source', 'Ticker', 'Headline', 
                'Published', 'URL', 'Summary'
            ])
        logging.info(f"Created news log file: {NEWS_LOG_FILE}")

def extract_tickers_from_text(text, ticker_list):
    """
    Extract ticker symbols from text that match our ticker list.
    
    Args:
        text (str): Text to search for tickers
        ticker_list (list): List of valid ticker symbols
        
    Returns:
        list: List of found ticker symbols
    """
    found_tickers = []
    
    # First, look for tickers with $ prefix (common in financial news)
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    for ticker in dollar_tickers:
        if ticker in ticker_list and ticker not in found_tickers:
            found_tickers.append(ticker)
    
    # Then look for standalone tickers (must be whole words)
    for ticker in ticker_list:
        # Match the ticker as a whole word
        pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(pattern, text) and ticker not in found_tickers:
            found_tickers.append(ticker)
    
    return found_tickers

def get_article_content(url):
    """
    Fetch and parse the full article content to look for tickers.
    
    Args:
        url (str): URL of the article
        
    Returns:
        str: Article content or empty string if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from paragraphs (most common for article content)
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            
            # If no paragraphs found, try getting the main content area
            if not content:
                # Try common content div classes/ids
                for selector in ['article', '.content', '#content', '.article-body', '.story-body']:
                    content_div = soup.select_one(selector)
                    if content_div:
                        content = content_div.get_text()
                        break
            
            return content
        return ""
    except Exception as e:
        logging.error(f"Error fetching article content from {url}: {str(e)}")
        return ""

def is_recent_article(published_time, max_age_seconds=MAX_AGE_SECONDS):
    """
    Check if an article was published within the specified time window.
    
    Args:
        published_time: The published time of the article
        max_age_seconds: Maximum age in seconds to consider an article recent
        
    Returns:
        bool: True if the article is recent, False otherwise
    """
    try:
        # Convert to datetime if it's a string
        if isinstance(published_time, str):
            published_dt = None
            
            # Try different date formats
            formats_to_try = [
                # RFC 822 format (common in RSS feeds)
                '%a, %d %b %Y %H:%M:%S %z',  # Wed, 26 Feb 2025 11:25:00 +0000
                '%a, %d %b %Y %H:%M:%S GMT',  # Wed, 26 Feb 2025 14:05 GMT
                '%a, %d %b %Y %H:%M GMT',     # Wed, 26 Feb 2025 14:05 GMT (without seconds)
                '%a, %d %b %Y %H:%M:%S UT',   # Wed, 26 Feb 2025 11:25:00 UT
                '%Y-%m-%dT%H:%M:%S%z',        # ISO format with timezone
                '%Y-%m-%dT%H:%M:%SZ',         # ISO format with Z
                '%Y-%m-%d %H:%M:%S',          # Simple format
            ]
            
            # Try each format until one works
            for date_format in formats_to_try:
                try:
                    if 'GMT' in published_time and 'GMT' in date_format:
                        published_dt = datetime.strptime(published_time, date_format)
                        published_dt = pytz.UTC.localize(published_dt)
                        break
                    elif 'UT' in published_time and 'UT' in date_format:
                        published_dt = datetime.strptime(published_time, date_format)
                        published_dt = pytz.UTC.localize(published_dt)
                        break
                    elif '+' in published_time or '-' in published_time and '%z' in date_format:
                        published_dt = datetime.strptime(published_time, date_format)
                        break
                    elif 'Z' in published_time and 'Z' in date_format:
                        published_dt = datetime.strptime(published_time, date_format)
                        published_dt = pytz.UTC.localize(published_dt)
                        break
                    elif 'T' not in published_time and 'GMT' not in published_time and 'UT' not in published_time:
                        published_dt = datetime.strptime(published_time, date_format)
                        published_dt = pytz.UTC.localize(published_dt)
                        break
                except ValueError:
                    continue
            
            # If none of the formats worked, try a more flexible approach
            if published_dt is None:
                # Handle "Wed, 26 Feb 2025 11:25:00 UT" format
                if "UT" in published_time:
                    try:
                        # Remove the "UT" and add "+0000"
                        fixed_time = published_time.replace(" UT", " +0000")
                        published_dt = datetime.strptime(fixed_time, '%a, %d %b %Y %H:%M:%S %z')
                    except ValueError:
                        pass
                
                # Handle "Wed, 26 Feb 2025 14:05 GMT" format
                if published_dt is None and "GMT" in published_time:
                    try:
                        # Remove the "GMT" and add "+0000"
                        fixed_time = published_time.replace(" GMT", " +0000")
                        published_dt = datetime.strptime(fixed_time, '%a, %d %b %Y %H:%M:%S %z')
                    except ValueError:
                        try:
                            # Try without seconds
                            fixed_time = published_time.replace(" GMT", ":00 +0000")
                            published_dt = datetime.strptime(fixed_time, '%a, %d %b %Y %H:%M:%S %z')
                        except ValueError:
                            pass
            
            # If we still couldn't parse the date, log and return False
            if published_dt is None:
                logging.warning(f"Could not parse date: {published_time}")
                return False
        else:
            published_dt = published_time
        
        # Ensure timezone is set
        if published_dt.tzinfo is None:
            published_dt = pytz.UTC.localize(published_dt)
        
        # Get current time in UTC
        now = datetime.now(pytz.UTC)
        
        # Calculate time difference in seconds
        time_diff = (now - published_dt).total_seconds()
        
        # Check if the article is recent enough
        return 0 <= time_diff <= max_age_seconds
    
    except Exception as e:
        logging.error(f"Error checking if article is recent: {str(e)}")
        return False

def parse_feed(feed_url, ticker_list, source_name, processed_urls):
    """
    Parse an RSS feed and extract news related to our tickers.
    
    Args:
        feed_url (str): URL of the RSS feed
        ticker_list (list): List of ticker symbols to look for
        source_name (str): Name of the news source
        processed_urls (set): Set of already processed URLs
        
    Returns:
        list: List of news items related to our tickers
    """
    try:
        feed = feedparser.parse(feed_url)
        
        if not feed.entries:
            logging.debug(f"No entries found in feed: {feed_url}")
            return []
        
        ticker_news = []
        
        for entry in feed.entries:
            # Skip already processed URLs
            if entry.link in processed_urls:
                continue
            
            # Extract data from feed entry
            title = entry.title
            summary = entry.summary if hasattr(entry, 'summary') else ""
            published = entry.published if hasattr(entry, 'published') else datetime.now(pytz.UTC).strftime('%a, %d %b %Y %H:%M:%S %z')
            url = entry.link
            
            # Check if the article is recent (within the last 90 seconds)
            if not is_recent_article(published):
                processed_urls.add(url)  # Mark as processed even if not recent
                continue
            
            # Check title and summary for tickers
            found_tickers = extract_tickers_from_text(title + " " + summary, ticker_list)
            
            # If no tickers found in title/summary, check the full article
            if not found_tickers:
                article_content = get_article_content(url)
                found_tickers = extract_tickers_from_text(article_content, ticker_list)
            
            # If tickers found, add to results
            if found_tickers:
                for ticker in found_tickers:
                    # Truncate title if it's too long to avoid logging errors
                    safe_title = title[:100] + '...' if len(title) > 100 else title
                    logging.info(f"Found news for {ticker}: {safe_title}")
                    
                    ticker_news.append({
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Source': source_name,
                        'Ticker': ticker,
                        'Headline': title,
                        'Published': published,
                        'URL': url,
                        'Summary': summary[:200] + '...' if len(summary) > 200 else summary
                    })
            
            # Mark URL as processed
            processed_urls.add(url)
        
        return ticker_news
    
    except Exception as e:
        logging.error(f"Error parsing feed {feed_url}: {str(e)}")
        return []

def log_ticker_news(news_items):
    """Log ticker news to CSV file."""
    if not news_items:
        return
    
    try:
        with open(NEWS_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in news_items:
                writer.writerow([
                    item['Timestamp'],
                    item['Source'],
                    item['Ticker'],
                    item['Headline'],
                    item['Published'],
                    item['URL'],
                    item['Summary']
                ])
        logging.info(f"Logged {len(news_items)} news items to {NEWS_LOG_FILE}")
    except Exception as e:
        logging.error(f"Error logging news items: {str(e)}")
        # Try to log with problematic characters removed
        try:
            with open(NEWS_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for item in news_items:
                    # Replace non-ASCII characters with '?' for problematic fields
                    writer.writerow([
                        item['Timestamp'],
                        item['Source'],
                        item['Ticker'],
                        item['Headline'].encode('ascii', 'replace').decode('ascii'),
                        item['Published'],
                        item['URL'],
                        item['Summary'].encode('ascii', 'replace').decode('ascii')
                    ])
            logging.info(f"Logged {len(news_items)} news items with ASCII encoding fallback")
        except Exception as e2:
            logging.error(f"Failed to log even with fallback encoding: {str(e2)}")

async def create_news_data_for_price_checker(news_item):
    """
    Create a news_data object compatible with price_checker.py
    
    Args:
        news_item (dict): News item from RSS feed
        
    Returns:
        dict: News data object for price_checker.py
    """
    # Format the news data to match what price_checker.py expects
    return {
        'symbol': news_item['Ticker'],
        'news_data': {
            'results': [
                {
                    'title': news_item['Headline'],
                    'published_utc': news_item['Published'],
                    'article_url': news_item['URL']
                }
            ]
        },
        'query_timestamp': datetime.now().isoformat()
    }

async def process_news_items(news_items):
    """
    Process news items and send to price checker
    
    Args:
        news_items (list): List of news items from RSS feeds
    """
    if not news_items:
        return
    
    for item in news_items:
        try:
            # Create news data object for price checker
            news_data = await create_news_data_for_price_checker(item)
            
            # Log that we're sending to price checker
            logging.info(f"Sending news for {item['Ticker']} to price checker: {item['Headline'][:50]}...")
            
            # Send to price checker
            await check_price_on_news(news_data)
            
        except Exception as e:
            logging.error(f"Error processing news item for price checker: {str(e)}")

def check_all_feeds(ticker_list, processed_urls):
    """
    Check all RSS feeds for news about our tickers.
    
    Args:
        ticker_list (list): List of ticker symbols to look for
        processed_urls (set): Set of already processed URLs
        
    Returns:
        list: Combined list of news items from all feeds
    """
    all_news = []
    feeds_checked = set()
    feeds_with_entries = set()
    feeds_without_entries = set()
    
    # Use ThreadPoolExecutor to check feeds concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_feed = {}
        
        # Submit tasks for each feed
        for source, feeds in RSS_FEEDS.items():
            for feed_url in feeds:
                future = executor.submit(parse_feed, feed_url, ticker_list, source, processed_urls)
                future_to_feed[future] = (source, feed_url)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_feed):
            source, feed_url = future_to_feed[future]
            try:
                feeds_checked.add(source)
                news_items = future.result()
                if news_items:
                    all_news.extend(news_items)
                    feeds_with_entries.add(source)
                    logging.info(f"Found {len(news_items)} ticker news items in {source} feed")
                else:
                    # Only add to feeds_without_entries if we know the feed was checked successfully
                    # (i.e., no exception was raised)
                    feeds_without_entries.add(source)
            except Exception as e:
                logging.error(f"Error processing {source} feed {feed_url}: {str(e)}")
    
    # Log summary of feeds checked
    if feeds_checked:
        logging.info(f"Successfully checked feeds from: {', '.join(sorted(feeds_checked))}")
    
    if feeds_without_entries:
        logging.info(f"No entries found in feeds from: {', '.join(sorted(feeds_without_entries))}")
    
    return all_news

async def main():
    """Main function to monitor RSS feeds for ticker news."""
    logging.info("Starting RSS News Monitor")
    
    # Ensure log directory exists
    ensure_directory_exists(LOGS_DIR)
    
    # Initialize news log file
    initialize_news_log()
    
    # Load ticker list
    ticker_list = load_tickers()
    if not ticker_list:
        logging.error("No tickers loaded. Exiting.")
        return
    
    # Set to track processed URLs to avoid duplicates
    processed_urls = set()
    
    try:
        while True:
            logging.info(f"Checking RSS feeds for news about {len(ticker_list)} tickers")
            
            # Check all feeds
            news_items = check_all_feeds(ticker_list, processed_urls)
            
            # Log found news
            if news_items:
                log_ticker_news(news_items)
                logging.info(f"Found and logged {len(news_items)} ticker news items")
                
                # Process news items and send to price checker
                await process_news_items(news_items)
            else:
                logging.info("No ticker news found matching our criteria")
            
            # Limit the size of processed_urls to prevent memory issues
            if len(processed_urls) > 10000:
                # Keep only the most recent 5000 URLs
                processed_urls = set(list(processed_urls)[-5000:])
            
            # Wait before next check
            logging.info(f"Waiting {CHECK_INTERVAL} seconds before next check")
            await asyncio.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        logging.info("RSS News Monitor stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 