import feedparser
import pandas as pd
import os
import time
import re
import logging
import csv
import pytz
import requests
from bs4 import BeautifulSoup
import sys
import io
import asyncio
import concurrent.futures
from urllib.parse import urlparse
import aiohttp
from price_checker import check_price_on_news
from datetime import datetime, timezone, timedelta
from dateutil import parser as date_parser

# Additional imports for web scraping
from typing import List, Dict, Any
import json
import time

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rss_news_monitor.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\data_files"
LOGS_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\logs"
TICKERS_FILE = os.path.join(DATA_DIR, "FV_master_u50float_u10price.csv")
NEWS_LOG_FILE = os.path.join(LOGS_DIR, "ticker_news_alerts.csv")
ALL_NEWS_LOG_FILE = os.path.join(LOGS_DIR, "all_news_articles.csv")
CHECK_INTERVAL = 30  # seconds
MAX_PROCESSED_URLS = 10000  # Limit memory usage
MAX_AGE_SECONDS = 90  # Only process articles published in the last 90 seconds

# RSS feed URLs
RSS_FEEDS = {
    "BusinessWire": "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEF9YXA==",
    "PRNewswire": "https://www.prnewswire.com/rss/news-releases-list.rss",
    "GlobeNewswire": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20Releases",
    "MarketWatch_Bulletins": "https://feeds.content.dowjones.io/public/rss/mw_bulletins",
    "MarketWatch_RealTime": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "MarketWatch_MarketPulse": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse"
}

# Web scraping URLs
SCRAPING_URLS = {
    "GlobeNewswire": "https://www.globenewswire.com/en/search/keyword/{ticker}?pageSize=10",
    "BusinessWire": "https://www.businesswire.com/portal/site/home/search/?searchType=all&searchTerm={ticker}",
    "PRNewswire": "https://www.prnewswire.com/search/news/?keyword={ticker}&pagesize=25"
}

# Date formats to try for parsing published dates
DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",           # "Wed, 26 Feb 2025 20:00:00 +0000"
    "%a, %d %b %Y %H:%M:%S %Z",           # "Wed, 26 Feb 2025 20:00:00 GMT"
    "%a, %d %b %Y %H:%M:%S UT",           # "Wed, 26 Feb 2025 13:00:00 UT"
    "%Y-%m-%dT%H:%M:%S%z",                # ISO format with timezone
    "%Y-%m-%dT%H:%M:%SZ",                 # ISO format UTC
    "%Y-%m-%d %H:%M:%S",                  # Simple datetime
]

def ensure_directory_exists(directory_path):
    """
    Ensure that the specified directory exists
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def load_tickers():
    """
    Load ticker symbols from CSV file
    """
    try:
        ensure_directory_exists(DATA_DIR)
        
        if not os.path.exists(TICKERS_FILE):
            logger.error(f"Tickers file not found: {TICKERS_FILE}")
            # Create a sample tickers file for testing
            sample_tickers = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']
            })
            sample_tickers.to_csv(TICKERS_FILE, index=False)
            logger.info(f"Created sample tickers file: {TICKERS_FILE}")
        
        # Read the tickers file
        tickers_df = pd.read_csv(TICKERS_FILE)
        
        # Extract the ticker symbols column
        if 'Symbol' in tickers_df.columns:
            ticker_list = tickers_df['Symbol'].tolist()
        else:
            # Try to find a column that might contain ticker symbols
            for col in tickers_df.columns:
                if any(re.match(r'^[A-Z]{1,5}$', str(val)) for val in tickers_df[col].dropna()):
                    ticker_list = tickers_df[col].dropna().tolist()
                    break
            else:
                logger.error(f"Could not find ticker symbols column in {TICKERS_FILE}")
                ticker_list = []
        
        # Convert to uppercase and remove any non-string values
        ticker_list = [str(ticker).upper() for ticker in ticker_list if isinstance(ticker, (str, int, float))]
        
        logger.info(f"Loaded {len(ticker_list)} tickers from {TICKERS_FILE}")
        return ticker_list
    
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        return []

def initialize_news_log():
    """
    Initialize the news log file with headers if it doesn't exist
    """
    try:
        ensure_directory_exists(LOGS_DIR)
        
        if not os.path.exists(NEWS_LOG_FILE):
            with open(NEWS_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Source', 'Ticker', 'Headline', 'Published', 'URL', 'Summary'])
            logger.info(f"Created news log file: {NEWS_LOG_FILE}")
            
        if not os.path.exists(ALL_NEWS_LOG_FILE):
            with open(ALL_NEWS_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Source', 'Headline', 'Published', 'URL', 'Summary', 'Matched_Tickers'])
            logger.info(f"Created all news log file: {ALL_NEWS_LOG_FILE}")
    
    except Exception as e:
        logger.error(f"Error initializing news log: {e}")

def extract_tickers_from_text(text, ticker_list):
    """
    Extract ticker symbols from text
    """
    if not text or not isinstance(text, str):
        return []
    
    # Convert text to uppercase for case-insensitive matching
    text_upper = text.upper()
    
    # Find all standalone ticker mentions (surrounded by non-alphanumeric characters)
    found_tickers = []
    
    for ticker in ticker_list:
        # Skip invalid tickers
        if not ticker or not isinstance(ticker, str):
            continue
            
        ticker = ticker.strip().upper()
        
        # Look for the ticker as a standalone word or with $ prefix
        pattern = r'(?:^|\W)[$]?(' + re.escape(ticker) + r')(?:$|\W)'
        
        if re.search(pattern, text_upper):
            found_tickers.append(ticker)
    
    return found_tickers

def get_article_content(url):
    """
    Fetch and parse the full article content to look for ticker mentions
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])
            
            return article_text
        else:
            logger.warning(f"Failed to fetch article content: {url}, Status: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Error fetching article content: {url}, Error: {e}")
        return None

def is_recent_article(published_time: str, max_age_seconds: int = MAX_AGE_SECONDS) -> bool:
    """
    Check if the article was published within the specified time window
    """
    if not published_time:
        logger.warning("No published time provided")
        return False
    
    try:
        # Try to parse the date using dateutil parser which handles most formats
        dt = date_parser.parse(published_time)
        
        # If the datetime has no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to UTC for comparison
        now_utc = datetime.now(timezone.utc)
        
        # Calculate the age of the article in seconds
        age_seconds = (now_utc - dt).total_seconds()
        
        # Check if the article is recent enough
        is_recent = age_seconds <= max_age_seconds
        
        if is_recent:
            logger.info(f"Article is recent: {published_time} (Age: {age_seconds:.1f} seconds)")
        else:
            logger.debug(f"Article is too old: {published_time} (Age: {age_seconds:.1f} seconds)")
        
        return is_recent
    
    except Exception as e:
        logger.warning(f"Error checking if article is recent: {e}, Date: {published_time}")
        return False

def parse_feed(feed_url, ticker_list, source_name, processed_urls):
    """
    Parse an RSS feed and extract news related to the ticker list
    """
    try:
        logger.info(f"Checking {source_name} feed...")
        feed = feedparser.parse(feed_url)
        
        if not feed.entries:
            logger.warning(f"No entries found in {source_name} feed")
            return []
        
        logger.info(f"Found {len(feed.entries)} entries in {source_name} feed")
        
        ticker_news = []
        all_articles = []
        
        for entry in feed.entries:
            try:
                # Extract relevant information
                title = entry.get('title', '')
                link = entry.get('link', '')
                summary = entry.get('summary', '')
                published = entry.get('published', '')
                
                # Skip if we've already processed this URL
                if link in processed_urls:
                    continue
                
                # Add to processed URLs
                processed_urls.add(link)
                
                # Limit the size of processed_urls to prevent memory issues
                if len(processed_urls) > MAX_PROCESSED_URLS:
                    # Remove oldest items (assuming they're added in order)
                    processed_urls = set(list(processed_urls)[-MAX_PROCESSED_URLS:])
                
                # Check if the article is recent
                if not is_recent_article(published):
                    continue
                
                # Combine title and summary for ticker extraction
                full_text = f"{title} {summary}"
                
                # Extract tickers from the combined text
                found_tickers = extract_tickers_from_text(full_text, ticker_list)
                
                # If no tickers found in title/summary, try to fetch the full article
                if not found_tickers and (source_name == "BusinessWire" or source_name == "PRNewswire" or source_name == "GlobeNewswire"):
                    article_content = get_article_content(link)
                    if article_content:
                        found_tickers = extract_tickers_from_text(article_content, ticker_list)
                
                # Log all articles regardless of ticker matches
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                all_articles.append({
                    'Timestamp': timestamp,
                    'Source': source_name,
                    'Headline': title,
                    'Published': published,
                    'URL': link,
                    'Summary': summary,
                    'Matched_Tickers': ','.join(found_tickers) if found_tickers else 'None'
                })
                
                # If tickers found, add to ticker news
                if found_tickers:
                    for ticker in found_tickers:
                        ticker_news.append({
                            'Timestamp': timestamp,
                            'Source': source_name,
                            'Ticker': ticker,
                            'Headline': title,
                            'Published': published,
                            'URL': link,
                            'Summary': summary
                        })
                        logger.info(f"Found news for {ticker}: {title}")
            
            except Exception as e:
                logger.warning(f"Error processing entry in {source_name} feed: {e}")
                continue
        
        # Log all articles to CSV
        if all_articles:
            log_all_articles(all_articles)
            
        return ticker_news
    
    except Exception as e:
        logger.error(f"Error parsing {source_name} feed: {e}")
        return []

def log_all_articles(articles):
    """
    Log all articles to CSV file, regardless of ticker matches
    """
    try:
        with open(ALL_NEWS_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for article in articles:
                try:
                    writer.writerow([
                        article['Timestamp'],
                        article['Source'],
                        article['Headline'],
                        article['Published'],
                        article['URL'],
                        article['Summary'],
                        article['Matched_Tickers']
                    ])
                except Exception as e:
                    # If there's an encoding error, try to write with ASCII encoding
                    logger.warning(f"Error writing article to CSV, trying ASCII fallback: {e}")
                    writer.writerow([
                        article['Timestamp'],
                        article['Source'],
                        str(article['Headline']).encode('ascii', 'replace').decode('ascii'),
                        article['Published'],
                        article['URL'],
                        str(article['Summary']).encode('ascii', 'replace').decode('ascii'),
                        article['Matched_Tickers']
                    ])
        
        logger.info(f"Logged {len(articles)} articles to {ALL_NEWS_LOG_FILE}")
    
    except Exception as e:
        logger.error(f"Error logging all articles: {e}")

def log_ticker_news(news_items):
    """
    Log ticker news to CSV file
    """
    if not news_items:
        return
    
    try:
        with open(NEWS_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in news_items:
                try:
                    writer.writerow([
                        item['Timestamp'],
                        item['Source'],
                        item['Ticker'],
                        item['Headline'],
                        item['Published'],
                        item['URL'],
                        item['Summary']
                    ])
                except Exception as e:
                    # If there's an encoding error, try to write with ASCII encoding
                    logger.warning(f"Error writing news item to CSV, trying ASCII fallback: {e}")
                    writer.writerow([
                        item['Timestamp'],
                        item['Source'],
                        item['Ticker'],
                        str(item['Headline']).encode('ascii', 'replace').decode('ascii'),
                        item['Published'],
                        item['URL'],
                        str(item['Summary']).encode('ascii', 'replace').decode('ascii')
                    ])
        
        logger.info(f"Logged {len(news_items)} news items to {NEWS_LOG_FILE}")
    
    except Exception as e:
        logger.error(f"Error logging ticker news: {e}")

async def create_news_data_for_price_checker(news_item):
    """
    Create a news data object for the price checker
    """
    try:
        symbol = news_item['Ticker']
        
        # Create a news data object for the price checker
        news_data = {
            'symbol': symbol,
            'news_data': {
                'results': [
                    {
                        'title': news_item['Headline'],
                        'published_utc': news_item['Published'],
                        'article_url': news_item['URL']
                    }
                ]
            }
        }
        
        return news_data
    
    except Exception as e:
        logger.error(f"Error creating news data for price checker: {e}")
        return None

async def process_news_items(news_items):
    """
    Process news items and check prices
    """
    if not news_items:
        return
    
    try:
        # Log the news items
        log_ticker_news(news_items)
        
        # Process each news item
        for item in news_items:
            try:
                # Create news data for price checker
                news_data = await create_news_data_for_price_checker(item)
                
                if news_data:
                    # Get the current time for tracking when the news was detected
                    news_detected_time = datetime.datetime.now()
                    
                    # Check price on news
                    await check_price_on_news(news_data)
                    
                    logger.info(f"Processed news item for {item['Ticker']}: {item['Headline']}")
            
            except Exception as e:
                logger.error(f"Error processing news item: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error processing news items: {e}")

def check_all_feeds(ticker_list, processed_urls):
    """
    Check all RSS feeds for ticker news
    """
    all_news = []
    feeds_checked = 0
    feeds_with_entries = 0
    
    # Use ThreadPoolExecutor for concurrent feed checking
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RSS_FEEDS)) as executor:
        # Create a dictionary to store futures
        future_to_source = {
            executor.submit(parse_feed, url, ticker_list, source, processed_urls): source
            for source, url in RSS_FEEDS.items()
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_source):
            source = future_to_source[future]
            feeds_checked += 1
            
            try:
                news_items = future.result()
                if news_items:
                    feeds_with_entries += 1
                    all_news.extend(news_items)
                    logger.info(f"Found {len(news_items)} news items in {source} feed")
                else:
                    logger.info(f"No matching news found in {source} feed")
            
            except Exception as e:
                logger.error(f"Error checking {source} feed: {e}")
    
    # Log summary message
    if feeds_checked > 0:
        if not all_news:
            logger.info(f"No News: Checked {feeds_checked} feeds, found 0 matching articles")
        else:
            logger.info(f"Found {len(all_news)} matching articles across {feeds_with_entries} feeds")
    
    return all_news

async def scrape_globenewswire(session: aiohttp.ClientSession, ticker_list: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape GlobeNewswire for recent news articles
    """
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        for ticker in ticker_list:
            url = SCRAPING_URLS["GlobeNewswire"].format(ticker=ticker)
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all news articles
                    articles = soup.find_all('div', class_='news-item')
                    
                    for article in articles:
                        try:
                            title = article.find('a', class_='news-item__headline').text.strip()
                            link = article.find('a', class_='news-item__headline')['href']
                            published = article.find('time')['datetime']
                            
                            # Check if article is recent enough
                            if is_recent_article(published):
                                news_items.append({
                                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'Source': 'GlobeNewswire',
                                    'Ticker': ticker,
                                    'Headline': title,
                                    'Published': published,
                                    'URL': f"https://www.globenewswire.com{link}",
                                    'Summary': ''
                                })
                        except Exception as e:
                            logger.warning(f"Error processing GlobeNewswire article: {e}")
                            continue
                            
    except Exception as e:
        logger.error(f"Error scraping GlobeNewswire: {e}")
    
    return news_items

async def scrape_businesswire(session: aiohttp.ClientSession, ticker_list: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape BusinessWire for recent news articles using ticker-specific searches
    """
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        for ticker in ticker_list:
            url = SCRAPING_URLS["BusinessWire"].format(ticker=ticker)
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all news articles
                    articles = soup.find_all('div', class_='bw-news-list-item')
                    
                    for article in articles:
                        try:
                            title = article.find('a', class_='bw-news-title').text.strip()
                            link = article.find('a', class_='bw-news-title')['href']
                            published = article.find('time')['datetime']
                            
                            # Check if article is recent enough
                            if is_recent_article(published):
                                news_items.append({
                                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'Source': 'BusinessWire',
                                    'Ticker': ticker,
                                    'Headline': title,
                                    'Published': published,
                                    'URL': link,
                                    'Summary': ''
                                })
                        except Exception as e:
                            logger.warning(f"Error processing BusinessWire article: {e}")
                            continue
                            
    except Exception as e:
        logger.error(f"Error scraping BusinessWire: {e}")
    
    return news_items

async def scrape_prnewswire(session: aiohttp.ClientSession, ticker_list: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape PRNewswire for recent news articles using ticker-specific searches
    """
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        for ticker in ticker_list:
            url = SCRAPING_URLS["PRNewswire"].format(ticker=ticker)
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all news articles
                    articles = soup.find_all('div', class_='news-release')
                    
                    for article in articles:
                        try:
                            title = article.find('h3').text.strip()
                            link = article.find('a')['href']
                            published = article.find('time')['datetime']
                            
                            # Check if article is recent enough
                            if is_recent_article(published):
                                news_items.append({
                                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'Source': 'PRNewswire',
                                    'Ticker': ticker,
                                    'Headline': title,
                                    'Published': published,
                                    'URL': f"https://www.prnewswire.com{link}",
                                    'Summary': ''
                                })
                        except Exception as e:
                            logger.warning(f"Error processing PRNewswire article: {e}")
                            continue
                            
    except Exception as e:
        logger.error(f"Error scraping PRNewswire: {e}")
    
    return news_items

async def scrape_all_sources(ticker_list: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape all news sources concurrently
    """
    all_news = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            scrape_globenewswire(session, ticker_list),
            scrape_businesswire(session, ticker_list),
            scrape_prnewswire(session, ticker_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            else:
                logger.error(f"Error in scraping task: {result}")
    
    return all_news

async def main():
    """
    Main function to monitor RSS feeds and web sources for ticker news
    """
    logger.info("Starting news monitor...")
    
    # Initialize the news log
    initialize_news_log()
    
    # Load ticker list
    ticker_list = load_tickers()
    
    if not ticker_list:
        logger.error("No tickers loaded, exiting")
        return
    
    # Set to track processed URLs
    processed_urls = set()
    
    try:
        while True:
            try:
                # Check RSS feeds
                logger.info("Checking RSS feeds...")
                rss_news_items = check_all_feeds(ticker_list, processed_urls)
                
                # Check web sources
                logger.info("Checking web sources...")
                web_news_items = await scrape_all_sources(ticker_list)
                
                # Combine news items, avoiding duplicates
                all_news_items = []
                seen_urls = set()
                
                for item in rss_news_items + web_news_items:
                    if item['URL'] not in seen_urls:
                        seen_urls.add(item['URL'])
                        all_news_items.append(item)
                
                if all_news_items:
                    logger.info(f"Found {len(all_news_items)} unique news items")
                    # Process news items
                    await process_news_items(all_news_items)
                else:
                    logger.info("No news items found")
                
                # Wait for the next check
                logger.info(f"Waiting {CHECK_INTERVAL} seconds until next check...")
                await asyncio.sleep(CHECK_INTERVAL)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        logger.info("News monitor stopped by user")

if __name__ == "__main__":
    asyncio.run(main()) 