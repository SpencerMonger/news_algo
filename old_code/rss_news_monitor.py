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
    """Load tickers from CSV file"""
    try:
        # Use os.path.join for proper path handling across operating systems
        csv_path = os.path.join('data_files', 'FV_master_u50float_u10price.csv')
        logger.info(f"Loading tickers from {csv_path}")
        
        # Verify file exists before trying to read it
        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Could not find {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get tickers from the 'Ticker' column
        tickers = df['Ticker'].tolist()
        
        # Clean the tickers (remove any whitespace, convert to uppercase)
        tickers = [str(ticker).strip().upper() for ticker in tickers if pd.notna(ticker)]
        
        logger.info(f"Loaded {len(tickers)} tickers. Sample: {tickers[:5]}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers from CSV: {e}")
        logger.error(f"Current working directory: {os.getcwd()}")
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
    """Extract tickers from text"""
    if not text or not ticker_list:
        return []
    
    # Add more detailed logging
    logger.debug(f"Searching text: {text[:200]}...")
    logger.debug(f"Number of tickers to check: {len(ticker_list)}")
    
    matched_tickers = []
    text = text.upper()  # Convert text to uppercase for case-insensitive matching
    
    for ticker in ticker_list:
        ticker = str(ticker).strip().upper()
        # Look for exact matches with word boundaries
        pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(pattern, text):
            matched_tickers.append(ticker)
            logger.debug(f"Found match for ticker: {ticker}")
    
    if matched_tickers:
        logger.info(f"Found matching tickers: {matched_tickers}")
    
    return matched_tickers

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
        # Convert string to datetime if it's not already
        if isinstance(published_time, str):
            published_dt = pd.to_datetime(published_time)
        else:
            published_dt = published_time  # Already a datetime object
            
        current_time = datetime.now(pytz.UTC)
        if not published_dt.tzinfo:
            published_dt = pytz.UTC.localize(published_dt)
            
        age = (current_time - published_dt).total_seconds()
        return age <= max_age_seconds
    except Exception as e:
        logger.error(f"Error checking article age: {e}")
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
    """Log all articles to CSV regardless of ticker matches"""
    if not articles:
        logger.info("No articles to log to all_news_articles.csv")
        return
        
    try:
        df = pd.DataFrame(articles)
        df.to_csv('all_news_articles.csv', mode='a', header=not os.path.exists('all_news_articles.csv'), index=False)
        logger.info(f"Successfully logged {len(articles)} articles to all_news_articles.csv")
    except Exception as e:
        logger.error(f"Error logging to all_news_articles.csv: {e}")

def log_ticker_news(news_items):
    """Log ticker-specific news to CSV"""
    if not news_items:
        return
        
    # Add logging to debug
    logger.info(f"Attempting to log {len(news_items)} ticker-related articles to ticker_news_alert.csv")
    
    try:
        df = pd.DataFrame(news_items)
        df.to_csv('ticker_news_alert.csv', mode='a', header=not os.path.exists('ticker_news_alert.csv'), index=False)
        logger.info(f"Successfully logged {len(news_items)} articles to ticker_news_alert.csv")
    except Exception as e:
        logger.error(f"Error logging to ticker_news_alert.csv: {e}")

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
    try:
        url = "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States"
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Error fetching GlobeNewswire feed: Status {response.status}")
                return []
            
            feed_content = await response.text()
            feed = feedparser.parse(feed_content)
            
            logger.info(f"Found {len(feed.entries)} entries in GlobeNewswire feed")
            
            articles = []
            for entry in feed.entries:
                try:
                    title = entry.get('title', '')
                    description = entry.get('description', '')
                    
                    # Search for tickers in both title and description
                    text_to_search = f"{title} {description}"
                    matched_tickers = extract_tickers_from_text(text_to_search, ticker_list)
                    
                    if matched_tickers:
                        article = {
                            'title': title,
                            'link': entry.get('link', ''),
                            'published_utc': entry.get('published', ''),
                            'source': 'GlobeNewswire',
                            'tickers': matched_tickers
                        }
                        articles.append(article)
                        logger.info(f"Found article matching tickers {matched_tickers}: {title}")
                
                except Exception as e:
                    logger.warning(f"Error processing entry in GlobeNewswire feed: {e}")
                    continue
            
            if not articles:
                logger.info("No matching news found in GlobeNewswire feed")
            else:
                logger.info(f"Found {len(articles)} matching articles in GlobeNewswire feed")
            
            return articles
            
    except Exception as e:
        logger.error(f"Error scraping GlobeNewswire: {e}")
        return []

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