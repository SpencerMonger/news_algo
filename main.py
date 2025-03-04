import logging
import asyncio
import signal
import sys
import os
import csv
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures

# Import functions from our modules
from price_checker import check_price_on_news, get_latest_trade, get_five_minute_bar
from rss_news_monitor import (
    load_tickers, 
    ensure_directory_exists, 
    extract_tickers_from_text,
    check_all_feeds,
    is_recent_article,
    RSS_FEEDS
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_price_monitor.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

# Constants
DATA_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\data_files"
LOGS_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\logs"
TICKERS_FILE = os.path.join(DATA_DIR, "FV_master_float_u10.csv")
NEWS_LOG_FILE = os.path.join(LOGS_DIR, "ticker_news_alerts.csv")
PRICE_MATCH_LOG_FILE = os.path.join(LOGS_DIR, "price_matches.csv")
CHECK_INTERVAL = 30  # seconds between checks
MAX_AGE_SECONDS = 90  # only return articles published in the last 90 seconds

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info("Shutdown signal received, finishing current cycle...")
    shutdown_requested = True

def create_price_match_log_if_not_exists():
    """Create the price match log file with headers if it doesn't exist"""
    ensure_directory_exists(LOGS_DIR)
    
    if not os.path.exists(PRICE_MATCH_LOG_FILE):
        with open(PRICE_MATCH_LOG_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 
                'symbol', 
                'headline', 
                'published_utc', 
                'news_detected_time',
                'price_match_time',
                'article_url', 
                'latest_price',
                'bar_high',
                'bar_low',
                'bar_open',
                'bar_close',
                'bar_volume'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            logger.info(f"Created price match log file: {PRICE_MATCH_LOG_FILE}")

def log_price_match(news_data, trade_data, bar_data, news_detected_time):
    """Log price match data to CSV file"""
    try:
        # Ensure the directory and file exist
        create_price_match_log_if_not_exists()
        
        # Get the article data
        article = news_data['news_data']['results'][0]
        
        # Prepare row data
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': news_data['symbol'],
            'headline': article['title'],
            'published_utc': article['published_utc'],
            'news_detected_time': news_detected_time,
            'price_match_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'article_url': article.get('article_url', 'N/A'),
            'latest_price': trade_data['p'],
            'bar_high': bar_data['h'],
            'bar_low': bar_data['l'],
            'bar_open': bar_data['o'],
            'bar_close': bar_data['c'],
            'bar_volume': bar_data['v']
        }
        
        # Append to CSV
        with open(PRICE_MATCH_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
            
        logger.info(f"PRICE MATCH LOGGED: {news_data['symbol']} - {article['title'][:50]}...")
        return True
    except Exception as e:
        logger.error(f"Error logging price match to CSV: {e}")
        return False

async def enhanced_price_checker(news_data, news_detected_time):
    """
    Enhanced version of price checker that logs matches with all timestamps
    
    Args:
        news_data: News data object
        news_detected_time: Timestamp when the news was detected
    """
    try:
        symbol = news_data['symbol']
        logger.info(f"Checking price for {symbol} after news detection")
        
        # Get latest trade
        trade_data = await get_latest_trade(symbol)
        if not trade_data:
            logger.warning(f"Could not get latest trade for {symbol}, aborting price check")
            return
        
        # Get 5-minute bar
        bar_data = await get_five_minute_bar(symbol)
        if not bar_data:
            logger.warning(f"Could not get 5-minute bar for {symbol}, aborting price check")
            return
        
        # Check if latest price is greater than bar high
        if trade_data['p'] > bar_data['h']:
            logger.info(f"ALERT: {symbol} price ${trade_data['p']} exceeds 5-min bar high ${bar_data['h']}")
            
            # Log the price match with all timestamps
            log_price_match(news_data, trade_data, bar_data, news_detected_time)
            
            # Create a trigger (from original price_checker)
            trigger_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "news": {
                    "title": news_data['news_data']['results'][0]['title'],
                    "published_utc": news_data['news_data']['results'][0]['published_utc'],
                    "article_url": news_data['news_data']['results'][0]['article_url'],
                    "news_detected_time": news_detected_time
                },
                "price_data": {
                    "latest_price": trade_data['p'],
                    "latest_trade_time": trade_data['t'],
                    "bar_high": bar_data['h'],
                    "bar_low": bar_data['l'],
                    "bar_open": bar_data['o'],
                    "bar_close": bar_data['c'],
                    "bar_volume": bar_data['v'],
                    "price_match_time": datetime.now().isoformat()
                }
            }
            
            # Save trigger to file
            try:
                os.makedirs('triggers', exist_ok=True)
                filename = f"triggers/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                import json
                with open(filename, 'w') as f:
                    json.dump(trigger_data, f, indent=2)
                logger.info(f"Trigger saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving trigger: {e}")
            
        else:
            logger.info(f"No trigger for {symbol}: Latest price ${trade_data['p']} <= Bar high ${bar_data['h']}")
            
    except Exception as e:
        logger.error(f"Error in enhanced price checker: {e}")

async def process_news_item(news_item):
    """
    Process a single news item from RSS feed
    
    Args:
        news_item: News item from RSS feed
    """
    try:
        # Record when we detected this news
        news_detected_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create news data object for price checker
        news_data = {
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
        
        # Log that we're sending to price checker
        logger.info(f"Sending news for {news_item['Ticker']} to price checker: {news_item['Headline'][:50]}...")
        
        # Send to enhanced price checker with news detection timestamp
        await enhanced_price_checker(news_data, news_detected_time)
        
    except Exception as e:
        logger.error(f"Error processing news item: {e}")

async def process_news_items(news_items):
    """
    Process multiple news items from RSS feeds
    
    Args:
        news_items: List of news items from RSS feeds
    """
    if not news_items:
        return
    
    # Process each news item
    for item in news_items:
        await process_news_item(item)

async def run_news_monitor():
    """Run the RSS news monitor"""
    logger.info("Starting RSS News Monitor")
    
    # Ensure log directory exists
    ensure_directory_exists(LOGS_DIR)
    
    # Load ticker list
    ticker_list = load_tickers()
    if not ticker_list:
        logger.error("No tickers loaded. Exiting.")
        return
    
    # Set to track processed URLs to avoid duplicates
    processed_urls = set()
    
    try:
        while not shutdown_requested:
            logger.info(f"Checking RSS feeds for news about {len(ticker_list)} tickers")
            
            # Check all feeds
            news_items = check_all_feeds(ticker_list, processed_urls)
            
            # Process found news
            if news_items:
                logger.info(f"Found {len(news_items)} ticker news items")
                
                # Process news items and send to price checker
                await process_news_items(news_items)
            else:
                logger.info("No ticker news found matching our criteria")
            
            # Limit the size of processed_urls to prevent memory issues
            if len(processed_urls) > 10000:
                # Keep only the most recent 5000 URLs
                processed_urls = set(list(processed_urls)[-5000:])
            
            # Wait before next check
            logger.info(f"Waiting {CHECK_INTERVAL} seconds before next check")
            await asyncio.sleep(CHECK_INTERVAL)
    
    except asyncio.CancelledError:
        logger.info("RSS News Monitor task cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in RSS News Monitor: {e}")

async def main():
    """
    Main function to run the news and price monitoring system
    """
    global shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting News and Price Monitoring System")
    
    # Print current time in both UTC and ET for reference
    utc_now = datetime.now(pytz.UTC)
    et_now = datetime.now(pytz.timezone('US/Eastern'))
    logger.info(f"Current time - UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}, ET: {et_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure log files exist
    create_price_match_log_if_not_exists()
    
    try:
        # Start the news monitoring process
        news_monitor_task = asyncio.create_task(run_news_monitor())
        
        # Wait for shutdown request
        while not shutdown_requested:
            await asyncio.sleep(1)
        
        # Cancel the news monitor task
        news_monitor_task.cancel()
        try:
            await news_monitor_task
        except asyncio.CancelledError:
            pass
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in main monitoring loop: {e}")
    finally:
        logger.info("Shutting down monitoring system")

if __name__ == "__main__":
    asyncio.run(main()) 