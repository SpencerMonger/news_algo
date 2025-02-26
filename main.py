import logging
import asyncio
import signal
import sys
import os
import csv
from datetime import datetime, timedelta
import pytz
from news_checker import monitor_news
from price_checker import check_price_on_news
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_price_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

# Constants
NEWS_LOG_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\logs"
NEWS_LOG_FILE = os.path.join(NEWS_LOG_DIR, "news_findings.csv")

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info("Shutdown signal received, finishing current cycle...")
    shutdown_requested = True

def ensure_log_directory():
    """Ensure the log directory exists"""
    if not os.path.exists(NEWS_LOG_DIR):
        os.makedirs(NEWS_LOG_DIR)
        logger.info(f"Created log directory: {NEWS_LOG_DIR}")

def create_csv_if_not_exists():
    """Create the CSV file with headers if it doesn't exist"""
    ensure_log_directory()
    
    if not os.path.exists(NEWS_LOG_FILE):
        with open(NEWS_LOG_FILE, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'headline', 'published_utc', 'article_url', 'time_diff_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            logger.info(f"Created news log file: {NEWS_LOG_FILE}")

def log_news_to_csv(news_data):
    """Log news data to CSV file"""
    try:
        # Ensure the directory and file exist
        create_csv_if_not_exists()
        
        # Get the first article (most recent)
        article = news_data['news_data']['results'][0]
        
        # Calculate time difference using Eastern Time
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        published_time = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
        published_time_et = published_time.astimezone(et)
        time_diff_seconds = (now - published_time_et).total_seconds()
        
        # Prepare row data
        row = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S ET'),
            'symbol': news_data['symbol'],
            'headline': article['title'],
            'published_utc': article['published_utc'],
            'article_url': article.get('article_url', 'N/A'),
            'time_diff_seconds': f"{time_diff_seconds:.1f}"
        }
        
        # Append to CSV
        with open(NEWS_LOG_FILE, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'headline', 'published_utc', 'article_url', 'time_diff_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
            
        logger.info(f"Logged news for {news_data['symbol']} to CSV: {article['title']}")
        return True
    except Exception as e:
        logger.error(f"Error logging news to CSV: {e}")
        return False

async def news_callback_wrapper(news_data):
    """Wrapper for the price checker callback that also logs to CSV"""
    # Log to CSV first
    log_news_to_csv(news_data)
    
    # Then pass to the price checker
    await check_price_on_news(news_data)

async def test_specific_ticker(symbol):
    """Test function to check a specific ticker for news"""
    from news_checker import get_recent_news
    
    logger.info(f"Testing specific ticker: {symbol}")
    
    # Print current time in both UTC and ET for reference
    utc_now = datetime.now(pytz.UTC)
    et_now = datetime.now(pytz.timezone('US/Eastern'))
    logger.info(f"Current time - UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}, ET: {et_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Use a longer time window for testing
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    
    # Check for news in the last hour
    one_hour_ago = now - timedelta(hours=1)
    one_hour_ago_utc = one_hour_ago.astimezone(pytz.UTC)
    
    logger.info(f"Checking for news for {symbol} since {one_hour_ago.strftime('%Y-%m-%d %H:%M:%S ET')}")
    
    # Construct URL for news endpoint
    url = f"http://3.128.134.41/v2/reference/news"
    
    params = {
        'ticker': symbol,
        'published_utc.gte': one_hour_ago_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        'limit': 10,
        'order': 'desc',
        'sort': 'published_utc',
        'apiKey': os.getenv('POLYGON_API_KEY')
    }
    
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, proxy=os.getenv('PROXY_URL')) as response:
            if response.status == 200:
                data = await response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    logger.info(f"Found {len(data['results'])} news articles for {symbol} in the last hour")
                    
                    for idx, article in enumerate(data['results']):
                        published_time = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                        published_time_et = published_time.astimezone(et)
                        time_diff = (now - published_time_et).total_seconds()
                        
                        logger.info(f"Article {idx+1}: {article['title']}")
                        logger.info(f"  Published: {published_time_et.strftime('%Y-%m-%d %H:%M:%S ET')} ({time_diff/60:.1f} minutes ago)")
                        logger.info(f"  URL: {article.get('article_url', 'N/A')}")
                        
                        # Highlight if this article would be caught by our 60-second filter
                        if time_diff <= 60:
                            logger.info(f"  ** This article would trigger an alert (published within last 60 seconds) **")
                else:
                    logger.info(f"No news found for {symbol} in the last hour")
            else:
                logger.error(f"Error fetching news for {symbol}: Status {response.status}")

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
    
    # Ensure log directory and file exist
    create_csv_if_not_exists()
    
    try:
        # Check if we should test a specific ticker first
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            if len(sys.argv) > 2:
                test_symbol = sys.argv[2]
            else:
                test_symbol = "MULN"  # Default test ticker
                
            await test_specific_ticker(test_symbol)
            return
        
        # Configuration
        check_interval = 20  # seconds between news check cycles
        batch_size = 100     # number of tickers to process in each batch
        
        logger.info(f"Configuration: check_interval={check_interval}s, batch_size={batch_size} tickers")
        logger.info(f"News findings will be logged to: {NEWS_LOG_FILE}")
        
        # Start the news monitoring process, passing the wrapper callback
        monitoring_task = asyncio.create_task(
            monitor_news(
                price_checker_callback=news_callback_wrapper,
                interval=check_interval,
                batch_size=batch_size
            )
        )
        
        # Check for shutdown request
        while not shutdown_requested:
            await asyncio.sleep(1)
        
        # Cancel the monitoring task
        monitoring_task.cancel()
        try:
            await monitoring_task
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