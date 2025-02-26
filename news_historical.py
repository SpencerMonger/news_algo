import logging
import aiohttp
import os
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import urllib.parse
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROXY_URL = os.getenv('PROXY_URL')
API_KEY = os.getenv('POLYGON_API_KEY')

# Default historical search parameters
HISTORICAL_TICKER = "MULN"
HISTORICAL_DATE = "2025-02-26"  # Format: YYYY-MM-DD

# Output directory
LOGS_DIR = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\logs"

# Known test cases with guaranteed news
TEST_CASES = [
    {"ticker": "AAPL", "date": "2023-09-12"},  # Apple iPhone 15 announcement
    {"ticker": "TSLA", "date": "2023-10-18"},  # Tesla earnings
    {"ticker": "NVDA", "date": "2023-11-21"},  # NVIDIA earnings
    {"ticker": "MULN", "date": "2023-05-04"},  # Try an older date for MULN
]

async def get_historical_news(ticker=None, date=None, debug_mode=False):
    """
    Get news for a specific ticker on a specific date
    """
    # Use provided parameters or defaults
    ticker = ticker or HISTORICAL_TICKER
    date = date or HISTORICAL_DATE
    
    try:
        # Set up timezone
        et = pytz.timezone('US/Eastern')
        
        # Create datetime objects for the start and end of the day
        start_date = datetime.strptime(f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S")
        start_date = et.localize(start_date)
        end_date = datetime.strptime(f"{date} 23:59:59", "%Y-%m-%d %H:%M:%S")
        end_date = et.localize(end_date)
        
        # Convert to UTC for API request
        start_date_utc = start_date.astimezone(pytz.UTC)
        end_date_utc = end_date.astimezone(pytz.UTC)
        
        # Format timestamps for API
        published_after = start_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        published_before = end_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        logger.info(f"Searching for {ticker} news on {date}")
        logger.info(f"Time range: {published_after} to {published_before} (UTC)")
        logger.info(f"Time range in ET: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')} (ET)")
        
        # Construct URL for news endpoint
        url = f"http://3.128.134.41/v2/reference/news"
        
        params = {
            'ticker': ticker,
            'published_utc.gte': published_after,
            'published_utc.lte': published_before,
            'limit': 50,  # Increased limit to get more articles
            'order': 'asc',  # Chronological order
            'sort': 'published_utc',
            'apiKey': API_KEY
        }
        
        # Log the full URL with parameters in debug mode
        if debug_mode:
            query_string = "&".join([f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items()])
            logger.info(f"Full API URL: {url}?{query_string}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, proxy=PROXY_URL) as response:
                status = response.status
                response_text = await response.text()
                
                if debug_mode:
                    logger.info(f"API Response Status: {status}")
                    logger.info(f"API Response Headers: {response.headers}")
                    logger.info(f"API Response Text: {response_text[:500]}...")  # First 500 chars
                
                if status == 200:
                    try:
                        data = json.loads(response_text)
                        
                        if debug_mode:
                            logger.info(f"API Response Status Field: {data.get('status')}")
                            logger.info(f"API Response Count: {data.get('count', 'N/A')}")
                            logger.info(f"API Response Next URL: {data.get('next_url', 'N/A')}")
                        
                        if data.get('results') and len(data['results']) > 0:
                            # Log all articles received
                            logger.info(f"Found {len(data['results'])} articles for {ticker} on {date}")
                            
                            # Ensure logs directory exists
                            os.makedirs(LOGS_DIR, exist_ok=True)
                            
                            # Save to JSON file for analysis
                            output_file = os.path.join(LOGS_DIR, f"{ticker}_{date}_news.json")
                            
                            with open(output_file, 'w') as f:
                                json.dump(data, f, indent=2)
                            logger.info(f"Saved news data to {output_file}")
                            
                            # Save to CSV for easier viewing
                            news_csv = os.path.join(LOGS_DIR, f"{ticker}_{date}_news.csv")
                            
                            # Extract relevant fields for CSV
                            news_data = []
                            for article in data['results']:
                                published_time = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                                published_time_et = published_time.astimezone(et)
                                
                                news_data.append({
                                    'title': article['title'],
                                    'published_utc': article['published_utc'],
                                    'published_et': published_time_et.strftime('%Y-%m-%d %H:%M:%S ET'),
                                    'article_url': article.get('article_url', 'N/A'),
                                    'source': article.get('publisher', {}).get('name', 'Unknown'),
                                    'tickers': ', '.join(article.get('tickers', [])),
                                    'description': article.get('description', 'N/A')
                                })
                            
                            # Save to CSV
                            pd.DataFrame(news_data).to_csv(news_csv, index=False)
                            logger.info(f"Saved news data to CSV: {news_csv}")
                            
                            # Display articles
                            for idx, article in enumerate(data['results']):
                                published_time = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                                published_time_et = published_time.astimezone(et)
                                
                                logger.info(f"Article {idx+1}: {article['title']}")
                                logger.info(f"  Published: {published_time_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
                                logger.info(f"  URL: {article.get('article_url', 'N/A')}")
                                if article.get('description'):
                                    logger.info(f"  Description: {article['description'][:100]}...")
                            
                            return data
                        else:
                            logger.info(f"No news found for {ticker} on {date}")
                            if debug_mode and data.get('results') is not None:
                                logger.info(f"API returned empty results array")
                            return None
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON response: {e}")
                        logger.error(f"Response text: {response_text[:500]}...")
                        return None
                else:
                    logger.warning(f"Error fetching news: Status {status}")
                    logger.warning(f"Response text: {response_text[:500]}...")
                    return None
                
    except Exception as e:
        logger.error(f"Error fetching historical news: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def test_today():
    """
    Test with today's date to see if we can get any news
    """
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Testing with today's date: {today}")
    
    # Try a few popular tickers that likely have news today
    tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "MULN"]
    
    for ticker in tickers:
        logger.info(f"Testing ticker: {ticker}")
        result = await get_historical_news(ticker=ticker, date=today, debug_mode=True)
        if result:
            logger.info(f"SUCCESS: Found news for {ticker} today!")
            return True
    
    logger.info("Could not find any news for today for any of the test tickers")
    return False

async def test_known_cases():
    """
    Test with known cases that should have news
    """
    logger.info("Testing with known historical dates that should have news")
    
    for case in TEST_CASES:
        ticker = case["ticker"]
        date = case["date"]
        logger.info(f"Testing case: {ticker} on {date}")
        result = await get_historical_news(ticker=ticker, date=date, debug_mode=True)
        if result:
            logger.info(f"SUCCESS: Found news for {ticker} on {date}!")
            return True
    
    logger.info("Could not find news for any of the known test cases")
    return False

async def main():
    """
    Main function for historical news retrieval
    """
    logger.info("Starting historical news retrieval")
    
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Parse command line arguments
    ticker = HISTORICAL_TICKER
    date = HISTORICAL_DATE
    debug_mode = False
    run_tests = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--debug":
            debug_mode = True
            logger.info("Debug mode enabled")
            if len(sys.argv) > 2:
                ticker = sys.argv[2]
            if len(sys.argv) > 3:
                date = sys.argv[3]
        elif sys.argv[1] == "--test":
            run_tests = True
            logger.info("Running tests to verify API connectivity")
        else:
            ticker = sys.argv[1]
            if len(sys.argv) > 2:
                date = sys.argv[2]
    
    if run_tests:
        # First try today's date
        today_result = await test_today()
        
        # If that fails, try known historical cases
        if not today_result:
            await test_known_cases()
        
        return
    
    # Get news data
    logger.info(f"Retrieving news for {ticker} on {date}")
    news_data = await get_historical_news(ticker=ticker, date=date, debug_mode=debug_mode)
    
    if news_data:
        logger.info(f"Successfully retrieved {len(news_data['results'])} news articles")
    else:
        logger.info("No news data retrieved")
    
    logger.info("Historical news retrieval complete")
    logger.info(f"Output files have been saved to: {LOGS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())