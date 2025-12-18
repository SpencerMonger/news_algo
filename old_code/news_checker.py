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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CSV_PATH = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\data_files\FV_master_float_u10.csv"
PROXY_URL = os.getenv('PROXY_URL')
API_KEY = os.getenv('POLYGON_API_KEY')

async def get_recent_news(symbol):
    """
    Get news for a specific symbol from the last 60 seconds
    """
    try:
        # Ensure we're using Eastern Time (ET) for all time calculations
        et = pytz.timezone('US/Eastern')
        # Get current time in ET
        now = datetime.now(et)
        
        # Calculate timestamp for 60 seconds ago
        sixty_seconds_ago = now - timedelta(seconds=60)
        
        # Store query timestamp for verification
        query_timestamp = now
        
        # Format timestamp for API - IMPORTANT: Polygon API expects UTC time
        # Convert ET time to UTC for the API request
        sixty_seconds_ago_utc = sixty_seconds_ago.astimezone(pytz.UTC)
        published_after = sixty_seconds_ago_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        logger.info(f"Checking for news after {published_after} (UTC) / {sixty_seconds_ago.strftime('%Y-%m-%dT%H:%M:%S')} (ET) for {symbol}")
        
        # Construct URL for news endpoint
        url = f"http://3.128.134.41/v2/reference/news"
        
        # Use a longer timeframe for the API request to ensure we don't miss anything
        # We'll filter more precisely in our code
        five_minutes_ago_utc = (now - timedelta(minutes=5)).astimezone(pytz.UTC)
        params = {
            'ticker': symbol,
            'published_utc.gte': five_minutes_ago_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'limit': 20,  # Increased limit to catch more potential articles
            'order': 'desc',
            'sort': 'published_utc',
            'apiKey': API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, proxy=PROXY_URL) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results') and len(data['results']) > 0:
                        # Log all articles received for debugging
                        logger.info(f"Received {len(data['results'])} articles for {symbol}")
                        for idx, article in enumerate(data['results']):
                            logger.info(f"Article {idx+1} for {symbol}: {article['title']} (published: {article['published_utc']})")
                        
                        # Double-check that articles are actually within the last 60 seconds
                        recent_articles = []
                        
                        for article in data['results']:
                            try:
                                # Parse the published_utc timestamp (which is in UTC)
                                published_time_utc = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                                
                                # Convert to ET for consistent comparison
                                published_time_et = published_time_utc.astimezone(et)
                                
                                # Calculate time difference in seconds
                                time_diff = (query_timestamp - published_time_et).total_seconds()
                                
                                # Log all articles with their time differences for debugging
                                logger.info(f"Article for {symbol} published {time_diff:.1f} seconds ago: {article['title']}")
                                
                                # Only include if published within the last 60 seconds
                                if time_diff <= 60 and time_diff >= 0:
                                    recent_articles.append(article)
                                    logger.info(f"MATCH: Article for {symbol} published {time_diff:.1f} seconds ago: {article['title']}")
                                else:
                                    logger.info(f"SKIPPING: Article for {symbol} published {time_diff:.1f} seconds ago: {article['title']}")
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error parsing article timestamp for {symbol}: {e}")
                        
                        # Only return if we have recent articles
                        if recent_articles:
                            logger.info(f"Found {len(recent_articles)} recent news articles for {symbol} within the last 60 seconds")
                            # Create a new data structure with only the recent articles
                            filtered_data = data.copy()
                            filtered_data['results'] = recent_articles
                            return {
                                'symbol': symbol,
                                'news_data': filtered_data,
                                'query_timestamp': query_timestamp.isoformat()
                            }
                        else:
                            logger.info(f"No articles within the last 60 seconds for {symbol}")
                            return None
                    else:
                        logger.debug(f"No recent news for {symbol}")
                        return None
                else:
                    logger.warning(f"Error fetching news for {symbol}: Status {response.status}")
                    return None
                
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return None

async def process_ticker_news(symbol, price_checker_callback):
    """
    Process news for a ticker and pass to price checker if news exists
    """
    news_data = await get_recent_news(symbol)
    if news_data:
        logger.info(f"Sending news data for {symbol} to price checker")
        await price_checker_callback(news_data)
    return news_data is not None

async def load_tickers():
    """
    Load tickers from CSV file
    """
    try:
        df = pd.read_csv(CSV_PATH)
        # Assuming the ticker column is named 'Symbol' - adjust if different
        if 'Symbol' in df.columns:
            return df['Symbol'].tolist()
        else:
            # Try to find a column that might contain ticker symbols
            potential_columns = ['Ticker', 'ticker', 'SYMBOL', 'symbol']
            for col in potential_columns:
                if col in df.columns:
                    logger.info(f"Using column '{col}' for ticker symbols")
                    return df[col].tolist()
            
            # If no obvious column found, use the first column
            logger.warning(f"No obvious ticker column found, using first column: {df.columns[0]}")
            return df[df.columns[0]].tolist()
    except Exception as e:
        logger.error(f"Error loading tickers from CSV: {e}")
        return []

async def process_ticker_batch(tickers, price_checker_callback):
    """
    Process a batch of tickers to limit concurrent requests
    """
    tasks = [process_ticker_news(ticker, price_checker_callback) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count tickers with news and exceptions
    tickers_with_news = 0
    exceptions = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            exceptions += 1
            logger.warning(f"Exception processing ticker {tickers[i]}: {str(result)}")
        elif result is True:
            tickers_with_news += 1
    
    return tickers_with_news, exceptions

async def monitor_news(price_checker_callback, interval=10, batch_size=50):
    """
    Continuously monitor news for all tickers at specified interval (10 seconds)
    With batching to prevent overwhelming the proxy server
    """
    cycle_count = 0
    start_time = datetime.now()
    
    while True:
        try:
            cycle_count += 1
            all_tickers = await load_tickers()
            logger.info(f"Monitoring news for {len(all_tickers)} tickers (cycle #{cycle_count})")
            
            # Process tickers in batches to avoid overwhelming the proxy
            total_tickers_with_news = 0
            total_exceptions = 0
            
            # Split tickers into batches
            for i in range(0, len(all_tickers), batch_size):
                batch = all_tickers[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(all_tickers) + batch_size - 1)//batch_size} ({len(batch)} tickers)")
                
                tickers_with_news, exceptions = await process_ticker_batch(batch, price_checker_callback)
                total_tickers_with_news += tickers_with_news
                total_exceptions += exceptions
                
                # Add a small delay between batches to avoid overwhelming the proxy
                if i + batch_size < len(all_tickers):
                    await asyncio.sleep(1)
            
            # Log results of this cycle
            if total_tickers_with_news > 0:
                logger.info(f"Found news for {total_tickers_with_news} tickers in this cycle")
            else:
                # Only log "No news found" every 6 cycles (about once per minute) to avoid log spam
                if cycle_count % 6 == 0:
                    runtime = datetime.now() - start_time
                    logger.info(f"No news found in cycle #{cycle_count}. Running for {runtime.total_seconds():.0f} seconds.")
            
            if total_exceptions > 0:
                logger.warning(f"{total_exceptions} exceptions occurred while processing tickers")
                
                # If we're getting a lot of exceptions, increase the delay between cycles
                if total_exceptions > batch_size * 0.5:  # If more than 50% of a batch is failing
                    logger.warning(f"High error rate detected. Consider increasing batch delay or reducing batch size.")
            
            # Log a summary every 30 cycles (about every 5 minutes)
            if cycle_count % 30 == 0:
                runtime = datetime.now() - start_time
                logger.info(f"=== SUMMARY: Completed {cycle_count} cycles over {runtime.total_seconds():.0f} seconds ===")
                logger.info(f"=== System is running normally, monitoring {len(all_tickers)} tickers in batches of {batch_size} ===")
            
            # Wait for next cycle
            logger.info(f"Waiting {interval} seconds for next news check cycle")
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error in news monitoring cycle: {e}")
            await asyncio.sleep(interval)

# For testing purposes
async def test_callback(news_data):
    logger.info(f"Test callback received news for {news_data['symbol']}")
    logger.info(f"First headline: {news_data['news_data']['results'][0]['title']}")
    logger.info(f"Published at: {news_data['news_data']['results'][0]['published_utc']}")
    logger.info(f"Query timestamp: {news_data.get('query_timestamp', 'Not available')}")

async def main():
    """
    Main function for testing
    """
    logger.info("Starting news checker test")
    
    # Test a specific ticker that should have news (MULN)
    test_symbol = os.getenv('SYMBOL', 'MULN')
    logger.info(f"Checking for recent news for {test_symbol}...")
    
    # Print current time in both UTC and ET for reference
    utc_now = datetime.now(pytz.UTC)
    et_now = datetime.now(pytz.timezone('US/Eastern'))
    logger.info(f"Current time - UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}, ET: {et_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    news = await get_recent_news(test_symbol)
    
    if news:
        logger.info(f"Found news for {test_symbol}")
        logger.info(f"First headline: {news['news_data']['results'][0]['title']}")
        logger.info(f"Published at: {news['news_data']['results'][0]['published_utc']}")
        logger.info(f"Query timestamp: {news.get('query_timestamp', 'Not available')}")
        
        # Calculate and display time difference for verification
        try:
            et = pytz.timezone('US/Eastern')
            published_time = datetime.fromisoformat(news['news_data']['results'][0]['published_utc'].replace('Z', '+00:00'))
            published_time_et = published_time.astimezone(et)
            query_time = datetime.fromisoformat(news['query_timestamp'])
            time_diff_seconds = (query_time - published_time_et).total_seconds()
            logger.info(f"Time difference: {time_diff_seconds:.1f} seconds")
        except Exception as e:
            logger.error(f"Error calculating time difference: {e}")
    else:
        logger.info(f"No recent news found for {test_symbol} in the last 60 seconds")
        
        # Get some older news to verify API is working
        logger.info(f"Checking for any recent news for {test_symbol} (last 24 hours)...")
        
        try:
            # Construct URL for news endpoint with a longer timeframe
            url = f"http://3.128.134.41/v2/reference/news"
            
            params = {
                'ticker': test_symbol,
                'limit': 5,  # Increased to show more articles
                'order': 'desc',
                'sort': 'published_utc',
                'apiKey': API_KEY
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, proxy=PROXY_URL) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            logger.info(f"Found {len(data['results'])} articles for {test_symbol} in the last 24 hours:")
                            
                            for idx, article in enumerate(data['results']):
                                logger.info(f"Article {idx+1}: {article['title']}")
                                logger.info(f"Published at: {article['published_utc']}")
                                
                                # Calculate how long ago it was published
                                try:
                                    et = pytz.timezone('US/Eastern')
                                    now = datetime.now(et)
                                    published_time = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                                    published_time_et = published_time.astimezone(et)
                                    time_diff_seconds = (now - published_time_et).total_seconds()
                                    
                                    if time_diff_seconds < 60:
                                        logger.info(f"Published {time_diff_seconds:.1f} seconds ago")
                                    elif time_diff_seconds < 3600:
                                        logger.info(f"Published {time_diff_seconds/60:.1f} minutes ago")
                                    else:
                                        logger.info(f"Published {time_diff_seconds/3600:.1f} hours ago")
                                except Exception as e:
                                    logger.error(f"Error calculating time difference: {e}")
                        else:
                            logger.info(f"No news articles found for {test_symbol} in the last 24 hours")
                    else:
                        logger.warning(f"Error fetching news for {test_symbol}: Status {response.status}")
        except Exception as e:
            logger.error(f"Error checking for older news: {e}")
    
    # Uncomment to test continuous monitoring
    # await monitor_news(test_callback)

if __name__ == "__main__":
    asyncio.run(main()) 