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
PROXY_URL = os.getenv('PROXY_URL')
API_KEY = os.getenv('POLYGON_API_KEY')

async def get_latest_trade(symbol):
    """
    Get the latest trade for a symbol
    """
    try:
        # Construct URL for latest trade endpoint
        url = f"http://3.128.134.41/v2/last/trade/{symbol}"
        
        params = {
            'apiKey': API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, proxy=PROXY_URL) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results'):
                        logger.info(f"Latest trade for {symbol}: ${data['results']['p']}")
                        return data['results']
                    else:
                        logger.warning(f"No trade data found for {symbol}")
                        return None
                else:
                    logger.warning(f"Error fetching latest trade for {symbol}: Status {response.status}")
                    return None
                
    except Exception as e:
        logger.error(f"Error fetching latest trade for {symbol}: {e}")
        return None

async def get_five_minute_bar(symbol):
    """
    Get the latest 5-minute aggregate bar for a symbol
    """
    try:
        # Calculate timestamp for current 5-minute bar
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        
        # Round down to the nearest 5-minute interval
        minutes = now.minute - (now.minute % 5)
        bar_start = now.replace(minute=minutes, second=0, microsecond=0) - timedelta(minutes=5)
        bar_end = now.replace(minute=minutes, second=0, microsecond=0)
        
        # Convert to milliseconds for API
        start_ms = int(bar_start.timestamp() * 1000)
        end_ms = int(bar_end.timestamp() * 1000)
        
        # Construct URL for aggregates endpoint
        url = f"http://3.128.134.41/v2/aggs/ticker/{symbol}/range/5/minute/{start_ms}/{end_ms}"
        
        params = {
            'adjusted': "true",
            'sort': 'desc',
            'limit': 1,
            'apiKey': API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, proxy=PROXY_URL) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results') and len(data['results']) > 0:
                        bar_data = data['results'][0]
                        logger.info(f"5-min bar for {symbol}: High ${bar_data['h']}, Close ${bar_data['c']}")
                        return bar_data
                    else:
                        logger.warning(f"No 5-minute bar data found for {symbol}")
                        return None
                else:
                    logger.warning(f"Error fetching 5-minute bar for {symbol}: Status {response.status}")
                    return None
                
    except Exception as e:
        logger.error(f"Error fetching 5-minute bar for {symbol}: {e}")
        return None

async def create_trigger(symbol, news_data, trade_data, bar_data):
    """
    Create a trigger for the websocket function
    """
    trigger_data = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "news": {
            "title": news_data['news_data']['results'][0]['title'],
            "published_utc": news_data['news_data']['results'][0]['published_utc'],
            "article_url": news_data['news_data']['results'][0]['article_url']
        },
        "price_data": {
            "latest_price": trade_data['p'],
            "latest_trade_time": trade_data['t'],
            "bar_high": bar_data['h'],
            "bar_low": bar_data['l'],
            "bar_open": bar_data['o'],
            "bar_close": bar_data['c'],
            "bar_volume": bar_data['v']
        }
    }
    
    logger.info(f"TRIGGER CREATED for {symbol}: Latest price ${trade_data['p']} > Bar high ${bar_data['h']}")
    
    # Here you would send this trigger to your websocket function
    # For now, we'll just save it to a file for testing
    try:
        os.makedirs('triggers', exist_ok=True)
        filename = f"triggers/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(trigger_data, f, indent=2)
        logger.info(f"Trigger saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving trigger: {e}")
    
    return trigger_data

async def check_price_on_news(news_data):
    """
    Main function to check price when news is detected
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
            await create_trigger(symbol, news_data, trade_data, bar_data)
        else:
            logger.info(f"No trigger for {symbol}: Latest price ${trade_data['p']} <= Bar high ${bar_data['h']}")
            
    except Exception as e:
        logger.error(f"Error in price checker: {e}")

async def main():
    """
    Main function for testing
    """
    logger.info("Starting price checker test")
    
    # Test with a mock news data
    test_symbol = os.getenv('SYMBOL', 'AMD')
    mock_news_data = {
        'symbol': test_symbol,
        'news_data': {
            'results': [
                {
                    'title': 'Test News Article',
                    'published_utc': datetime.now().isoformat(),
                    'article_url': 'https://example.com/test-article'
                }
            ]
        }
    }
    
    await check_price_on_news(mock_news_data)

if __name__ == "__main__":
    asyncio.run(main())
