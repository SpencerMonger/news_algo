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
import time
from typing import List, Dict, Any, Optional
from clickhouse_setup import ClickHouseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolygonPriceChecker:
    def __init__(self, clickhouse_manager: ClickHouseManager):
        self.ch_manager = clickhouse_manager
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.session = None
        
        if not self.polygon_api_key:
            logger.error("POLYGON_API_KEY environment variable not set")
            raise ValueError("Polygon API key is required")
        
        # Use PROXY_URL if available, otherwise use official Polygon API
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.base_url = proxy_url.rstrip('/')
            logger.info(f"Using proxy URL: {self.base_url}")
        else:
            self.base_url = "https://api.polygon.io"
            logger.info("Using official Polygon API")
        
    async def initialize(self):
        """Initialize the price checker"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("Polygon price checker initialized")

    async def get_previous_day_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get previous day's daily bar data for baseline comparison"""
        try:
            # Get yesterday's date for the daily bar
            today = datetime.now(pytz.UTC)
            yesterday = today - timedelta(days=1)
            
            # Format date for Polygon API
            yesterday_date = yesterday.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{yesterday_date}/{yesterday_date}"
            params = {
                'apikey': self.polygon_api_key,
                'adjusted': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('resultsCount', 0) > 0 and 'results' in data:
                        result = data['results'][0]
                        parsed_data = {
                            'previous_close': result.get('c', 0.0),
                            'previous_high': result.get('h', 0.0),
                            'previous_low': result.get('l', 0.0),
                            'previous_open': result.get('o', 0.0),
                            'previous_volume': result.get('v', 0),
                            'previous_date': yesterday_date,
                            'timestamp': datetime.fromtimestamp(result.get('t', 0) / 1000, tz=pytz.UTC)
                        }
                        logger.info(f"Previous day data for {ticker}: Close=${parsed_data['previous_close']}, Volume={parsed_data['previous_volume']:,}")
                        return parsed_data
                    else:
                        logger.warning(f"No previous day data available for {ticker}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get previous day data for {ticker}: HTTP {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting previous day data for {ticker}: {e}")
            return None

    async def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current/latest price for ticker"""
        try:
            # Use the last trade endpoint for most recent price
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {
                'apikey': self.polygon_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'results' in data and data['results']:
                        result = data['results']
                        parsed_data = {
                            'price': result.get('p', 0.0),
                            'timestamp': datetime.fromtimestamp(result.get('t', 0) / 1000000000, tz=pytz.UTC),  # nanoseconds to seconds
                            'size': result.get('s', 0)
                        }
                        return parsed_data
                    else:
                        logger.warning(f"No current price data available for {ticker}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get current price for {ticker}: HTTP {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return None

    async def get_current_day_volume(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current day's volume data for volume spike detection"""
        try:
            # Get today's date for the daily bar (partial day)
            today = datetime.now(pytz.UTC)
            today_date = today.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{today_date}/{today_date}"
            params = {
                'apikey': self.polygon_api_key,
                'adjusted': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('resultsCount', 0) > 0 and 'results' in data:
                        result = data['results'][0]
                        parsed_data = {
                            'current_volume': result.get('v', 0),
                            'current_high': result.get('h', 0.0),
                            'current_low': result.get('l', 0.0),
                            'current_open': result.get('o', 0.0),
                            'date': today_date,
                            'timestamp': datetime.fromtimestamp(result.get('t', 0) / 1000, tz=pytz.UTC)
                        }
                        return parsed_data
                    else:
                        logger.warning(f"No current day volume data available for {ticker}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get current day volume for {ticker}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current day volume for {ticker}: {e}")
            return None

    async def check_price_move(self, ticker: str, news_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if current price is above previous day's close with volume spike"""
        start_time = time.time()
        
        try:
            # Get current price, previous day data, and current day volume concurrently
            current_price_task = self.get_current_price(ticker)
            previous_day_task = self.get_previous_day_data(ticker)
            current_volume_task = self.get_current_day_volume(ticker)
            
            current_price_data, previous_day_data, current_volume_data = await asyncio.gather(
                current_price_task, previous_day_task, current_volume_task, return_exceptions=True
            )
            
            # Check for errors
            if isinstance(current_price_data, Exception):
                logger.error(f"Error getting current price for {ticker}: {current_price_data}")
                return None
            if isinstance(previous_day_data, Exception):
                logger.error(f"Error getting previous day data for {ticker}: {previous_day_data}")
                return None
            if isinstance(current_volume_data, Exception):
                logger.debug(f"Could not get current volume for {ticker}: {current_volume_data}")
                current_volume_data = None
                
            if not current_price_data or not previous_day_data:
                return None
            
            current_price = current_price_data['price']
            previous_close = previous_day_data['previous_close']
            previous_volume = previous_day_data['previous_volume']
            
            # Calculate price change percentage
            price_change_pct = ((current_price - previous_close) / previous_close) * 100
            
            # Check volume spike if we have current volume data
            volume_spike_detected = False
            volume_change_pct = 0.0
            current_volume = 0
            
            if current_volume_data:
                current_volume = current_volume_data['current_volume']
                if previous_volume > 0:
                    volume_change_pct = ((current_volume - previous_volume) / previous_volume) * 100
                    # Require at least 50% volume increase for spike detection
                    volume_spike_detected = volume_change_pct >= 50.0
            
            # Check if current price is above previous day's close by meaningful amount
            min_price_increase_pct = 2.0  # Require at least 2% price increase
            
            # Trigger if we have both price move AND volume spike (or if volume data unavailable)
            price_trigger = price_change_pct >= min_price_increase_pct
            volume_trigger = volume_spike_detected or current_volume_data is None  # Allow if no volume data
            
            if price_trigger and volume_trigger:
                # Calculate timing metrics
                price_check_latency_ms = int((time.time() - start_time) * 1000)
                news_to_price_check_delay = datetime.now(pytz.UTC) - news_data['detected_at']
                news_to_price_check_delay_ms = int(news_to_price_check_delay.total_seconds() * 1000)
                
                move_data = {
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'news_headline': news_data['headline'],
                    'news_published_utc': news_data['published_utc'],
                    'news_article_url': news_data['article_url'],
                    'current_price': current_price,
                    'current_price_timestamp': current_price_data['timestamp'],
                    'previous_close': previous_day_data['previous_close'],
                    'previous_high': previous_day_data['previous_high'],
                    'previous_low': previous_day_data['previous_low'],
                    'previous_open': previous_day_data['previous_open'],
                    'previous_volume': previous_day_data['previous_volume'],
                    'current_volume': current_volume,
                    'previous_date': previous_day_data['previous_date'],
                    'price_change_percentage': price_change_pct,
                    'volume_change_percentage': volume_change_pct,
                    'volume_spike_detected': 1 if volume_spike_detected else 0,
                    'price_above_previous_close': 1,
                    'price_check_latency_ms': price_check_latency_ms,
                    'news_to_price_check_delay_ms': news_to_price_check_delay_ms
                }
                
                volume_info = f", Vol: {current_volume:,} vs {previous_volume:,} ({volume_change_pct:+.1f}%)" if current_volume_data else ", Vol: N/A"
                logger.info(f"PRICE MOVE DETECTED: {ticker} - Current: ${current_price:.4f} vs Previous Close: ${previous_close:.4f} (+{price_change_pct:.2f}%){volume_info}")
                return move_data
            else:
                reason = []
                if not price_trigger:
                    reason.append(f"price {price_change_pct:+.2f}% < {min_price_increase_pct}%")
                if not volume_trigger and current_volume_data:
                    reason.append(f"volume {volume_change_pct:+.1f}% < 50%")
                
                logger.debug(f"No trigger for {ticker}: {', '.join(reason)}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking price move for {ticker}: {e}")
            return None

    async def get_recent_news_tickers(self, minutes_back: int = 5) -> List[Dict[str, Any]]:
        """Get unique tickers from recent breaking news"""
        try:
            query = f"""
            SELECT DISTINCT 
                ticker,
                headline,
                published_utc,
                article_url,
                detected_at
            FROM News.breaking_news 
            WHERE detected_at >= now() - INTERVAL {minutes_back} MINUTE
            ORDER BY detected_at DESC
            """
            
            result = self.ch_manager.client.query(query)
            
            news_items = []
            for row in result.result_rows:
                news_items.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'published_utc': row[2],
                    'article_url': row[3],
                    'detected_at': row[4]
                })
            
            logger.info(f"Found {len(news_items)} recent news items for price checking")
            return news_items
            
        except Exception as e:
            logger.error(f"Error getting recent news tickers: {e}")
            return []

    async def process_recent_news(self, minutes_back: int = 5) -> int:
        """Process recent news items and check for price moves"""
        try:
            # Get recent news tickers
            news_items = await self.get_recent_news_tickers(minutes_back)
            
            if not news_items:
                return 0
            
            # Check price moves for each ticker
            price_moves = []
            for news_item in news_items:
                move_data = await self.check_price_move(news_item['ticker'], news_item)
                if move_data:
                    price_moves.append(move_data)
                    
                # Small delay to avoid hitting rate limits
                await asyncio.sleep(0.1)
            
            # Insert price moves into database
            if price_moves:
                inserted_count = self.ch_manager.insert_price_moves(price_moves)
                logger.info(f"Inserted {inserted_count} price moves into database")
                return inserted_count
            else:
                logger.info("No price moves detected for recent news")
                return 0
                
        except Exception as e:
            logger.error(f"Error processing recent news: {e}")
            return 0

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            logger.info("Price checker session closed")

async def main():
    """Test the price checker"""
    # Setup ClickHouse
    from clickhouse_setup import setup_clickhouse_database
    ch_manager = setup_clickhouse_database()
    
    # Create and test price checker
    price_checker = PolygonPriceChecker(ch_manager)
    
    try:
        await price_checker.initialize()
        
        # Test processing recent news
        result = await price_checker.process_recent_news(minutes_back=60)  # Test with 1 hour back
        logger.info(f"Processed {result} price moves")
        
    except KeyboardInterrupt:
        logger.info("Price checker stopped by user")
    finally:
        await price_checker.cleanup()
        ch_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
