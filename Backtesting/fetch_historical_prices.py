#!/usr/bin/env python3
"""
Historical Price Data Fetcher
Fetches 10-second aggregate bars from Polygon API for all tickers in float_list
Only fetches data during trading hours (7am-9:30am EST)
Stores data in News.historical_price table for backtesting
"""

import asyncio
import aiohttp
import logging
import os
import sys
import time
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, Set
import pytz
from concurrent.futures import ThreadPoolExecutor
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalPriceFetcher:
    """
    Fetches historical price data from Polygon API for backtesting
    """
    
    def __init__(self, days_back: int = 180):
        self.ch_manager = None
        self.session = None
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.days_back = days_back  # How many days back to fetch data
        
        if not self.polygon_api_key:
            logger.error("POLYGON_API_KEY environment variable not set")
            raise ValueError("Polygon API key is required")
        
        # Use PROXY_URL if available
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.base_url = proxy_url.rstrip('/')
            logger.info(f"Using proxy URL: {self.base_url}")
        else:
            self.base_url = "https://api.polygon.io"
            logger.info("Using official Polygon API")
        
        # EST timezone for trading hours
        self.est_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC
        
        # Trading hours (7am-9:30am EST)
        self.trading_start_time = "07:00:00"
        self.trading_end_time = "09:30:00"
        
        # Stats tracking
        self.stats = {
            'tickers_processed': 0,
            'tickers_failed': 0,
            'total_bars_fetched': 0,
            'total_bars_stored': 0,
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time()
        }
        
        # Rate limiting
        self.rate_limit_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    async def initialize(self):
        """Initialize the price fetcher"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Drop and recreate historical_price table to avoid duplicates
            logger.info("🗑️ Dropping existing historical_price table to avoid duplicates...")
            self.ch_manager.client.command("DROP TABLE IF EXISTS News.historical_price")
            
            historical_price_table = """
            CREATE TABLE News.historical_price (
                ticker String,
                timestamp DateTime64(3, 'UTC'),
                date Date,
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume Nullable(UInt64),
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(date)
            ORDER BY (ticker, timestamp)
            SETTINGS index_granularity = 8192
            """
            
            self.ch_manager.client.command(historical_price_table)
            logger.info("✅ Created fresh historical_price table")
            
            # Create HTTP session for Polygon API
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Test Polygon API connectivity
            if not await self.test_polygon_api():
                logger.error("❌ Failed to connect to Polygon API")
                return False
            
            logger.info("✅ Historical Price Fetcher initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing price fetcher: {e}")
            return False

    async def test_polygon_api(self) -> bool:
        """Test Polygon API connectivity"""
        try:
            logger.info("🔍 Testing Polygon API connection...")
            
            # Use aggregates endpoint for testing
            test_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            test_url = f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/{test_date}/{test_date}"
            params = {'apikey': self.polygon_api_key}
            
            async with self.session.get(test_url, params=params) as response:
                if response.status == 200:
                    logger.info("✅ Polygon API connection successful")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Polygon API test failed: {response.status} - {response_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Polygon API test failed: {e}")
            return False

    async def get_tickers_from_database(self, ticker_limit: int = None, single_ticker: str = None) -> List[str]:
        """Get ticker list from float_list table"""
        try:
            if single_ticker:
                # If single ticker specified, just return that one
                logger.info(f"📊 Processing single ticker: {single_ticker}")
                return [single_ticker.upper()]
            
            # Original logic for getting all tickers
            query = """
            SELECT DISTINCT ticker 
            FROM News.float_list 
            WHERE ticker IS NOT NULL 
            AND ticker != ''
            ORDER BY ticker
            """
            
            if ticker_limit:
                query += f" LIMIT {ticker_limit}"
            
            result = self.ch_manager.client.query(query)
            tickers = [row[0] for row in result.result_rows]
            
            logger.info(f"📊 Found {len(tickers)} tickers from float_list table")
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting tickers from database: {e}")
            return []

    def get_trading_date_range(self) -> List[date]:
        """Get list of trading dates (weekdays only) for the past N days"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.days_back)
        
        trading_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Only include weekdays (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"📅 Processing {len(trading_dates)} trading days from {start_date} to {end_date}")
        return trading_dates

    def filter_trading_hours(self, bars: List[Dict[str, Any]], trade_date: date) -> List[Dict[str, Any]]:
        """Filter bars to only include trading hours (7am-9:30am EST)"""
        filtered_bars = []
        
        # Create EST timezone-aware datetime objects for the trading day
        start_time_est = self.est_tz.localize(
            datetime.combine(trade_date, datetime.strptime(self.trading_start_time, "%H:%M:%S").time())
        )
        end_time_est = self.est_tz.localize(
            datetime.combine(trade_date, datetime.strptime(self.trading_end_time, "%H:%M:%S").time())
        )
        
        # Convert to UTC for comparison
        start_time_utc = start_time_est.astimezone(self.utc_tz)
        end_time_utc = end_time_est.astimezone(self.utc_tz)
        
        for bar in bars:
            bar_timestamp = bar['timestamp']
            if start_time_utc <= bar_timestamp <= end_time_utc:
                filtered_bars.append(bar)
        
        return filtered_bars

    async def fetch_ticker_bars_for_date(self, ticker: str, trade_date: date) -> List[Dict[str, Any]]:
        """Fetch 10-second bars for a specific ticker and date"""
        async with self.rate_limit_semaphore:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.last_request_time = time.time()
            self.stats['api_calls'] += 1
            
            try:
                # Format date for Polygon API
                date_str = trade_date.strftime('%Y-%m-%d')
                
                # Polygon aggregates endpoint - 10 second bars
                url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/10/second/{date_str}/{date_str}"
                params = {
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50000,  # Get all bars for the day
                    'apikey': self.polygon_api_key
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'OK' and data.get('results'):
                            bars = []
                            for bar in data['results']:
                                bar_timestamp = datetime.fromtimestamp(bar['t'] / 1000, tz=self.utc_tz)
                                
                                bars.append({
                                    'ticker': ticker,
                                    'timestamp': bar_timestamp,
                                    'date': trade_date,
                                    'open': bar['o'],
                                    'high': bar['h'],
                                    'low': bar['l'],
                                    'close': bar['c'],
                                    'volume': bar['v']
                                })
                            
                            # Filter to trading hours only
                            filtered_bars = self.filter_trading_hours(bars, trade_date)
                            
                            logger.debug(f"📊 {ticker} {date_str}: {len(filtered_bars)} bars in trading hours (filtered from {len(bars)} total)")
                            return filtered_bars
                        else:
                            logger.debug(f"No bar data for {ticker} on {date_str}")
                            return []
                            
                    elif response.status == 429:
                        logger.warning(f"Rate limited for {ticker} on {date_str} - waiting before retry")
                        await asyncio.sleep(2)
                        return []
                    else:
                        response_text = await response.text()
                        logger.warning(f"Polygon API error for {ticker} on {date_str}: {response.status} - {response_text}")
                        self.stats['api_errors'] += 1
                        return []
                        
            except Exception as e:
                logger.error(f"Error fetching bars for {ticker} on {trade_date}: {e}")
                self.stats['api_errors'] += 1
                return []

    async def fetch_ticker_historical_data(self, ticker: str, trading_dates: List[date]) -> List[Dict[str, Any]]:
        """Fetch all historical data for a single ticker across all trading dates"""
        logger.info(f"📈 Fetching historical data for {ticker} ({len(trading_dates)} days)")
        
        all_bars = []
        
        # Process dates in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(trading_dates), batch_size):
            batch_dates = trading_dates[i:i + batch_size]
            
            # Fetch data for this batch of dates
            batch_tasks = [self.fetch_ticker_bars_for_date(ticker, trade_date) for trade_date in batch_dates]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Exception fetching {ticker} data for {batch_dates[j]}: {result}")
                    continue
                if result:
                    all_bars.extend(result)
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        self.stats['total_bars_fetched'] += len(all_bars)
        logger.info(f"✅ {ticker}: Fetched {len(all_bars)} bars across {len(trading_dates)} days")
        
        return all_bars

    async def store_price_data(self, price_data: List[Dict[str, Any]]):
        """Store price data in ClickHouse, batched by date to avoid partition issues"""
        if not price_data:
            return
        
        try:
            # Group data by date to avoid too many partitions in single insert
            data_by_date = {}
            
            for bar in price_data:
                # Validate and clean data
                volume = bar.get('volume', 0)
                if volume is None or volume < 0:
                    volume = None  # Use None for nullable column
                else:
                    volume = int(volume)  # Ensure integer type
                
                # Ensure all price fields are valid floats
                open_price = float(bar.get('open', 0.0))
                high_price = float(bar.get('high', 0.0))
                low_price = float(bar.get('low', 0.0))
                close_price = float(bar.get('close', 0.0))
                
                # Skip bars with invalid price data
                if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                    logger.debug(f"Skipping bar with invalid price data: {bar}")
                    continue
                
                # Group by date
                bar_date = bar['date']
                if bar_date not in data_by_date:
                    data_by_date[bar_date] = []
                
                data_by_date[bar_date].append((
                    bar['ticker'],
                    bar['timestamp'],
                    bar['date'],
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    datetime.now()
                ))
            
            if not data_by_date:
                logger.warning("No valid price data to insert after validation")
                return
            
            # Insert data for each date separately to avoid partition limit
            total_inserted = 0
            for trade_date, date_data in data_by_date.items():
                try:
                    self.ch_manager.client.insert(
                        'News.historical_price',
                        date_data,
                        column_names=['ticker', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'created_at']
                    )
                    total_inserted += len(date_data)
                    logger.debug(f"✅ Stored {len(date_data)} bars for {trade_date}")
                    
                except Exception as date_error:
                    logger.error(f"Error storing data for {trade_date}: {date_error}")
                    continue
            
            self.stats['total_bars_stored'] += total_inserted
            logger.info(f"✅ Stored {total_inserted} price bars across {len(data_by_date)} dates")
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
            # Log some sample data for debugging
            if price_data:
                logger.debug(f"Sample data that failed: {price_data[0]}")

    async def process_ticker(self, ticker: str, trading_dates: List[date]):
        """Process a single ticker - fetch and store all its historical data"""
        try:
            # Fetch all historical data for this ticker
            price_data = await self.fetch_ticker_historical_data(ticker, trading_dates)
            
            if price_data:
                # Store in database
                await self.store_price_data(price_data)
                logger.info(f"✅ COMPLETED: {ticker} - {len(price_data)} bars stored")
            else:
                logger.warning(f"⚠️ NO DATA: {ticker} - no price data found")
            
            self.stats['tickers_processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ FAILED: {ticker} - {e}")
            self.stats['tickers_failed'] += 1

    async def run_historical_price_fetch(self, ticker_limit: int = None, single_ticker: str = None):
        """Run the complete historical price fetching process"""
        try:
            logger.info("🚀 Starting Historical Price Data Fetching...")
            
            # Initialize 
            if not await self.initialize():
                logger.error("Failed to initialize price fetcher")
                return False
            
            # Get ticker list
            tickers = await self.get_tickers_from_database(ticker_limit=ticker_limit, single_ticker=single_ticker)
            if not tickers:
                logger.error("No tickers found to process")
                return False
            
            # Process all tickers
            logger.info(f"📊 Processing {len(tickers)} tickers for {self.days_back} days of data...")
            
            # Get trading date range
            trading_dates = self.get_trading_date_range()
            if not trading_dates:
                logger.error("No trading dates found")
                return False
            
            logger.info(f"📊 Processing {len(tickers)} tickers for {len(trading_dates)} trading days")
            logger.info(f"📊 Expected API calls: ~{len(tickers) * len(trading_dates)}")
            
            # Process tickers with controlled concurrency
            semaphore = asyncio.Semaphore(3)  # Limit concurrent ticker processing
            
            async def process_with_semaphore(ticker):
                async with semaphore:
                    await self.process_ticker(ticker, trading_dates)
            
            # Process all tickers
            ticker_tasks = [process_with_semaphore(ticker) for ticker in tickers]
            await asyncio.gather(*ticker_tasks, return_exceptions=True)
            
            # Final statistics
            elapsed = time.time() - self.stats['start_time']
            logger.info("🎉 HISTORICAL PRICE FETCHING COMPLETE!")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Tickers processed: {self.stats['tickers_processed']}")
            logger.info(f"  • Tickers failed: {self.stats['tickers_failed']}")
            logger.info(f"  • Total bars fetched: {self.stats['total_bars_fetched']}")
            logger.info(f"  • Total bars stored: {self.stats['total_bars_stored']}")
            logger.info(f"  • API calls made: {self.stats['api_calls']}")
            logger.info(f"  • API errors: {self.stats['api_errors']}")
            logger.info(f"  • Time elapsed: {elapsed/60:.1f} minutes")
            logger.info(f"  • Average bars per ticker: {self.stats['total_bars_stored']/max(self.stats['tickers_processed'], 1):.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in historical price fetching: {e}")
            return False
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("✅ Historical price fetcher cleanup completed")

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch Historical Price Data for Backtesting')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process (for testing)')
    parser.add_argument('--days', type=int, default=180, help='Number of days back to fetch data (default: 180)')
    parser.add_argument('--ticker', type=str, help='Process a single ticker (e.g., AAPL)')
    
    args = parser.parse_args()
    
    # Show what will be processed
    if args.ticker:
        print(f"🎯 Processing single ticker: {args.ticker.upper()}")
    elif args.limit:
        print(f"🎯 Processing up to {args.limit} tickers")
    else:
        print("🎯 Processing all tickers from float_list table")
    
    print(f"📅 Fetching {args.days} days of historical data")
    print()
    
    fetcher = HistoricalPriceFetcher(days_back=args.days)
    success = await fetcher.run_historical_price_fetch(ticker_limit=args.limit, single_ticker=args.ticker)
    
    if success:
        print("\n✅ Historical price data fetching completed successfully!")
    else:
        print("\n❌ Historical price data fetching failed!")

if __name__ == "__main__":
    asyncio.run(main()) 