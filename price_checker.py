#!/usr/bin/env python3
"""
Continuous Price Monitor - OPTIMIZED for minimal latency
Monitors breaking news tickers and tracks price changes in real-time
"""

import logging
import aiohttp
import os
import asyncio
import time
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Set
from clickhouse_setup import ClickHouseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContinuousPriceMonitor:
    def __init__(self):
        self.ch_manager = None
        self.session = None
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.active_tickers: Set[str] = set()  # Track tickers directly from breaking_news
        
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
        
        # Stats
        self.stats = {
            'tickers_monitored': 0,
            'price_checks': 0,
            'alerts_triggered': 0,
            'start_time': time.time()
        }

    async def initialize(self):
        """Initialize the monitor"""
        # FIXED: Don't call setup_clickhouse_database() as it WIPES breaking_news table!
        # Instead, create direct connection and only create needed tables
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        
        # Only create essential tables (no monitored_tickers table needed)
        await self.create_essential_tables()
        
        # OPTIMIZED: Faster timeout for quicker API responses
        timeout = aiohttp.ClientTimeout(total=10, connect=3)  # Reduced from 30s to 10s
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("🚀 Price monitor initialized - PRESERVES existing breaking_news data")

    async def create_essential_tables(self):
        """Create only essential tables for optimized flow"""
        try:
            # Only need price_tracking and news_alert tables
            price_tracking_sql = """
            CREATE TABLE IF NOT EXISTS News.price_tracking (
                timestamp DateTime DEFAULT now(),
                ticker String,
                price Float64,
                volume UInt64,
                source String DEFAULT 'polygon'
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 7 DAY
            """
            self.ch_manager.client.command(price_tracking_sql)
            logger.info("Created price_tracking table")

            news_alert_sql = """
            CREATE TABLE IF NOT EXISTS News.news_alert (
                ticker String,
                timestamp DateTime DEFAULT now(),
                alert UInt8 DEFAULT 1,
                price Float64
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 30 DAY
            """
            self.ch_manager.client.command(news_alert_sql)
            logger.info("Created news_alert table")
            
        except Exception as e:
            logger.error(f"Error creating essential tables: {e}")
            raise

    async def get_active_tickers_from_breaking_news(self) -> Set[str]:
        """Get tickers directly from breaking_news table - ELIMINATES monitored_tickers bottleneck"""
        try:
            # OPTIMIZED: Get tickers from recent news with minimal data transfer
            query = """
            SELECT DISTINCT ticker
            FROM News.breaking_news 
            WHERE timestamp >= now() - INTERVAL 5 MINUTE
            AND ticker != ''
            ORDER BY ticker
            """
            
            result = self.ch_manager.client.query(query)
            current_tickers = {row[0] for row in result.result_rows}
            
            # Only log detailed info if debug level
            if logger.isEnabledFor(logging.DEBUG) and result.result_rows:
                logger.debug(f"📰 Found {len(current_tickers)} active tickers in last 5 minutes")
            
            # Update active tickers
            self.active_tickers = current_tickers
            return current_tickers
                
        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            return set()

    async def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current price for ticker"""
        try:
            # Use the real-time quote endpoint instead of last trade
            url = f"{self.base_url}/v2/last/nbbo/{ticker}"
            params = {'apikey': self.polygon_api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'results' in data and data['results']:
                        result = data['results']
                        # Use bid/ask midpoint for current price
                        bid = result.get('P', 0.0)  # bid price
                        ask = result.get('p', 0.0)  # ask price
                        
                        if bid > 0 and ask > 0:
                            current_price = (bid + ask) / 2
                            return {
                                'price': current_price,
                                'bid': bid,
                                'ask': ask,
                                'timestamp': datetime.now(pytz.UTC)
                            }
                else:
                    logger.debug(f"API returned status {response.status} for {ticker}")
                    
        except Exception as e:
            logger.debug(f"Error getting price for {ticker}: {e}")
        
        return None

    async def track_prices_parallel(self):
        """Get current prices for all active tickers in PARALLEL - much faster"""
        if not self.active_tickers:
            return
        
        # Create parallel price fetching tasks
        price_tasks = [self.get_current_price(ticker) for ticker in self.active_tickers]
        
        # Execute all price requests in parallel
        price_results = await asyncio.gather(*price_tasks, return_exceptions=True)
        
        # Process results and prepare batch insert
        price_data = []
        successful_prices = 0
        
        for i, (ticker, price_result) in enumerate(zip(self.active_tickers, price_results)):
            if isinstance(price_result, Exception):
                logger.debug(f"Error getting price for {ticker}: {price_result}")
                continue
                
            if price_result:
                price_data.append((
                    datetime.now(),
                    ticker,
                    price_result['price'],
                    0,  # Set volume to 0 since we're using quotes not trades
                    'polygon'
                ))
                successful_prices += 1
        
        # Batch insert price data
        if price_data:
            self.ch_manager.client.insert(
                'News.price_tracking',
                price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
            )
            
            self.stats['price_checks'] += len(price_data)
            logger.info(f"⚡ PARALLEL: Tracked {successful_prices}/{len(self.active_tickers)} ticker prices")
        else:
            logger.warning("No price data retrieved for any tickers")

    async def check_price_alerts_optimized(self):
        """Check for 5%+ price increases - IMMEDIATE trigger within 2 minutes of first price"""
        try:
            if not self.active_tickers:
                return
                
            # Convert set to list for SQL IN clause
            ticker_list = list(self.active_tickers)
            ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
            
            # OPTIMIZED: Compare current price to FIRST price recorded, but ONLY within 2 minutes
            # This captures immediate breaking news moves and ignores delayed reactions
            query = f"""
            SELECT 
                ticker,
                current_price,
                first_price,
                ((current_price - first_price) / first_price) * 100 as change_pct,
                price_count,
                first_timestamp,
                current_timestamp,
                dateDiff('second', first_timestamp, current_timestamp) as seconds_elapsed
            FROM (
                SELECT 
                    ticker,
                    argMax(price, timestamp) as current_price,
                    argMin(price, timestamp) as first_price,
                    argMax(timestamp, timestamp) as current_timestamp,
                    argMin(timestamp, timestamp) as first_timestamp,
                    count() as price_count
                FROM News.price_tracking 
                WHERE timestamp >= now() - INTERVAL 15 MINUTE
                AND ticker IN ({ticker_placeholders})
                GROUP BY ticker
                HAVING first_price > 0 AND price_count >= 2
            )
            WHERE change_pct >= 5.0 
            AND seconds_elapsed <= 120
            ORDER BY change_pct DESC
            """
            
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                # Prepare batch insert for news_alert table
                alert_data = []
                
                for row in result.result_rows:
                    ticker, current_price, first_price, change_pct, price_count, first_timestamp, current_timestamp, seconds_elapsed = row
                    
                    logger.info(f"🚨 IMMEDIATE ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from first price ${first_price:.4f}) in {seconds_elapsed}s [{price_count} price points]")
                    
                    # Add to alert data for batch insert
                    alert_data.append((ticker, datetime.now(), 1, current_price))
                    
                    # Log to price_move table
                    await self.log_price_alert(ticker, current_price, first_price, change_pct)
                    
                    self.stats['alerts_triggered'] += 1
                
                # Batch insert all alerts
                if alert_data:
                    self.ch_manager.client.insert(
                        'News.news_alert',
                        alert_data,
                        column_names=['ticker', 'timestamp', 'alert', 'price']
                    )
                    logger.info(f"✅ BREAKING NEWS ALERTS: Inserted {len(alert_data)} immediate alerts (within 2min window)")
                
        except Exception as e:
            logger.error(f"Error checking price alerts: {e}")

    async def log_price_alert(self, ticker: str, current_price: float, prev_price: float, change_pct: float):
        """Log price alert to database"""
        try:
            # Get latest news for this ticker
            news_query = """
            SELECT headline, article_url, published_utc
            FROM News.breaking_news 
            WHERE ticker = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            news_result = self.ch_manager.client.query(news_query, parameters=[ticker])
            
            if news_result.result_rows:
                headline, url, published_utc = news_result.result_rows[0]
            else:
                headline, url, published_utc = "No recent news", "", datetime.now()
            
            # Insert price move
            values = [(
                datetime.now(),  # timestamp
                ticker,          # ticker
                headline,        # headline
                published_utc,   # published_utc
                url,             # article_url
                current_price,   # latest_price
                prev_price,      # previous_close
                change_pct,      # price_change_percentage
                0,               # volume_change_percentage
                datetime.now()   # detected_at
            )]
            
            self.ch_manager.client.insert(
                'News.price_move',
                values,
                column_names=['timestamp', 'ticker', 'headline', 'published_utc', 'article_url', 'latest_price', 'previous_close', 'price_change_percentage', 'volume_change_percentage', 'detected_at']
            )
            
        except Exception as e:
            logger.error(f"Error logging price alert: {e}")

    async def report_stats(self):
        """Report monitoring statistics"""
        runtime = time.time() - self.stats['start_time']
        
        logger.info(f"📊 OPTIMIZED MONITOR STATS:")
        logger.info(f"   Runtime: {runtime/60:.1f} minutes")
        logger.info(f"   Active Tickers: {len(self.active_tickers)}")
        logger.info(f"   Price Checks: {self.stats['price_checks']}")
        logger.info(f"   Alerts Triggered: {self.stats['alerts_triggered']}")

    async def ultra_fast_monitoring_loop(self):
        """OPTIMIZED monitoring loop - eliminates all unnecessary delays"""
        logger.info("🚀 Starting ULTRA-FAST price monitoring - IMMEDIATE first check, then 1 SECOND CYCLES!")
        
        cycle = 0
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                # Get active tickers directly from breaking_news (no intermediate table)
                previous_tickers = self.active_tickers.copy()
                await self.get_active_tickers_from_breaking_news()
                
                # Check if we have new tickers - if so, do IMMEDIATE price check
                new_tickers = self.active_tickers - previous_tickers
                if new_tickers:
                    logger.info(f"🎯 NEW TICKERS DETECTED: {new_tickers} - IMMEDIATE PRICE CHECK!")
                    # Don't wait - do immediate price tracking for new tickers
                    await self.track_prices_parallel()
                    await self.check_price_alerts_optimized()
                elif self.active_tickers:
                    # Regular price tracking for existing tickers
                    await self.track_prices_parallel()
                    await self.check_price_alerts_optimized()
                else:
                    logger.debug(f"⏳ Cycle {cycle}: No active tickers to monitor")
                
                # Report stats every minute (60 cycles at 1s intervals)
                if cycle % 60 == 0:
                    await self.report_stats()
                
                cycle_time = time.time() - cycle_start
                logger.debug(f"⚡ Cycle {cycle} completed in {cycle_time:.3f}s")
                
                # IMMEDIATE first check, then 1-second intervals for ULTRA-FAST response
                if cycle == 1:
                    logger.info("⚡ IMMEDIATE FIRST CHECK COMPLETE - Now switching to 1-second intervals")
                    await asyncio.sleep(0.05)  # Just a tiny pause, then immediate next check
                elif new_tickers:
                    # If we found new tickers, do another immediate check
                    logger.info("⚡ NEW TICKERS - IMMEDIATE NEXT CHECK!")
                    await asyncio.sleep(0.1)  # Very short pause for new ticker immediate response
                else:
                    # Sleep for remainder of 1-second interval
                    sleep_time = max(0, 1.0 - cycle_time)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("Price monitor cleanup completed")

    async def start(self):
        """Start the optimized monitor"""
        try:
            await self.initialize()
            await self.ultra_fast_monitoring_loop()
        except KeyboardInterrupt:
            logger.info("Price monitor stopped by user")
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    logger.info("🚀 Starting OPTIMIZED Continuous Price Monitor")
    logger.info("⚡ ULTRA-FAST: Direct breaking_news → price_tracking → news_alert")
    
    monitor = ContinuousPriceMonitor()
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
