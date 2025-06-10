#!/usr/bin/env python3
"""
Continuous Price Monitor
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
from typing import List, Dict, Any, Optional
from clickhouse_setup import setup_clickhouse_database

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
        self.monitored_tickers = set()
        
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
        self.ch_manager = setup_clickhouse_database()
        
        # Drop and recreate monitoring tables for fresh start
        await self.reset_monitoring_tables()
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("Continuous price monitor initialized")

    async def reset_monitoring_tables(self):
        """Drop and recreate monitoring tables for a fresh start"""
        try:
            logger.info("Resetting monitoring tables...")
            
            # Drop the monitoring tables
            tables_to_drop = ['monitored_tickers', 'price_tracking', 'news_alert']
            
            for table in tables_to_drop:
                try:
                    self.ch_manager.client.command(f"DROP TABLE IF EXISTS News.{table}")
                    logger.info(f"Dropped table: {table}")
                except Exception as e:
                    logger.warning(f"Error dropping table {table}: {e}")
            
            # Recreate the tables
            await self.recreate_monitoring_tables()
            
        except Exception as e:
            logger.error(f"Error resetting monitoring tables: {e}")
            raise

    async def recreate_monitoring_tables(self):
        """Recreate the monitoring tables"""
        try:
            # Recreate monitored_tickers table
            monitored_tickers_sql = """
            CREATE TABLE IF NOT EXISTS News.monitored_tickers (
                ticker String,
                first_seen DateTime DEFAULT now(),
                news_headline String,
                news_url String,
                active UInt8 DEFAULT 1,
                last_updated DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(last_updated)
            ORDER BY ticker
            """
            self.ch_manager.client.command(monitored_tickers_sql)
            logger.info("Recreated monitored_tickers table")

            # Recreate price_tracking table
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
            logger.info("Recreated price_tracking table")

            # Recreate news_alert table
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
            logger.info("Recreated news_alert table")
            
            logger.info("All monitoring tables reset successfully")
            
        except Exception as e:
            logger.error(f"Error recreating monitoring tables: {e}")
            raise

    async def scan_for_new_tickers(self):
        """Scan breaking_news for new tickers to monitor"""
        try:
            # Get tickers from recent news (last 2 hours)
            query = """
            SELECT DISTINCT 
                ticker,
                argMax(headline, timestamp) as latest_headline,
                argMax(article_url, timestamp) as latest_url
            FROM News.breaking_news 
            WHERE timestamp >= now() - INTERVAL 2 HOUR
            AND ticker != ''
            GROUP BY ticker
            """
            
            result = self.ch_manager.client.query(query)
            
            new_tickers = 0
            for row in result.result_rows:
                ticker, headline, url = row
                
                if ticker not in self.monitored_tickers:
                    # Add to monitored_tickers table
                    values = [(ticker, headline, url)]
                    
                    self.ch_manager.client.insert(
                        'News.monitored_tickers',
                        values,
                        column_names=['ticker', 'news_headline', 'news_url']
                    )
                    
                    self.monitored_tickers.add(ticker)
                    new_tickers += 1
                    logger.info(f"Added {ticker} to monitoring list")
            
            if new_tickers > 0:
                self.stats['tickers_monitored'] = len(self.monitored_tickers)
                logger.info(f"Added {new_tickers} new tickers. Total monitoring: {len(self.monitored_tickers)}")
                
        except Exception as e:
            logger.error(f"Error scanning for new tickers: {e}")

    async def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current price for ticker"""
        try:
            # Use the real-time quote endpoint instead of last trade
            url = f"{self.base_url}/v2/last/nbbo/{ticker}"
            params = {'apikey': self.polygon_api_key}
            
            # Add timeout to prevent hanging (using wait_for for compatibility)
            try:
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
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting price for {ticker}")
                    
        except Exception as e:
            logger.debug(f"Error getting price for {ticker}: {e}")
        
        return None

    async def track_prices(self):
        """Get current prices for all monitored tickers and store them"""
        if not self.monitored_tickers:
            return
        
        price_data = []
        successful_prices = 0
        
        for ticker in self.monitored_tickers:
            price_info = await self.get_current_price(ticker)
            
            if price_info:
                price_data.append((
                    datetime.now(),
                    ticker,
                    price_info['price'],
                    0,  # Set volume to 0 since we're using quotes not trades
                    'polygon'
                ))
                successful_prices += 1
                
            await asyncio.sleep(0.1)  # Rate limiting
        
        # Batch insert price data
        if price_data:
            self.ch_manager.client.insert(
                'News.price_tracking',
                price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
            )
            
            self.stats['price_checks'] += len(price_data)
            logger.info(f"Tracked {successful_prices}/{len(self.monitored_tickers)} ticker prices")
        else:
            logger.warning("No price data retrieved for any tickers")

    async def check_price_alerts(self):
        """Check for 5%+ price increases from the lowest price in last 5 minutes"""
        try:
            # Query for price changes from lowest price in last 5 minutes
            query = """
            SELECT 
                ticker,
                current_price,
                min_price_5min,
                ((current_price - min_price_5min) / min_price_5min) * 100 as change_pct
            FROM (
                SELECT 
                    ticker,
                    argMax(price, timestamp) as current_price,
                    min(price) as min_price_5min
                FROM News.price_tracking 
                WHERE timestamp >= now() - INTERVAL 5 MINUTE
                AND ticker IN (SELECT ticker FROM News.monitored_tickers WHERE active = 1)
                GROUP BY ticker
                HAVING min_price_5min > 0
            )
            WHERE change_pct >= 5.0
            ORDER BY change_pct DESC
            """
            
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                # Prepare batch insert for news_alert table
                alert_data = []
                
                for row in result.result_rows:
                    ticker, current_price, min_price_5min, change_pct = row
                    
                    logger.info(f"🚨 PRICE ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from 5min ago)")
                    
                    # Add to alert data for batch insert
                    alert_data.append((ticker, datetime.now(), 1, current_price))
                    
                    # Log to price_move table
                    await self.log_price_alert(ticker, current_price, min_price_5min, change_pct)
                    
                    self.stats['alerts_triggered'] += 1
                
                # Batch insert all alerts
                if alert_data:
                    self.ch_manager.client.insert(
                        'News.news_alert',
                        alert_data,
                        column_names=['ticker', 'timestamp', 'alert', 'price']
                    )
                    logger.info(f"Inserted {len(alert_data)} alerts into news_alert table")
                
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
            
            # Insert price move - using correct column names for the price_move table
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

    async def load_existing_monitored_tickers(self):
        """Load existing monitored tickers from database"""
        try:
            logger.debug("Querying monitored_tickers table...")
            query = "SELECT ticker FROM News.monitored_tickers WHERE active = 1"
            result = self.ch_manager.client.query(query)
            
            logger.debug(f"Query returned {len(result.result_rows)} rows")
            for row in result.result_rows:
                self.monitored_tickers.add(row[0])
            
            logger.info(f"Loaded {len(self.monitored_tickers)} existing monitored tickers")
            logger.debug("Finished loading monitored tickers")
            
        except Exception as e:
            logger.error(f"Error loading monitored tickers: {e}")
            raise  # Re-raise to see if this is causing the hang

    async def report_stats(self):
        """Report monitoring statistics"""
        runtime = time.time() - self.stats['start_time']
        
        logger.info(f"📊 MONITOR STATS:")
        logger.info(f"   Runtime: {runtime/60:.1f} minutes")
        logger.info(f"   Tickers Monitored: {self.stats['tickers_monitored']}")
        logger.info(f"   Price Checks: {self.stats['price_checks']}")
        logger.info(f"   Alerts Triggered: {self.stats['alerts_triggered']}")

    async def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting continuous price monitoring...")
        
        # Load existing monitored tickers
        await self.load_existing_monitored_tickers()
        
        cycle = 0
        
        while True:
            try:
                cycle += 1
                
                # Check for new tickers every 10 seconds (every 2 cycles)
                if cycle % 2 == 0:
                    await self.scan_for_new_tickers()
                    logger.info(f"Cycle {cycle}: Scanned for new tickers, monitoring {len(self.monitored_tickers)} total")
                
                # Track prices every 5 seconds
                await self.track_prices()
                await self.check_price_alerts()
                
                # Report stats every 2 minutes (24 cycles)
                if cycle % 24 == 0:
                    await self.report_stats()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("Price monitor cleanup completed")

    async def start(self):
        """Start the monitor"""
        try:
            await self.initialize()
            await self.monitoring_loop()
        except KeyboardInterrupt:
            logger.info("Price monitor stopped by user")
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    logger.info("🚀 Starting Continuous Price Monitor")
    logger.info("Monitoring breaking news tickers for 5%+ price increases")
    
    monitor = ContinuousPriceMonitor()
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
