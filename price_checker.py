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

# GLOBAL NOTIFICATION QUEUE for immediate ticker notifications
ticker_notification_queue = asyncio.Queue()

class ContinuousPriceMonitor:
    def __init__(self):
        self.ch_manager = None
        self.session = None
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.active_tickers: Set[str] = set()  # Track tickers directly from breaking_news
        self.ticker_timestamps: Dict[str, datetime] = {}  # Track when tickers were added
        self.ready_event = asyncio.Event()  # Signal when monitor is ready for new tickers
        
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
        """Initialize the price monitoring system"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create essential tables
            await self.create_essential_tables()
            
            # Load active tickers from breaking_news
            self.active_tickers = await self.get_active_tickers_from_breaking_news()
            logger.info(f"✅ ZERO-LAG Price Monitor initialized with FILE TRIGGERS ONLY - {len(self.active_tickers)} active tickers!")
            
            # OPTIMIZED: Aggressive timeouts for consistent 2-second polling cycles
            # Individual requests must complete quickly to avoid blocking polling
            timeout = aiohttp.ClientTimeout(
                total=2.0,      # 2 second total timeout - matches polling interval
                connect=0.5,    # 0.5 second connect timeout - proxy connection must be quick
                sock_read=1.5   # 1.5 second read timeout - proxy processing must be fast
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=50,           # INCREASED: More concurrent connections for multiple operations
                    limit_per_host=20,  # INCREASED: More connections per host for parallel API calls
                    ttl_dns_cache=300,  # DNS cache for 5 minutes
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
            )
            
        except Exception as e:
            logger.error(f"Error initializing price monitor: {e}")
            raise

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
        """Get tickers directly from breaking_news table - OPTIMIZED to avoid API interference"""
        try:
            query_start = time.time()
            
            # OPTIMIZED: Use simpler query without FINAL to avoid slow table merges
            # Only check for very recent tickers to minimize query time
            query = """
            SELECT DISTINCT ticker
            FROM News.breaking_news
            WHERE detected_at >= now() - INTERVAL 5 MINUTE
            AND ticker != ''
            LIMIT 100
            """
            
            result = self.ch_manager.client.query(query)
            current_tickers = {row[0] for row in result.result_rows}
            
            query_time = time.time() - query_start
            
            # Only log if there are changes or query is slow
            if current_tickers != self.active_tickers or query_time > 0.1:
                logger.info(f"🔍 TICKER QUERY: Found {len(current_tickers)} tickers in {query_time:.3f}s")
                if current_tickers != self.active_tickers:
                    logger.info(f"🔍 CURRENT TICKERS: {sorted(current_tickers)}")
                    logger.info(f"🔍 PREVIOUS TICKERS: {sorted(self.active_tickers)}")
                    
                    # Show recent records only if tickers changed
                    debug_query = """
                    SELECT ticker, detected_at, headline
                    FROM News.breaking_news
                    WHERE detected_at >= now() - INTERVAL 5 MINUTE
                    AND ticker != ''
                    ORDER BY detected_at DESC
                    LIMIT 3
                    """
                    debug_result = self.ch_manager.client.query(debug_query)
                    logger.info(f"🔍 RECENT DATABASE RECORDS:")
                    for i, row in enumerate(debug_result.result_rows):
                        ticker, detected_at, headline = row
                        logger.info(f"   {i+1}. {ticker} - detected_at: {detected_at} - headline: {headline[:50]}...")
            
            self.active_tickers = current_tickers
            return current_tickers
            
        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            return set()

    async def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current price for ticker using ONLY last trade endpoint - no unreliable fallbacks"""
        try:
            # Use Last Trade endpoint only - most accurate actual price
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {'apikey': self.polygon_api_key}
            
            start_time = time.time()
            try:
                async with self.session.get(url, params=params) as response:
                    api_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'results' in data and data['results']:
                            result = data['results']
                            price = result.get('p', 0.0)  # trade price
                            
                            if price > 0:
                                logger.debug(f"⚡ {ticker}: ${price:.4f} (TRADE) in {api_time:.3f}s")
                                return {
                                    'price': price,
                                    'timestamp': datetime.now(pytz.UTC),
                                    'source': 'trade'
                                }
                    elif response.status == 429:
                        logger.debug(f"⚠️ Rate limited for {ticker} - skipping this cycle")
                    else:
                        logger.debug(f"Trade API returned status {response.status} for {ticker} - skipping")
            except asyncio.TimeoutError:
                logger.debug(f"⏱️ TIMEOUT for {ticker} - skipping this cycle")
            except Exception as e:
                logger.debug(f"Trade error for {ticker}: {e} - skipping")
            
            # No fallback - just skip failed requests to maintain clean intervals
            total_time = time.time() - start_time
            logger.debug(f"❌ Skipping {ticker} this cycle (failed in {total_time:.3f}s)")
                    
        except Exception as e:
            logger.debug(f"Fatal error getting price for {ticker}: {e} - skipping")
        
        return None

    async def track_prices_parallel(self):
        """Get current prices for all active tickers in PARALLEL with OPTIMIZED timeouts"""
        if not self.active_tickers:
            return
        
        # OPTIMIZED: Track timing for performance monitoring
        start_time = time.time()
        
        # Create parallel price fetching tasks
        price_tasks = [self.get_current_price(ticker) for ticker in self.active_tickers]
        
        # Execute all price requests in parallel with REDUCED timeout
        try:
            price_results = await asyncio.wait_for(
                asyncio.gather(*price_tasks, return_exceptions=True),
                timeout=2.0  # 2 seconds max - matches polling interval
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ BULK TIMEOUT: Price fetching took >2s for {len(self.active_tickers)} tickers - SKIPPING THIS CYCLE")
            return
        
        # Process results and prepare batch insert
        price_data = []
        successful_prices = 0
        failed_tickers = []
        
        for i, (ticker, price_result) in enumerate(zip(self.active_tickers, price_results)):
            if isinstance(price_result, Exception):
                logger.debug(f"Exception getting price for {ticker}: {price_result}")
                failed_tickers.append(ticker)
                continue
                
            if price_result:
                price_data.append((
                    datetime.now(),
                    ticker,
                    price_result['price'],
                    0,  # Set volume to 0 since we're using quotes not trades
                    price_result.get('source', 'polygon')
                ))
                successful_prices += 1
            else:
                failed_tickers.append(ticker)
        
        # Batch insert price data
        if price_data:
            self.ch_manager.client.insert(
                'News.price_tracking',
                price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
            )
            
            total_time = time.time() - start_time
            self.stats['price_checks'] += len(price_data)
            logger.info(f"⚡ PARALLEL: Tracked {successful_prices}/{len(self.active_tickers)} ticker prices in {total_time:.3f}s")
            
            if failed_tickers:
                logger.debug(f"⚠️ Failed to get prices for: {failed_tickers}")
        else:
            total_time = time.time() - start_time
            logger.warning(f"❌ No price data retrieved for any tickers in {total_time:.3f}s - API issues or rate limiting")
            
            # If all tickers fail, add a brief delay to avoid hammering the API
            if len(self.active_tickers) > 0:
                logger.info("💤 Adding 2s delay due to API failures to avoid rate limiting")
                await asyncio.sleep(2)

    async def check_price_alerts_optimized(self):
        """Check for 5%+ price increases - IMMEDIATE trigger within 2 minutes of first price - LIMITED to 5 signals per ticker"""
        try:
            if not self.active_tickers:
                return
                
            # Convert set to list for SQL IN clause
            ticker_list = list(self.active_tickers)
            ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
            
            # OPTIMIZED: Compare current price to FIRST price recorded, but ONLY within 2 minutes
            # This captures immediate breaking news moves and ignores delayed reactions
            # LIMITED: Only process tickers with fewer than 5 existing alerts
            query = f"""
            SELECT 
                ticker,
                current_price,
                first_price,
                ((current_price - first_price) / first_price) * 100 as change_pct,
                price_count,
                first_timestamp,
                current_timestamp,
                dateDiff('second', first_timestamp, current_timestamp) as seconds_elapsed,
                existing_alerts
            FROM (
                SELECT 
                    p.ticker,
                    argMax(p.price, p.timestamp) as current_price,
                    argMin(p.price, p.timestamp) as first_price,
                    argMax(p.timestamp, p.timestamp) as current_timestamp,
                    argMin(p.timestamp, p.timestamp) as first_timestamp,
                    count() as price_count,
                    COALESCE(a.alert_count, 0) as existing_alerts
                FROM News.price_tracking p
                LEFT JOIN (
                    SELECT ticker, count() as alert_count
                    FROM News.news_alert
                    WHERE timestamp >= now() - INTERVAL 2 MINUTE
                    GROUP BY ticker
                ) a ON p.ticker = a.ticker
                WHERE p.timestamp >= now() - INTERVAL 15 MINUTE
                AND p.ticker IN ({ticker_placeholders})
                AND COALESCE(a.alert_count, 0) < 5
                GROUP BY p.ticker, a.alert_count
                HAVING first_price > 0 AND price_count >= 2
            )
            WHERE change_pct >= 5.0 
            AND seconds_elapsed <= 30
            ORDER BY change_pct DESC
            """
            
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                # Prepare batch insert for news_alert table
                alert_data = []
                
                for row in result.result_rows:
                    ticker, current_price, first_price, change_pct, price_count, first_timestamp, current_timestamp, seconds_elapsed, existing_alerts = row
                    
                    logger.info(f"🚨 IMMEDIATE ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from first price ${first_price:.4f}) in {seconds_elapsed}s [{price_count} price points] [Alert #{existing_alerts + 1}/5]")
                    
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
                    logger.info(f"✅ BREAKING NEWS ALERTS: Inserted {len(alert_data)} immediate alerts (within 2min window, max 5 per ticker)")
            else:
                # Check if we're skipping tickers due to alert limit
                limit_check_query = f"""
                SELECT ticker, count() as alert_count
                FROM News.news_alert
                WHERE timestamp >= now() - INTERVAL 2 MINUTE
                AND ticker IN ({ticker_placeholders})
                GROUP BY ticker
                HAVING alert_count >= 5
                """
                
                limit_result = self.ch_manager.client.query(limit_check_query)
                if limit_result.result_rows:
                    limited_tickers = [row[0] for row in limit_result.result_rows]
                    logger.debug(f"🔒 ALERT LIMIT: Skipping {len(limited_tickers)} tickers that already have 5+ alerts: {limited_tickers}")
                
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

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("Price monitor cleanup completed")

    async def start(self):
        """Start the continuous price monitoring system"""
        try:
            logger.info("🚀 Starting ZERO-LAG Price Monitor with FILE TRIGGERS + CONTINUOUS POLLING!")
            await self.initialize()
            
            # Test API connectivity
            logger.info("🔌 Testing API connectivity...")
            await self.test_api_connectivity()
            
            # Start monitoring with file triggers for notifications and polling for price inserts
            logger.info("⚡ Starting monitoring: FILE TRIGGERS for notifications + CONTINUOUS POLLING for price inserts...")
            logger.info("✅ Price Monitor operational - Clean separation: notifications vs price tracking!")
            
            # Run BOTH file trigger monitor AND continuous polling in parallel
            # File triggers = immediate ticker notifications (add to active_tickers)
            # Continuous polling = ALL price database operations (every 2 seconds)
            await asyncio.gather(
                self.file_trigger_monitor_async(),      # Ticker notifications only
                self.continuous_polling_loop()          # ALL price database operations
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        except Exception as e:
            logger.error(f"Fatal error in price monitor: {e}")
            raise
        finally:
            await self.cleanup()

    async def test_api_connectivity(self):
        """Test API connectivity with a simple request"""
        test_ticker = "AAPL"  # Use AAPL as a test ticker
        logger.info(f"🔬 Testing API connectivity with {test_ticker}...")
        
        try:
            start_time = time.time()
            result = await self.get_current_price(test_ticker)
            test_time = time.time() - start_time
            
            if result:
                logger.info(f"✅ API TEST SUCCESS: {test_ticker} = ${result['price']:.4f} in {test_time:.3f}s")
            else:
                logger.warning(f"⚠️ API TEST FAILED: No price data for {test_ticker} in {test_time:.3f}s")
                logger.warning("🚨 API connectivity issues detected - price monitoring may be slow")
        except Exception as e:
            logger.error(f"❌ API TEST ERROR: {e}")
            logger.warning("🚨 Severe API issues detected - price monitoring will likely fail")

    async def file_trigger_monitor_async(self):
        """Async file trigger monitor that runs in main event loop - ONLY adds tickers to polling queue"""
        import os
        import json
        import glob
        
        trigger_dir = "triggers"
        logger.info("🚀 Starting ASYNC FILE TRIGGER MONITOR - NOTIFICATION ONLY (no direct price inserts)")
        
        while True:
            try:
                # Check for immediate trigger files
                trigger_pattern = os.path.join(trigger_dir, "immediate_*.json")
                trigger_files = glob.glob(trigger_pattern)
                
                if trigger_files:
                    # Process triggers ONE AT A TIME for consistent timing
                    # Sort by creation time to ensure fair processing order
                    trigger_files.sort(key=os.path.getctime)
                    
                    logger.info(f"🔥 ASYNC MONITOR: FOUND {len(trigger_files)} IMMEDIATE TRIGGER FILES!")
                    
                    for trigger_file in trigger_files:
                        try:
                            # Read trigger data
                            with open(trigger_file, 'r') as f:
                                trigger_data = json.load(f)
                            
                            ticker = trigger_data['ticker']
                            logger.info(f"⚡ ASYNC MONITOR: Processing trigger for {ticker}")
                            
                            # FIXED: ONLY add to active tickers - NO direct price inserts
                            # Let the continuous_polling_loop handle ALL price database operations
                            self.active_tickers.add(ticker)
                            self.ticker_timestamps[ticker] = datetime.now()  # Track when ticker was added
                            logger.info(f"🎯 ASYNC MONITOR: Added {ticker} to active tracking (polling will handle price inserts)")
                            
                            # Remove trigger file after successful processing
                            os.remove(trigger_file)
                            logger.info(f"✅ ASYNC MONITOR: Processed and removed trigger for {ticker}")
                            
                        except Exception as e:
                            logger.error(f"ASYNC MONITOR: Error processing trigger file {trigger_file}: {e}")
                            try:
                                os.remove(trigger_file)
                            except:
                                pass
                
                # Check every 1ms - MAXIMUM PRIORITY for trigger processing
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"ASYNC MONITOR: Error in file trigger monitor: {e}")
                await asyncio.sleep(0.001)

    async def cleanup_old_tickers(self):
        """Remove tickers older than 30 minutes from active tracking"""
        if not self.ticker_timestamps:
            return
            
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=30)
        
        old_tickers = []
        for ticker, timestamp in self.ticker_timestamps.items():
            if timestamp < cutoff_time:
                old_tickers.append(ticker)
        
        if old_tickers:
            for ticker in old_tickers:
                self.active_tickers.discard(ticker)
                del self.ticker_timestamps[ticker]
            
            logger.info(f"🧹 CLEANUP: Removed {len(old_tickers)} old tickers from active tracking: {old_tickers}")

    async def continuous_polling_loop(self):
        """Continuous polling loop for regular price updates every 2 seconds - ZERO DATABASE QUERIES"""
        logger.info("🔄 Starting CONTINUOUS POLLING LOOP - 2 second intervals, FILE TRIGGERS ONLY (no database queries)")
        
        cycle = 0
        last_cleanup = time.time()
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                # ELIMINATED: No more database queries - rely ONLY on file triggers
                # File triggers will add tickers immediately to active_tickers
                # This eliminates ALL database interference with API calls
                
                # Clean up old tickers every 5 minutes
                if time.time() - last_cleanup > 300:  # 5 minutes
                    await self.cleanup_old_tickers()
                    last_cleanup = time.time()
                
                if self.active_tickers:
                    logger.info(f"🔄 POLLING CYCLE {cycle}: Checking prices for {len(self.active_tickers)} active tickers")
                    
                    # Track prices for all active tickers
                    await self.track_prices_parallel()
                    await self.check_price_alerts_optimized()
                    
                    cycle_time = time.time() - cycle_start
                    logger.info(f"✅ POLLING CYCLE {cycle}: Completed in {cycle_time:.3f}s")
                else:
                    logger.debug(f"⏳ POLLING CYCLE {cycle}: No active tickers")
                
                # Wait 2 seconds before next cycle
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error in continuous polling loop: {e}")
                await asyncio.sleep(2.0)  # Continue with 2-second intervals even on error


# GLOBAL NOTIFICATION FUNCTION for immediate ticker notifications
async def notify_new_ticker(ticker: str, timestamp: datetime = None):
    """Send immediate notification when new ticker is detected - ELIMINATES POLLING LAG"""
    if timestamp is None:
        timestamp = datetime.now()
    
    notification = {
        'ticker': ticker,
        'timestamp': timestamp
    }
    
    try:
        # Non-blocking put - if queue is full, skip (shouldn't happen with immediate processing)
        ticker_notification_queue.put_nowait(notification)
        logger.info(f"📢 IMMEDIATE NOTIFICATION SENT: {ticker} at {timestamp}")
    except asyncio.QueueFull:
        logger.warning(f"⚠️ Notification queue full, skipping {ticker}")
    except Exception as e:
        logger.error(f"Error sending ticker notification: {e}")


async def main():
    """Main function"""
    logger.info("🚀 Starting ZERO-LAG Continuous Price Monitor with IMMEDIATE NOTIFICATIONS")
    logger.info("⚡ ZERO-LAG: Direct article insertion → immediate notification → instant price tracking")
    
    monitor = ContinuousPriceMonitor()
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
