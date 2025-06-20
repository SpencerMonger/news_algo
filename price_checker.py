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

# DATABASE-BASED IMMEDIATE NOTIFICATION SYSTEM for cross-process coordination
class DatabaseNotificationSystem:
    def __init__(self, ch_manager):
        self.ch_manager = ch_manager
        self.last_notification_check = datetime.now()
        
    async def create_notification_table(self):
        """Create immediate notification table for cross-process coordination"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS News.immediate_notifications (
                id UUID DEFAULT generateUUIDv4(),
                ticker String,
                timestamp DateTime64(3) DEFAULT now64(),
                processed UInt8 DEFAULT 0,
                created_at DateTime64(3) DEFAULT now64()
            ) ENGINE = MergeTree()
            ORDER BY (ticker, created_at)
            PARTITION BY toYYYYMM(created_at)
            TTL created_at + INTERVAL 1 HOUR
            """
            self.ch_manager.client.command(create_table_sql)
            logger.info("Created immediate_notifications table for cross-process coordination")
        except Exception as e:
            logger.error(f"Error creating notification table: {e}")
            
    async def send_immediate_notification(self, ticker: str, timestamp: datetime = None):
        """Send immediate notification via database - works across processes"""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            notification_data = [(
                ticker,
                timestamp,
                0,  # not processed yet
                datetime.now()
            )]
            
            self.ch_manager.client.insert(
                'News.immediate_notifications',
                notification_data,
                column_names=['ticker', 'timestamp', 'processed', 'created_at']
            )
            logger.info(f"📢 DB IMMEDIATE NOTIFICATION: {ticker} at {timestamp}")
        except Exception as e:
            logger.error(f"Error sending DB notification: {e}")
    
    async def get_pending_notifications(self):
        """Get unprocessed immediate notifications"""
        try:
            query = """
            SELECT ticker, timestamp, id
            FROM News.immediate_notifications
            WHERE processed = 0
            AND created_at >= now() - INTERVAL 10 MINUTE
            ORDER BY created_at ASC
            """
            
            result = self.ch_manager.client.query(query)
            return result.result_rows
        except Exception as e:
            logger.error(f"Error getting pending notifications: {e}")
            return []
    
    async def mark_notification_processed(self, notification_id: str):
        """Mark notification as processed"""
        try:
            update_sql = f"""
            ALTER TABLE News.immediate_notifications
            UPDATE processed = 1
            WHERE id = '{notification_id}'
            """
            self.ch_manager.client.command(update_sql)
        except Exception as e:
            logger.error(f"Error marking notification processed: {e}")

# Global database notification system
db_notification_system = None

class ContinuousPriceMonitor:
    def __init__(self):
        self.ch_manager = None
        self.session = None
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.active_tickers: Set[str] = set()  # Track tickers directly from breaking_news
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
            'immediate_notifications': 0,
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
        
        # ULTRA-OPTIMIZED: Aggressive timeouts for FASTEST possible API responses
        timeout = aiohttp.ClientTimeout(
            total=3.0,      # Reduced from 5.0 - total timeout
            connect=1.0,    # Reduced from 2.0 - connection timeout  
            sock_read=2.0   # Reduced from 3.0 - socket read timeout
        )
        
        # Create HTTP session with optimized settings
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=100,           # Increased from 50 - more concurrent connections
                limit_per_host=50,   # Increased from 20 - more per host
                ttl_dns_cache=300,   # DNS cache TTL
                use_dns_cache=True,  # Enable DNS caching
                keepalive_timeout=30 # Keep connections alive longer
            )
        )
        
        logger.info("✅ ULTRA-AGGRESSIVE Price Monitor initialized - 100ms detection ready!")

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
            query_start = time.time()
            
            # FIXED: Use FINAL to ensure we see latest data even before table merges
            # ReplacingMergeTree requires FINAL for immediate visibility of new inserts
            query = """
            SELECT DISTINCT ticker
            FROM News.breaking_news FINAL
            WHERE detected_at >= now() - INTERVAL 10 MINUTE
            AND ticker != ''
            ORDER BY ticker
            """
            
            result = self.ch_manager.client.query(query)
            current_tickers = {row[0] for row in result.result_rows}
            
            query_time = time.time() - query_start
            
            # DETAILED LOGGING: Show exactly what's happening with timing
            if current_tickers != self.active_tickers:
                logger.info(f"🔍 TICKER QUERY: Found {len(current_tickers)} tickers in {query_time:.3f}s")
                logger.info(f"🔍 CURRENT TICKERS: {sorted(current_tickers)}")
                logger.info(f"🔍 PREVIOUS TICKERS: {sorted(self.active_tickers)}")
                
                # Show the actual database records for debugging
                debug_query = """
                SELECT ticker, detected_at, timestamp, headline
                FROM News.breaking_news FINAL
                WHERE detected_at >= now() - INTERVAL 10 MINUTE
                AND ticker != ''
                ORDER BY detected_at DESC
                LIMIT 5
                """
                debug_result = self.ch_manager.client.query(debug_query)
                logger.info(f"🔍 RECENT DATABASE RECORDS:")
                for i, row in enumerate(debug_result.result_rows):
                    ticker, detected_at, timestamp, headline = row
                    logger.info(f"   {i+1}. {ticker} - detected_at: {detected_at} - headline: {headline[:50]}...")
            
            self.active_tickers = current_tickers
            return current_tickers
            
        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            return set()

    async def immediate_notification_handler(self):
        """Handle immediate ticker notifications from database - WORKS ACROSS PROCESSES"""
        logger.info("🚀 Starting DATABASE-BASED immediate notification handler - CROSS-PROCESS ZERO LAG!")
        
        while True:
            try:
                # Check for pending notifications in database every 50ms
                pending_notifications = await db_notification_system.get_pending_notifications()
                
                if pending_notifications:
                    logger.info(f"🔥 FOUND {len(pending_notifications)} IMMEDIATE NOTIFICATIONS!")
                    
                    for notification_row in pending_notifications:
                        ticker, timestamp, notification_id = notification_row
                        
                        self.stats['immediate_notifications'] += 1
                        logger.info(f"⚡ IMMEDIATE DB NOTIFICATION: {ticker} detected at {timestamp} - INSTANT PRICE CHECK!")
                        
                        # Add to active tickers immediately
                        self.active_tickers.add(ticker)
                        
                        # IMMEDIATE price check - no delays whatsoever
                        await self.track_single_ticker_immediate(ticker)
                        await self.check_price_alerts_optimized()
                        
                        # Second immediate check after 50ms
                        await asyncio.sleep(0.05)
                        await self.track_single_ticker_immediate(ticker)
                        await self.check_price_alerts_optimized()
                        
                        # Third immediate check after 100ms
                        await asyncio.sleep(0.1)
                        await self.track_single_ticker_immediate(ticker)
                        await self.check_price_alerts_optimized()
                        
                        # Mark as processed
                        await db_notification_system.mark_notification_processed(notification_id)
                        
                        logger.info(f"✅ IMMEDIATE DB PROCESSING COMPLETE for {ticker} - ZERO LAG ACHIEVED!")
                
                # Check every 50ms for immediate response
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in immediate DB notification handler: {e}")
                await asyncio.sleep(0.1)

    async def track_single_ticker_immediate(self, ticker: str):
        """Track price for a single ticker immediately - optimized for speed"""
        try:
            start_time = time.time()
            
            price_result = await self.get_current_price(ticker)
            
            if price_result:
                # Immediate insertion
                price_data = [(
                    datetime.now(),
                    ticker,
                    price_result['price'],
                    0,  # Set volume to 0 since we're using quotes not trades
                    price_result.get('source', 'polygon')
                )]
                
                self.ch_manager.client.insert(
                    'News.price_tracking',
                    price_data,
                    column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
                )
                
                total_time = time.time() - start_time
                self.stats['price_checks'] += 1
                logger.info(f"⚡ IMMEDIATE: {ticker} price ${price_result['price']:.4f} tracked in {total_time:.3f}s")
            else:
                total_time = time.time() - start_time
                logger.warning(f"❌ IMMEDIATE: Failed to get price for {ticker} in {total_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Error tracking immediate price for {ticker}: {e}")

    async def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current price for ticker with ULTRA-FAST timeout and multiple fallback strategies"""
        try:
            # Strategy 1: Try NBBO (real-time quotes) first - fastest
            url = f"{self.base_url}/v2/last/nbbo/{ticker}"
            params = {'apikey': self.polygon_api_key}
            
            start_time = time.time()
            try:
                async with self.session.get(url, params=params) as response:
                    api_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'results' in data and data['results']:
                            result = data['results']
                            # Use bid/ask midpoint for current price
                            bid = result.get('P', 0.0)  # bid price
                            ask = result.get('p', 0.0)  # ask price
                            
                            if bid > 0 and ask > 0:
                                current_price = (bid + ask) / 2
                                logger.debug(f"⚡ {ticker}: ${current_price:.4f} (NBBO) in {api_time:.3f}s")
                                return {
                                    'price': current_price,
                                    'bid': bid,
                                    'ask': ask,
                                    'timestamp': datetime.now(pytz.UTC),
                                    'source': 'nbbo'
                                }
                    elif response.status == 429:
                        logger.warning(f"⚠️ Rate limited for {ticker} - trying fallback")
                    else:
                        logger.debug(f"NBBO API returned status {response.status} for {ticker}")
            except asyncio.TimeoutError:
                logger.debug(f"⏱️ NBBO TIMEOUT for {ticker} - trying fallback")
            except Exception as e:
                logger.debug(f"NBBO error for {ticker}: {e} - trying fallback")
            
            # Strategy 2: Fallback to last trade endpoint
            fallback_url = f"{self.base_url}/v2/last/trade/{ticker}"
            try:
                async with self.session.get(fallback_url, params=params) as response:
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
                    else:
                        logger.debug(f"Trade API returned status {response.status} for {ticker}")
            except asyncio.TimeoutError:
                logger.debug(f"⏱️ TRADE TIMEOUT for {ticker}")
            except Exception as e:
                logger.debug(f"Trade error for {ticker}: {e}")
            
            # Strategy 3: If both fail, log and return None (don't block other tickers)
            total_time = time.time() - start_time
            logger.warning(f"❌ ALL ENDPOINTS FAILED for {ticker} in {total_time:.3f}s")
                    
        except Exception as e:
            logger.debug(f"Fatal error getting price for {ticker}: {e}")
        
        return None

    async def track_prices_parallel(self):
        """Get current prices for all active tickers in PARALLEL with ULTRA-FAST timeouts"""
        if not self.active_tickers:
            return
        
        # ULTRA-FAST: Track timing for optimization
        start_time = time.time()
        
        # Create parallel price fetching tasks
        price_tasks = [self.get_current_price(ticker) for ticker in self.active_tickers]
        
        # Execute all price requests in parallel with timeout
        try:
            price_results = await asyncio.wait_for(
                asyncio.gather(*price_tasks, return_exceptions=True),
                timeout=10.0  # INCREASED: Max 10 seconds for ALL price requests combined
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ BULK TIMEOUT: Price fetching took >10s for {len(self.active_tickers)} tickers - CONTINUING ANYWAY")
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
                logger.warning(f"⚠️ Failed to get prices for: {failed_tickers}")
        else:
            total_time = time.time() - start_time
            logger.warning(f"❌ No price data retrieved for any tickers in {total_time:.3f}s - API issues or rate limiting")
            
            # If all tickers fail, add a brief delay to avoid hammering the API
            if len(self.active_tickers) > 0:
                logger.info("💤 Adding 2s delay due to API failures to avoid rate limiting")
                await asyncio.sleep(2)

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
        logger.info(f"   Immediate Notifications: {self.stats['immediate_notifications']}")
        logger.info(f"   Alerts Triggered: {self.stats['alerts_triggered']}")

    async def ultra_fast_monitoring_loop(self):
        """ULTRA-AGGRESSIVE monitoring loop - IMMEDIATE detection within 100ms"""
        logger.info("🚀 Starting ULTRA-AGGRESSIVE polling monitor - 100ms detection cycles!")
        
        cycle = 0
        consecutive_empty_cycles = 0
        start_time = time.time()
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                runtime = time.time() - start_time
                
                # Get active tickers directly from breaking_news (ultra-aggressive polling)
                previous_tickers = self.active_tickers.copy()
                await self.get_active_tickers_from_breaking_news()
                
                # Signal ready after first cycle completes
                if cycle == 1:
                    self.ready_event.set()
                    logger.info("✅ ULTRA-AGGRESSIVE POLLING READY - 100ms detection cycles!")
                
                # Check if we have new tickers
                new_tickers = self.active_tickers - previous_tickers
                if new_tickers:
                    consecutive_empty_cycles = 0
                    logger.info(f"🔥 ULTRA-FAST DETECTION: {new_tickers} - IMMEDIATE PRICE TRACKING!")
                    
                    # IMMEDIATE processing for new tickers
                    for ticker in new_tickers:
                        logger.info(f"⚡ INSTANT PROCESSING: {ticker}")
                        await self.track_single_ticker_immediate(ticker)
                        await self.check_price_alerts_optimized()
                        
                        # Second check after 50ms
                        await asyncio.sleep(0.05)
                        await self.track_single_ticker_immediate(ticker)
                        await self.check_price_alerts_optimized()
                        
                        # Third check after 100ms
                        await asyncio.sleep(0.1)
                        await self.track_single_ticker_immediate(ticker)
                        await self.check_price_alerts_optimized()
                        
                        logger.info(f"✅ INSTANT PROCESSING COMPLETE: {ticker}")
                    
                elif self.active_tickers:
                    # Regular price tracking for existing tickers
                    await self.track_prices_parallel()
                    await self.check_price_alerts_optimized()
                    consecutive_empty_cycles = 0
                else:
                    consecutive_empty_cycles += 1
                    if consecutive_empty_cycles <= 3:
                        logger.debug(f"⏳ Cycle {cycle}: No active tickers")
                
                # Report stats every 2 minutes
                if cycle % 600 == 0:  # Adjusted for faster cycles
                    await self.report_stats()
                
                cycle_time = time.time() - cycle_start
                logger.debug(f"⚡ Cycle {cycle} completed in {cycle_time:.3f}s")
                
                # ULTRA-AGGRESSIVE TIMING:
                # First 60 seconds: 100ms cycles (MAXIMUM SPEED)
                # After 60 seconds: 200ms cycles (still very fast)
                # After 5 minutes: 250ms cycles (normal speed)
                if runtime < 60:
                    target_cycle_time = 0.1  # 100ms - ULTRA AGGRESSIVE for first minute
                    if cycle <= 10:
                        logger.info(f"🔥 ULTRA-AGGRESSIVE MODE: Cycle {cycle} - 100ms intervals")
                elif runtime < 300:
                    target_cycle_time = 0.2  # 200ms - still very fast for 5 minutes
                else:
                    target_cycle_time = 0.25  # 250ms - normal speed after 5 minutes
                
                sleep_time = max(0, target_cycle_time - cycle_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in ultra-aggressive monitoring loop: {e}")
                await asyncio.sleep(0.1)

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
            logger.info("🚀 Starting ULTRA-AGGRESSIVE Price Monitor - 100ms detection cycles!")
            await self.initialize()
            
            # Test API connectivity
            logger.info("🔌 Testing API connectivity...")
            await self.test_api_connectivity()
            
            # Start ULTRA-AGGRESSIVE monitoring (100ms cycles for first minute)
            logger.info("⚡ Starting ULTRA-AGGRESSIVE monitoring with 100ms detection cycles...")
            
            # Single task: ultra-aggressive polling only
            monitoring_task = asyncio.create_task(self.ultra_fast_monitoring_loop())
            
            logger.info("✅ ULTRA-AGGRESSIVE Price Monitor fully operational - MAXIMUM SPEED!")
            
            # Wait for monitoring task
            await monitoring_task
            
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
    logger.info("🚀 Starting OPTIMIZED Continuous Price Monitor with IMMEDIATE NOTIFICATIONS")
    logger.info("⚡ ZERO-LAG: Direct article insertion → immediate notification → instant price tracking")
    
    monitor = ContinuousPriceMonitor()
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
