#!/usr/bin/env python3
"""
Continuous Price Monitor - HYBRID WEBSOCKET APPROACH
Monitors breaking news tickers and tracks price changes in real-time via WebSocket streaming
Falls back to REST API if WebSocket fails
"""

import logging
import aiohttp
import os
import asyncio
import time
import websockets
import json
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
        
        # NEW: WebSocket components
        self.websocket = None
        self.websocket_authenticated = False
        self.websocket_price_buffer = {}  # Buffer prices between database writes
        self.websocket_url = "wss://socket.polygon.io/stocks"  # Real-time WebSocket URL
        self.websocket_subscriptions = set()  # Track current subscriptions
        self.websocket_enabled = False  # Track if WebSocket is operational
        self.websocket_reconnect_delay = 1.0  # Reconnection delay (exponential backoff)
        self.websocket_max_reconnect_delay = 60.0  # Maximum reconnection delay
        self.use_websocket_data = False  # Flag to determine data source
        
        if not self.polygon_api_key:
            logger.error("POLYGON_API_KEY environment variable not set")
            raise ValueError("Polygon API key is required")
        
        # Use PROXY_URL if available (for REST API fallback)
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.base_url = proxy_url.rstrip('/')
            logger.info(f"Using proxy URL for REST fallback: {self.base_url}")
        else:
            self.base_url = "https://api.polygon.io"
            logger.info("Using official Polygon API for REST fallback")
        
        # Stats
        self.stats = {
            'tickers_monitored': 0,
            'price_checks': 0,
            'alerts_triggered': 0,
            'websocket_messages': 0,
            'websocket_reconnections': 0,
            'rest_api_fallbacks': 0,
            'start_time': time.time()
        }

    async def initialize(self):
        """Initialize the price monitoring system with WebSocket and REST fallback"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create essential tables
            await self.create_essential_tables()
            
            # Load active tickers from breaking_news
            self.active_tickers = await self.get_active_tickers_from_breaking_news()
            logger.info(f"✅ HYBRID Price Monitor initialized with WebSocket + REST fallback - {len(self.active_tickers)} active tickers!")
            
            # NEW: Initialize WebSocket connection
            logger.info("🔌 Setting up WebSocket connection...")
            await self.setup_websocket_connection()
            
            # Keep aiohttp session for REST API fallback
            timeout = aiohttp.ClientTimeout(
                total=2.0,      # 2 second total timeout - matches polling interval
                connect=0.5,    # 0.5 second connect timeout
                sock_read=1.5   # 1.5 second read timeout
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=50,           # Concurrent connections for REST fallback
                    limit_per_host=20,  # Connections per host for REST fallback
                    ttl_dns_cache=300,  # DNS cache for 5 minutes
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
            )
            
        except Exception as e:
            logger.error(f"Error initializing price monitor: {e}")
            raise

    async def setup_websocket_connection(self):
        """Setup WebSocket connection to Polygon"""
        try:
            logger.info(f"🔌 Connecting to Polygon WebSocket: {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("✅ WebSocket connection established")
            
            # Authenticate WebSocket
            success = await self.authenticate_websocket()
            if success:
                self.websocket_enabled = True
                self.use_websocket_data = True
                self.websocket_reconnect_delay = 1.0  # Reset reconnection delay
                logger.info("🚀 WebSocket authentication successful - REAL-TIME MODE ACTIVE")
            else:
                logger.error("❌ WebSocket authentication failed - falling back to REST API")
                await self.close_websocket()
                self.use_websocket_data = False
                
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            logger.info("🔄 Falling back to REST API mode")
            self.websocket = None
            self.websocket_enabled = False
            self.use_websocket_data = False

    async def authenticate_websocket(self):
        """Authenticate WebSocket connection with API key"""
        if not self.websocket:
            return False
        
        try:
            # Send authentication message
            auth_message = {
                "action": "auth",
                "params": self.polygon_api_key
            }
            
            logger.info("🔐 Sending WebSocket authentication...")
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for authentication response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            auth_response = json.loads(response)
            
            logger.info(f"📨 Auth response: {auth_response}")
            
            # Handle both single message and array of messages
            if isinstance(auth_response, list):
                for msg in auth_response:
                    if msg.get("status") == "auth_success":
                        self.websocket_authenticated = True
                        return True
                    elif msg.get("status") == "connected":
                        # Wait for auth_success message
                        try:
                            auth_confirm = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                            auth_confirm_data = json.loads(auth_confirm)
                            if isinstance(auth_confirm_data, list):
                                for confirm_msg in auth_confirm_data:
                                    if confirm_msg.get("status") == "auth_success":
                                        self.websocket_authenticated = True
                                        return True
                            elif auth_confirm_data.get("status") == "auth_success":
                                self.websocket_authenticated = True
                                return True
                        except asyncio.TimeoutError:
                            logger.error("❌ Timeout waiting for auth confirmation")
            else:
                if auth_response.get("status") == "auth_success":
                    self.websocket_authenticated = True
                    return True
            
            return False
            
        except asyncio.TimeoutError:
            logger.error("❌ WebSocket authentication timeout")
            return False
        except Exception as e:
            logger.error(f"❌ WebSocket authentication error: {e}")
            return False

    async def close_websocket(self):
        """Close WebSocket connection safely"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass  # Ignore errors during close
            finally:
                self.websocket = None
                self.websocket_authenticated = False
                self.websocket_enabled = False

    async def reconnect_websocket(self):
        """Reconnect WebSocket with exponential backoff"""
        logger.info(f"🔄 Attempting WebSocket reconnection in {self.websocket_reconnect_delay:.1f}s...")
        await asyncio.sleep(self.websocket_reconnect_delay)
        
        # Exponential backoff
        self.websocket_reconnect_delay = min(
            self.websocket_reconnect_delay * 2,
            self.websocket_max_reconnect_delay
        )
        self.stats['websocket_reconnections'] += 1
        
        # Close existing connection
        await self.close_websocket()
        
        # Attempt to reconnect
        await self.setup_websocket_connection()

    async def update_websocket_subscriptions(self):
        """Update WebSocket subscriptions based on active tickers"""
        if not self.websocket_enabled or not self.websocket_authenticated:
            return
        
        try:
            # Calculate new subscriptions needed
            current_subscriptions = set(self.websocket_subscriptions)
            needed_subscriptions = set()
            
            # Create subscription strings for active tickers
            for ticker in self.active_tickers:
                needed_subscriptions.add(f"T.{ticker}")  # Trades - primary source
                needed_subscriptions.add(f"Q.{ticker}")  # Quotes - fallback source
            
            # Subscribe to new tickers
            new_subscriptions = needed_subscriptions - current_subscriptions
            if new_subscriptions:
                subscribe_message = {
                    "action": "subscribe",
                    "params": ",".join(new_subscriptions)
                }
                
                logger.info(f"📡 Subscribing to {len(new_subscriptions)} new WebSocket streams: {new_subscriptions}")
                await self.websocket.send(json.dumps(subscribe_message))
                
                # Update subscription tracking immediately (fire-and-forget)
                self.websocket_subscriptions.update(new_subscriptions)
                logger.info(f"✅ Subscribed to {len(new_subscriptions)} WebSocket streams")
            
            # Unsubscribe from old tickers
            old_subscriptions = current_subscriptions - needed_subscriptions
            if old_subscriptions:
                unsubscribe_message = {
                    "action": "unsubscribe",
                    "params": ",".join(old_subscriptions)
                }
                
                logger.info(f"📡 Unsubscribing from {len(old_subscriptions)} WebSocket streams")
                await self.websocket.send(json.dumps(unsubscribe_message))
                self.websocket_subscriptions -= old_subscriptions
                
        except Exception as e:
            logger.error(f"Error updating WebSocket subscriptions: {e}")

    async def handle_websocket_message(self, message):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            self.stats['websocket_messages'] += 1
            
            # Handle arrays and single messages
            if isinstance(data, list):
                for msg in data:
                    await self.process_single_websocket_message(msg)
            else:
                await self.process_single_websocket_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing WebSocket JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    async def process_single_websocket_message(self, msg):
        """Process a single WebSocket message"""
        try:
            event_type = msg.get("ev")
            symbol = msg.get("sym", msg.get("S", "UNKNOWN"))
            timestamp = msg.get("t", msg.get("s", 0))
            
            # Only process messages for active tickers
            if symbol not in self.active_tickers:
                return
            
            current_time = datetime.now(pytz.UTC)
            
            if event_type == "T":  # Trade message
                price = msg.get("p", 0.0)
                size = msg.get("s", 0)
                
                if price > 0:
                    # Add to price buffer
                    if symbol not in self.websocket_price_buffer:
                        self.websocket_price_buffer[symbol] = []
                    
                    self.websocket_price_buffer[symbol].append({
                        'price': price,
                        'volume': size,
                        'timestamp': current_time,
                        'source': 'websocket_trade',
                        'websocket_timestamp': timestamp
                    })
                    
                    logger.debug(f"📈 WEBSOCKET TRADE {symbol}: ${price:.4f} (size: {size})")
                
            elif event_type == "status":
                logger.debug(f"📡 WebSocket status: {msg}")
                
        except Exception as e:
            logger.error(f"Error processing single WebSocket message: {e}")

    async def process_websocket_prices(self):
        """Process buffered WebSocket prices and insert to database"""
        if not self.websocket_price_buffer:
            return
        
        try:
            start_time = time.time()
            
            # Convert buffer to database format
            price_data = []
            processed_tickers = set()
            
            for ticker, prices in self.websocket_price_buffer.items():
                if not prices:
                    continue
                
                # Use the most recent price for each ticker
                latest_price_info = prices[-1]  # Get latest price
                
                price_data.append((
                    latest_price_info['timestamp'],
                    ticker,
                    latest_price_info['price'],
                    latest_price_info['volume'],
                    latest_price_info['source']
                ))
                processed_tickers.add(ticker)
            
            if price_data:
                # Get sentiment data for enrichment (same as REST approach)
                sentiment_data = {}
                if processed_tickers:
                    try:
                        ticker_list = list(processed_tickers)
                        ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
                        
                        sentiment_query = f"""
                        SELECT 
                            ticker,
                            sentiment,
                            recommendation,
                            confidence
                        FROM (
                            SELECT 
                                ticker,
                                sentiment,
                                recommendation,
                                confidence,
                                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analyzed_at DESC) as rn
                            FROM News.breaking_news
                            WHERE ticker IN ({ticker_placeholders})
                            AND analyzed_at >= now() - INTERVAL 1 HOUR
                            AND sentiment != ''
                            AND recommendation != ''
                        ) ranked
                        WHERE rn = 1
                        """
                        
                        sentiment_result = self.ch_manager.client.query(sentiment_query)
                        for row in sentiment_result.result_rows:
                            ticker, sentiment, recommendation, confidence = row
                            sentiment_data[ticker] = {
                                'sentiment': sentiment,
                                'recommendation': recommendation,
                                'confidence': confidence
                            }
                    except Exception as e:
                        logger.debug(f"Error getting sentiment data: {e}")
                
                # Prepare enriched price data with sentiment
                enriched_price_data = []
                for price_row in price_data:
                    timestamp, ticker, price, volume, source = price_row
                    
                    # Get sentiment for this ticker (or use defaults)
                    ticker_sentiment = sentiment_data.get(ticker, {
                        'sentiment': 'neutral',
                        'recommendation': 'HOLD',
                        'confidence': 'low'
                    })
                    
                    # Add sentiment fields to price data
                    enriched_price_data.append((
                        timestamp,
                        ticker,
                        price,
                        volume,
                        source,
                        ticker_sentiment['sentiment'],
                        ticker_sentiment['recommendation'],
                        ticker_sentiment['confidence']
                    ))
                
                # Batch insert enriched price data with sentiment
                self.ch_manager.client.insert(
                    'News.price_tracking',
                    enriched_price_data,
                    column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 'sentiment', 'recommendation', 'confidence']
                )
                
                total_time = time.time() - start_time
                self.stats['price_checks'] += len(enriched_price_data)
                
                # Enhanced logging with sentiment info
                sentiment_count = len([t for t in sentiment_data.values() if t['recommendation'] != 'HOLD'])
                logger.info(f"🌐 WEBSOCKET: Processed {len(enriched_price_data)} price updates in {total_time:.3f}s")
                if sentiment_count > 0:
                    logger.info(f"🧠 SENTIMENT: {sentiment_count} tickers have non-neutral sentiment analysis")
            
            # Clear processed data from buffer
            self.websocket_price_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error processing WebSocket prices: {e}")

    # Keep existing REST API methods for fallback
    async def get_price_with_double_call(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Make double API call for new tickers - discard first (garbage), use second (correct)"""
        url = f"{self.base_url}/v2/last/trade/{ticker}"
        params = {'apikey': self.polygon_api_key}
        start_time = time.time()
        
        try:
            # First call - expect garbage data, discard it
            async with self.session.get(url, params=params) as response1:
                if response1.status == 200:
                    garbage_data = await response1.json()
                    if 'results' in garbage_data and garbage_data['results']:
                        garbage_price = garbage_data['results'].get('p', 0.0)
                        logger.debug(f"🗑️ {ticker}: Discarding first call garbage price: ${garbage_price:.4f}")
            
            # Small delay between calls to avoid rate limiting
            await asyncio.sleep(0.1)
            
            # Second call - expect correct data, use this one
            async with self.session.get(url, params=params) as response2:
                api_time = time.time() - start_time
                
                if response2.status == 200:
                    data = await response2.json()
                    
                    if 'results' in data and data['results']:
                        result = data['results']
                        price = result.get('p', 0.0)
                        
                        if price > 0:
                            logger.info(f"✅ {ticker}: ${price:.4f} (REST DOUBLE-CALL VERIFIED) in {api_time:.3f}s")
                            self.stats['rest_api_fallbacks'] += 1
                            return {
                                'price': price,
                                'timestamp': datetime.now(pytz.UTC),
                                'source': 'rest_fallback_verified'
                            }
                elif response2.status == 429:
                    logger.debug(f"⚠️ Rate limited for {ticker} on second call - skipping this cycle")
                else:
                    logger.debug(f"Second API call returned status {response2.status} for {ticker} - skipping")
                    
        except asyncio.TimeoutError:
            logger.debug(f"⏱️ TIMEOUT for {ticker} double call - skipping this cycle")
        except Exception as e:
            logger.debug(f"Double call error for {ticker}: {e} - skipping")
        
        return None

    async def get_current_price_rest_fallback(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current price for ticker using REST API as fallback when WebSocket fails"""
        try:
            # Check if this is a newly added ticker (within 10 seconds)
            is_new_ticker = False
            if ticker in self.ticker_timestamps:
                time_since_added = datetime.now() - self.ticker_timestamps[ticker]
                is_new_ticker = time_since_added.total_seconds() <= 10
            
            if is_new_ticker:
                # NEW TICKER: Use double call to avoid garbage data
                logger.debug(f"🔄 REST FALLBACK: Making double API call for {ticker} to avoid garbage data")
                return await self.get_price_with_double_call(ticker)
            
            # EXISTING TICKER: Use single API call (normal flow)
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
                                logger.debug(f"⚡ {ticker}: ${price:.4f} (REST FALLBACK) in {api_time:.3f}s")
                                self.stats['rest_api_fallbacks'] += 1
                                return {
                                    'price': price,
                                    'timestamp': datetime.now(pytz.UTC),
                                    'source': 'rest_fallback'
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

    async def track_prices_rest_fallback(self):
        """Get current prices for all active tickers using REST API as fallback"""
        if not self.active_tickers:
            return
        
        # OPTIMIZED: Track timing for performance monitoring
        start_time = time.time()
        
        logger.info(f"🔄 REST FALLBACK: Processing {len(self.active_tickers)} tickers via REST API")
        
        # Create parallel price fetching tasks
        price_tasks = [self.get_current_price_rest_fallback(ticker) for ticker in self.active_tickers]
        
        # Execute all price requests in parallel with REDUCED timeout
        try:
            price_results = await asyncio.wait_for(
                asyncio.gather(*price_tasks, return_exceptions=True),
                timeout=2.0  # 2 seconds max - matches polling interval
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ REST FALLBACK TIMEOUT: Price fetching took >2s for {len(self.active_tickers)} tickers - SKIPPING THIS CYCLE")
            return
        
        # Process results and prepare batch insert (same logic as WebSocket)
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
                    0,  # Set volume to 0 since we're using trades not volume data
                    price_result.get('source', 'rest_fallback')
                ))
                successful_prices += 1
            else:
                failed_tickers.append(ticker)
        
        # Batch insert price data (same sentiment enrichment as WebSocket)
        if price_data:
            # Get sentiment data for each ticker before inserting
            sentiment_data = {}
            if self.active_tickers:
                try:
                    # Get latest sentiment analysis for all active tickers
                    ticker_list = list(self.active_tickers)
                    ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
                    
                    sentiment_query = f"""
                    SELECT 
                        ticker,
                        sentiment,
                        recommendation,
                        confidence
                    FROM (
                        SELECT 
                            ticker,
                            sentiment,
                            recommendation,
                            confidence,
                            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analyzed_at DESC) as rn
                        FROM News.breaking_news
                        WHERE ticker IN ({ticker_placeholders})
                        AND analyzed_at >= now() - INTERVAL 1 HOUR
                        AND sentiment != ''
                        AND recommendation != ''
                    ) ranked
                    WHERE rn = 1
                    """
                    
                    sentiment_result = self.ch_manager.client.query(sentiment_query)
                    for row in sentiment_result.result_rows:
                        ticker, sentiment, recommendation, confidence = row
                        sentiment_data[ticker] = {
                            'sentiment': sentiment,
                            'recommendation': recommendation,
                            'confidence': confidence
                        }
                except Exception as e:
                    logger.debug(f"Error getting sentiment data: {e}")
            
            # Prepare enriched price data with sentiment
            enriched_price_data = []
            for price_row in price_data:
                timestamp, ticker, price, volume, source = price_row
                
                # Get sentiment for this ticker (or use defaults)
                ticker_sentiment = sentiment_data.get(ticker, {
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low'
                })
                
                # Add sentiment fields to price data
                enriched_price_data.append((
                    timestamp,
                    ticker,
                    price,
                    volume,
                    source,
                    ticker_sentiment['sentiment'],
                    ticker_sentiment['recommendation'],
                    ticker_sentiment['confidence']
                ))
            
            # Insert enriched price data with sentiment
            self.ch_manager.client.insert(
                'News.price_tracking',
                enriched_price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 'sentiment', 'recommendation', 'confidence']
            )
            
            total_time = time.time() - start_time
            self.stats['price_checks'] += len(price_data)
            
            # Enhanced logging with sentiment info
            sentiment_count = len([t for t in sentiment_data.values() if t['recommendation'] != 'HOLD'])
            logger.info(f"🔄 REST FALLBACK: Tracked {successful_prices}/{len(self.active_tickers)} ticker prices in {total_time:.3f}s")
            if sentiment_count > 0:
                logger.info(f"🧠 SENTIMENT: {sentiment_count} tickers have non-neutral sentiment analysis")
            
            if failed_tickers:
                logger.debug(f"⚠️ Failed to get prices for: {failed_tickers}")
        else:
            total_time = time.time() - start_time
            logger.warning(f"❌ No price data retrieved for any tickers in {total_time:.3f}s - API issues or rate limiting")

    async def websocket_listener(self):
        """Dedicated WebSocket listener that runs continuously"""
        logger.info("👂 Starting WebSocket listener for real-time price streaming...")
        
        while True:
            try:
                if not self.websocket_enabled or not self.websocket:
                    # Wait for WebSocket to be available or attempt reconnection
                    await asyncio.sleep(5)
                    if not self.websocket_enabled:
                        logger.info("🔄 Attempting WebSocket reconnection...")
                        await self.reconnect_websocket()
                    continue
                
                try:
                    # Listen for WebSocket messages with timeout
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                    await self.handle_websocket_message(message)
                    
                except asyncio.TimeoutError:
                    # No message received in 30 seconds - send ping to keep connection alive
                    try:
                        await self.websocket.ping()
                        logger.debug("📡 WebSocket ping sent")
                    except:
                        logger.warning("❌ WebSocket ping failed - connection may be lost")
                        self.websocket_enabled = False
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("❌ WebSocket connection closed unexpectedly")
                    self.websocket_enabled = False
                    await self.reconnect_websocket()
                    
            except Exception as e:
                logger.error(f"Error in WebSocket listener: {e}")
                self.websocket_enabled = False
                await asyncio.sleep(1)

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
                source String DEFAULT 'polygon',
                sentiment String DEFAULT 'neutral',
                recommendation String DEFAULT 'HOLD',
                confidence String DEFAULT 'low'
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

    async def check_price_alerts_optimized(self):
        """Check for 5%+ price increases WITH sentiment analysis - ONLY trigger alerts when price moves 5% AND sentiment is 'BUY' with high confidence"""
        try:
            if not self.active_tickers:
                return
                
            # Convert set to list for SQL IN clause
            ticker_list = list(self.active_tickers)
            ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
            
            # ENHANCED: Include sentiment analysis in price alert logic
            # Only trigger alerts when:
            # 1. Price moves 5%+ within 2 minutes (existing logic)
            # 2. AND sentiment is 'BUY' with 'high' confidence (from price_tracking table)
            query = f"""
            WITH ticker_first_timestamps AS (
                SELECT 
                    ticker,
                    min(timestamp) as first_timestamp,
                    min(timestamp) + INTERVAL 40 SECOND as cutoff_timestamp
                FROM News.price_tracking
                WHERE ticker IN ({ticker_placeholders})
                GROUP BY ticker
            ),
            price_data AS (
                SELECT 
                    pt.ticker as ticker,
                    argMax(pt.price, pt.timestamp) as current_price,
                    argMin(pt.price, pt.timestamp) as first_price,
                    max(pt.timestamp) as current_timestamp,
                    min(pt.timestamp) as first_timestamp,
                    count() as price_count,
                    COALESCE(a.alert_count, 0) as existing_alerts,
                    argMax(pt.sentiment, pt.timestamp) as sentiment,
                    argMax(pt.recommendation, pt.timestamp) as recommendation,
                    argMax(pt.confidence, pt.timestamp) as confidence
                FROM News.price_tracking pt
                INNER JOIN ticker_first_timestamps tft ON pt.ticker = tft.ticker
                LEFT JOIN (
                    SELECT ticker, count() as alert_count
                    FROM News.news_alert
                    WHERE timestamp >= now() - INTERVAL 2 MINUTE
                    GROUP BY ticker
                ) a ON pt.ticker = a.ticker
                WHERE pt.ticker IN ({ticker_placeholders})
                AND COALESCE(a.alert_count, 0) < 8
                AND pt.timestamp <= tft.cutoff_timestamp
                GROUP BY pt.ticker, a.alert_count
                HAVING first_price > 0 AND price_count >= 2
            )
            SELECT 
                ticker,
                current_price,
                first_price,
                ((current_price - first_price) / first_price) * 100 as change_pct,
                price_count,
                first_timestamp,
                current_timestamp,
                dateDiff('second', first_timestamp, current_timestamp) as seconds_elapsed,
                existing_alerts,
                sentiment,
                recommendation,
                confidence
            FROM price_data
            WHERE ((current_price - first_price) / first_price) * 100 >= 5.0 
            AND dateDiff('second', first_timestamp, current_timestamp) <= 40
            AND current_price < 20.0
            AND (recommendation = 'BUY' AND confidence = 'high')
            ORDER BY change_pct DESC
            """
            
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                # Prepare batch insert for news_alert table
                alert_data = []
                
                for row in result.result_rows:
                    ticker, current_price, first_price, change_pct, price_count, first_timestamp, current_timestamp, seconds_elapsed, existing_alerts, sentiment, recommendation, confidence = row
                    
                    # Determine data source for logging
                    data_source_emoji = "🌐" if self.use_websocket_data else "🔄"
                    
                    # Enhanced logging with sentiment information
                    if sentiment and recommendation:
                        sentiment_info = f"Sentiment: {sentiment}, Recommendation: {recommendation} ({confidence} confidence)"
                        logger.info(f"🚨 SENTIMENT-ENHANCED ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from ${first_price:.4f}) in {seconds_elapsed}s")
                        logger.info(f"   📊 {sentiment_info}")
                        logger.info(f"   {data_source_emoji} Data Source: {'WebSocket' if self.use_websocket_data else 'REST API'}")
                        logger.info(f"   📈 Price Data: [{price_count} points] [Alert #{existing_alerts + 1}/8]")
                    else:
                        logger.info(f"🚨 PRICE-ONLY ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from ${first_price:.4f}) in {seconds_elapsed}s")
                        logger.info(f"   ⚠️ No sentiment data available - using price-only logic")
                        logger.info(f"   {data_source_emoji} Data Source: {'WebSocket' if self.use_websocket_data else 'REST API'}")
                        logger.info(f"   📈 Price Data: [{price_count} points] [Alert #{existing_alerts + 1}/8]")
                    
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
                    logger.info(f"✅ SENTIMENT-ENHANCED ALERTS: Inserted {len(alert_data)} alerts with sentiment analysis")
            else:
                # Enhanced debug logging to show why no alerts were triggered
                # First check if there are any price movements that would qualify
                debug_query = f"""
                SELECT 
                    ticker,
                    ((argMax(price, timestamp) - argMin(price, timestamp)) / argMin(price, timestamp)) * 100 as change_pct,
                    count() as price_count,
                    dateDiff('second', argMin(timestamp, timestamp), argMax(timestamp, timestamp)) as seconds_elapsed
                FROM News.price_tracking
                WHERE timestamp >= now() - INTERVAL 15 MINUTE
                AND ticker IN ({ticker_placeholders})
                GROUP BY ticker
                HAVING change_pct >= 5.0 AND price_count >= 2
                AND dateDiff('second', argMin(timestamp, timestamp), argMax(timestamp, timestamp)) <= 40
                """
                
                debug_result = self.ch_manager.client.query(debug_query)
                if debug_result.result_rows:
                    logger.info(f"💡 SENTIMENT FILTER: Found {len(debug_result.result_rows)} tickers with price moves but no favorable sentiment")
                    for row in debug_result.result_rows:
                        ticker, change_pct, price_count, seconds_elapsed = row
                        logger.info(f"   📊 {ticker}: +{change_pct:.2f}% in {seconds_elapsed}s - blocked by sentiment filter")
                        
                        # Check what sentiment data exists for this ticker in price_tracking
                        sentiment_debug_query = f"""
                        SELECT 
                            argMax(sentiment, timestamp) as latest_sentiment,
                            argMax(recommendation, timestamp) as latest_recommendation,
                            argMax(confidence, timestamp) as latest_confidence,
                            argMax(timestamp, timestamp) as latest_timestamp
                        FROM News.price_tracking
                        WHERE ticker = '{ticker}'
                        AND timestamp >= now() - INTERVAL 1 HOUR
                        GROUP BY ticker
                        """
                        
                        sentiment_debug_result = self.ch_manager.client.query(sentiment_debug_query)
                        if sentiment_debug_result.result_rows:
                            sentiment, recommendation, confidence, timestamp = sentiment_debug_result.result_rows[0]
                            logger.info(f"      🧠 Available sentiment: {sentiment}, {recommendation} ({confidence} confidence) at {timestamp}")
                        else:
                            logger.info(f"      ❌ No sentiment data found for {ticker} in price_tracking")
                
                # Check if we're skipping tickers due to alert limit
                limit_check_query = f"""
                SELECT ticker, count() as alert_count
                FROM News.news_alert
                WHERE timestamp >= now() - INTERVAL 2 MINUTE
                AND ticker IN ({ticker_placeholders})
                GROUP BY ticker
                HAVING alert_count >= 8
                """
                
                limit_result = self.ch_manager.client.query(limit_check_query)
                if limit_result.result_rows:
                    limited_tickers = [row[0] for row in limit_result.result_rows]
                    logger.debug(f"🔒 ALERT LIMIT: Skipping {len(limited_tickers)} tickers that already have 8+ alerts: {limited_tickers}")
                
        except Exception as e:
            logger.error(f"Error checking sentiment-enhanced price alerts: {e}")

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
        
        logger.info(f"📊 HYBRID MONITOR STATS:")
        logger.info(f"   Runtime: {runtime/60:.1f} minutes")
        logger.info(f"   Active Tickers: {len(self.active_tickers)}")
        logger.info(f"   Price Checks: {self.stats['price_checks']}")
        logger.info(f"   Alerts Triggered: {self.stats['alerts_triggered']}")
        logger.info(f"   WebSocket Messages: {self.stats['websocket_messages']}")
        logger.info(f"   WebSocket Reconnections: {self.stats['websocket_reconnections']}")
        logger.info(f"   REST API Fallbacks: {self.stats['rest_api_fallbacks']}")
        logger.info(f"   Data Source: {'WebSocket' if self.use_websocket_data else 'REST API'}")

    async def cleanup(self):
        """Clean up resources"""
        if self.websocket:
            await self.close_websocket()
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("Hybrid price monitor cleanup completed")

    async def start(self):
        """Start the continuous price monitoring system with WebSocket hybrid approach"""
        try:
            logger.info("🚀 Starting HYBRID Price Monitor with WebSocket + REST fallback!")
            await self.initialize()
            
            # Test connectivity based on available data source
            if self.use_websocket_data:
                logger.info("🌐 WebSocket mode active - testing subscription capability...")
                # WebSocket is already tested in initialize()
            else:
                logger.info("🔄 REST API fallback mode - testing API connectivity...")
                await self.test_api_connectivity()
            
            logger.info("⚡ Starting hybrid monitoring: WebSocket real-time + REST fallback + File triggers...")
            logger.info("✅ Hybrid Price Monitor operational!")
            
            # Run ALL components in parallel:
            # 1. WebSocket listener for real-time price streaming
            # 2. File trigger monitor for immediate ticker notifications
            # 3. Continuous polling loop for database operations and subscription management
            await asyncio.gather(
                self.websocket_listener(),              # Real-time WebSocket price streaming
                self.file_trigger_monitor_async(),      # Ticker notifications only
                self.continuous_polling_loop()          # Database operations + subscription management
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        except Exception as e:
            logger.error(f"Fatal error in hybrid price monitor: {e}")
            raise
        finally:
            await self.cleanup()

    async def test_api_connectivity(self):
        """Test REST API connectivity with a simple request (fallback mode)"""
        test_ticker = "AAPL"  # Use AAPL as a test ticker
        logger.info(f"🔬 Testing REST API connectivity with {test_ticker}...")
        
        try:
            start_time = time.time()
            result = await self.get_current_price_rest_fallback(test_ticker)
            test_time = time.time() - start_time
            
            if result:
                logger.info(f"✅ REST API TEST SUCCESS: {test_ticker} = ${result['price']:.4f} in {test_time:.3f}s")
            else:
                logger.warning(f"⚠️ REST API TEST FAILED: No price data for {test_ticker} in {test_time:.3f}s")
                logger.warning("🚨 API connectivity issues detected - price monitoring may be slow")
        except Exception as e:
            logger.error(f"❌ REST API TEST ERROR: {e}")
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
                            
                            # Add to active tickers for both WebSocket and REST tracking
                            self.active_tickers.add(ticker)
                            self.ticker_timestamps[ticker] = datetime.now()  # Track when ticker was added
                            logger.info(f"🎯 ASYNC MONITOR: Added {ticker} to active tracking")
                            
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
        """Continuous polling loop - WebSocket subscription management + database operations every 2 seconds"""
        logger.info("🔄 Starting HYBRID POLLING LOOP - WebSocket subscription management + database operations")
        
        cycle = 0
        last_cleanup = time.time()
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                # Clean up old tickers every 5 minutes
                if time.time() - last_cleanup > 300:  # 5 minutes
                    await self.cleanup_old_tickers()
                    last_cleanup = time.time()
                
                if self.active_tickers:
                    logger.info(f"🔄 HYBRID CYCLE {cycle}: Managing {len(self.active_tickers)} active tickers")
                    
                    if self.use_websocket_data and self.websocket_enabled:
                        # WebSocket mode: Update subscriptions and process buffered prices
                        await self.update_websocket_subscriptions()
                        await self.process_websocket_prices()
                        logger.debug(f"🌐 WebSocket cycle: Subscriptions updated, prices processed")
                    else:
                        # REST fallback mode: Track prices via API calls
                        await self.track_prices_rest_fallback()
                        logger.debug(f"🔄 REST fallback cycle: API prices tracked")
                    
                    # Always check for alerts (regardless of data source)
                    await self.check_price_alerts_optimized()
                    
                    cycle_time = time.time() - cycle_start
                    data_source = "WebSocket" if self.use_websocket_data else "REST"
                    logger.info(f"✅ HYBRID CYCLE {cycle}: Completed ({data_source}) in {cycle_time:.3f}s")
                else:
                    logger.debug(f"⏳ HYBRID CYCLE {cycle}: No active tickers")
                
                # Wait 2 seconds before next cycle
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error in hybrid polling loop: {e}")
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
    logger.info("🚀 Starting HYBRID Continuous Price Monitor with WebSocket + REST fallback")
    logger.info("⚡ HYBRID MODE: WebSocket real-time streaming → immediate price data → instant alerts")
    logger.info("🔄 FALLBACK: REST API calls if WebSocket fails → reliable operation guaranteed")
    
    monitor = ContinuousPriceMonitor()
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
