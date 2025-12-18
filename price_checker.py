#!/usr/bin/env python3
"""
Continuous Price Monitor - IBKR TWS API Implementation

Monitors breaking news tickers and tracks price changes in real-time via IBKR TWS API.
Replaces Polygon WebSocket as the primary data source.

Requirements:
    - TWS or IB Gateway must be running
    - API must be enabled in TWS settings
    - client_id=10 (to avoid conflict with tradehead using client_id=1)

Environment Variables:
    - IBKR_HOST: TWS/Gateway host (default: 127.0.0.1)
    - IBKR_PORT: TWS/Gateway port (7497=paper, 7496=live)
    - IBKR_CLIENT_ID: Unique client ID (default: 10)
"""

import logging
import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Set, Optional

import pytz
from dotenv import load_dotenv

from clickhouse_setup import ClickHouseManager
from ibkr_client import IBKRClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce ibapi logging verbosity (very spammy at DEBUG level)
logging.getLogger('ibapi').setLevel(logging.WARNING)
logging.getLogger('ibapi.client').setLevel(logging.WARNING)
logging.getLogger('ibapi.wrapper').setLevel(logging.WARNING)
logging.getLogger('ibapi.decoder').setLevel(logging.WARNING)
logging.getLogger('ibapi.connection').setLevel(logging.WARNING)
logging.getLogger('ibapi.reader').setLevel(logging.WARNING)

# GLOBAL NOTIFICATION QUEUE for immediate ticker notifications
ticker_notification_queue = asyncio.Queue()


class ContinuousPriceMonitor:
    """
    IBKR-based price monitor for tracking breaking news tickers.
    
    Replaces Polygon WebSocket with IBKR TWS API for real-time price data.
    """
    
    def __init__(self):
        self.ch_manager = None
        self.active_tickers: Set[str] = set()
        self.ticker_timestamps: Dict[str, datetime] = {}
        self.ready_event = asyncio.Event()
        
        # IBKR client (replaces Polygon WebSocket)
        self.ibkr_client: Optional[IBKRClient] = None
        self.ibkr_connected = False
        
        # Load IBKR configuration from environment
        self.ibkr_host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.ibkr_port = int(os.getenv('IBKR_PORT', '7497'))
        self.ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', '10'))
        
        logger.info(f"IBKR Config: {self.ibkr_host}:{self.ibkr_port} (client_id={self.ibkr_client_id})")
        
        # Stats
        self.stats = {
            'tickers_monitored': 0,
            'price_checks': 0,
            'alerts_triggered': 0,
            'ibkr_ticks_received': 0,
            'ibkr_reconnections': 0,
            'start_time': time.time()
        }

    async def initialize(self):
        """Initialize the price monitoring system with IBKR connection"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create essential tables
            await self.create_essential_tables()
            
            # Load active tickers from breaking_news
            self.active_tickers = await self.get_active_tickers_from_breaking_news()
            
            # Initialize IBKR connection
            logger.info("🔌 Initializing IBKR TWS API connection...")
            await self.setup_ibkr_connection()
            
            logger.info(f"✅ IBKR Price Monitor initialized - {len(self.active_tickers)} active tickers")
            
        except Exception as e:
            logger.error(f"Error initializing price monitor: {e}")
            raise

    async def setup_ibkr_connection(self):
        """Setup connection to IBKR TWS/Gateway"""
        try:
            self.ibkr_client = IBKRClient(
                host=self.ibkr_host,
                port=self.ibkr_port,
                client_id=self.ibkr_client_id
            )
            
            # Connect (runs in separate thread)
            success = self.ibkr_client.connect_and_run()
            
            if success:
                self.ibkr_connected = True
                logger.info(f"✅ IBKR connection established on port {self.ibkr_port}")
                logger.info(f"   Mode: {'PAPER TRADING' if self.ibkr_port == 7497 else 'LIVE TRADING'}")
            else:
                raise ConnectionError("Failed to connect to IBKR TWS/Gateway")
                
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            logger.error("   Ensure TWS or IB Gateway is running and API is enabled")
            raise

    async def update_ibkr_subscriptions(self):
        """Update IBKR subscriptions based on active tickers"""
        if not self.ibkr_connected or not self.ibkr_client:
            return
        
        try:
            self.ibkr_client.update_subscriptions(self.active_tickers)
            
            # Remove failed tickers from active tracking (they won't get price data)
            failed_tickers = self.ibkr_client.get_failed_tickers()
            for ticker in list(failed_tickers.keys()):
                if ticker in self.active_tickers:
                    self.active_tickers.discard(ticker)
                    if ticker in self.ticker_timestamps:
                        del self.ticker_timestamps[ticker]
                    logger.info(f"⛔ Removed {ticker} from active tracking - IBKR cannot find security")
                    
        except Exception as e:
            logger.error(f"Error updating IBKR subscriptions: {e}")

    async def process_ibkr_prices(self):
        """Process buffered IBKR prices and insert to database"""
        if not self.ibkr_client:
            return
        
        try:
            start_time = time.time()
            
            # CRITICAL: Check 60-second window BEFORE inserting price data
            valid_tickers = await self.get_tickers_within_60_second_window()
            
            if not valid_tickers:
                logger.debug("⏰ No tickers within 60-second window")
                return
            
            # Get price data from IBKR buffer (only for valid tickers)
            price_buffer = self.ibkr_client.get_buffer_for_tickers(valid_tickers)
            
            if not price_buffer:
                return
            
            # Convert buffer to database format
            price_data = []
            processed_tickers = set()
            
            for ticker, prices in price_buffer.items():
                if not prices:
                    continue
                
                # Use the most recent price for each ticker
                latest_price_info = prices[-1]
                
                price_data.append((
                    latest_price_info['timestamp'],
                    ticker,
                    latest_price_info['price'],
                    latest_price_info['volume'],
                    latest_price_info['source']
                ))
                processed_tickers.add(ticker)
                self.stats['ibkr_ticks_received'] += len(prices)
            
            if price_data:
                # Get sentiment data for enrichment
                sentiment_data = await self._get_sentiment_data(processed_tickers)
                
                # Prepare enriched price data with sentiment
                enriched_price_data = []
                for price_row in price_data:
                    timestamp, ticker, price, volume, source = price_row
                    
                    ticker_sentiment = sentiment_data.get(ticker, {
                        'sentiment': 'neutral',
                        'recommendation': 'HOLD',
                        'confidence': 'low'
                    })
                    
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
                
                # Batch insert enriched price data
                self.ch_manager.client.insert(
                    'News.price_tracking',
                    enriched_price_data,
                    column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 
                                 'sentiment', 'recommendation', 'confidence']
                )
                
                total_time = time.time() - start_time
                self.stats['price_checks'] += len(enriched_price_data)
                
                logger.info(f"📊 IBKR: Processed {len(enriched_price_data)} price updates in {total_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing IBKR prices: {e}")

    async def _get_sentiment_data(self, tickers: Set[str]) -> Dict[str, Dict]:
        """Get latest sentiment data for tickers"""
        sentiment_data = {}
        
        if not tickers:
            return sentiment_data
        
        try:
            ticker_list = list(tickers)
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
            
            result = self.ch_manager.client.query(sentiment_query)
            for row in result.result_rows:
                ticker, sentiment, recommendation, confidence = row
                sentiment_data[ticker] = {
                    'sentiment': sentiment,
                    'recommendation': recommendation,
                    'confidence': confidence
                }
        except Exception as e:
            logger.debug(f"Error getting sentiment data: {e}")
        
        return sentiment_data

    async def reconnect_ibkr(self):
        """Reconnect to IBKR with backoff"""
        self.stats['ibkr_reconnections'] += 1
        
        # Close existing connection if any
        if self.ibkr_client:
            try:
                self.ibkr_client.disconnect_safely()
            except:
                pass
        
        self.ibkr_connected = False
        
        # Wait before reconnecting
        await asyncio.sleep(5.0)
        
        try:
            await self.setup_ibkr_connection()
        except Exception as e:
            logger.error(f"IBKR reconnection failed: {e}")

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
                source String DEFAULT 'ibkr',
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
        """
        Check for 5%+ price increases WITH sentiment analysis - Individual timestamp alerts with deduplication
        Each qualifying row generates one trade signal once.
        """
        try:
            if not self.active_tickers:
                return
                
            # Convert set to list for SQL IN clause
            ticker_list = list(self.active_tickers)
            ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
            
            # Individual timestamp query - matches test pattern exactly
            query = f"""
            WITH ticker_second_prices AS (
                SELECT 
                    ticker,
                    least(
                        max(CASE WHEN rn = 2 THEN price END),
                        max(CASE WHEN rn = 3 THEN price END)
                    ) as second_price
                FROM (
                    SELECT 
                        ticker,
                        price,
                        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp ASC) as rn
                    FROM News.price_tracking
                    WHERE ticker IN ({ticker_placeholders})
                ) ranked
                WHERE rn IN (2, 3)
                GROUP BY ticker
            ),
            ticker_first_3_volume AS (
                SELECT 
                    ticker,
                    sum(volume) as first_3_volume_total
                FROM (
                    SELECT 
                        ticker,
                        volume,
                        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp ASC) as rn
                    FROM News.price_tracking
                    WHERE ticker IN ({ticker_placeholders})
                ) ranked
                WHERE rn <= 3
                GROUP BY ticker
            ),
            price_analysis AS (
                SELECT 
                    pt.ticker,
                    pt.price as current_price,
                    COALESCE(tsp.second_price, 5.10) as baseline_price,
                    pt.timestamp as current_timestamp,
                    min(pt.timestamp) OVER (PARTITION BY pt.ticker) as first_timestamp,
                    ROW_NUMBER() OVER (PARTITION BY pt.ticker ORDER BY pt.timestamp ASC) as price_count,
                    pt.sentiment,
                    pt.recommendation,
                    pt.confidence,
                    ((pt.price - COALESCE(tsp.second_price, 5.10)) / COALESCE(tsp.second_price, 5.10)) * 100 as change_pct,
                    dateDiff('second', min(pt.timestamp) OVER (PARTITION BY pt.ticker), pt.timestamp) as seconds_elapsed
                FROM News.price_tracking pt
                LEFT JOIN ticker_second_prices tsp ON pt.ticker = tsp.ticker
                WHERE pt.ticker IN ({ticker_placeholders})
            )
            SELECT 
                pa.ticker,
                current_price,
                baseline_price,
                current_timestamp,
                first_timestamp,
                price_count,
                0 as existing_alerts,
                sentiment,
                recommendation,
                confidence,
                change_pct,
                seconds_elapsed,
                fld.strength_score
            FROM price_analysis pa
            INNER JOIN ticker_first_3_volume tv ON pa.ticker = tv.ticker
            LEFT JOIN News.float_list_detailed_dedup fld ON pa.ticker = fld.ticker
            WHERE price_count >= 3
            AND change_pct >= 5.0
            AND seconds_elapsed <= 60
            AND current_price < 11.0
            AND current_price >= 0.40
            AND recommendation = 'BUY'
            ORDER BY current_timestamp ASC
            """
            
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                # Check for existing alerts to handle deduplication
                existing_alerts_query = f"SELECT ticker, timestamp FROM News.news_alert WHERE ticker IN ({ticker_placeholders})"
                existing_result = self.ch_manager.client.query(existing_alerts_query)
                existing_alert_timestamps = {(row[0], row[1]) for row in existing_result.result_rows}
                
                # Prepare batch insert for news_alert table
                alert_data = []
                
                for row in result.result_rows:
                    ticker, current_price, baseline_price, current_timestamp, first_timestamp, price_count, existing_alerts, sentiment, recommendation, confidence, change_pct, seconds_elapsed, strength_score = row
                    
                    # Skip if alert already exists for this exact timestamp
                    if (ticker, current_timestamp) in existing_alert_timestamps:
                        logger.info(f"⏸️ DEDUPLICATION: Skipping {ticker} at {current_timestamp} - alert already exists")
                        continue
                    
                    # Determine alert value based on strength_score
                    # If strength_score >= 4: alert = 1 (High Priority)
                    # If strength_score < 4: alert = 2 (Lower Priority)
                    # If strength_score is NULL (no data): alert = 3 (Unknown)
                    if strength_score is None:
                        alert_value = 3
                        strength_info = f"Strength Score: N/A (Alert Type: 3 - Unknown Strength)"
                    elif strength_score >= 4:
                        alert_value = 1
                        strength_info = f"Strength Score: {strength_score} (Alert Type: 1 - High Priority)"
                    else:
                        alert_value = 2
                        strength_info = f"Strength Score: {strength_score} (Alert Type: 2 - Lower Priority)"
                    
                    # Enhanced logging with sentiment information
                    if sentiment and recommendation:
                        sentiment_info = f"Sentiment: {sentiment}, Recommendation: {recommendation} ({confidence} confidence)"
                        logger.info(f"🚨 IBKR ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from ${baseline_price:.4f}) in {seconds_elapsed}s")
                        logger.info(f"   📊 {sentiment_info}")
                        logger.info(f"   💪 {strength_info}")
                        logger.info(f"   🌐 Data Source: IBKR TWS API")
                        logger.info(f"   📈 Price sequence: #{price_count}")
                        logger.info(f"   🕐 Time Window: {first_timestamp} → {current_timestamp} ({seconds_elapsed}s)")
                    else:
                        logger.info(f"🚨 IBKR ALERT: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from ${baseline_price:.4f}) in {seconds_elapsed}s")
                        logger.info(f"   ⚠️ No sentiment data available - using price-only logic")
                        logger.info(f"   💪 {strength_info}")
                        logger.info(f"   🌐 Data Source: IBKR TWS API")
                        logger.info(f"   📈 Price sequence: #{price_count}")
                        logger.info(f"   🕐 Time Window: {first_timestamp} → {current_timestamp} ({seconds_elapsed}s)")
                    
                    # Add to alert data for batch insert
                    alert_data.append((ticker, current_timestamp, alert_value, current_price))
                    
                    # Log to price_move table
                    await self.log_price_alert(ticker, current_price, baseline_price, change_pct)
                    
                    self.stats['alerts_triggered'] += 1
                
                # Batch insert all alerts
                if alert_data:
                    self.ch_manager.client.insert(
                        'News.news_alert',
                        alert_data,
                        column_names=['ticker', 'timestamp', 'alert', 'price']
                    )
                    logger.info(f"✅ IBKR ALERTS: Created {len(alert_data)} new alerts with deduplication")
            else:
                # Log when tickers fail to meet requirements
                logger.info(f"❌ REQUIREMENTS CHECK: {len(self.active_tickers)} active tickers failed to meet one or more requirements")
                # Enhanced debug logging
                await self._log_debug_info(ticker_placeholders)
                
        except Exception as e:
            logger.error(f"Error checking sentiment-enhanced price alerts: {e}")

    async def _log_debug_info(self, ticker_placeholders: str):
        """Log debug information when no alerts are triggered"""
        try:
            # First check if there are any price movements that would qualify
            debug_query = f"""
            WITH ticker_first_timestamps AS (
                SELECT ticker, min(timestamp) as first_timestamp
                FROM News.price_tracking
                WHERE ticker IN ({ticker_placeholders})
                GROUP BY ticker
            ),
            ticker_second_prices AS (
                SELECT 
                    ticker,
                    least(
                        max(CASE WHEN rn = 2 THEN price END),
                        max(CASE WHEN rn = 3 THEN price END)
                    ) as second_price
                FROM (
                    SELECT 
                        ticker,
                        price,
                        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp ASC) as rn
                    FROM News.price_tracking
                    WHERE ticker IN ({ticker_placeholders})
                ) ranked
                WHERE rn IN (2, 3)
                GROUP BY ticker
            )
            SELECT 
                pt.ticker,
                ((argMax(pt.price, pt.timestamp) - COALESCE(tsp.second_price, argMin(pt.price, pt.timestamp))) / COALESCE(tsp.second_price, argMin(pt.price, pt.timestamp))) * 100 as change_pct,
                count() as price_count,
                dateDiff('second', tft.first_timestamp, max(pt.timestamp)) as seconds_elapsed
            FROM News.price_tracking pt
            INNER JOIN ticker_first_timestamps tft ON pt.ticker = tft.ticker
            LEFT JOIN ticker_second_prices tsp ON pt.ticker = tsp.ticker
            WHERE pt.ticker IN ({ticker_placeholders})
            AND pt.timestamp <= tft.first_timestamp + INTERVAL 60 SECOND
            GROUP BY pt.ticker, tft.first_timestamp, tsp.second_price
            HAVING change_pct >= 5.0 AND price_count >= 3
            AND seconds_elapsed <= 60
            """
            
            debug_result = self.ch_manager.client.query(debug_query)
            if debug_result.result_rows:
                logger.info(f"💡 60-SECOND WINDOW ENFORCED: Found {len(debug_result.result_rows)} tickers with price moves but no favorable sentiment")
                for row in debug_result.result_rows:
                    ticker, change_pct, price_count, seconds_elapsed = row
                    logger.info(f"   📊 {ticker}: +{change_pct:.2f}% in {seconds_elapsed}s - blocked by sentiment filter (within 60s window)")
                    
                    # Check what sentiment data exists for this ticker in price_tracking
                    sentiment_debug_query = f"""
                    SELECT 
                        argMax(sentiment, timestamp) as latest_sentiment,
                        argMax(recommendation, timestamp) as latest_recommendation,
                        argMax(confidence, timestamp) as latest_confidence,
                        max(timestamp) as latest_timestamp
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
            logger.debug(f"Error in debug logging: {e}")

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
            
            # Convert published_utc string to datetime if needed
            if isinstance(published_utc, str):
                try:
                    published_utc = datetime.fromisoformat(published_utc.replace('Z', '+00:00'))
                except:
                    published_utc = datetime.now()
            
            # Insert price move - using correct schema column names
            current_timestamp = datetime.now()
            values = [(
                current_timestamp,     # timestamp
                ticker,                # ticker
                headline,              # news_headline
                published_utc,         # news_published_utc
                url,                   # news_article_url
                current_price,         # current_price
                current_timestamp,     # current_price_timestamp
                0.0,                   # minute_30_high (placeholder)
                0.0,                   # minute_30_low (placeholder)
                0.0,                   # minute_30_open (placeholder)
                0.0,                   # minute_30_close (placeholder)
                0,                     # minute_30_volume (placeholder)
                current_timestamp,     # minute_30_timestamp (placeholder)
                change_pct,            # price_move_percentage
                1,                     # price_above_30min_high (default 1)
                0,                     # price_check_latency_ms (placeholder)
                0                      # news_to_price_check_delay_ms (placeholder)
            )]
            
            self.ch_manager.client.insert(
                'News.price_move',
                values,
                column_names=[
                    'timestamp', 'ticker', 'news_headline', 'news_published_utc', 'news_article_url',
                    'current_price', 'current_price_timestamp', 'minute_30_high', 'minute_30_low',
                    'minute_30_open', 'minute_30_close', 'minute_30_volume', 'minute_30_timestamp',
                    'price_move_percentage', 'price_above_30min_high', 'price_check_latency_ms',
                    'news_to_price_check_delay_ms'
                ]
            )
            
        except Exception as e:
            logger.error(f"Error logging price alert: {e}")

    async def report_stats(self):
        """Report monitoring statistics"""
        runtime = time.time() - self.stats['start_time']
        
        logger.info(f"📊 IBKR MONITOR STATS:")
        logger.info(f"   Runtime: {runtime/60:.1f} minutes")
        logger.info(f"   Active Tickers: {len(self.active_tickers)}")
        logger.info(f"   Price Checks: {self.stats['price_checks']}")
        logger.info(f"   Alerts Triggered: {self.stats['alerts_triggered']}")
        logger.info(f"   IBKR Ticks Received: {self.stats['ibkr_ticks_received']}")
        logger.info(f"   IBKR Reconnections: {self.stats['ibkr_reconnections']}")
        logger.info(f"   Mode: {'PAPER' if self.ibkr_port == 7497 else 'LIVE'}")

    async def cleanup(self):
        """Clean up resources"""
        if self.ibkr_client:
            try:
                self.ibkr_client.disconnect_safely()
            except:
                pass
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("IBKR price monitor cleanup completed")

    async def start(self):
        """Start the continuous price monitoring system with IBKR"""
        try:
            logger.info("🚀 Starting IBKR Price Monitor!")
            await self.initialize()
            
            # Verify IBKR connection
            if not self.ibkr_connected:
                raise ConnectionError("IBKR connection not established")
            
            logger.info(f"⚡ IBKR Mode: Port {self.ibkr_port} ({'PAPER' if self.ibkr_port == 7497 else 'LIVE'})")
            logger.info("✅ IBKR Price Monitor operational!")
            
            # Run file trigger monitor and polling loop in parallel
            # Note: No WebSocket listener needed - IBKR callbacks run in separate thread
            await asyncio.gather(
                self.file_trigger_monitor_async(),
                self.continuous_polling_loop()
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        except Exception as e:
            logger.error(f"Fatal error in IBKR price monitor: {e}")
            raise
        finally:
            await self.cleanup()

    async def file_trigger_monitor_async(self):
        """Async file trigger monitor that runs in main event loop - ONLY adds tickers to polling queue"""
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
                            
                            # Add to active tickers for IBKR tracking
                            self.active_tickers.add(ticker)
                            self.ticker_timestamps[ticker] = datetime.now()
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
        """Continuous polling loop - IBKR subscription management + database operations every 2 seconds"""
        logger.info("🔄 Starting IBKR POLLING LOOP - subscription management + database operations")
        
        cycle = 0
        last_cleanup = time.time()
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                # Clean up old tickers every 5 minutes
                if time.time() - last_cleanup > 300:
                    await self.cleanup_old_tickers()
                    last_cleanup = time.time()
                
                # Check IBKR connection health
                if not self.ibkr_connected or not self.ibkr_client or not self.ibkr_client.connected:
                    logger.warning("❌ IBKR disconnected - attempting reconnect...")
                    await self.reconnect_ibkr()
                    await asyncio.sleep(2.0)
                    continue
                
                if self.active_tickers:
                    logger.debug(f"🔄 IBKR CYCLE {cycle}: Managing {len(self.active_tickers)} active tickers")
                    
                    # Update IBKR subscriptions
                    await self.update_ibkr_subscriptions()
                    
                    # Process buffered prices from IBKR
                    await self.process_ibkr_prices()
                    
                    # Check for alerts
                    await self.check_price_alerts_optimized()
                    
                    cycle_time = time.time() - cycle_start
                    logger.info(f"✅ IBKR CYCLE {cycle}: Completed in {cycle_time:.3f}s")
                else:
                    logger.debug(f"⏳ IBKR CYCLE {cycle}: No active tickers")
                
                # Wait 2 seconds before next cycle
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error in IBKR polling loop: {e}")
                await asyncio.sleep(2.0)

    async def get_tickers_within_60_second_window(self) -> Set[str]:
        """
        Get tickers whose FIRST price timestamp is within the last 60 seconds.
        This prevents price data insertion for tickers that have exceeded the 60-second trading window.
        IMPORTANT: New tickers with NO price data yet should be allowed to start tracking.
        """
        try:
            if not self.active_tickers:
                return set()
            
            # Convert active tickers to list for SQL query
            ticker_list = list(self.active_tickers)
            ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
            
            # Get first timestamp for each active ticker
            query = f"""
            SELECT 
                ticker, 
                min(timestamp) as first_timestamp,
                dateDiff('second', min(timestamp), now()) as seconds_since_first
            FROM News.price_tracking
            WHERE ticker IN ({ticker_placeholders})
            GROUP BY ticker
            HAVING seconds_since_first <= 60
            """
            
            result = self.ch_manager.client.query(query)
            
            # Extract tickers that are still within the 60-second window
            valid_tickers_with_data = set()
            
            for row in result.result_rows:
                ticker, first_timestamp, seconds_since_first = row
                valid_tickers_with_data.add(ticker)
                logger.debug(f"✅ WINDOW VALID: {ticker} - {seconds_since_first}s since first price")
            
            # CRITICAL FIX: Include tickers that have NO price data yet (new tickers)
            tickers_with_no_data = self.active_tickers - valid_tickers_with_data
            
            # Check which of these actually have NO data vs expired data
            if tickers_with_no_data:
                # Check if these tickers have ANY price data at all
                no_data_placeholders = ','.join([f"'{ticker}'" for ticker in tickers_with_no_data])
                check_query = f"""
                SELECT DISTINCT ticker
                FROM News.price_tracking
                WHERE ticker IN ({no_data_placeholders})
                """
                
                check_result = self.ch_manager.client.query(check_query)
                tickers_with_any_data = {row[0] for row in check_result.result_rows}
                
                # Tickers with no data at all = new tickers that should be allowed
                truly_new_tickers = tickers_with_no_data - tickers_with_any_data
                
                # Tickers with data but not in valid window = expired tickers
                expired_tickers = tickers_with_any_data
                
                if truly_new_tickers:
                    logger.info(f"🆕 NEW TICKERS: {len(truly_new_tickers)} tickers have no price data yet, allowing tracking: {sorted(truly_new_tickers)}")
                
                if expired_tickers:
                    logger.info(f"⏰ WINDOW EXPIRED: {len(expired_tickers)} tickers exceeded 60s window: {sorted(expired_tickers)}")
                    # Remove expired tickers from active tracking
                    for ticker in expired_tickers:
                        self.active_tickers.discard(ticker)
                        if ticker in self.ticker_timestamps:
                            del self.ticker_timestamps[ticker]
                
                # Valid tickers = those within window + truly new tickers
                valid_tickers = valid_tickers_with_data | truly_new_tickers
            else:
                valid_tickers = valid_tickers_with_data
            
            if valid_tickers:
                logger.debug(f"✅ TRACKING ALLOWED: {len(valid_tickers)} tickers ready for price tracking: {sorted(valid_tickers)}")
            
            return valid_tickers
            
        except Exception as e:
            logger.error(f"Error checking 60-second window: {e}")
            # On error, return all active tickers to be safe and allow tracking
            logger.warning(f"⚠️ Window check failed, allowing all {len(self.active_tickers)} active tickers to continue tracking")
            return self.active_tickers


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
    logger.info("🚀 Starting IBKR Continuous Price Monitor")
    logger.info("⚡ IBKR MODE: Real-time streaming via TWS API → immediate price data → instant alerts")
    
    monitor = ContinuousPriceMonitor()
    await monitor.start()


if __name__ == "__main__":
    asyncio.run(main())
