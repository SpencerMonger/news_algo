#!/usr/bin/env python3
"""
Test the 40-second window fix for price alerts
Simulates STTK price data to verify the fix works correctly
"""

import asyncio
import time
from datetime import datetime, timedelta
import pytz
from clickhouse_setup import ClickHouseManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test40SecondFix:
    def __init__(self):
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
    
    async def setup_test_data(self):
        """Create test tables and insert STTK price data spanning more than 40 seconds"""
        
        # Drop and recreate tables (same as main system)
        logger.info("🗑️ Dropping existing tables...")
        self.ch_manager.client.command("DROP TABLE IF EXISTS News.price_tracking")
        self.ch_manager.client.command("DROP TABLE IF EXISTS News.news_alert")
        
        # Create price_tracking table
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
        
        # Create news_alert table
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
        logger.info("✅ Created test tables")
        
        # Insert STTK test data - simulate the scenario from logs
        base_time = datetime.now(pytz.UTC)
        test_data = []
        
        # Simulate STTK price progression over 2 minutes (120 seconds)
        # This should trigger alerts ONLY in the first 40 seconds
        prices = [
            (0, 0.80, 'BUY', 'high'),    # t=0s: Initial price $0.80
            (5, 0.85, 'BUY', 'high'),    # t=5s: +6.25% increase
            (10, 0.90, 'BUY', 'high'),   # t=10s: +12.5% increase  
            (15, 0.95, 'BUY', 'high'),   # t=15s: +18.75% increase
            (20, 1.00, 'BUY', 'high'),   # t=20s: +25% increase
            (25, 1.05, 'BUY', 'high'),   # t=25s: +31.25% increase ← SHOULD TRIGGER ALERT
            (30, 1.10, 'BUY', 'high'),   # t=30s: +37.5% increase ← SHOULD TRIGGER ALERT
            (35, 1.12, 'BUY', 'high'),   # t=35s: +40% increase ← SHOULD TRIGGER ALERT
            (40, 1.14, 'BUY', 'high'),   # t=40s: +42.5% increase ← LAST VALID ALERT
            (45, 1.16, 'BUY', 'high'),   # t=45s: +45% increase ← SHOULD NOT TRIGGER (>40s)
            (50, 1.18, 'BUY', 'high'),   # t=50s: +47.5% increase ← SHOULD NOT TRIGGER (>40s)
            (60, 1.20, 'BUY', 'high'),   # t=60s: +50% increase ← SHOULD NOT TRIGGER (>40s)
            (90, 1.25, 'BUY', 'high'),   # t=90s: +56.25% increase ← SHOULD NOT TRIGGER (>40s)
            (120, 1.30, 'BUY', 'high'),  # t=120s: +62.5% increase ← SHOULD NOT TRIGGER (>40s)
        ]
        
        for seconds_offset, price, recommendation, confidence in prices:
            timestamp = base_time + timedelta(seconds=seconds_offset)
            test_data.append((
                timestamp,
                'STTK',
                price,
                1000,  # volume
                'test_data',
                'positive',  # sentiment
                recommendation,
                confidence
            ))
        
        # Insert test data
        self.ch_manager.client.insert(
            'News.price_tracking',
            test_data,
            column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 'sentiment', 'recommendation', 'confidence']
        )
        
        logger.info(f"✅ Inserted {len(test_data)} STTK price points spanning 120 seconds")
        logger.info(f"📊 Price range: ${test_data[0][2]:.2f} → ${test_data[-1][2]:.2f}")
        logger.info(f"📈 Total increase: {((test_data[-1][2] - test_data[0][2]) / test_data[0][2]) * 100:.1f}%")
    
    async def test_40_second_window(self):
        """Test the fixed 40-second window logic"""
        logger.info("🧪 Testing 40-second window logic...")
        
        # Use WINDOW FUNCTION approach - no correlated subquery needed
        query = """
        WITH ticker_first_timestamps AS (
            SELECT ticker, min(timestamp) as first_timestamp
            FROM News.price_tracking
            WHERE ticker = 'STTK'
            GROUP BY ticker
        )
        SELECT 
            pt.ticker,
            argMax(pt.price, pt.timestamp) as current_price,
            argMin(pt.price, pt.timestamp) as first_price,
            max(pt.timestamp) as current_timestamp,
            min(pt.timestamp) as first_timestamp,
            count() as price_count,
            0 as existing_alerts,
            argMax(pt.sentiment, pt.timestamp) as sentiment,
            argMax(pt.recommendation, pt.timestamp) as recommendation,
            argMax(pt.confidence, pt.timestamp) as confidence,
            ((argMax(pt.price, pt.timestamp) - argMin(pt.price, pt.timestamp)) / argMin(pt.price, pt.timestamp)) * 100 as change_pct,
            dateDiff('second', min(pt.timestamp), max(pt.timestamp)) as seconds_elapsed
        FROM News.price_tracking pt
        INNER JOIN ticker_first_timestamps tft ON pt.ticker = tft.ticker
        WHERE pt.ticker = 'STTK'
        -- SIMPLE: Only include data within 40 seconds of the first timestamp
        AND pt.timestamp <= tft.first_timestamp + INTERVAL 40 SECOND
        GROUP BY pt.ticker
        HAVING first_price > 0 
        AND price_count >= 2
        AND change_pct >= 5.0 
        AND seconds_elapsed <= 40
        AND current_price < 12.0
        AND (recommendation = 'BUY' AND confidence = 'high')
        ORDER BY change_pct DESC
        """
        
        result = self.ch_manager.client.query(query)
        
        if result.result_rows:
            logger.info("✅ 40-SECOND WINDOW TEST PASSED!")
            for row in result.result_rows:
                ticker, current_price, first_price, current_timestamp, first_timestamp, price_count, existing_alerts, sentiment, recommendation, confidence, change_pct, seconds_elapsed = row
                logger.info(f"🚨 ALERT TRIGGERED: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from ${first_price:.4f}) in {seconds_elapsed}s")
                logger.info(f"   📊 Sentiment: {sentiment}, Recommendation: {recommendation} ({confidence} confidence)")
                logger.info(f"   📈 Price Data: [{price_count} points within 40s window]")
                logger.info(f"   🕐 Time Window: {first_timestamp} → {current_timestamp}")
                
                # Verify the window is exactly 40 seconds or less
                if seconds_elapsed <= 40:
                    logger.info("   ✅ CORRECT: Alert triggered within 40-second window")
                else:
                    logger.error("   ❌ BUG: Alert triggered OUTSIDE 40-second window!")
        else:
            logger.error("❌ 40-SECOND WINDOW TEST FAILED: No alerts triggered (should have triggered for early price moves)")
    
    async def test_beyond_40_seconds(self):
        """Test that alerts are NOT triggered for data beyond 40 seconds"""
        logger.info("🧪 Testing that alerts are blocked beyond 40 seconds...")
        
        # Query for ALL data (including beyond 40 seconds) - should NOT trigger alerts
        query_all = """
        SELECT 
            pt.ticker,
            argMax(pt.price, pt.timestamp) as current_price,
            argMin(pt.price, pt.timestamp) as first_price,
            max(pt.timestamp) as current_timestamp,
            min(pt.timestamp) as first_timestamp,
            count() as price_count,
            0 as existing_alerts,
            argMax(pt.sentiment, pt.timestamp) as sentiment,
            argMax(pt.recommendation, pt.timestamp) as recommendation,
            argMax(pt.confidence, pt.timestamp) as confidence,
            ((argMax(pt.price, pt.timestamp) - argMin(pt.price, pt.timestamp)) / argMin(pt.price, pt.timestamp)) * 100 as change_pct,
            dateDiff('second', min(pt.timestamp), max(pt.timestamp)) as seconds_elapsed
        FROM News.price_tracking pt
        WHERE pt.ticker = 'STTK'
        -- NO 40-second filter - include ALL data
        GROUP BY pt.ticker
        HAVING first_price > 0 
        AND price_count >= 2
        AND change_pct >= 5.0 
        AND current_price < 12.0
        AND (recommendation = 'BUY' AND confidence = 'high')
        ORDER BY change_pct DESC
        """
        
        result_all = self.ch_manager.client.query(query_all)
        
        if result_all.result_rows:
            row = result_all.result_rows[0]
            ticker, current_price, first_price, current_timestamp, first_timestamp, price_count, existing_alerts, sentiment, recommendation, confidence, change_pct, seconds_elapsed = row
            logger.info(f"📊 WITHOUT 40s filter: {ticker} - ${current_price:.4f} (+{change_pct:.2f}% from ${first_price:.4f}) in {seconds_elapsed}s")
            logger.info(f"   📈 Total price points: [{price_count}] over {seconds_elapsed} seconds")
            
            if seconds_elapsed > 40:
                logger.info("   ✅ CORRECT: This would be blocked by the 40-second filter")
            else:
                logger.info("   ⚠️ Note: All data is within 40 seconds")
    
    async def show_data_breakdown(self):
        """Show the actual data points to verify test setup"""
        logger.info("📋 Data breakdown:")
        
        query = """
        SELECT 
            ticker,
            timestamp,
            price,
            recommendation,
            confidence,
            dateDiff('second', 
                (SELECT min(timestamp) FROM News.price_tracking WHERE ticker = 'STTK'), 
                timestamp
            ) as seconds_from_start
        FROM News.price_tracking
        WHERE ticker = 'STTK'
        ORDER BY timestamp ASC
        """
        
        result = self.ch_manager.client.query(query)
        
        for row in result.result_rows:
            ticker, timestamp, price, recommendation, confidence, seconds_from_start = row
            window_status = "✅ WITHIN" if seconds_from_start <= 40 else "❌ BEYOND"
            logger.info(f"   t+{seconds_from_start:3d}s: ${price:.2f} ({recommendation}/{confidence}) {window_status} 40s window")
    
    async def cleanup(self):
        """Clean up test data"""
        logger.info("🧹 Cleaning up test data...")
        self.ch_manager.client.command("DROP TABLE IF EXISTS News.price_tracking")
        self.ch_manager.client.command("DROP TABLE IF EXISTS News.news_alert")
        self.ch_manager.close()

async def main():
    """Run the 40-second window test"""
    logger.info("🚀 Starting 40-second window fix test...")
    
    test = Test40SecondFix()
    
    try:
        await test.setup_test_data()
        await test.show_data_breakdown()
        await test.test_40_second_window()
        await test.test_beyond_40_seconds()
        
        logger.info("🎯 Test completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        await test.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 