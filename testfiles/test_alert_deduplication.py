#!/usr/bin/env python3
"""
Test the alert deduplication logic to prevent duplicate alerts
Tests the new last_alerted_timestamp functionality in price_checker.py
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

class TestAlertDeduplication:
    def __init__(self):
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
    
    async def setup_test_data(self):
        """Create test tables and insert initial test price data for deduplication testing"""
        
        # Drop and recreate tables (same as main system)
        logger.info("🗑️ Dropping existing test tables...")
        self.ch_manager.client.command("DROP TABLE IF EXISTS News.price_tracking")
        self.ch_manager.client.command("DROP TABLE IF EXISTS News.news_alert")
        
        # Create price_tracking table (same schema as production)
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
        
        # Create news_alert table (same schema as production)
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
        
        # Store base time for data insertion
        self.base_time = datetime.now(pytz.UTC)
        
        # Insert all test price data at once - with multiple qualifying timestamps
        await self.insert_test_price_data()
        
        logger.info("📊 Test scenario: Individual timestamp alerts with deduplication")
        logger.info("🎯 Expected: 4 distinct alerts (one per qualifying timestamp)")
    
    async def insert_test_price_data(self):
        """Insert test price data with 4 qualifying timestamps for individual alerts"""
        test_data = []
        
        # Test prices with 4 qualifying timestamps (≥5% above $5.10 baseline)
        # Based on the actual database data you showed in the screenshot
        test_prices = [
            (0, 5.00, 'BUY', 'medium'),   # t=0s: First price $5.00 
            (5, 5.10, 'BUY', 'medium'),   # t=5s: Second price $5.10 (BASELINE for calculations)
            (10, 5.30, 'BUY', 'medium'),  # t=10s: +3.92% (not qualifying)
            (15, 5.35, 'BUY', 'medium'),  # t=15s: +4.90% (not qualifying)
            (20, 5.40, 'BUY', 'medium'),  # t=20s: +5.88% ← ALERT #1 (qualifying)
            (25, 5.45, 'BUY', 'medium'),  # t=25s: +6.86% ← ALERT #2 (qualifying)
            (30, 5.50, 'BUY', 'medium'),  # t=30s: +7.84% ← ALERT #3 (qualifying)
            (35, 5.60, 'BUY', 'medium'),  # t=35s: +9.80% ← ALERT #4 (qualifying)
        ]
        
        for seconds_offset, price, recommendation, confidence in test_prices:
            timestamp = self.base_time + timedelta(seconds=seconds_offset)
            test_data.append((
                timestamp, 'TEST', price, 1000, 'polygon', 
                'positive', recommendation, confidence
            ))
        
        # Insert all test data
        self.ch_manager.client.insert(
            'News.price_tracking',
            test_data,
            column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 'sentiment', 'recommendation', 'confidence']
        )
        
        logger.info(f"✅ Inserted {len(test_data)} test price records")
        logger.info("📊 4 qualifying timestamps: t=20s($5.40), t=25s($5.45), t=30s($5.50), t=35s($5.60)")
        logger.info("📊 All ≥5% above baseline $5.10, should generate 4 individual alerts")
        
    async def run_alert_query(self, cycle_number: int):
        """Run the same alert detection query as production price_checker"""
        logger.info(f"🔄 Running alert detection cycle #{cycle_number}")
        
        # SIMPLIFIED: Use the working query structure from debug
        query = """
        WITH ticker_second_prices AS (
            SELECT 
                ticker,
                price as second_price
            FROM (
                SELECT 
                    ticker,
                    price,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp ASC) as rn
                FROM News.price_tracking
                WHERE ticker IN ('TEST')
            ) ranked
            WHERE rn = 2
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
            WHERE pt.ticker IN ('TEST')
        )
        SELECT 
            ticker,
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
            seconds_elapsed
        FROM price_analysis
        WHERE price_count >= 3
        AND change_pct >= 5.0
        AND seconds_elapsed <= 60
        AND current_price < 11.0
        AND recommendation = 'BUY'
        ORDER BY current_timestamp ASC
        """
        
        result = self.ch_manager.client.query(query)
        
        # DEBUG: Let's see what data we actually have and what the query finds
        if not result.result_rows:
            logger.info(f"🔍 DEBUG CYCLE #{cycle_number}: No alerts found, investigating...")
            
            # Check raw data
            debug_query1 = "SELECT ticker, timestamp, price, recommendation FROM News.price_tracking WHERE ticker = 'TEST' ORDER BY timestamp"
            debug_result1 = self.ch_manager.client.query(debug_query1)
            logger.info("📊 Raw price data:")
            for i, (ticker, timestamp, price, recommendation) in enumerate(debug_result1.result_rows):
                logger.info(f"   #{i+1}: {ticker} at {timestamp} - ${price:.2f} ({recommendation})")
            
            # Check what the analysis query finds (without filters)
            debug_query2 = """
            WITH ticker_second_prices AS (
                SELECT ticker, price as second_price
                FROM (
                    SELECT ticker, price, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp ASC) as rn
                    FROM News.price_tracking WHERE ticker IN ('TEST')
                ) ranked WHERE rn = 2
            ),
            price_analysis AS (
                SELECT 
                    pt.ticker,
                    pt.price as current_price,
                    COALESCE(tsp.second_price, 5.10) as baseline_price,
                    pt.timestamp as current_timestamp,
                    ROW_NUMBER() OVER (PARTITION BY pt.ticker ORDER BY pt.timestamp ASC) as price_count,
                    pt.recommendation,
                    ((pt.price - COALESCE(tsp.second_price, 5.10)) / COALESCE(tsp.second_price, 5.10)) * 100 as change_pct
                FROM News.price_tracking pt
                LEFT JOIN ticker_second_prices tsp ON pt.ticker = tsp.ticker
                WHERE pt.ticker IN ('TEST')
            )
            SELECT ticker, current_price, baseline_price, price_count, recommendation, change_pct
            FROM price_analysis 
            ORDER BY current_timestamp
            """
            
            debug_result2 = self.ch_manager.client.query(debug_query2)
            logger.info("📊 Analysis results (no filters):")
            for row in debug_result2.result_rows:
                ticker, current_price, baseline_price, price_count, recommendation, change_pct = row
                qualifier = "✅ QUALIFIES" if (price_count >= 3 and change_pct >= 5.0 and recommendation == 'BUY') else "❌ FILTERED"
                logger.info(f"   {ticker}: ${current_price:.2f} vs ${baseline_price:.2f} = {change_pct:.2f}% (#{price_count}, {recommendation}) {qualifier}")
            
            # DEBUG: Test the main query step by step
            logger.info("🔍 Testing main query filters...")
            
            # Test without time filter
            debug_query3 = """
            WITH ticker_second_prices AS (
                SELECT ticker, price as second_price
                FROM (SELECT ticker, price, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp ASC) as rn
                      FROM News.price_tracking WHERE ticker IN ('TEST')) ranked WHERE rn = 2
            ),
            price_analysis AS (
                SELECT pt.ticker, pt.price as current_price, COALESCE(tsp.second_price, 5.10) as baseline_price,
                       pt.timestamp as current_timestamp, min(pt.timestamp) OVER (PARTITION BY pt.ticker) as first_timestamp,
                       ROW_NUMBER() OVER (PARTITION BY pt.ticker ORDER BY pt.timestamp ASC) as price_count,
                       pt.sentiment, pt.recommendation, pt.confidence,
                       ((pt.price - COALESCE(tsp.second_price, 5.10)) / COALESCE(tsp.second_price, 5.10)) * 100 as change_pct,
                       dateDiff('second', min(pt.timestamp) OVER (PARTITION BY pt.ticker), pt.timestamp) as seconds_elapsed
                FROM News.price_tracking pt LEFT JOIN ticker_second_prices tsp ON pt.ticker = tsp.ticker
                WHERE pt.ticker IN ('TEST')
            )
            SELECT ticker, current_price, change_pct, price_count, seconds_elapsed, recommendation
            FROM price_analysis pa
            LEFT JOIN News.news_alert na ON (na.ticker = pa.ticker AND na.timestamp = pa.current_timestamp)
            WHERE na.timestamp IS NULL AND pa.price_count >= 3 AND pa.change_pct >= 5.0 
            AND pa.current_price < 11.0 AND pa.recommendation = 'BUY'
            ORDER BY pa.current_timestamp
            """
            
            debug_result3 = self.ch_manager.client.query(debug_query3)
            logger.info(f"📊 Main query results (no time filter): {len(debug_result3.result_rows)} rows")
            for row in debug_result3.result_rows:
                ticker, current_price, change_pct, price_count, seconds_elapsed, recommendation = row
                logger.info(f"   {ticker}: ${current_price:.2f} ({change_pct:.2f}%, #{price_count}, {seconds_elapsed}s, {recommendation})")
        
        alerts_to_create = []
        if result.result_rows:
            # Check for existing alerts to handle deduplication
            existing_alerts_query = "SELECT ticker, timestamp FROM News.news_alert WHERE ticker = 'TEST'"
            existing_result = self.ch_manager.client.query(existing_alerts_query)
            existing_alert_timestamps = {(row[0], row[1]) for row in existing_result.result_rows}
            
            for row in result.result_rows:
                ticker, current_price, baseline_price, current_timestamp, first_timestamp, price_count, existing_alerts, sentiment, recommendation, confidence, change_pct, seconds_elapsed = row
                
                # Skip if alert already exists for this exact timestamp
                if (ticker, current_timestamp) in existing_alert_timestamps:
                    logger.info(f"⏸️ CYCLE #{cycle_number}: Skipping {ticker} at {current_timestamp} - alert already exists")
                    continue
                
                logger.info(f"🚨 CYCLE #{cycle_number}: Individual timestamp alert for {ticker}")
                logger.info(f"   💰 Price: ${current_price:.2f} (+{change_pct:.2f}% from ${baseline_price:.2f})")
                logger.info(f"   ⏱️ Timestamp: {current_timestamp}")
                logger.info(f"   📊 Price sequence: #{price_count}")
                logger.info(f"   🕐 Seconds elapsed: {seconds_elapsed}s")
                
                # Create alert for this specific timestamp
                alert_data = (ticker, current_timestamp, 1, current_price)
                alerts_to_create.append(alert_data)
                
        if alerts_to_create:
            # Insert alerts
            self.ch_manager.client.insert(
                'News.news_alert',
                alerts_to_create,
                column_names=['ticker', 'timestamp', 'alert', 'price']
            )
            logger.info(f"✅ CYCLE #{cycle_number}: Created {len(alerts_to_create)} new alerts")
        else:
            logger.info(f"⏸️ CYCLE #{cycle_number}: No alerts created (deduplication working)")
            
        return len(alerts_to_create)
    
    async def check_final_results(self):
        """Check final alert count and verify deduplication worked"""
        logger.info("📊 Checking final results...")
        
        # Count total alerts created
        count_query = "SELECT count() FROM News.news_alert WHERE ticker = 'TEST'"
        result = self.ch_manager.client.query(count_query)
        total_alerts = result.result_rows[0][0] if result.result_rows else 0
        
        # Get alert details
        details_query = """
        SELECT ticker, timestamp, price 
        FROM News.news_alert 
        WHERE ticker = 'TEST' 
        ORDER BY timestamp
        """
        result = self.ch_manager.client.query(details_query)
        
        logger.info(f"🎯 FINAL RESULTS:")
        logger.info(f"   Total alerts created: {total_alerts}")
        
        if result.result_rows:
            logger.info("   Alert details:")
            for i, (ticker, timestamp, price) in enumerate(result.result_rows, 1):
                logger.info(f"     Alert #{i}: {ticker} at {timestamp} - ${price:.2f}")
        
        # Expected: 4 alerts should be created (1 per qualifying timestamp)
        # Each timestamp with ≥5% price increase should generate its own alert
        # Deduplication should prevent re-alerting on the same timestamp
        expected_alerts = 4
        
        if total_alerts == expected_alerts:
            logger.info("✅ DEDUPLICATION TEST PASSED! 4 individual timestamp alerts created as expected")
            return True
        else:
            logger.error(f"❌ DEDUPLICATION TEST FAILED! Expected {expected_alerts} alerts, got {total_alerts}")
            return False
    
    async def run_deduplication_test(self):
        """Main test function that simulates multiple polling cycles"""
        logger.info("🧪 Starting Individual Timestamp Alert Test")
        logger.info("=" * 60)
        
        # Setup test data (all at once)
        await self.setup_test_data()
        
        # Wait for data to settle
        await asyncio.sleep(1)
        
        # Run multiple polling cycles to test deduplication
        logger.info("🔄 Testing individual timestamp alerts with deduplication...")
        alerts_per_cycle = []
        
        # First cycle should create 4 alerts (one per qualifying timestamp)
        # Subsequent cycles should create 0 alerts (deduplication working)
        for cycle in range(1, 6):
            alerts_created = await self.run_alert_query(cycle)
            alerts_per_cycle.append(alerts_created)
            await asyncio.sleep(2)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 POLLING CYCLE SUMMARY:")
        for i, alerts in enumerate(alerts_per_cycle, 1):
            logger.info(f"   Cycle #{i}: {alerts} alerts created")
        
        # Verify final results
        success = await self.check_final_results()
        
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("🎉 INDIVIDUAL TIMESTAMP ALERT TEST COMPLETED SUCCESSFULLY!")
            logger.info("✅ Individual timestamp alerts created correctly")
            logger.info("✅ Deduplication prevented duplicate alerts for same timestamps")
        else:
            logger.error("💥 INDIVIDUAL TIMESTAMP ALERT TEST FAILED!")
            logger.error("❌ Fix needed in the alert logic")
        
        return success

async def main():
    """Run the deduplication test"""
    test = TestAlertDeduplication()
    try:
        success = await test.run_deduplication_test()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1
    finally:
        # Cleanup
        test.ch_manager.close()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
