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
        """Create test tables and insert test price data for deduplication testing"""
        
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
        
        # Insert TEST ticker price data for deduplication scenarios
        base_time = datetime.now(pytz.UTC)
        test_data = []
        
        # SCENARIO 1: Multiple price points that should trigger alerts
        # All within 60 seconds, all qualify (5%+ move, BUY sentiment)
        prices = [
            (0, 5.00, 'BUY', 'medium'),   # t=0s: Baseline $5.00 
            (5, 5.10, 'BUY', 'medium'),   # t=5s: +2% (not qualifying)
            (10, 5.30, 'BUY', 'medium'),  # t=10s: +6% ← SHOULD TRIGGER ALERT #1
            (15, 5.35, 'BUY', 'medium'),  # t=15s: +7% ← SHOULD TRIGGER ALERT #2  
            (20, 5.40, 'BUY', 'medium'),  # t=20s: +8% ← SHOULD TRIGGER ALERT #3
            (25, 5.45, 'BUY', 'medium'),  # t=25s: +9% ← SHOULD TRIGGER ALERT #4
            (30, 5.50, 'BUY', 'medium'),  # t=30s: +10% ← SHOULD TRIGGER ALERT #5
        ]
        
        for seconds_offset, price, recommendation, confidence in prices:
            timestamp = base_time + timedelta(seconds=seconds_offset)
            test_data.append((
                timestamp, 'TEST', price, 1000, 'polygon', 
                'positive', recommendation, confidence
            ))
        
        # Insert the test data
        self.ch_manager.client.insert(
            'News.price_tracking',
            test_data,
            column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 'sentiment', 'recommendation', 'confidence']
        )
        
        logger.info(f"✅ Inserted {len(test_data)} test price records for TEST ticker")
        logger.info("📊 Test scenario: Price moves from $5.00 to $5.50 (+10%) over 30 seconds")
        logger.info("🎯 Expected: 5 distinct alerts (one per qualifying timestamp)")
        
    async def run_alert_query(self, cycle_number: int):
        """Run the same alert detection query as production price_checker"""
        logger.info(f"🔄 Running alert detection cycle #{cycle_number}")
        
        # Exact same query structure as price_checker.py check_price_alerts_optimized()
        query = """
        WITH ticker_first_timestamps AS (
            SELECT ticker, min(timestamp) as first_timestamp
            FROM News.price_tracking
            WHERE ticker IN ('TEST')
            GROUP BY ticker
        ),
        ticker_second_prices AS (
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
        )
        SELECT 
            pt.ticker,
            argMax(pt.price, pt.timestamp) as current_price,
            COALESCE(tsp.second_price, argMin(pt.price, pt.timestamp)) as baseline_price,
            max(pt.timestamp) as current_timestamp,
            tft.first_timestamp as first_timestamp,
            count() as price_count,
            COALESCE(a.alert_count, 0) as existing_alerts,
            argMax(pt.sentiment, pt.timestamp) as sentiment,
            argMax(pt.recommendation, pt.timestamp) as recommendation,
            argMax(pt.confidence, pt.timestamp) as confidence,
            ((argMax(pt.price, pt.timestamp) - COALESCE(tsp.second_price, argMin(pt.price, pt.timestamp))) / COALESCE(tsp.second_price, argMin(pt.price, pt.timestamp))) * 100 as change_pct,
            dateDiff('second', tft.first_timestamp, max(pt.timestamp)) as seconds_elapsed,
            a.last_alerted_timestamp
        FROM News.price_tracking pt
        INNER JOIN ticker_first_timestamps tft ON pt.ticker = tft.ticker
        LEFT JOIN ticker_second_prices tsp ON pt.ticker = tsp.ticker
        LEFT JOIN (
            SELECT ticker, 
                   count() as alert_count,
                   argMax(timestamp, timestamp) as last_alerted_timestamp
            FROM News.news_alert
            WHERE timestamp >= now() - INTERVAL 2 MINUTE
            GROUP BY ticker
        ) a ON pt.ticker = a.ticker
        WHERE pt.ticker IN ('TEST')
        AND COALESCE(a.alert_count, 0) < 8
        -- SIMPLE: Only include data within 60 seconds of the first timestamp
        AND pt.timestamp <= tft.first_timestamp + INTERVAL 60 SECOND
        GROUP BY pt.ticker, a.alert_count, tft.first_timestamp, tsp.second_price, a.last_alerted_timestamp
        HAVING baseline_price > 0
        AND (a.last_alerted_timestamp IS NULL OR current_timestamp > a.last_alerted_timestamp) 
        AND price_count >= 3
        AND change_pct >= 5.0 
        AND seconds_elapsed <= 60
        AND current_price < 11.0
        AND recommendation = 'BUY'
        ORDER BY change_pct DESC
        """
        
        result = self.ch_manager.client.query(query)
        
        alerts_to_create = []
        if result.result_rows:
            for row in result.result_rows:
                ticker, current_price, baseline_price, current_timestamp, first_timestamp, price_count, existing_alerts, sentiment, recommendation, confidence, change_pct, seconds_elapsed, last_alerted_timestamp = row
                
                logger.info(f"🚨 CYCLE #{cycle_number}: Alert condition met for {ticker}")
                logger.info(f"   💰 Price: ${current_price:.2f} (+{change_pct:.2f}% from ${baseline_price:.2f})")
                logger.info(f"   ⏱️ Current timestamp: {current_timestamp}")
                logger.info(f"   📅 Last alerted timestamp: {last_alerted_timestamp}")
                logger.info(f"   📊 Existing alerts: {existing_alerts}/8")
                
                # Create the alert (same as production) - use current_timestamp for deduplication
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
        
        # Expected: Only 1 alert should be created despite multiple polling cycles
        # Because the deduplication logic should prevent re-alerting on the same current_timestamp
        expected_alerts = 1
        
        if total_alerts == expected_alerts:
            logger.info("✅ DEDUPLICATION TEST PASSED! Only 1 alert created as expected")
            return True
        else:
            logger.error(f"❌ DEDUPLICATION TEST FAILED! Expected {expected_alerts} alerts, got {total_alerts}")
            return False
    
    async def run_deduplication_test(self):
        """Main test function that simulates multiple polling cycles"""
        logger.info("🧪 Starting Alert Deduplication Test")
        logger.info("=" * 60)
        
        # Setup test data
        await self.setup_test_data()
        
        # Wait for data to settle
        await asyncio.sleep(1)
        
        # Simulate 5 polling cycles (like production runs every 2 seconds)
        alerts_created_per_cycle = []
        
        for cycle in range(1, 6):
            alerts_created = await self.run_alert_query(cycle)
            alerts_created_per_cycle.append(alerts_created)
            
            # Wait 2 seconds between cycles (simulate production timing)
            await asyncio.sleep(2)
        
        # Check results
        logger.info("\n" + "=" * 60)
        logger.info("📊 POLLING CYCLE SUMMARY:")
        for i, alerts in enumerate(alerts_created_per_cycle, 1):
            logger.info(f"   Cycle #{i}: {alerts} alerts created")
        
        # Verify final results
        success = await self.check_final_results()
        
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("🎉 DEDUPLICATION TEST COMPLETED SUCCESSFULLY!")
            logger.info("✅ The last_alerted_timestamp logic is working correctly")
        else:
            logger.error("💥 DEDUPLICATION TEST FAILED!")
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
