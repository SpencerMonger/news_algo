#!/usr/bin/env python3
"""
Fix price_tracking table schema and clear old NBBO data
This script will recreate the price_tracking table with the correct schema
"""

import logging
from clickhouse_setup import ClickHouseManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_price_tracking_table():
    """Fix the price_tracking table schema"""
    ch_manager = ClickHouseManager()
    
    try:
        # Connect to ClickHouse
        ch_manager.connect()
        logger.info("Connected to ClickHouse")
        
        # Check current table structure
        logger.info("Checking current price_tracking table structure...")
        try:
            result = ch_manager.client.query("DESCRIBE TABLE News.price_tracking")
            logger.info("Current table structure:")
            for row in result.result_rows:
                logger.info(f"  {row}")
        except Exception as e:
            logger.info(f"Table doesn't exist or error: {e}")
        
        # Drop the existing table
        logger.info("Dropping existing price_tracking table...")
        ch_manager.client.command("DROP TABLE IF EXISTS News.price_tracking")
        logger.info("✅ Dropped existing price_tracking table")
        
        # Recreate with correct schema
        logger.info("Creating new price_tracking table with correct schema...")
        price_tracking_sql = """
        CREATE TABLE News.price_tracking (
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
        
        ch_manager.client.command(price_tracking_sql)
        logger.info("✅ Created new price_tracking table")
        
        # Verify new table structure
        logger.info("Verifying new table structure...")
        result = ch_manager.client.query("DESCRIBE TABLE News.price_tracking")
        logger.info("New table structure:")
        for row in result.result_rows:
            logger.info(f"  {row}")
        
        logger.info("🎉 Price tracking table schema fix completed!")
        logger.info("🔍 Next price data will use:")
        logger.info("   - 'trade' as source when using last trade endpoint")
        logger.info("   - 'polygon' as fallback source")
        logger.info("   - No more 'nbbo' entries")
        
    except Exception as e:
        logger.error(f"Error fixing price_tracking table: {e}")
        raise
    finally:
        ch_manager.close()

if __name__ == "__main__":
    print("🔧 FIXING PRICE_TRACKING TABLE SCHEMA")
    print("This will:")
    print("1. Drop the existing price_tracking table (with old 'nbbo' data)")
    print("2. Recreate it with the correct schema")
    print("3. Future price data will use 'trade' or 'polygon' as source")
    print()
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        fix_price_tracking_table()
        print("\n✅ Schema fix completed! Run your price checker to see new 'trade' sources.")
    else:
        print("❌ Cancelled") 