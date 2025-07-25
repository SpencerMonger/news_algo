#!/usr/bin/env python3
"""
NewsHead Testing Table Creator
Creates a news_testing table that mirrors the breaking_news table structure
and copies all existing data for testing purposes.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
import clickhouse_connect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news_testing_creation.log')
    ]
)
logger = logging.getLogger(__name__)

class NewsTestingTableCreator:
    """Creates and populates the news_testing table for testing purposes"""
    
    def __init__(self):
        self.host = os.getenv('CLICKHOUSE_HOST', 'localhost')
        self.port = int(os.getenv('CLICKHOUSE_PORT', 8123))
        self.username = os.getenv('CLICKHOUSE_USER', 'default')
        self.password = os.getenv('CLICKHOUSE_PASSWORD', '')
        self.database = 'News'
        self.client = None
        
    def connect(self) -> bool:
        """Establish connection to ClickHouse"""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                compress=True,
                send_receive_timeout=60
            )
            logger.info(f"✅ Connected to ClickHouse at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to ClickHouse: {e}")
            return False

    def verify_source_table_exists(self) -> bool:
        """Verify that the breaking_news table exists"""
        try:
            query = "SELECT COUNT(*) FROM News.breaking_news LIMIT 1"
            result = self.client.query(query)
            logger.info("✅ Source table 'breaking_news' exists and is accessible")
            return True
        except Exception as e:
            logger.error(f"❌ Source table 'breaking_news' not accessible: {e}")
            return False

    def get_source_table_count(self) -> int:
        """Get the total count of records in breaking_news table"""
        try:
            query = "SELECT COUNT(*) FROM News.breaking_news"
            result = self.client.query(query)
            count = result.result_rows[0][0] if result.result_rows else 0
            logger.info(f"📊 Source table contains {count:,} records")
            return count
        except Exception as e:
            logger.error(f"❌ Error getting source table count: {e}")
            return 0

    def drop_existing_testing_table(self):
        """Drop existing news_testing table if it exists"""
        try:
            self.client.command("DROP TABLE IF EXISTS News.news_testing")
            logger.info("🗑️ Dropped existing news_testing table")
        except Exception as e:
            logger.error(f"❌ Error dropping existing table: {e}")
            raise

    def create_news_testing_table(self) -> bool:
        """Create the news_testing table with identical structure to breaking_news"""
        create_table_sql = """
        CREATE TABLE News.news_testing (
            id UUID DEFAULT generateUUIDv4(),
            timestamp DateTime64(3) DEFAULT now64(),
            source String,
            ticker String,
            headline String,
            published_utc String,
            article_url String,
            summary String,
            full_content String,
            detected_at DateTime64(3) DEFAULT now64(),
            processing_latency_ms UInt32,
            market_relevant UInt8 DEFAULT 0,
            
            -- Performance tracking
            source_check_time DateTime64(3),
            content_hash String,
            
            -- Classification fields
            news_type String DEFAULT 'other',
            urgency_score UInt8 DEFAULT 0,
            
            -- Sentiment analysis fields
            sentiment String DEFAULT 'neutral',
            recommendation String DEFAULT 'HOLD',
            confidence String DEFAULT 'low',
            explanation String DEFAULT '',
            analysis_time_ms UInt32 DEFAULT 0,
            analyzed_at DateTime64(3) DEFAULT now64(),
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 3,
            INDEX idx_source (source) TYPE set(100) GRANULARITY 1,
            INDEX idx_content_hash (content_hash) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_sentiment (sentiment) TYPE set(10) GRANULARITY 1,
            INDEX idx_recommendation (recommendation) TYPE set(10) GRANULARITY 1
        ) 
        ENGINE = ReplacingMergeTree(detected_at)
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (content_hash, article_url)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("✅ news_testing table created successfully with identical structure")
            return True
        except Exception as e:
            logger.error(f"❌ Error creating news_testing table: {e}")
            return False

    def copy_data_to_testing_table(self) -> int:
        """Copy all data from breaking_news to news_testing table"""
        try:
            # Use INSERT INTO ... SELECT for efficient bulk copy
            copy_sql = """
            INSERT INTO News.news_testing 
            SELECT * FROM News.breaking_news
            """
            
            logger.info("🔄 Starting data copy operation...")
            start_time = datetime.now()
            
            self.client.command(copy_sql)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Verify the copy was successful
            copied_count = self.get_testing_table_count()
            logger.info(f"✅ Data copy completed in {duration:.2f} seconds")
            logger.info(f"📊 Copied {copied_count:,} records to news_testing table")
            
            return copied_count
            
        except Exception as e:
            logger.error(f"❌ Error copying data: {e}")
            return 0

    def get_testing_table_count(self) -> int:
        """Get the total count of records in news_testing table"""
        try:
            query = "SELECT COUNT(*) FROM News.news_testing"
            result = self.client.query(query)
            count = result.result_rows[0][0] if result.result_rows else 0
            return count
        except Exception as e:
            logger.error(f"❌ Error getting testing table count: {e}")
            return 0

    def verify_data_integrity(self) -> bool:
        """Verify that the data was copied correctly"""
        try:
            source_count = self.get_source_table_count()
            testing_count = self.get_testing_table_count()
            
            if source_count == testing_count:
                logger.info(f"✅ Data integrity verified: {testing_count:,} records match")
                return True
            else:
                logger.error(f"❌ Data integrity check failed: source={source_count:,}, testing={testing_count:,}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verifying data integrity: {e}")
            return False

    def show_testing_table_sample(self, limit: int = 5):
        """Show a sample of records from the news_testing table"""
        try:
            query = f"""
            SELECT 
                ticker, 
                headline, 
                source, 
                timestamp,
                sentiment,
                recommendation
            FROM News.news_testing 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            """
            
            result = self.client.query(query)
            
            if result.result_rows:
                logger.info(f"📋 Sample records from news_testing table (showing {len(result.result_rows)} records):")
                for i, row in enumerate(result.result_rows, 1):
                    ticker, headline, source, timestamp, sentiment, recommendation = row
                    headline_truncated = headline[:50] + "..." if len(headline) > 50 else headline
                    logger.info(f"  {i}. {ticker} | {headline_truncated} | {source} | {timestamp} | {sentiment}/{recommendation}")
            else:
                logger.warning("⚠️ No records found in news_testing table")
                
        except Exception as e:
            logger.error(f"❌ Error showing sample records: {e}")

    def close(self):
        """Close the ClickHouse connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 ClickHouse connection closed")

def main():
    """Main execution function"""
    logger.info("🚀 Starting news_testing table creation process")
    
    creator = NewsTestingTableCreator()
    
    try:
        # Step 1: Connect to ClickHouse
        if not creator.connect():
            logger.error("❌ Connection failed. Exiting.")
            return False
        
        # Step 2: Verify source table exists
        if not creator.verify_source_table_exists():
            logger.error("❌ Source table verification failed. Exiting.")
            return False
        
        # Step 3: Get source data count
        source_count = creator.get_source_table_count()
        if source_count == 0:
            logger.warning("⚠️ Source table is empty. Continuing anyway.")
        
        # Step 4: Drop existing testing table (if any)
        creator.drop_existing_testing_table()
        
        # Step 5: Create new testing table
        if not creator.create_news_testing_table():
            logger.error("❌ Table creation failed. Exiting.")
            return False
        
        # Step 6: Copy data if source table has records
        if source_count > 0:
            copied_count = creator.copy_data_to_testing_table()
            if copied_count == 0:
                logger.error("❌ Data copy failed. Exiting.")
                return False
            
            # Step 7: Verify data integrity
            if not creator.verify_data_integrity():
                logger.error("❌ Data integrity verification failed.")
                return False
        else:
            logger.info("ℹ️ No data to copy from empty source table")
        
        # Step 8: Show sample records
        creator.show_testing_table_sample()
        
        logger.info("🎉 news_testing table creation process completed successfully!")
        logger.info("💡 You can now use 'News.news_testing' table for testing purposes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in main process: {e}")
        return False
    
    finally:
        creator.close()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⛔ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1) 