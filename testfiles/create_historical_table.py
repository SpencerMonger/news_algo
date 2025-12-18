#!/usr/bin/env python3
"""
Standalone script to create and populate breaking_news_historical table.

This script combines data from breaking_news and news_alert tables to create
a historical record of all processed articles and their alert outcomes.

Usage:
    python3 create_historical_table.py              # Create and populate table
    python3 create_historical_table.py --recreate   # Drop existing table and recreate
"""

import logging
import clickhouse_connect
from datetime import datetime
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalTableManager:
    """Manages the creation and population of breaking_news_historical table"""
    
    def __init__(self):
        """Initialize ClickHouse connection"""
        try:
            self.client = clickhouse_connect.get_client(
                host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
                port=int(os.getenv('CLICKHOUSE_PORT', 8123)),
                username=os.getenv('CLICKHOUSE_USER', 'default'),
                password=os.getenv('CLICKHOUSE_PASSWORD', ''),
                database='News'
            )
            logger.info("✅ Connected to ClickHouse")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ClickHouse: {e}")
            raise
    
    def drop_historical_table(self):
        """Drop the breaking_news_historical table if it exists"""
        try:
            logger.info("🗑️ Dropping existing breaking_news_historical table...")
            drop_sql = "DROP TABLE IF EXISTS News.breaking_news_historical"
            self.client.command(drop_sql)
            logger.info("✅ Table dropped successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error dropping table: {e}")
            return False
    
    def create_historical_table(self):
        """Create the breaking_news_historical table"""
        try:
            logger.info("📋 Creating breaking_news_historical table...")
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS News.breaking_news_historical (
                -- Original breaking_news fields
                id UUID,
                timestamp DateTime64(3),
                source String,
                ticker String,
                headline String,
                published_utc String,
                article_url String,
                summary String,
                full_content String,
                detected_at DateTime64(3),
                processing_latency_ms UInt32,
                market_relevant UInt8,
                source_check_time DateTime64(3),
                content_hash String,
                news_type String,
                urgency_score UInt8,
                
                -- Sentiment analysis fields
                sentiment String,
                recommendation String,
                confidence String,
                explanation String,
                analysis_time_ms UInt32,
                analyzed_at DateTime64(3),
                
                -- Alert outcome field (from news_alert table)
                alert_generated UInt8 DEFAULT 0,
                
                -- Historical tracking
                archived_at DateTime64(3) DEFAULT now64(),
                
                INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
                INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 1,
                INDEX idx_content_hash (content_hash) TYPE bloom_filter GRANULARITY 1,
                INDEX idx_sentiment (sentiment) TYPE set(10) GRANULARITY 1,
                INDEX idx_recommendation (recommendation) TYPE set(10) GRANULARITY 1,
                INDEX idx_alert_generated (alert_generated) TYPE set(2) GRANULARITY 1
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp, content_hash)
            PARTITION BY toYYYYMM(timestamp)
            """
            
            self.client.command(create_table_sql)
            logger.info("✅ breaking_news_historical table created/verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating historical table: {e}")
            return False
    
    def check_alert_table(self):
        """Check the news_alert table to understand its contents"""
        try:
            logger.info("🔍 Checking news_alert table contents...")
            
            # Check total alerts
            count_query = "SELECT COUNT(*) FROM News.news_alert"
            result = self.client.query(count_query)
            total_alerts = result.result_rows[0][0] if result.result_rows else 0
            logger.info(f"   - Total alerts in news_alert table: {total_alerts}")
            
            if total_alerts == 0:
                logger.info("   ⚠️ No alerts in news_alert table - all articles will show alert_generated=0")
                return True
            
            # Check sample alerts
            sample_query = """
            SELECT ticker, timestamp, price, alert
            FROM News.news_alert
            ORDER BY timestamp DESC
            LIMIT 5
            """
            result = self.client.query(sample_query)
            if result.result_rows:
                logger.info("   - Sample of recent alerts:")
                for ticker, timestamp, price, alert in result.result_rows:
                    logger.info(f"      {ticker} | {timestamp} | ${price:.2f} | alert={alert}")
            
            # Check for problematic alerts (null/default timestamps or prices)
            problem_query = """
            SELECT COUNT(*) 
            FROM News.news_alert 
            WHERE timestamp < toDateTime('2020-01-01 00:00:00')
               OR price <= 0
               OR alert != 1
            """
            result = self.client.query(problem_query)
            problem_count = result.result_rows[0][0] if result.result_rows else 0
            if problem_count > 0:
                logger.warning(f"   ⚠️ Found {problem_count} problematic alerts (bad timestamp/price/alert values)")
                logger.info(f"      These will be filtered out during join")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking alert table: {e}")
            return False
    
    def test_join_logic(self):
        """Test the join logic to see what it produces"""
        try:
            logger.info("🧪 Testing join logic...")
            
            # Test the subquery that gets distinct tickers from news_alert
            subquery_test = """
            SELECT COUNT(*) as distinct_ticker_count
            FROM (
                SELECT DISTINCT ticker
                FROM News.news_alert
                WHERE alert = 1
            )
            """
            result = self.client.query(subquery_test)
            ticker_count = result.result_rows[0][0] if result.result_rows else 0
            logger.info(f"   - Distinct tickers in news_alert: {ticker_count}")
            
            # Test the actual LEFT JOIN to see what happens
            join_test = """
            SELECT 
                bn.ticker,
                alert_tickers.ticker as joined_ticker,
                CASE WHEN alert_tickers.ticker IS NOT NULL AND alert_tickers.ticker != '' THEN 1 ELSE 0 END as would_be_alert_generated
            FROM News.breaking_news bn
            LEFT JOIN (
                SELECT DISTINCT ticker
                FROM News.news_alert
                WHERE alert = 1
            ) AS alert_tickers ON bn.ticker = alert_tickers.ticker
            LIMIT 3
            """
            result = self.client.query(join_test)
            if result.result_rows:
                logger.info("   - Sample LEFT JOIN results:")
                for bn_ticker, joined_ticker, alert_val in result.result_rows:
                    logger.info(f"      bn.ticker={bn_ticker}, joined_ticker={joined_ticker}, alert_generated={alert_val}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing join logic: {e}")
            return False
    
    def populate_historical_table(self):
        """Populate historical table by combining breaking_news and news_alert data"""
        try:
            logger.info("📊 Populating breaking_news_historical table...")
            
            # Simple query: just check if ticker exists in news_alert table
            insert_query = """
            INSERT INTO News.breaking_news_historical (
                id, timestamp, source, ticker, headline, published_utc, article_url,
                summary, full_content, detected_at, processing_latency_ms, market_relevant,
                source_check_time, content_hash, news_type, urgency_score,
                sentiment, recommendation, confidence, explanation, analysis_time_ms, analyzed_at,
                alert_generated,
                archived_at
            )
            SELECT 
                bn.id,
                bn.timestamp,
                bn.source,
                bn.ticker,
                bn.headline,
                bn.published_utc,
                bn.article_url,
                bn.summary,
                bn.full_content,
                bn.detected_at,
                bn.processing_latency_ms,
                bn.market_relevant,
                bn.source_check_time,
                bn.content_hash,
                bn.news_type,
                bn.urgency_score,
                bn.sentiment,
                bn.recommendation,
                bn.confidence,
                bn.explanation,
                bn.analysis_time_ms,
                bn.analyzed_at,
                
                -- Simple: if ticker exists in news_alert, set to 1, else 0
                -- Check for both NULL and empty string (ClickHouse returns '' for no match)
                CASE 
                    WHEN alert_tickers.ticker IS NOT NULL AND alert_tickers.ticker != '' THEN 1 
                    ELSE 0 
                END as alert_generated,
                
                now64() as archived_at
                
            FROM News.breaking_news bn
            LEFT JOIN (
                -- Get distinct tickers from news_alert
                SELECT DISTINCT ticker
                FROM News.news_alert
                WHERE alert = 1
            ) AS alert_tickers ON bn.ticker = alert_tickers.ticker
            
            ORDER BY bn.timestamp DESC
            """
            
            self.client.command(insert_query)
            
            # Get count of records inserted
            count_query = "SELECT COUNT(*) FROM News.breaking_news_historical"
            result = self.client.query(count_query)
            record_count = result.result_rows[0][0] if result.result_rows else 0
            
            logger.info(f"✅ Successfully populated breaking_news_historical with {record_count} records")
            
            # DEBUG: Check actual values in the table
            debug_query = """
            SELECT 
                ticker, 
                alert_generated
            FROM News.breaking_news_historical
            LIMIT 5
            """
            debug_result = self.client.query(debug_query)
            if debug_result.result_rows:
                logger.info("🐛 DEBUG - First 5 records actual values:")
                for ticker, alert_gen in debug_result.result_rows:
                    status = "✅ ALERT" if alert_gen else "❌ NO ALERT"
                    logger.info(f"   {ticker} → alert_generated={alert_gen} ({status})")
            
            # Get alert statistics
            alert_stats_query = """
            SELECT 
                COUNT(*) as total_articles,
                SUM(alert_generated) as articles_with_alerts,
                ROUND((SUM(alert_generated) / COUNT(*)) * 100, 2) as alert_percentage
            FROM News.breaking_news_historical
            """
            stats_result = self.client.query(alert_stats_query)
            if stats_result.result_rows:
                total, with_alerts, percentage = stats_result.result_rows[0]
                logger.info(f"📈 Statistics:")
                logger.info(f"   - Total articles: {total}")
                logger.info(f"   - Articles with alerts: {with_alerts}")
                logger.info(f"   - Alert rate: {percentage}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error populating historical table: {e}")
            return False
    
    def verify_data(self):
        """Verify the historical table data"""
        try:
            logger.info("🔍 Verifying historical table data...")
            
            # Sample query to show recent records
            sample_query = """
            SELECT 
                ticker,
                headline,
                detected_at,
                sentiment,
                recommendation,
                confidence,
                alert_generated
            FROM News.breaking_news_historical
            ORDER BY detected_at DESC
            LIMIT 5
            """
            
            result = self.client.query(sample_query)
            
            if result.result_rows:
                logger.info("📋 Sample of recent records:")
                for row in result.result_rows:
                    ticker, headline, detected, sentiment, recommendation, confidence, alert_gen = row
                    alert_status = "✅ ALERT" if alert_gen else "❌ NO ALERT"
                    logger.info(f"   {ticker} | {detected} | {sentiment}/{recommendation}/{confidence} | {alert_status}")
            else:
                logger.warning("⚠️ No records found in historical table")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying data: {e}")
            return False
    
    def close(self):
        """Close ClickHouse connection"""
        try:
            self.client.close()
            logger.info("👋 ClickHouse connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("🚀 Breaking News Historical Table Creation Script")
    logger.info("="*80)
    
    # Check for command line arguments
    recreate = '--recreate' in sys.argv
    
    try:
        # Initialize manager
        manager = HistoricalTableManager()
        
        # Drop table if recreate flag is set
        if recreate:
            logger.info("♻️ Recreate mode enabled - dropping existing table first")
            if not manager.drop_historical_table():
                logger.error("❌ Failed to drop existing table. Exiting.")
                return
        
        # Create historical table
        if not manager.create_historical_table():
            logger.error("❌ Failed to create historical table. Exiting.")
            return
        
        # Check alert table contents (for debugging)
        manager.check_alert_table()
        
        # Test the join logic (for debugging)
        manager.test_join_logic()
        
        # Populate historical table
        if not manager.populate_historical_table():
            logger.error("❌ Failed to populate historical table. Exiting.")
            return
        
        # Verify data
        manager.verify_data()
        
        # Close connection
        manager.close()
        
        logger.info("="*80)
        logger.info("✅ Historical table creation completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Script execution failed: {e}")
        raise


if __name__ == "__main__":
    main()

