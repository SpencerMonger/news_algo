#!/usr/bin/env python3
"""
Test script to generate dummy entries in News.news_alert table
Uses random tickers from float_list table to create test alerts
"""

import logging
import random
from datetime import datetime, timedelta
from clickhouse_setup import setup_clickhouse_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsAlertTestGenerator:
    def __init__(self):
        self.ch_manager = None
        
    def initialize(self):
        """Initialize ClickHouse connection"""
        try:
            self.ch_manager = setup_clickhouse_database()
            logger.info("✅ Connected to ClickHouse database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ClickHouse: {e}")
            raise
    
    def get_random_tickers(self, count: int = 5) -> list:
        """Get random tickers from the float_list table"""
        try:
            # Query random tickers from float_list
            query = f"""
            SELECT ticker FROM News.float_list 
            ORDER BY rand() 
            LIMIT {count}
            """
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                tickers = [row[0] for row in result.result_rows]
                logger.info(f"📊 Retrieved {len(tickers)} random tickers: {tickers}")
                return tickers
            else:
                logger.warning("⚠️ No tickers found in float_list table, using fallback tickers")
                # Fallback to common test tickers if float_list is empty
                return ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA'][:count]
                
        except Exception as e:
            logger.error(f"❌ Error getting random tickers: {e}")
            # Return fallback tickers
            return ['TEST1', 'TEST2', 'TEST3', 'TEST4', 'TEST5'][:count]
    
    def generate_test_alerts(self, num_alerts: int = 3):
        """Generate test entries in the news_alert table"""
        try:
            # Get random tickers
            random_tickers = self.get_random_tickers(num_alerts)
            
            # Prepare alert data
            alert_data = []
            for ticker in random_tickers:
                # Generate slightly random timestamps (within last 5 minutes)
                random_offset = random.randint(0, 300)  # 0-5 minutes ago
                timestamp = datetime.now() - timedelta(seconds=random_offset)
                
                alert_data.append((ticker, timestamp, 1))
                logger.info(f"🚨 Preparing test alert: {ticker} at {timestamp}")
            
            # Batch insert all alerts into news_alert table
            if alert_data:
                self.ch_manager.client.insert(
                    'News.news_alert',
                    alert_data,
                    column_names=['ticker', 'timestamp', 'alert']
                )
                
                logger.info(f"✅ Successfully inserted {len(alert_data)} test alerts into News.news_alert table")
                
                # Show what was inserted
                self.show_recent_alerts()
                
            else:
                logger.warning("⚠️ No alert data to insert")
                
        except Exception as e:
            logger.error(f"❌ Error generating test alerts: {e}")
            raise
    
    def show_recent_alerts(self, limit: int = 10):
        """Show recent alerts from the news_alert table"""
        try:
            query = f"""
            SELECT ticker, timestamp, alert 
            FROM News.news_alert 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            """
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                logger.info(f"📋 Recent alerts (last {len(result.result_rows)}):")
                for i, (ticker, timestamp, alert) in enumerate(result.result_rows, 1):
                    logger.info(f"   {i}. {ticker} - {timestamp} (alert={alert})")
            else:
                logger.info("📋 No alerts found in news_alert table")
                
        except Exception as e:
            logger.error(f"❌ Error querying recent alerts: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.ch_manager:
            self.ch_manager.close()
            logger.info("🔒 Database connection closed")

def main():
    """Main function to run the test alert generator"""
    logger.info("🚀 Starting News Alert Test Generator")
    
    generator = NewsAlertTestGenerator()
    
    try:
        # Initialize connection
        generator.initialize()
        
        # Show current state before generating
        logger.info("📊 Current state of news_alert table:")
        generator.show_recent_alerts()
        
        # Generate test alerts
        num_alerts = 3  # Change this number to generate more/fewer alerts
        logger.info(f"🎯 Generating {num_alerts} test alerts...")
        generator.generate_test_alerts(num_alerts)
        
        logger.info("✅ Test alert generation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main() 