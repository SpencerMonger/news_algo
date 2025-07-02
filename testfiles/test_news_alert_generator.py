#!/usr/bin/env python3
"""
Test script to generate dummy entries in News.news_alert table
Uses random tickers from float_list table to create test alerts
"""

import logging
import random
import time
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
            # Query random tickers with their prices from float_list
            query = f"""
            SELECT ticker, price FROM News.float_list 
            ORDER BY rand() 
            LIMIT {count}
            """
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                ticker_data = [(row[0], row[1]) for row in result.result_rows]  # (ticker, price) tuples
                tickers = [data[0] for data in ticker_data]
                logger.info(f"📊 Retrieved {len(tickers)} random tickers with prices: {ticker_data}")
                return ticker_data
            else:
                logger.warning("⚠️ No tickers found in float_list table, using fallback tickers")
                # Fallback to common test tickers with default prices if float_list is empty
                fallback_data = [('AAPL', 150.0), ('TSLA', 200.0), ('MSFT', 300.0), ('GOOGL', 100.0), ('NVDA', 400.0)]
                return fallback_data[:count]
                
        except Exception as e:
            logger.error(f"❌ Error getting random tickers: {e}")
            # Return fallback tickers with default prices
            fallback_data = [('TEST1', 5.0), ('TEST2', 10.0), ('TEST3', 15.0), ('TEST4', 20.0), ('TEST5', 25.0)]
            return fallback_data[:count]
    
    def generate_test_alerts(self, num_alerts: int = 3):
        """Generate test entries in the news_alert table"""
        try:
            # Get one random ticker with its actual price
            ticker_data = self.get_random_tickers(1)
            ticker, price = ticker_data[0]
            
            logger.info(f"🎯 Creating {num_alerts} alerts for ticker: {ticker} (price: ${price})")
            
            # Create alerts one by one with 2-second intervals
            for i in range(num_alerts):
                # Use the current time as the timestamp
                timestamp = datetime.now()
                
                # Insert single alert
                alert_data = [(ticker, timestamp, 1, price)]
                
                self.ch_manager.client.insert(
                    'News.news_alert',
                    alert_data,
                    column_names=['ticker', 'timestamp', 'alert', 'price']
                )
                
                logger.info(f"🚨 Inserted alert {i+1}/{num_alerts}: {ticker} at {timestamp} (price: ${price})")
                
                # Wait 2 seconds before next insertion (except for the last one)
                if i < num_alerts - 1:
                    logger.info("⏱️ Waiting 2 seconds before next insertion...")
                    time.sleep(2)
            
            logger.info(f"✅ Successfully inserted {num_alerts} test alerts for {ticker}")
            
            # Show what was inserted
            self.show_recent_alerts()
                
        except Exception as e:
            logger.error(f"❌ Error generating test alerts: {e}")
            raise
    
    def show_recent_alerts(self, limit: int = 10):
        """Show recent alerts from the news_alert table"""
        try:
            query = f"""
            SELECT ticker, timestamp, alert, price 
            FROM News.news_alert 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            """
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                logger.info(f"📋 Recent alerts (last {len(result.result_rows)}):")
                for i, (ticker, timestamp, alert, price) in enumerate(result.result_rows, 1):
                    logger.info(f"   {i}. {ticker} - {timestamp} (alert={alert}) (price=${price})")
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