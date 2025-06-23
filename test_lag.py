#!/usr/bin/env python3
"""
Test script to measure the exact lag between article insertion and price tracking handoff.
This will help identify where the delay is occurring.
"""

import asyncio
import time
import json
import os
import threading
from datetime import datetime
from clickhouse_setup import ClickHouseManager
import logging
import aiohttp
import pytz

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestPriceChecker:
    """Minimal price checker for testing lag"""
    def __init__(self, test_ticker):
        self.test_ticker = test_ticker
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        self.session = None
        self.polygon_api_key = "GtmktmIJcz6MaL8Yfkx4qzLpLZMbkJRH"
        self.base_url = "https://api.polygon.io"
        self.running = True
        
    async def initialize(self):
        """Initialize the price checker"""
        timeout = aiohttp.ClientTimeout(total=3.0)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
    async def get_current_price(self, ticker: str):
        """Get current price for ticker - simplified version"""
        try:
            url = f"{self.base_url}/v2/last/nbbo/{ticker}"
            params = {'apikey': self.polygon_api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data and data['results']:
                        result = data['results']
                        bid = result.get('P', 0.0)
                        ask = result.get('p', 0.0)
                        if bid > 0 and ask > 0:
                            return (bid + ask) / 2
                        
            # Fallback to fake price for testing
            return 1.0 + (hash(ticker) % 100) / 100  # Fake but consistent price
            
        except Exception as e:
            logger.debug(f"Price API error for {ticker}: {e}, using fake price")
            return 1.0 + (hash(ticker) % 100) / 100  # Fake but consistent price
    
    async def track_price_immediate(self, ticker: str):
        """Track price immediately and insert into database"""
        try:
            price = await self.get_current_price(ticker)
            
            # Insert into price_tracking table
            price_data = [(
                datetime.now(),
                ticker,
                price,
                0,  # volume
                'test_api'
            )]
            
            self.ch_manager.client.insert(
                'News.price_tracking',
                price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
            )
            
            logger.info(f"💰 TEST PRICE TRACKED: {ticker} = ${price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking price for {ticker}: {e}")
            return False
    
    async def monitor_trigger_files(self):
        """Monitor trigger files and process them immediately"""
        trigger_dir = "triggers"
        logger.info(f"🔍 TEST PRICE CHECKER: Monitoring trigger files for {self.test_ticker}")
        
        while self.running:
            try:
                import glob
                trigger_pattern = os.path.join(trigger_dir, f"immediate_{self.test_ticker}_*.json")
                trigger_files = glob.glob(trigger_pattern)
                
                if trigger_files:
                    for trigger_file in trigger_files:
                        try:
                            # Read trigger data
                            with open(trigger_file, 'r') as f:
                                trigger_data = json.load(f)
                            
                            ticker = trigger_data['ticker']
                            logger.info(f"⚡ TEST PRICE CHECKER: Processing trigger for {ticker}")
                            
                            # Track price immediately
                            await self.track_price_immediate(ticker)
                            
                            # Remove trigger file
                            os.remove(trigger_file)
                            logger.info(f"✅ TEST PRICE CHECKER: Processed and removed trigger for {ticker}")
                            
                        except Exception as e:
                            logger.error(f"Error processing trigger file {trigger_file}: {e}")
                            try:
                                os.remove(trigger_file)
                            except:
                                pass
                
                await asyncio.sleep(0.001)  # Check every 1ms
                
            except Exception as e:
                logger.error(f"Error in trigger file monitor: {e}")
                await asyncio.sleep(0.001)
    
    async def start(self):
        """Start the test price checker"""
        await self.initialize()
        await self.monitor_trigger_files()
    
    async def stop(self):
        """Stop the test price checker"""
        self.running = False
        if self.session:
            await self.session.close()

class LagTester:
    def __init__(self):
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()  # Ensure connection is established
        self.ch_manager.create_database()  # Ensure database exists
        self.test_ticker = f"TEST{int(time.time() % 10000)}"  # Unique test ticker
        self.insertion_time = None
        self.trigger_file_created = None
        self.db_notification_sent = None
        self.price_tracking_started = None
        self.price_checker = None
        
    def insert_test_article(self):
        """Insert a test article and measure timing"""
        logger.info(f"🧪 INSERTING TEST ARTICLE for ticker: {self.test_ticker}")
        
        # Record insertion start time
        self.insertion_time = time.time()
        insertion_timestamp = datetime.now()
        
        # Create test article data
        test_article = {
            'timestamp': insertion_timestamp,
            'source': 'LAG_TEST',
            'ticker': self.test_ticker,
            'headline': f'LAG TEST: Testing detection speed for {self.test_ticker}',
            'published_utc': insertion_timestamp.isoformat(),
            'article_url': f'https://test.com/lag-test-{self.test_ticker}',
            'summary': 'This is a test article to measure detection lag',
            'full_content': 'Test content for lag measurement',
            'detected_at': insertion_timestamp,
            'processing_latency_ms': 0,
            'market_relevant': 1,
            'source_check_time': insertion_timestamp,
            'content_hash': f'test_hash_{self.test_ticker}_{int(time.time())}',
            'news_type': 'test',
            'urgency_score': 10
        }
        
        # Insert the article using the same method as the real system
        inserted_count = self.ch_manager.insert_articles([test_article])
        
        insertion_complete_time = time.time()
        insertion_duration = insertion_complete_time - self.insertion_time
        
        logger.info(f"📝 ARTICLE INSERTED: {self.test_ticker} in {insertion_duration:.3f}s")
        logger.info(f"⏰ INSERTION_TIME: {self.insertion_time:.6f}")
        
        return inserted_count > 0
    
    def monitor_trigger_files(self):
        """Monitor for trigger file creation"""
        trigger_dir = "triggers"
        if not os.path.exists(trigger_dir):
            os.makedirs(trigger_dir)
            
        trigger_pattern = f"immediate_{self.test_ticker}_*.json"
        
        start_monitor = time.time()
        logger.info(f"👀 MONITORING for trigger file: {trigger_pattern}")
        
        while time.time() - start_monitor < 5:  # Monitor for 5 seconds max
            import glob
            trigger_files = glob.glob(os.path.join(trigger_dir, trigger_pattern))
            
            if trigger_files:
                self.trigger_file_created = time.time()
                trigger_lag = self.trigger_file_created - self.insertion_time
                
                logger.info(f"📁 TRIGGER FILE CREATED: {trigger_files[0]}")
                logger.info(f"⏰ TRIGGER_FILE_TIME: {self.trigger_file_created:.6f}")
                logger.info(f"🕐 TRIGGER_LAG: {trigger_lag:.3f}s")
                
                # Read trigger file content
                try:
                    with open(trigger_files[0], 'r') as f:
                        trigger_data = json.load(f)
                    logger.info(f"📄 TRIGGER_CONTENT: {trigger_data}")
                except Exception as e:
                    logger.error(f"Error reading trigger file: {e}")
                
                return True
            
            time.sleep(0.001)  # Check every 1ms
        
        logger.warning(f"⚠️ NO TRIGGER FILE CREATED after 5 seconds")
        return False
    
    def monitor_price_tracking(self):
        """Monitor for price tracking to start"""
        start_monitor = time.time()
        logger.info(f"💰 MONITORING for price tracking: {self.test_ticker}")
        
        while time.time() - start_monitor < 10:  # Monitor for 10 seconds max
            # Check price_tracking table for our test ticker
            try:
                query = """
                SELECT timestamp, ticker, price, source
                FROM News.price_tracking
                WHERE ticker = %s
                ORDER BY timestamp DESC
                LIMIT 1
                """
                
                result = self.ch_manager.client.query(query, [self.test_ticker])
                
                if result.result_rows:
                    self.price_tracking_started = time.time()
                    price_lag = self.price_tracking_started - self.insertion_time
                    
                    row = result.result_rows[0]
                    timestamp, ticker, price, source = row
                    
                    logger.info(f"💰 PRICE TRACKING STARTED: {ticker} = ${price}")
                    logger.info(f"⏰ PRICE_TRACKING_TIME: {self.price_tracking_started:.6f}")
                    logger.info(f"🕐 TOTAL_LAG: {price_lag:.3f}s")
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Error checking price tracking: {e}")
            
            time.sleep(0.01)  # Check every 10ms
        
        logger.warning(f"⚠️ NO PRICE TRACKING STARTED after 10 seconds")
        return False
    
    async def run_async_test(self):
        """Run the async part of the test with price checker"""
        # Start the test price checker
        self.price_checker = TestPriceChecker(self.test_ticker)
        price_checker_task = asyncio.create_task(self.price_checker.start())
        
        # Give price checker time to initialize
        await asyncio.sleep(0.1)
        
        # Insert test article (this will create trigger file)
        if not self.insert_test_article():
            logger.error("❌ Failed to insert test article")
            await self.price_checker.stop()
            return
        
        # Monitor for price tracking (the price checker should process the trigger)
        price_tracking_started = self.monitor_price_tracking()
        
        # Stop the price checker
        await self.price_checker.stop()
        price_checker_task.cancel()
        
        return price_tracking_started
    
    def run_test(self):
        """Run the complete lag test"""
        logger.info("🚀 STARTING COMPLETE LAG TEST WITH PRICE CHECKER")
        logger.info("=" * 60)
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            price_tracking_started = loop.run_until_complete(self.run_async_test())
        finally:
            loop.close()
        
        # Monitor for trigger file creation in parallel
        trigger_thread = threading.Thread(target=self.monitor_trigger_files)
        trigger_thread.daemon = True
        trigger_thread.start()
        trigger_thread.join(timeout=1)
        
        # Print summary
        self.print_summary()
        
        # Cleanup
        self.cleanup()
    
    def print_summary(self):
        """Print test summary"""
        logger.info("=" * 60)
        logger.info("📊 COMPLETE LAG TEST SUMMARY")
        logger.info("=" * 60)
        
        if self.insertion_time:
            logger.info(f"📝 Article Insertion: {datetime.fromtimestamp(self.insertion_time).strftime('%H:%M:%S.%f')[:-3]}")
        
        if self.trigger_file_created:
            trigger_lag = self.trigger_file_created - self.insertion_time
            logger.info(f"📁 Trigger File Created: {datetime.fromtimestamp(self.trigger_file_created).strftime('%H:%M:%S.%f')[:-3]} (+{trigger_lag:.3f}s)")
        else:
            logger.info(f"📁 Trigger File Created: ❌ NOT DETECTED")
        
        if self.price_tracking_started:
            total_lag = self.price_tracking_started - self.insertion_time
            logger.info(f"💰 Price Tracking Started: {datetime.fromtimestamp(self.price_tracking_started).strftime('%H:%M:%S.%f')[:-3]} (+{total_lag:.3f}s)")
        else:
            logger.info(f"💰 Price Tracking Started: ❌ NOT DETECTED")
        
        logger.info("=" * 60)
        
        if self.price_tracking_started:
            total_lag = self.price_tracking_started - self.insertion_time
            if total_lag < 0.1:
                logger.info(f"✅ EXCELLENT: Total lag {total_lag:.3f}s < 100ms")
            elif total_lag < 0.5:
                logger.info(f"⚠️ ACCEPTABLE: Total lag {total_lag:.3f}s < 500ms")
            else:
                logger.info(f"❌ TOO SLOW: Total lag {total_lag:.3f}s > 500ms")
        else:
            logger.info(f"❌ FAILED: Price tracking never started")
    
    def cleanup(self):
        """Clean up test data"""
        logger.info(f"🧹 CLEANING UP test data for {self.test_ticker}")
        
        try:
            # Remove trigger files
            import glob
            trigger_files = glob.glob(f"triggers/immediate_{self.test_ticker}_*.json")
            for trigger_file in trigger_files:
                os.remove(trigger_file)
                logger.info(f"🗑️ Removed trigger file: {trigger_file}")
        except Exception as e:
            logger.warning(f"Error cleaning trigger files: {e}")
        
        # Note: We'll leave the test data in the database for analysis

def main():
    """Main function"""
    tester = LagTester()
    tester.run_test()

if __name__ == "__main__":
    main() 