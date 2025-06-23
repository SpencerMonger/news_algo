#!/usr/bin/env python3
"""
Simple synchronous lag test to measure exact timing
"""

import time
import json
import os
import glob
from datetime import datetime
from clickhouse_setup import ClickHouseManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lag():
    """Simple lag test"""
    
    # Initialize
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    ch_manager.create_database()
    
    test_ticker = f"TEST{int(time.time() % 10000)}"
    logger.info(f"🧪 TESTING LAG FOR: {test_ticker}")
    
    # Ensure triggers directory exists
    trigger_dir = "triggers"
    if not os.path.exists(trigger_dir):
        os.makedirs(trigger_dir)
    
    # Step 1: Record start time and insert article
    insertion_start = time.time()
    logger.info(f"📝 INSERTING ARTICLE: {test_ticker} at {insertion_start:.6f}")
    
    test_article = {
        'timestamp': datetime.now(),
        'source': 'LAG_TEST',
        'ticker': test_ticker,
        'headline': f'LAG TEST: Testing detection speed for {test_ticker}',
        'published_utc': datetime.now().isoformat(),
        'article_url': f'https://test.com/lag-test-{test_ticker}',
        'summary': 'This is a test article to measure detection lag',
        'full_content': 'Test content for lag measurement',
        'detected_at': datetime.now(),
        'processing_latency_ms': 0,
        'market_relevant': 1,
        'source_check_time': datetime.now(),
        'content_hash': f'test_hash_{test_ticker}_{int(time.time())}',
        'news_type': 'test',
        'urgency_score': 10
    }
    
    # Insert article
    inserted_count = ch_manager.insert_articles([test_article])
    insertion_complete = time.time()
    
    logger.info(f"✅ ARTICLE INSERTED: {test_ticker} in {insertion_complete - insertion_start:.6f}s")
    
    # Step 2: Immediately check for trigger file
    trigger_pattern = os.path.join(trigger_dir, f"immediate_{test_ticker}_*.json")
    trigger_found_time = None
    
    logger.info(f"🔍 CHECKING FOR TRIGGER: {trigger_pattern}")
    
    # Check for trigger file with 1ms precision
    for i in range(5000):  # Check for 5 seconds
        trigger_files = glob.glob(trigger_pattern)
        if trigger_files:
            trigger_found_time = time.time()
            trigger_lag = trigger_found_time - insertion_start
            
            logger.info(f"📁 TRIGGER FOUND: {trigger_files[0]} after {trigger_lag:.6f}s")
            
            # Read trigger content
            with open(trigger_files[0], 'r') as f:
                trigger_data = json.load(f)
            logger.info(f"📄 TRIGGER DATA: {trigger_data}")
            
            # Step 3: Simulate price tracking
            price_track_start = time.time()
            
            # Insert fake price data
            price_data = [(
                datetime.now(),
                test_ticker,
                1.234,  # fake price
                0,  # volume
                'test_api'
            )]
            
            ch_manager.client.insert(
                'News.price_tracking',
                price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
            )
            
            price_track_complete = time.time()
            total_lag = price_track_complete - insertion_start
            
            logger.info(f"💰 PRICE TRACKED: {test_ticker} = $1.234 after {total_lag:.6f}s")
            
            # Step 4: Summary
            logger.info("=" * 60)
            logger.info("📊 SIMPLE LAG TEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"📝 Article Insertion: {insertion_complete - insertion_start:.6f}s")
            logger.info(f"📁 Trigger Detection: {trigger_lag:.6f}s")
            logger.info(f"💰 Price Tracking: {total_lag:.6f}s")
            logger.info("=" * 60)
            
            if total_lag < 0.1:
                logger.info(f"✅ EXCELLENT: Total lag {total_lag:.6f}s < 100ms")
            elif total_lag < 0.5:
                logger.info(f"⚠️ ACCEPTABLE: Total lag {total_lag:.6f}s < 500ms")
            else:
                logger.info(f"❌ TOO SLOW: Total lag {total_lag:.6f}s > 500ms")
            
            # Cleanup
            os.remove(trigger_files[0])
            logger.info(f"🧹 CLEANED UP: {trigger_files[0]}")
            
            return total_lag
        
        time.sleep(0.001)  # Check every 1ms
    
    logger.error(f"❌ NO TRIGGER FILE FOUND after 5 seconds")
    return None

if __name__ == "__main__":
    test_lag() 