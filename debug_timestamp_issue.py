#!/usr/bin/env python3
"""
Debug timestamp issue - trace exactly what happens during insertion
"""

import time
from datetime import datetime, timezone
from clickhouse_setup import ClickHouseManager

def debug_timestamps():
    """Debug the exact timestamp values and insertion process"""
    
    # Connect to ClickHouse
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    
    print("🔍 Debugging timestamp insertion process...")
    print(f"Current time: {datetime.now()}")
    
    # Create a test article with known timestamp
    test_time = datetime.now()
    print(f"Creating test article with timestamp: {test_time}")
    
    test_article = {
        'timestamp': test_time,
        'source': 'DEBUG_TEST',
        'ticker': 'DEBUG',
        'headline': 'Debug Test Article',
        'published_utc': 'DEBUG_TIME',
        'article_url': 'https://debug.test',
        'summary': 'Debug test',
        'full_content': 'Debug test content',
        'detected_at': datetime.now(),
        'processing_latency_ms': 0,
        'market_relevant': 1,
        'source_check_time': datetime.now(),
        'content_hash': f'debug_{int(time.time())}',
        'news_type': 'debug',
        'urgency_score': 5
    }
    
    print(f"Test article timestamp BEFORE insertion: {test_article['timestamp']}")
    
    # Insert the test article
    try:
        inserted_count = ch_manager.insert_articles([test_article])
        print(f"Inserted {inserted_count} test articles")
        
        # Immediately query back what was inserted
        time.sleep(0.1)  # Small delay to ensure insertion is complete
        
        query = """
        SELECT timestamp, source, ticker, headline, detected_at
        FROM News.breaking_news 
        WHERE ticker = 'DEBUG'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        result = ch_manager.client.query(query)
        if result.result_rows:
            db_timestamp = result.result_rows[0][0]
            print(f"Test article timestamp AFTER insertion: {db_timestamp}")
            print(f"Time difference: {(db_timestamp - test_time).total_seconds():.3f} seconds")
            
            if abs((db_timestamp - test_time).total_seconds()) > 1:
                print("❌ TIMESTAMP MISMATCH DETECTED!")
            else:
                print("✅ Timestamps match correctly")
        else:
            print("❌ Test article not found in database")
            
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("\n" + "="*50)
    print("🔍 Checking recent LEE articles...")
    
    # Get recent LEE articles with all timestamp fields
    query = """
    SELECT timestamp, detected_at, source, ticker, headline, published_utc
    FROM News.breaking_news 
    WHERE ticker = 'LEE'
    ORDER BY timestamp DESC
    LIMIT 3
    """
    
    result = ch_manager.client.query(query)
    for i, row in enumerate(result.result_rows):
        timestamp, detected_at, source, ticker, headline, published_utc = row
        print(f"LEE Article #{i+1}:")
        print(f"  timestamp: {timestamp}")
        print(f"  detected_at: {detected_at}")
        print(f"  published_utc: {published_utc}")
        print(f"  source: {source}")
        print(f"  headline: {headline[:50]}...")
        print()
    
    # Clean up test data
    try:
        ch_manager.client.command("DELETE FROM News.breaking_news WHERE ticker = 'DEBUG'")
        print("🧹 Cleaned up test data")
    except Exception as e:
        print(f"Error cleaning up: {e}")
    
    ch_manager.close()

if __name__ == "__main__":
    debug_timestamps() 