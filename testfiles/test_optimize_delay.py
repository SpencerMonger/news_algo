#!/usr/bin/env python3
"""
Test to measure the delay caused by OPTIMIZE TABLE FINAL operation
"""

import time
import asyncio
from datetime import datetime
from clickhouse_setup import ClickHouseManager

def test_optimize_delay():
    """Test how long OPTIMIZE TABLE FINAL takes and if it blocks queries"""
    
    # Connect to ClickHouse
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    
    print("🔬 Testing OPTIMIZE TABLE FINAL delay...")
    
    # Insert a test article
    test_article = [{
        'timestamp': datetime.now(),
        'source': 'OPTIMIZE_TEST',
        'ticker': 'OPTTEST',
        'headline': 'Optimize Test Article',
        'published_utc': '12:00 ET',
        'article_url': 'https://test.com/optimize',
        'summary': 'Testing optimize delay',
        'full_content': 'Full content for optimize test',
        'detected_at': datetime.now(),
        'processing_latency_ms': 0,
        'market_relevant': 1,
        'source_check_time': datetime.now(),
        'content_hash': 'optimize_test_hash',
        'news_type': 'other',
        'urgency_score': 5
    }]
    
    # Time the insertion without OPTIMIZE
    print("\n1️⃣ Testing insertion WITHOUT OPTIMIZE TABLE FINAL...")
    start_time = time.time()
    
    # Insert without calling force_merge
    data_rows = [[
        test_article[0]['timestamp'],
        test_article[0]['source'],
        test_article[0]['ticker'] + '_NO_OPT',  # Different ticker
        test_article[0]['headline'],
        test_article[0]['published_utc'],
        test_article[0]['article_url'],
        test_article[0]['summary'],
        test_article[0]['full_content'],
        test_article[0]['detected_at'],
        test_article[0]['processing_latency_ms'],
        test_article[0]['market_relevant'],
        test_article[0]['source_check_time'],
        test_article[0]['content_hash'] + '_no_opt',
        test_article[0]['news_type'],
        test_article[0]['urgency_score']
    ]]
    
    columns = [
        'timestamp', 'source', 'ticker', 'headline', 'published_utc',
        'article_url', 'summary', 'full_content', 'detected_at',
        'processing_latency_ms', 'market_relevant', 'source_check_time',
        'content_hash', 'news_type', 'urgency_score'
    ]
    
    ch_manager.client.insert('News.breaking_news', data_rows, column_names=columns)
    
    insertion_time = time.time() - start_time
    print(f"   ✅ Insertion without OPTIMIZE: {insertion_time*1000:.1f}ms")
    
    # Test immediate query availability
    query_start = time.time()
    result = ch_manager.client.query("SELECT COUNT(*) FROM News.breaking_news WHERE ticker = 'OPTTEST_NO_OPT'")
    query_time = time.time() - query_start
    count = result.result_rows[0][0] if result.result_rows else 0
    print(f"   ✅ Immediate query after insertion: {query_time*1000:.1f}ms, found {count} records")
    
    # Now test WITH OPTIMIZE TABLE FINAL
    print("\n2️⃣ Testing insertion WITH OPTIMIZE TABLE FINAL...")
    start_time = time.time()
    
    # Insert with OPTIMIZE
    data_rows[0][2] = 'OPTTEST_WITH_OPT'  # Different ticker
    data_rows[0][12] = 'optimize_test_hash_with_opt'  # Different hash
    
    ch_manager.client.insert('News.breaking_news', data_rows, column_names=columns)
    insertion_time = time.time() - start_time
    print(f"   ✅ Insertion part: {insertion_time*1000:.1f}ms")
    
    # Now run OPTIMIZE TABLE FINAL
    optimize_start = time.time()
    ch_manager.client.command("OPTIMIZE TABLE News.breaking_news FINAL")
    optimize_time = time.time() - optimize_start
    print(f"   ⚠️ OPTIMIZE TABLE FINAL: {optimize_time*1000:.1f}ms")
    
    # Test query availability after optimize
    query_start = time.time()
    result = ch_manager.client.query("SELECT COUNT(*) FROM News.breaking_news WHERE ticker = 'OPTTEST_WITH_OPT'")
    query_time = time.time() - query_start
    count = result.result_rows[0][0] if result.result_rows else 0
    print(f"   ✅ Query after OPTIMIZE: {query_time*1000:.1f}ms, found {count} records")
    
    total_time_with_optimize = insertion_time + optimize_time
    print(f"\n📊 SUMMARY:")
    print(f"   Without OPTIMIZE: {insertion_time*1000:.1f}ms total")
    print(f"   With OPTIMIZE: {total_time_with_optimize*1000:.1f}ms total")
    print(f"   OPTIMIZE overhead: {optimize_time*1000:.1f}ms")
    print(f"   Slowdown factor: {total_time_with_optimize/insertion_time:.1f}x")
    
    if optimize_time > 0.1:  # More than 100ms
        print(f"   🚨 OPTIMIZE TABLE FINAL is causing significant delay!")
    else:
        print(f"   ✅ OPTIMIZE TABLE FINAL delay is acceptable")
    
    # Cleanup
    ch_manager.close()

if __name__ == "__main__":
    test_optimize_delay() 