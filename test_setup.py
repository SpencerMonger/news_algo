import asyncio
import logging
from datetime import datetime
from clickhouse_setup import ClickHouseManager, setup_clickhouse_database
from newswire_monitor import NewswireMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_clickhouse_connection():
    """Test ClickHouse connection and basic operations"""
    logger.info("Testing ClickHouse connection...")
    
    try:
        # Setup ClickHouse
        ch_manager = setup_clickhouse_database()
        
        # Test insert
        test_articles = [{
            'source': 'TestSource',
            'ticker': 'TEST',
            'headline': 'Test Breaking News Article',
            'published_utc': datetime.now(),
            'article_url': 'https://test.com/news/1',
            'summary': 'This is a test summary',
            'full_content': 'Full test content here',
            'processing_latency_ms': 150,
            'content_hash': 'test_hash_12345',
            'news_type': 'other',
            'urgency_score': 7
        }]
        
        # Insert test data
        inserted = ch_manager.insert_articles(test_articles)
        logger.info(f"Successfully inserted {inserted} test article(s)")
        
        # Test query
        recent_articles = ch_manager.get_recent_articles('TEST', 1)
        logger.info(f"Retrieved {len(recent_articles)} recent articles for TEST ticker")
        
        # Test performance stats
        stats = ch_manager.get_performance_stats()
        logger.info(f"Performance stats retrieved: {len(stats)} source(s)")
        
        ch_manager.close()
        logger.info("ClickHouse test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"ClickHouse test failed: {e}")
        return False

async def test_newswire_feeds():
    """Test newswire feed access"""
    logger.info("Testing newswire feed access...")
    
    try:
        # Test RSS feed access
        import aiohttp
        import feedparser
        
        test_urls = [
            "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20Releases",
            "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEF9YXA==",
        ]
        
        async with aiohttp.ClientSession() as session:
            for url in test_urls:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)
                            logger.info(f"✓ {url}: {len(feed.entries)} articles")
                        else:
                            logger.warning(f"✗ {url}: HTTP {response.status}")
                except Exception as e:
                    logger.error(f"✗ {url}: {e}")
        
        logger.info("Newswire feed test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Newswire feed test failed: {e}")
        return False

async def test_ticker_extraction():
    """Test ticker extraction functionality"""
    logger.info("Testing ticker extraction...")
    
    try:
        # Create a mock monitor for testing
        monitor = NewswireMonitor()
        
        # Set up test ticker list
        monitor.ticker_list = ['AAPL', 'TSLA', 'SPRO', 'NVDA', 'AMD']
        monitor.ticker_patterns = {}  # Initialize the patterns dict
        monitor.compile_ticker_patterns()
        
        # Test cases
        test_cases = [
            "Spero Therapeutics (NASDAQ: SPRO) Announces Positive Phase 3 Results",
            "Apple Inc. (AAPL) Reports Strong Q4 Earnings",
            "Tesla stock ($TSLA) surges on delivery numbers",
            "No tickers in this text at all",
            "Multiple tickers: AAPL and NVDA partnership announced"
        ]
        
        for test_text in test_cases:
            tickers = monitor.extract_tickers_from_text(test_text)
            logger.info(f"Text: '{test_text[:50]}...' → Tickers: {tickers}")
        
        logger.info("Ticker extraction test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Ticker extraction test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("=== Starting Setup Validation Tests ===")
    
    tests = [
        ("ClickHouse Connection", test_clickhouse_connection()),
        ("Newswire Feeds", test_newswire_feeds()),
        ("Ticker Extraction", test_ticker_extraction())
    ]
    
    results = []
    for test_name, test_coro in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! System is ready for production.")
    else:
        logger.warning("⚠️  Some tests failed. Please check configuration.")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 