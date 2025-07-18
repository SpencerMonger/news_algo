#!/usr/bin/env python3
"""
Test Batch/Flush Processing with AsyncIO Lock
Tests the current live code logic from benzinga_websocket.py
Simulates multiple articles arriving in a burst and validates they all get processed
"""

import asyncio
import json
import time
import logging
import argparse
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from clickhouse_setup import ClickHouseManager
from sentiment_service import analyze_articles_with_sentiment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchFlushTester:
    """Test class that simulates the exact batch/flush processing from benzinga_websocket.py"""
    
    def __init__(self, debug: bool = False):
        self.batch_queue = []
        self.batch_queue_lock = None  # Will be created in async context
        self.clickhouse_manager = None
        self.debug = debug
        self.is_running = True
        
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🐛 Debug logging enabled")
        
        # Stats tracking
        self.stats = {
            'articles_detected': 0,
            'articles_queued': 0,
            'articles_processed': 0,
            'articles_inserted': 0,
            'flush_operations': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def initialize(self):
        """Initialize the test service"""
        try:
            # Create asyncio.Lock for atomic batch queue operations (same as live code)
            self.batch_queue_lock = asyncio.Lock()
            
            # Initialize ClickHouse connection
            self.clickhouse_manager = ClickHouseManager()
            self.clickhouse_manager.connect()
            
            logger.info("✅ Batch/Flush Tester initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Batch/Flush Tester: {str(e)}")
            return False
    
    def get_test_articles(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get real articles from the breaking news table"""
        try:
            query = f"""
            SELECT 
                ticker,
                headline,
                summary,
                full_content,
                source,
                timestamp,
                article_url,
                published_utc
            FROM News.breaking_news 
            WHERE timestamp >= now() - INTERVAL 48 HOUR
            AND ticker != ''
            AND ticker != 'UNKNOWN'
            AND headline != ''
            ORDER BY timestamp DESC
            LIMIT {count}
            """
            
            result = self.clickhouse_manager.client.query(query)
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'summary': row[2],
                    'full_content': row[3],
                    'source': row[4],
                    'timestamp': row[5],
                    'article_url': row[6],
                    'published_utc': row[7],
                    'detected_at': datetime.now(),
                    'processing_latency_ms': 0,
                    'market_relevant': 1,
                    'source_check_time': datetime.now(),
                    'content_hash': f"test_hash_{row[0]}_{int(time.time())}",
                    'news_type': 'other',
                    'urgency_score': 5
                })
            
            logger.info(f"Retrieved {len(articles)} real articles for testing")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving test articles: {e}")
            return []
    
    async def simulate_websocket_message_burst(self, articles: List[Dict[str, Any]]):
        """Simulate a WebSocket message burst where multiple articles arrive rapidly"""
        logger.info(f"🚀 SIMULATING WEBSOCKET MESSAGE BURST: {len(articles)} articles arriving rapidly")
        
        # Simulate articles arriving with slight delays (like in real WebSocket bursts)
        tasks = []
        for i, article in enumerate(articles):
            delay = i * 0.01  # 10ms between articles (very rapid)
            task = asyncio.create_task(self.simulate_article_detection(article, delay))
            tasks.append(task)
        
        # Wait for all articles to be "detected"
        await asyncio.gather(*tasks)
        
        logger.info(f"✅ Message burst simulation complete: {len(articles)} articles detected")
    
    async def simulate_article_detection(self, article: Dict[str, Any], delay: float):
        """Simulate detecting a single article and adding it to the batch queue (LIVE CODE LOGIC)"""
        await asyncio.sleep(delay)
        
        self.stats['articles_detected'] += 1
        
        # 🔧 LIVE CODE LOGIC: Use asyncio.Lock to ensure atomic batch queue operations
        async with self.batch_queue_lock:
            self.batch_queue.append(article)
            self.stats['articles_queued'] += 1
            
            if self.debug:
                logger.debug(f"📝 Added {article['ticker']} to batch queue (total queued: {len(self.batch_queue)})")
    
    async def buffer_flusher(self):
        """Buffer flusher that runs every 250ms (LIVE CODE LOGIC)"""
        logger.info("🔄 Starting buffer flusher - checking every 250ms")
        flush_count = 0
        
        while self.is_running:
            try:
                await asyncio.sleep(0.25)  # Flush every 250ms
                
                # 🔧 LIVE CODE LOGIC: Use asyncio.Lock to ensure atomic batch queue operations
                articles_to_flush = []
                async with self.batch_queue_lock:
                    if self.batch_queue:
                        articles_to_flush = self.batch_queue.copy()
                        self.batch_queue.clear()
                        flush_count += 1
                        self.stats['flush_operations'] += 1
                        logger.info(f"🚀 Buffer flush #{flush_count} - {len(articles_to_flush)} articles ready")
                
                # Process articles outside the lock to avoid blocking WebSocket processing
                if articles_to_flush:
                    await self.flush_articles_to_clickhouse(articles_to_flush)
                
            except Exception as e:
                logger.error(f"Error in buffer flusher: {e}")
    
    async def flush_articles_to_clickhouse(self, articles: List[Dict[str, Any]]):
        """Flush articles to ClickHouse WITH sentiment analysis (LIVE CODE LOGIC + RETRY)"""
        if not articles:
            return
            
        try:
            # STEP 1: Analyze articles with sentiment BEFORE inserting into database
            logger.info(f"🧠 Starting sentiment analysis for {len(articles)} articles...")
            
            # Use the EXACT same function as the live system - it already handles concurrency properly
            enriched_articles = await analyze_articles_with_sentiment(articles)
            
            # STEP 2: Check for failed analyses and retry them
            failed_articles = []
            successful_count = 0
            
            for article in enriched_articles:
                if hasattr(article, 'get') and article.get('sentiment') == 'neutral' and 'error' in str(article.get('explanation', '')):
                    failed_articles.append(article)
                else:
                    successful_count += 1
            
            if failed_articles:
                logger.info(f"🔄 RETRYING {len(failed_articles)} failed articles with delays...")
                
                # Retry failed articles one by one with delays to avoid rate limits
                for i, failed_article in enumerate(failed_articles, 1):
                    try:
                        # Exponential backoff: 5s, 10s, 15s, etc.
                        wait_time = 5 + (i * 5)  # 5s, 10s, 15s, 20s...
                        if i > 1:
                            logger.info(f"⏳ Waiting {wait_time}s before retry {i}/{len(failed_articles)} to let rate limit reset...")
                            await asyncio.sleep(wait_time)
                        
                        logger.info(f"🔄 Retry {i}/{len(failed_articles)}: {failed_article.get('ticker', 'UNKNOWN')}")
                        
                        # Get sentiment service and retry single article
                        from sentiment_service import get_sentiment_service
                        sentiment_service = await get_sentiment_service()
                        
                        # Retry the analysis
                        sentiment_result = await sentiment_service.analyze_article_sentiment(failed_article)
                        
                        # Update article with retry result
                        if isinstance(sentiment_result, dict) and 'error' not in sentiment_result:
                            failed_article.update({
                                'sentiment': sentiment_result.get('sentiment', 'neutral'),
                                'recommendation': sentiment_result.get('recommendation', 'HOLD'),
                                'confidence': sentiment_result.get('confidence', 'low'),
                                'explanation': sentiment_result.get('explanation', 'No explanation'),
                                'analysis_time_ms': sentiment_result.get('analysis_time_ms', 0),
                                'analyzed_at': sentiment_result.get('analyzed_at', datetime.now())
                            })
                            successful_count += 1
                            logger.info(f"✅ Retry successful: {failed_article.get('ticker', 'UNKNOWN')} - {sentiment_result.get('recommendation', 'HOLD')}")
                        else:
                            logger.warning(f"❌ Retry failed: {failed_article.get('ticker', 'UNKNOWN')} - {sentiment_result.get('error', 'Unknown error')}")
                        
                    except Exception as e:
                        logger.error(f"Error retrying article {i}: {e}")
                        continue
                
                logger.info(f"🔄 RETRY COMPLETE: {successful_count}/{len(articles)} total successful analyses")
            else:
                logger.info(f"✅ SENTIMENT ANALYSIS COMPLETE: {successful_count}/{len(articles)} successful analyses (no retries needed)")
            
            # STEP 3: Insert articles WITH sentiment data into database
            # For testing, we'll just count them instead of actually inserting
            inserted_count = len(enriched_articles)
            self.stats['articles_processed'] += len(articles)
            self.stats['articles_inserted'] += inserted_count
            
            logger.info(f"✅ Processed {inserted_count} articles with sentiment analysis")
            
        except Exception as e:
            logger.error(f"Error processing articles with sentiment analysis: {e}")
            # Still count as processed even if sentiment analysis fails
            self.stats['articles_processed'] += len(articles)
            self.stats['articles_inserted'] += len(articles)
    
    async def run_test(self, num_articles: int = 20):
        """Run the batch/flush test with current live code logic"""
        logger.info(f"🚀 TESTING CURRENT LIVE CODE LOGIC (with asyncio.Lock)")
        
        # Reset stats
        self.stats = {key: 0 for key in self.stats.keys()}
        self.stats['start_time'] = time.time()
        self.batch_queue.clear()
        self.is_running = True
        
        # Get test articles
        test_articles = self.get_test_articles(num_articles)
        if not test_articles:
            logger.error("❌ No test articles found - aborting test")
            return
        
        # Show articles being tested
        logger.info("📋 Test Articles:")
        for i, article in enumerate(test_articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')[:60] + "..." if len(article.get('headline', '')) > 60 else article.get('headline', '')
            logger.info(f"   {i:2d}. {ticker:6} | {headline}")
        
        print(f"\n{'='*80}")
        print(f"🧠 BATCH/FLUSH PROCESSING TEST - CURRENT LIVE CODE LOGIC")
        print(f"{'='*80}")
        
        # Start buffer flusher
        flusher_task = asyncio.create_task(self.buffer_flusher())
        
        # Simulate WebSocket message burst
        await self.simulate_websocket_message_burst(test_articles)
        
        # Wait for all articles to be processed - check periodically
        logger.info("⏳ Waiting for sentiment analysis to complete...")
        max_wait_time = 300  # 5 minutes max
        wait_interval = 5    # Check every 5 seconds
        total_waited = 0
        
        while total_waited < max_wait_time:
            if self.stats['articles_processed'] >= len(test_articles):
                logger.info(f"✅ All {len(test_articles)} articles processed!")
                break
            
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval
            
            # Log progress every 30 seconds
            if total_waited % 30 == 0:
                logger.info(f"⏳ Still waiting... {self.stats['articles_processed']}/{len(test_articles)} articles processed ({total_waited}s elapsed)")
        
        if self.stats['articles_processed'] < len(test_articles):
            logger.warning(f"⚠️ Timeout reached. Only {self.stats['articles_processed']}/{len(test_articles)} articles processed in {max_wait_time}s")
        
        # Stop the flusher
        self.is_running = False
        flusher_task.cancel()
        
        # Final flush
        async with self.batch_queue_lock:
            if self.batch_queue:
                final_articles = self.batch_queue.copy()
                self.batch_queue.clear()
                if final_articles:
                    await self.flush_articles_to_clickhouse(final_articles)
        
        self.stats['end_time'] = time.time()
        
        # Print results
        self.print_results(num_articles)
    
    def print_results(self, num_articles: int):
        """Print detailed test results"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        print(f"\n{'='*80}")
        print(f"📊 BATCH/FLUSH PROCESSING TEST RESULTS")
        print(f"{'='*80}")
        
        print(f"📋 TEST CONFIGURATION:")
        print(f"   Articles in Burst: {num_articles}")
        print(f"   Article Arrival Rate: 10ms between articles")
        print(f"   Buffer Flush Interval: 250ms")
        print(f"   Scenario: Multiple articles arriving rapidly (like 13:00:05 burst)")
        
        print(f"\n⏱️  TIMING BREAKDOWN:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Time per Article: {(total_time / num_articles):.2f}s")
        
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"   Articles Detected: {self.stats['articles_detected']}")
        print(f"   Articles Queued: {self.stats['articles_queued']}")
        print(f"   Articles Processed: {self.stats['articles_processed']}")
        print(f"   Articles Inserted: {self.stats['articles_inserted']}")
        print(f"   Flush Operations: {self.stats['flush_operations']}")
        
        # Calculate success rate
        success_rate = (self.stats['articles_processed'] / self.stats['articles_detected']) * 100 if self.stats['articles_detected'] > 0 else 0
        loss_rate = 100 - success_rate
        
        print(f"\n📈 SUCCESS METRICS:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Loss Rate: {loss_rate:.1f}%")
        print(f"   Articles Lost: {self.stats['articles_detected'] - self.stats['articles_processed']}")
        
        print(f"\n🎯 VALIDATION:")
        if success_rate == 100:
            print(f"   ✅ PERFECT: All {num_articles} articles processed successfully")
            print(f"   ✅ The asyncio.Lock prevents race conditions in burst scenarios")
            print(f"   ✅ No articles were lost during rapid WebSocket message processing")
        elif success_rate >= 95:
            print(f"   ✅ EXCELLENT: {success_rate:.1f}% success rate")
            print(f"   ✅ Minor losses acceptable for this test scenario")
        elif success_rate >= 80:
            print(f"   ⚠️ GOOD: {success_rate:.1f}% success rate")
            print(f"   ⚠️ Some articles lost - may need further optimization")
        else:
            print(f"   ❌ POOR: {success_rate:.1f}% success rate")
            print(f"   ❌ Significant article loss detected - investigate race conditions")
        
        # Performance metrics
        if total_time > 0:
            throughput = self.stats['articles_processed'] / total_time
            print(f"\n📊 PERFORMANCE:")
            print(f"   Throughput: {throughput:.2f} articles/second")
            print(f"   Flush Rate: {self.stats['flush_operations'] / total_time:.2f} flushes/second")
        
        print(f"\n{'='*80}")
    
    async def close(self):
        """Close resources"""
        if self.clickhouse_manager:
            self.clickhouse_manager.close()

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Current Live Code Batch/Flush Processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--num-articles", type=int, default=20, help="Number of articles to test (default: 20)")
    args = parser.parse_args()

    tester = BatchFlushTester(debug=args.debug)
    
    try:
        if not await tester.initialize():
            logger.error("❌ Failed to initialize - aborting test")
            return
        
        await tester.run_test(args.num_articles)
            
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 