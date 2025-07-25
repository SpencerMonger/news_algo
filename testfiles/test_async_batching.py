#!/usr/bin/env python3
"""
Test Asynchronous Individual Processing vs Batch Processing
Compares the current batch processing approach with the new individual processing approach.

Current approach: Collect batch → Analyze all → Insert all
New approach: Collect batch → Analyze individually → Insert immediately when each completes

This test simulates the 250ms buffer flush cycle with varying batch sizes.
"""

import asyncio
import json
import time
import logging
import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from clickhouse_setup import ClickHouseManager
from sentiment_service import analyze_articles_with_sentiment, get_sentiment_service, clear_sentiment_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingComparisonTester:
    """Test class comparing batch vs individual processing approaches"""
    
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
        
        # Stats tracking for both approaches
        self.batch_stats = {
            'articles_detected': 0,
            'articles_queued': 0,
            'articles_processed': 0,
            'articles_inserted': 0,
            'flush_operations': 0,
            'total_processing_time': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.individual_stats = {
            'articles_detected': 0,
            'articles_queued': 0,
            'articles_processed': 0,
            'articles_inserted': 0,
            'flush_operations': 0,
            'total_processing_time': 0,
            'first_insert_time': None,
            'last_insert_time': None,
            'start_time': None,
            'end_time': None
        }

    async def initialize(self):
        """Initialize the test environment"""
        logger.info("🚀 Initializing processing comparison tester...")
        
        # Create ClickHouse connection
        self.clickhouse_manager = ClickHouseManager()
        self.clickhouse_manager.connect()
        
        # Create asyncio.Lock for atomic batch queue operations
        self.batch_queue_lock = asyncio.Lock()
        
        logger.info("✅ Tester initialized successfully")

    def get_test_articles(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get real articles from the breaking news table for testing"""
        try:
            self.clickhouse_manager.connect()
            
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
                    # Add required fields for processing
                    'detected_at': datetime.now(),
                    'processing_latency_ms': 0,
                    'market_relevant': 1,
                    'source_check_time': datetime.now(),
                    'content_hash': f"test_hash_{row[0]}_{int(time.time())}_{random.randint(1000, 9999)}",
                    'news_type': 'other',
                    'urgency_score': 5
                })
            
            logger.info(f"Retrieved {len(articles)} real articles for testing")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving test articles: {e}")
            return []

    # ============================================================================
    # CURRENT BATCH PROCESSING APPROACH (Synchronous)
    # ============================================================================
    
    async def test_batch_processing(self, articles: List[Dict[str, Any]]):
        """Test the current batch processing approach: Collect all → Analyze all → Insert all"""
        logger.info("🔄 TESTING CURRENT BATCH PROCESSING APPROACH")
        logger.info("   Approach: Collect batch → Analyze ALL → Insert ALL")
        
        # Reset stats
        self.batch_stats = {key: 0 for key in self.batch_stats.keys()}
        self.batch_stats['start_time'] = time.time()
        
        # Simulate article detection burst
        await self.simulate_article_burst_batch(articles)
        
        # Simulate buffer flush (current approach)
        await self.flush_batch_synchronous(articles)
        
        self.batch_stats['end_time'] = time.time()
        self.batch_stats['total_processing_time'] = self.batch_stats['end_time'] - self.batch_stats['start_time']
        
        logger.info(f"✅ BATCH PROCESSING COMPLETE: {self.batch_stats['total_processing_time']:.2f}s total")

    async def simulate_article_burst_batch(self, articles: List[Dict[str, Any]]):
        """Simulate articles arriving in a burst (same as real WebSocket burst)"""
        logger.info(f"📥 Simulating article burst: {len(articles)} articles arriving rapidly")
        
        for i, article in enumerate(articles):
            delay = i * 0.01  # 10ms between articles (rapid burst)
            await asyncio.sleep(delay)
            self.batch_stats['articles_detected'] += 1
            self.batch_stats['articles_queued'] += 1

    async def flush_batch_synchronous(self, articles: List[Dict[str, Any]]):
        """Current batch processing: Analyze ALL articles, then insert ALL"""
        logger.info(f"🧠 BATCH APPROACH: Starting sentiment analysis for {len(articles)} articles...")
        
        start_time = time.time()
        
        try:
            # STEP 1: Analyze ALL articles with sentiment (current approach)
            enriched_articles = await analyze_articles_with_sentiment(articles)
            
            analysis_time = time.time() - start_time
            logger.info(f"🧠 BATCH ANALYSIS COMPLETE: {len(enriched_articles)} articles analyzed in {analysis_time:.2f}s")
            
            # STEP 2: "Insert" all articles at once (simulated for testing)
            # In real system, this would be: self.clickhouse_manager.insert_articles(enriched_articles)
            insert_start = time.time()
            await asyncio.sleep(0.1)  # Simulate database insertion time
            insert_time = time.time() - insert_start
            
            self.batch_stats['articles_processed'] = len(articles)
            self.batch_stats['articles_inserted'] = len(enriched_articles)
            self.batch_stats['flush_operations'] = 1
            
            logger.info(f"💾 BATCH INSERT COMPLETE: {len(enriched_articles)} articles inserted in {insert_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.batch_stats['articles_processed'] = len(articles)
            self.batch_stats['articles_inserted'] = len(articles)

    # ============================================================================
    # NEW INDIVIDUAL PROCESSING APPROACH (Asynchronous)
    # ============================================================================
    
    async def test_individual_processing(self, articles: List[Dict[str, Any]]):
        """Test the new individual processing approach: Collect batch → Analyze individually → Insert immediately"""
        logger.info("🚀 TESTING NEW INDIVIDUAL PROCESSING APPROACH")
        logger.info("   Approach: Collect batch → Analyze INDIVIDUALLY → Insert IMMEDIATELY when each completes")
        
        # Reset stats
        self.individual_stats = {key: 0 for key in self.individual_stats.keys()}
        self.individual_stats['start_time'] = time.time()
        
        # Simulate article detection burst
        await self.simulate_article_burst_individual(articles)
        
        # Simulate buffer flush (new approach)
        await self.flush_individual_asynchronous(articles)
        
        self.individual_stats['end_time'] = time.time()
        self.individual_stats['total_processing_time'] = self.individual_stats['end_time'] - self.individual_stats['start_time']
        
        # Calculate time to first and last insert
        if self.individual_stats['first_insert_time'] and self.individual_stats['last_insert_time']:
            time_to_first = self.individual_stats['first_insert_time'] - self.individual_stats['start_time']
            time_to_last = self.individual_stats['last_insert_time'] - self.individual_stats['start_time']
            logger.info(f"⚡ INDIVIDUAL PROCESSING COMPLETE: First insert in {time_to_first:.2f}s, last insert in {time_to_last:.2f}s")
        else:
            logger.info(f"✅ INDIVIDUAL PROCESSING COMPLETE: {self.individual_stats['total_processing_time']:.2f}s total")

    async def simulate_article_burst_individual(self, articles: List[Dict[str, Any]]):
        """Simulate articles arriving in a burst (same timing as batch)"""
        logger.info(f"📥 Simulating article burst: {len(articles)} articles arriving rapidly")
        
        for i, article in enumerate(articles):
            delay = i * 0.01  # 10ms between articles (rapid burst)
            await asyncio.sleep(delay)
            self.individual_stats['articles_detected'] += 1
            self.individual_stats['articles_queued'] += 1

    async def flush_individual_asynchronous(self, articles: List[Dict[str, Any]]):
        """NEW individual processing: Analyze each article individually, insert immediately when complete"""
        logger.info(f"🚀 INDIVIDUAL APPROACH: Starting individual analysis for {len(articles)} articles...")
        
        # Create tasks to analyze each article individually
        analysis_tasks = []
        for i, article in enumerate(articles):
            task = asyncio.create_task(self.analyze_and_insert_individual(article, i))
            analysis_tasks.append(task)
        
        # Start all analyses concurrently (but each will insert individually when complete)
        await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        logger.info(f"🚀 INDIVIDUAL PROCESSING: All {len(articles)} articles started processing")

    async def analyze_and_insert_individual(self, article: Dict[str, Any], index: int):
        """Analyze a single article and insert immediately when complete"""
        ticker = article.get('ticker', 'UNKNOWN')
        
        try:
            # STEP 1: Analyze sentiment for this individual article
            logger.info(f"🧠 #{index+1:2d} ANALYZING: {ticker}")
            
            sentiment_service = await get_sentiment_service()
            sentiment_start = time.time()
            sentiment_result = await sentiment_service.analyze_article_sentiment(article)
            sentiment_time = time.time() - sentiment_start
            
            # Update article with sentiment data
            if isinstance(sentiment_result, dict) and 'error' not in sentiment_result:
                article.update({
                    'sentiment': sentiment_result.get('sentiment', 'neutral'),
                    'recommendation': sentiment_result.get('recommendation', 'HOLD'),
                    'confidence': sentiment_result.get('confidence', 'low'),
                    'explanation': sentiment_result.get('explanation', 'No explanation'),
                    'analysis_time_ms': sentiment_result.get('analysis_time_ms', 0),
                    'analyzed_at': sentiment_result.get('analyzed_at', datetime.now())
                })
                logger.info(f"✅ #{index+1:2d} ANALYZED:  {ticker} -> {sentiment_result.get('recommendation', 'HOLD')} ({sentiment_time:.1f}s)")
            else:
                # Default sentiment for failed analysis
                article.update({
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': 'Analysis failed',
                    'analysis_time_ms': 0,
                    'analyzed_at': datetime.now()
                })
                logger.warning(f"⚠️ #{index+1:2d} FAILED:    {ticker} -> Analysis failed ({sentiment_time:.1f}s)")
            
            # STEP 2: Insert immediately when analysis is complete
            insert_start = time.time()
            await asyncio.sleep(0.05)  # Simulate individual database insertion time
            insert_time = time.time() - insert_start
            
            # Track timing
            current_time = time.time()
            if self.individual_stats['first_insert_time'] is None:
                self.individual_stats['first_insert_time'] = current_time
            self.individual_stats['last_insert_time'] = current_time
            
            self.individual_stats['articles_processed'] += 1
            self.individual_stats['articles_inserted'] += 1
            
            logger.info(f"💾 #{index+1:2d} INSERTED:  {ticker} -> Database ({insert_time:.1f}s)")
            
        except Exception as e:
            logger.error(f"❌ #{index+1:2d} ERROR:     {ticker} -> {str(e)}")
            self.individual_stats['articles_processed'] += 1

    # ============================================================================
    # SIMULATION AND COMPARISON
    # ============================================================================
    
    async def simulate_websocket_patterns(self, num_tests: int = 5):
        """Simulate different WebSocket arrival patterns and compare both approaches"""
        logger.info(f"🧪 SIMULATING {num_tests} DIFFERENT WEBSOCKET PATTERNS")
        
        patterns = [
            {'name': 'Single Article', 'size': 1},
            {'name': 'Small Burst', 'size': 3},
            {'name': 'Medium Burst', 'size': 8},
            {'name': 'Large Burst', 'size': 15},
            {'name': 'Mega Burst', 'size': 25}
        ]
        
        results = []
        
        for i, pattern in enumerate(patterns[:num_tests], 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 TEST {i}/{num_tests}: {pattern['name']} ({pattern['size']} articles)")
            logger.info(f"{'='*80}")
            
            # Get test articles for this pattern
            test_articles = self.get_test_articles(pattern['size'])
            if not test_articles:
                logger.error(f"❌ No test articles available for {pattern['name']}")
                continue
            
            # Show articles being tested
            logger.info("📋 Test Articles:")
            for j, article in enumerate(test_articles, 1):
                ticker = article.get('ticker', 'UNKNOWN')
                headline = article.get('headline', '')[:50] + "..." if len(article.get('headline', '')) > 50 else article.get('headline', '')
                logger.info(f"   {j:2d}. {ticker:6} | {headline}")
            
            logger.info(f"\n🔄 Testing Pattern: {pattern['name']}")
            
            # Test batch processing approach
            logger.info(f"\n--- BATCH PROCESSING TEST ---")
            await self.test_batch_processing(test_articles.copy())
            
            # Small delay between tests
            await asyncio.sleep(2)
            
            # CLEAR SENTIMENT CACHE to ensure fair comparison
            logger.info(f"\n🧹 CLEARING SENTIMENT CACHE for fair comparison...")
            await clear_sentiment_cache()
            
            # Test individual processing approach
            logger.info(f"\n--- INDIVIDUAL PROCESSING TEST ---")
            await self.test_individual_processing(test_articles.copy())
            
            # Record results
            result = {
                'pattern': pattern['name'],
                'size': pattern['size'],
                'batch_time': self.batch_stats['total_processing_time'],
                'individual_time': self.individual_stats['total_processing_time'],
                'individual_first_insert': (self.individual_stats['first_insert_time'] - self.individual_stats['start_time']) if self.individual_stats['first_insert_time'] else None,
                'individual_last_insert': (self.individual_stats['last_insert_time'] - self.individual_stats['start_time']) if self.individual_stats['last_insert_time'] else None
            }
            results.append(result)
            
            # Print comparison for this pattern
            self.print_pattern_comparison(result)
            
            # Delay between patterns
            if i < num_tests:
                logger.info(f"\n⏳ Waiting 3 seconds before next pattern...")
                await asyncio.sleep(3)
        
        # Print final summary
        self.print_final_summary(results)

    def print_pattern_comparison(self, result):
        """Print comparison results for a single pattern"""
        logger.info(f"\n📊 COMPARISON RESULTS: {result['pattern']}")
        logger.info(f"{'='*60}")
        logger.info(f"📦 Batch Processing:      {result['batch_time']:.2f}s total")
        logger.info(f"🚀 Individual Processing: {result['individual_time']:.2f}s total")
        
        if result['individual_first_insert']:
            logger.info(f"⚡ Time to first insert:  {result['individual_first_insert']:.2f}s")
            logger.info(f"⏱️ Time to last insert:   {result['individual_last_insert']:.2f}s")
            
            time_savings = result['batch_time'] - result['individual_first_insert']
            percentage_faster = (time_savings / result['batch_time']) * 100 if result['batch_time'] > 0 else 0
            
            if time_savings > 0:
                logger.info(f"🎯 FIRST INSERT ADVANTAGE: {time_savings:.2f}s faster ({percentage_faster:.1f}% improvement)")
            else:
                logger.info(f"⚠️ FIRST INSERT SLOWER: {abs(time_savings):.2f}s slower")
        
        logger.info(f"{'='*60}")

    def print_final_summary(self, results):
        """Print final summary comparing all patterns"""
        logger.info(f"\n🏆 FINAL SUMMARY: BATCH vs INDIVIDUAL PROCESSING")
        logger.info(f"{'='*80}")
        
        logger.info(f"{'Pattern':<15} {'Size':<4} {'Batch':<8} {'Individual':<10} {'First Insert':<12} {'Advantage':<10}")
        logger.info(f"{'-'*80}")
        
        for result in results:
            pattern = result['pattern'][:14]
            size = result['size']
            batch_time = result['batch_time']
            individual_time = result['individual_time']
            first_insert = result['individual_first_insert']
            
            if first_insert:
                advantage = batch_time - first_insert
                advantage_str = f"+{advantage:.1f}s" if advantage > 0 else f"{advantage:.1f}s"
            else:
                advantage_str = "N/A"
            
            logger.info(f"{pattern:<15} {size:<4} {batch_time:<8.2f} {individual_time:<10.2f} {first_insert:<12.2f} {advantage_str:<10}")
        
        # Calculate averages
        avg_batch = sum(r['batch_time'] for r in results) / len(results)
        avg_individual = sum(r['individual_time'] for r in results) / len(results)
        avg_first_insert = sum(r['individual_first_insert'] for r in results if r['individual_first_insert']) / len([r for r in results if r['individual_first_insert']])
        
        logger.info(f"{'-'*80}")
        logger.info(f"{'AVERAGES':<15} {'N/A':<4} {avg_batch:<8.2f} {avg_individual:<10.2f} {avg_first_insert:<12.2f}")
        
        # Key insights
        logger.info(f"\n🎯 KEY INSIGHTS:")
        logger.info(f"   • Individual processing provides first results {avg_batch - avg_first_insert:.2f}s faster on average")
        logger.info(f"   • Total processing time difference: {avg_individual - avg_batch:.2f}s")
        logger.info(f"   • Best use case: Large bursts where users need immediate feedback")
        logger.info(f"   • Trade-off: More database connections but faster user experience")

async def main():
    """Main function for testing"""
    parser = argparse.ArgumentParser(description='Test Asynchronous Individual Processing vs Batch Processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--tests', type=int, default=5, help='Number of different patterns to test (1-5)')
    parser.add_argument('--pattern', type=str, choices=['single', 'small', 'medium', 'large', 'mega'], 
                       help='Test only a specific pattern')
    
    args = parser.parse_args()
    
    # Create and initialize tester
    tester = ProcessingComparisonTester(debug=args.debug)
    await tester.initialize()
    
    try:
        if args.pattern:
            # Test specific pattern
            pattern_map = {
                'single': {'name': 'Single Article', 'size': 1},
                'small': {'name': 'Small Burst', 'size': 3},
                'medium': {'name': 'Medium Burst', 'size': 8},
                'large': {'name': 'Large Burst', 'size': 15},
                'mega': {'name': 'Mega Burst', 'size': 25}
            }
            
            pattern = pattern_map[args.pattern]
            logger.info(f"🎯 Testing specific pattern: {pattern['name']}")
            
            test_articles = tester.get_test_articles(pattern['size'])
            if test_articles:
                logger.info(f"\n🔄 BATCH PROCESSING TEST")
                await tester.test_batch_processing(test_articles.copy())
                
                # CLEAR SENTIMENT CACHE to ensure fair comparison
                logger.info(f"\n🧹 CLEARING SENTIMENT CACHE for fair comparison...")
                await clear_sentiment_cache()
                
                logger.info(f"\n🚀 INDIVIDUAL PROCESSING TEST")
                await tester.test_individual_processing(test_articles.copy())
                
                result = {
                    'pattern': pattern['name'],
                    'size': pattern['size'],
                    'batch_time': tester.batch_stats['total_processing_time'],
                    'individual_time': tester.individual_stats['total_processing_time'],
                    'individual_first_insert': (tester.individual_stats['first_insert_time'] - tester.individual_stats['start_time']) if tester.individual_stats['first_insert_time'] else None,
                    'individual_last_insert': (tester.individual_stats['last_insert_time'] - tester.individual_stats['start_time']) if tester.individual_stats['last_insert_time'] else None
                }
                
                tester.print_pattern_comparison(result)
            else:
                logger.error("❌ No test articles available")
        else:
            # Test multiple patterns
            await tester.simulate_websocket_patterns(args.tests)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        if tester.clickhouse_manager:
            tester.clickhouse_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 