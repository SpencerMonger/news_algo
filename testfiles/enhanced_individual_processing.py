#!/usr/bin/env python3
"""
Enhanced Individual Processing with Zero Article Loss
Implements a robust retry strategy that ensures no articles are ever lost
while maintaining the speed advantages of individual processing.

Key Features:
1. Immediate insertion of successful analyses
2. Smart retry queue for failed analyses
3. Progressive backoff to avoid rate limits
4. Zero article loss guarantee
5. Detailed tracking and logging
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetryReason(Enum):
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    UNKNOWN = "unknown"

@dataclass
class RetryItem:
    article: Dict[str, Any]
    attempt_count: int
    last_attempt_time: float
    retry_reason: RetryReason
    original_error: str
    next_retry_time: float

class EnhancedIndividualProcessor:
    """
    Enhanced individual processing with zero article loss guarantee
    """
    
    def __init__(self, max_retries: int = 8, base_delay: float = 3.0):  # Increased retries and base delay
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Tracking
        self.processed_articles: Set[str] = set()  # content_hash tracking
        self.retry_queue: List[RetryItem] = []
        self.retry_queue_lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            'articles_received': 0,
            'articles_analyzed_successfully': 0,
            'articles_inserted_immediately': 0,
            'articles_failed_initial': 0,
            'articles_retried': 0,
            'articles_recovered_on_retry': 0,
            'articles_failed_permanently': 0,
            'total_retry_attempts': 0,
            'rate_limit_hits': 0,
            'zero_loss_guarantee': True,  # This should always be True
            'real_analysis_success_rate': 0.0  # Track real vs default sentiment
        }
        
    async def process_article_batch(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of articles with individual processing and zero loss guarantee
        
        Returns summary of processing results
        """
        logger.info(f"🚀 ENHANCED INDIVIDUAL PROCESSING: Starting {len(articles)} articles")
        
        self.stats['articles_received'] += len(articles)
        start_time = time.time()
        
        # Create individual processing tasks
        processing_tasks = []
        for i, article in enumerate(articles):
            task = asyncio.create_task(self._process_single_article(article, i))
            processing_tasks.append(task)
        
        # Start all processing tasks concurrently
        await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Process retry queue until empty or max attempts reached
        await self._process_retry_queue()
        
        # Final verification - ensure zero loss
        await self._verify_zero_loss(articles)
        
        processing_time = time.time() - start_time
        
        # Return comprehensive summary
        return {
            'processing_time': processing_time,
            'stats': self.stats.copy(),
            'retry_queue_final_size': len(self.retry_queue),
            'zero_loss_achieved': self.stats['zero_loss_guarantee']
        }
    
    async def _process_single_article(self, article: Dict[str, Any], index: int):
        """Process a single article with immediate insertion on success"""
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}_{index}")
        
        try:
            logger.info(f"🧠 #{index+1:2d} ANALYZING: {ticker}")
            
            # Simulate sentiment analysis (replace with real sentiment service)
            analysis_result = await self._simulate_sentiment_analysis(article)
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Immediate insertion
                await self._insert_article_immediately(article, analysis_result, index)
                self.stats['articles_analyzed_successfully'] += 1
                self.stats['articles_inserted_immediately'] += 1
                self.processed_articles.add(content_hash)
                
                logger.info(f"✅ #{index+1:2d} SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')} (inserted immediately)")
                
            else:
                # FAILURE: Add to retry queue
                await self._add_to_retry_queue(article, analysis_result, index)
                self.stats['articles_failed_initial'] += 1
                
                logger.warning(f"⚠️ #{index+1:2d} QUEUED FOR RETRY: {ticker} -> {analysis_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            # EXCEPTION: Add to retry queue
            error_result = {'error': f'Exception: {str(e)}'}
            await self._add_to_retry_queue(article, error_result, index)
            self.stats['articles_failed_initial'] += 1
            
            logger.error(f"❌ #{index+1:2d} EXCEPTION: {ticker} -> {str(e)}")
    
    async def _simulate_sentiment_analysis(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate sentiment analysis with realistic failure patterns
        Replace this with actual sentiment service call
        """
        # Simulate processing time
        await asyncio.sleep(0.1 + (0.5 * asyncio.get_event_loop().time() % 1))
        
        # Simulate different failure modes
        import random
        failure_chance = random.random()
        
        if failure_chance < 0.15:  # 15% rate limit
            self.stats['rate_limit_hits'] += 1
            return {'error': 'HTTP 429: Rate limit exceeded', 'retry_reason': RetryReason.RATE_LIMIT}
        elif failure_chance < 0.20:  # 5% API error
            return {'error': 'HTTP 500: Internal server error', 'retry_reason': RetryReason.API_ERROR}
        elif failure_chance < 0.23:  # 3% timeout
            return {'error': 'Request timeout', 'retry_reason': RetryReason.TIMEOUT}
        elif failure_chance < 0.25:  # 2% parse error
            return {'error': 'JSON parsing failed', 'retry_reason': RetryReason.PARSE_ERROR}
        else:
            # Success
            return {
                'sentiment': random.choice(['positive', 'negative', 'neutral']),
                'recommendation': random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': random.choice(['high', 'medium', 'low']),
                'explanation': 'Simulated analysis result',
                'analysis_time_ms': int((time.time() % 1) * 1000)
            }
    
    async def _insert_article_immediately(self, article: Dict[str, Any], analysis_result: Dict[str, Any], index: int):
        """Insert article immediately with sentiment data"""
        # Update article with sentiment data
        article.update({
            'sentiment': analysis_result.get('sentiment', 'neutral'),
            'recommendation': analysis_result.get('recommendation', 'HOLD'),
            'confidence': analysis_result.get('confidence', 'low'),
            'explanation': analysis_result.get('explanation', 'No explanation'),
            'analysis_time_ms': analysis_result.get('analysis_time_ms', 0),
            'analyzed_at': datetime.now()
        })
        
        # Simulate database insertion
        await asyncio.sleep(0.05)  # Simulate DB insert time
        
        logger.info(f"💾 #{index+1:2d} INSERTED: {article.get('ticker', 'UNKNOWN')} -> Database")
    
    async def _add_to_retry_queue(self, article: Dict[str, Any], analysis_result: Dict[str, Any], index: int):
        """Add failed article to retry queue with smart backoff calculation"""
        retry_reason = analysis_result.get('retry_reason', RetryReason.UNKNOWN)
        error_msg = analysis_result.get('error', 'Unknown error')
        
        # Calculate next retry time based on failure type - MORE AGGRESSIVE DELAYS
        now = time.time()
        if retry_reason == RetryReason.RATE_LIMIT:
            # Much longer delay for rate limits - let the API cool down
            next_retry_time = now + (self.base_delay * 5)  # 15 seconds
        elif retry_reason == RetryReason.API_ERROR:
            # Longer delay for API errors - server might be struggling
            next_retry_time = now + (self.base_delay * 3)  # 9 seconds
        elif retry_reason == RetryReason.TIMEOUT:
            # Medium delay for timeouts - network might be slow
            next_retry_time = now + (self.base_delay * 2)  # 6 seconds
        else:
            # Standard delay for other errors
            next_retry_time = now + self.base_delay  # 3 seconds
        
        retry_item = RetryItem(
            article=article,
            attempt_count=1,
            last_attempt_time=now,
            retry_reason=retry_reason,
            original_error=error_msg,
            next_retry_time=next_retry_time
        )
        
        async with self.retry_queue_lock:
            self.retry_queue.append(retry_item)
    
    async def _process_retry_queue(self):
        """Process the retry queue until empty or max attempts reached"""
        logger.info(f"🔄 PROCESSING RETRY QUEUE: {len(self.retry_queue)} articles to retry")
        
        retry_round = 1
        
        while self.retry_queue and retry_round <= self.max_retries:
            logger.info(f"🔄 RETRY ROUND {retry_round}: Processing {len(self.retry_queue)} articles")
            
            # Get items ready for retry
            now = time.time()
            ready_items = []
            waiting_items = []
            
            async with self.retry_queue_lock:
                for item in self.retry_queue:
                    if now >= item.next_retry_time:
                        ready_items.append(item)
                    else:
                        waiting_items.append(item)
                
                # Update retry queue to only waiting items
                self.retry_queue = waiting_items
            
            if not ready_items:
                # Wait for next items to be ready
                if waiting_items:
                    min_wait_time = min(item.next_retry_time - now for item in waiting_items)
                    logger.info(f"⏳ Waiting {min_wait_time:.1f}s for next retry batch...")
                    await asyncio.sleep(min_wait_time)
                    continue
                else:
                    break
            
            # Process ready items with STAGGERED DELAYS to avoid rate limit cascades
            retry_tasks = []
            for i, item in enumerate(ready_items):
                # Stagger retries by 0.5s each to avoid hitting rate limits simultaneously
                stagger_delay = i * 0.5
                task = asyncio.create_task(self._retry_single_article_with_stagger(item, retry_round, stagger_delay))
                retry_tasks.append(task)
            
            await asyncio.gather(*retry_tasks, return_exceptions=True)
            
            retry_round += 1
        
        # Handle permanently failed articles
        await self._handle_permanent_failures()
    
    async def _retry_single_article_with_stagger(self, retry_item: RetryItem, retry_round: int, stagger_delay: float):
        """Retry a single article with staggered delay to avoid rate limit cascades"""
        if stagger_delay > 0:
            logger.info(f"⏳ STAGGER DELAY: {retry_item.article.get('ticker', 'UNKNOWN')} waiting {stagger_delay:.1f}s to avoid rate limit cascade")
            await asyncio.sleep(stagger_delay)
        
        await self._retry_single_article(retry_item, retry_round)
    
    async def _retry_single_article(self, retry_item: RetryItem, retry_round: int):
        """Retry a single article with exponential backoff"""
        article = retry_item.article
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}")
        
        # Skip if already processed
        if content_hash in self.processed_articles:
            return
        
        try:
            logger.info(f"🔄 RETRY {retry_round}: {ticker} (attempt {retry_item.attempt_count + 1})")
            
            self.stats['total_retry_attempts'] += 1
            
            # Retry sentiment analysis
            analysis_result = await self._simulate_sentiment_analysis(article)
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Insert immediately
                await self._insert_article_immediately(article, analysis_result, 0)
                self.stats['articles_analyzed_successfully'] += 1
                self.stats['articles_recovered_on_retry'] += 1
                self.processed_articles.add(content_hash)
                
                logger.info(f"✅ RETRY SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')} (recovered on retry)")
                
            else:
                # STILL FAILING: Update retry item
                retry_item.attempt_count += 1
                retry_item.last_attempt_time = time.time()
                retry_item.retry_reason = analysis_result.get('retry_reason', RetryReason.UNKNOWN)
                
                # MORE AGGRESSIVE EXPONENTIAL BACKOFF
                base_backoff = 2 ** min(retry_item.attempt_count, 4)  # Cap at 16x multiplier
                
                # Different multipliers based on error type
                if retry_item.retry_reason == RetryReason.RATE_LIMIT:
                    # Extra aggressive for rate limits - these need time to reset
                    backoff_multiplier = base_backoff * 4  # 8s, 16s, 32s, 64s
                elif retry_item.retry_reason == RetryReason.API_ERROR:
                    # Aggressive for API errors - server issues need time
                    backoff_multiplier = base_backoff * 2  # 4s, 8s, 16s, 32s
                else:
                    # Standard exponential backoff for other errors
                    backoff_multiplier = base_backoff  # 2s, 4s, 8s, 16s
                
                retry_item.next_retry_time = time.time() + (self.base_delay * backoff_multiplier)
                
                # Add back to retry queue if under max attempts
                if retry_item.attempt_count < self.max_retries:
                    async with self.retry_queue_lock:
                        self.retry_queue.append(retry_item)
                    
                    logger.warning(f"⚠️ RETRY FAILED: {ticker} -> Will retry again in {self.base_delay * backoff_multiplier:.1f}s (attempt {retry_item.attempt_count}/{self.max_retries})")
                else:
                    logger.error(f"❌ MAX RETRIES EXCEEDED: {ticker} -> Moving to permanent failure handling")
                
        except Exception as e:
            logger.error(f"❌ RETRY EXCEPTION: {ticker} -> {str(e)}")
            
            # Update retry item for exception
            retry_item.attempt_count += 1
            retry_item.last_attempt_time = time.time()
            retry_item.next_retry_time = time.time() + (self.base_delay * (2 ** retry_item.attempt_count))
            
            if retry_item.attempt_count < self.max_retries:
                async with self.retry_queue_lock:
                    self.retry_queue.append(retry_item)
    
    async def _handle_permanent_failures(self):
        """Handle articles that permanently failed analysis - ZERO LOSS GUARANTEE"""
        if not self.retry_queue:
            return
        
        logger.warning(f"🚨 HANDLING {len(self.retry_queue)} PERMANENT FAILURES with default sentiment")
        
        for retry_item in self.retry_queue:
            article = retry_item.article
            ticker = article.get('ticker', 'UNKNOWN')
            content_hash = article.get('content_hash', f"hash_{ticker}")
            
            # Skip if already processed
            if content_hash in self.processed_articles:
                continue
            
            # Insert with default sentiment - ZERO LOSS GUARANTEE
            default_analysis = {
                'sentiment': 'neutral',
                'recommendation': 'HOLD',
                'confidence': 'low',
                'explanation': f'Analysis failed after {retry_item.attempt_count} attempts: {retry_item.original_error}',
                'analysis_time_ms': 0
            }
            
            await self._insert_article_immediately(article, default_analysis, 0)
            self.processed_articles.add(content_hash)
            self.stats['articles_failed_permanently'] += 1
            
            logger.warning(f"🛡️ ZERO LOSS: {ticker} -> Inserted with default sentiment (HOLD)")
        
        # Clear retry queue
        self.retry_queue.clear()
        
        logger.info(f"✅ ZERO LOSS GUARANTEE: All articles processed (some with default sentiment)")
    
    async def _verify_zero_loss(self, original_articles: List[Dict[str, Any]]):
        """Verify that all articles were processed - CRITICAL SAFETY CHECK"""
        expected_count = len(original_articles)
        processed_count = len(self.processed_articles)
        
        if processed_count != expected_count:
            self.stats['zero_loss_guarantee'] = False
            logger.error(f"🚨 ZERO LOSS VIOLATION: Expected {expected_count}, processed {processed_count}")
            
            # Find missing articles
            expected_hashes = {
                article.get('content_hash', f"hash_{article.get('ticker', 'UNKNOWN')}_{i}")
                for i, article in enumerate(original_articles)
            }
            missing_hashes = expected_hashes - self.processed_articles
            
            logger.error(f"🚨 MISSING ARTICLES: {missing_hashes}")
            raise Exception(f"Zero loss guarantee violated: {len(missing_hashes)} articles not processed")
        else:
            logger.info(f"✅ ZERO LOSS VERIFIED: All {expected_count} articles processed successfully")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        total_articles = self.stats['articles_received']
        success_rate = (self.stats['articles_analyzed_successfully'] / total_articles * 100) if total_articles > 0 else 0
        immediate_success_rate = (self.stats['articles_inserted_immediately'] / total_articles * 100) if total_articles > 0 else 0
        recovery_rate = (self.stats['articles_recovered_on_retry'] / max(1, self.stats['articles_failed_initial']) * 100)
        
        # Calculate real analysis success rate (successful analyses vs defaults)
        real_analysis_success_rate = (self.stats['articles_analyzed_successfully'] / total_articles * 100) if total_articles > 0 else 0
        default_insertion_rate = (self.stats['articles_failed_permanently'] / total_articles * 100) if total_articles > 0 else 0
        
        return {
            'total_articles': total_articles,
            'success_rate': f"{success_rate:.1f}%",
            'immediate_success_rate': f"{immediate_success_rate:.1f}%",
            'recovery_rate': f"{recovery_rate:.1f}%",
            'real_analysis_success_rate': f"{real_analysis_success_rate:.1f}%",  # KEY METRIC
            'default_insertion_rate': f"{default_insertion_rate:.1f}%",  # MINIMIZE THIS
            'zero_loss_achieved': self.stats['zero_loss_guarantee'],
            'detailed_stats': self.stats
        }

# Example usage and testing
async def test_enhanced_processing():
    """Test the enhanced individual processing"""
    
    # Create test articles
    test_articles = []
    for i in range(10):
        test_articles.append({
            'ticker': f'TEST{i:02d}',
            'headline': f'Test headline {i}',
            'summary': f'Test summary {i}',
            'content_hash': f'test_hash_{i}',
            'article_url': f'https://example.com/article{i}',
            'timestamp': datetime.now()
        })
    
    # Process with enhanced individual processing - MORE AGGRESSIVE SETTINGS
    processor = EnhancedIndividualProcessor(max_retries=6, base_delay=2.0)
    
    logger.info("🚀 STARTING ENHANCED INDIVIDUAL PROCESSING TEST (AGGRESSIVE RETRY)")
    
    result = await processor.process_article_batch(test_articles)
    
    logger.info("📊 PROCESSING COMPLETE")
    logger.info(f"Processing Time: {result['processing_time']:.2f}s")
    logger.info(f"Zero Loss Achieved: {result['zero_loss_achieved']}")
    
    summary = processor.get_summary()
    logger.info(f"🎯 REAL ANALYSIS SUCCESS: {summary['real_analysis_success_rate']} (KEY METRIC)")
    logger.info(f"⚠️ DEFAULT INSERTIONS: {summary['default_insertion_rate']} (MINIMIZE THIS)")
    logger.info(f"⚡ Immediate Success Rate: {summary['immediate_success_rate']}")
    logger.info(f"🔄 Recovery Rate: {summary['recovery_rate']}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_enhanced_processing()) 