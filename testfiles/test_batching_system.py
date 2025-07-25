#!/usr/bin/env python3
"""
REAL SYSTEM Individual Processing Test
Tests the enhanced individual processing with REAL API calls and REAL database operations.

REAL SYSTEM SIMULATION:
1. Read articles from 'breaking_news' table (simulates WebSocket input)
2. Run REAL Claude API sentiment analysis
3. Insert into 'news_testing' table (real database operations)

This is the ACTUAL test we need to validate the individual processing approach.
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Import real system components
from clickhouse_setup import ClickHouseManager
from sentiment_service import get_sentiment_service, clear_sentiment_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

class LoadBalancedSentimentService:
    """Wrapper for sentiment service with load balancing across multiple API keys"""
    
    def __init__(self):
        self.api_keys = []
        self.sentiment_services = []
        self.current_key_index = 0
        self.key_usage_stats = {}
        
    async def initialize(self):
        """Initialize multiple sentiment services with different API keys"""
        # Get all available API keys
        api_key_1 = os.getenv('ANTHROPIC_API_KEY')
        api_key_2 = os.getenv('ANTHROPIC_API_KEY2')
        api_key_3 = os.getenv('ANTHROPIC_API_KEY3')
        
        # Debug logging to see what keys are available
        logger.info(f"🔍 DEBUG: API KEY 1 = {'✅ Found' if api_key_1 else '❌ Missing'}")
        logger.info(f"🔍 DEBUG: API KEY 2 = {'✅ Found' if api_key_2 else '❌ Missing'}")
        logger.info(f"🔍 DEBUG: API KEY 3 = {'✅ Found' if api_key_3 else '❌ Missing'}")
        
        available_keys = []
        if api_key_1:
            available_keys.append(('KEY1', api_key_1))
        if api_key_2:
            available_keys.append(('KEY2', api_key_2))
        if api_key_3:
            available_keys.append(('KEY3', api_key_3))
            
        if not available_keys:
            raise Exception("No API keys found in environment variables")
            
        logger.info(f"🔑 LOAD BALANCING: Found {len(available_keys)} API keys")
        
        # Initialize sentiment services for each key
        for key_name, api_key in available_keys:
            # Temporarily set the API key in environment
            original_key = os.environ.get('ANTHROPIC_API_KEY')
            os.environ['ANTHROPIC_API_KEY'] = api_key
            
            try:
                # Clear any existing sentiment cache to ensure fresh initialization
                await clear_sentiment_cache()
                service = await get_sentiment_service()
                self.sentiment_services.append(service)
                self.api_keys.append(key_name)
                self.key_usage_stats[key_name] = {
                    'requests': 0,
                    'successes': 0,
                    'rate_limits': 0,
                    'errors': 0
                }
                logger.info(f"✅ {key_name}: Sentiment service initialized successfully")
            except Exception as e:
                logger.error(f"❌ {key_name}: Failed to initialize - {e}")
                # Don't fail completely, continue with other keys
            finally:
                # Restore original key
                if original_key:
                    os.environ['ANTHROPIC_API_KEY'] = original_key
        
        if not self.sentiment_services:
            raise Exception("No sentiment services could be initialized")
            
        logger.info(f"🎯 LOAD BALANCING: {len(self.sentiment_services)} services ready for round-robin")
        
    def get_next_service(self):
        """Get next sentiment service using round-robin load balancing"""
        if not self.sentiment_services:
            raise Exception("No sentiment services available")
            
        service = self.sentiment_services[self.current_key_index]
        key_name = self.api_keys[self.current_key_index]
        
        # Move to next service for next request
        self.current_key_index = (self.current_key_index + 1) % len(self.sentiment_services)
        
        return service, key_name
    
    async def analyze_article_sentiment(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze article sentiment with load balancing across API keys"""
        service, key_name = self.get_next_service()
        
        # Update usage stats
        self.key_usage_stats[key_name]['requests'] += 1
        
        try:
            result = await service.analyze_article_sentiment(article)
            
            if result and 'error' not in result:
                self.key_usage_stats[key_name]['successes'] += 1
                logger.debug(f"🔑 {key_name}: SUCCESS for {article.get('ticker', 'UNKNOWN')}")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                if 'HTTP 429' in error_msg or 'Rate limit' in error_msg:
                    self.key_usage_stats[key_name]['rate_limits'] += 1
                    logger.warning(f"🔑 {key_name}: RATE LIMITED for {article.get('ticker', 'UNKNOWN')}")
                else:
                    self.key_usage_stats[key_name]['errors'] += 1
                    logger.warning(f"🔑 {key_name}: ERROR for {article.get('ticker', 'UNKNOWN')} - {error_msg}")
            
            return result
            
        except Exception as e:
            self.key_usage_stats[key_name]['errors'] += 1
            logger.error(f"🔑 {key_name}: EXCEPTION for {article.get('ticker', 'UNKNOWN')} - {str(e)}")
            return {'error': f'Exception with {key_name}: {str(e)}'}
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        total_requests = sum(stats['requests'] for stats in self.key_usage_stats.values())
        
        return {
            'total_requests': total_requests,
            'keys_used': len(self.api_keys),
            'key_stats': self.key_usage_stats.copy(),
            'load_distribution': {
                key: f"{(stats['requests'] / max(1, total_requests) * 100):.1f}%"
                for key, stats in self.key_usage_stats.items()
            }
        }
    
    async def cleanup(self):
        """Clean up all sentiment services"""
        for service in self.sentiment_services:
            try:
                await service.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up sentiment service: {e}")

class RealSystemIndividualProcessor:
    """
    REAL SYSTEM individual processing with zero article loss guarantee
    Uses actual Claude API calls and database operations
    """
    
    def __init__(self, max_retries: int = 8, base_delay: float = 3.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Real system components
        self.clickhouse_manager = ClickHouseManager()
        self.sentiment_service = None  # Will be LoadBalancedSentimentService
        
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
            'zero_loss_guarantee': True,
            'real_analysis_success_rate': 0.0,
            'total_processing_time': 0.0,
            'first_insert_time': None,
            'last_insert_time': None,
            'load_balancing_stats': {}  # New: track load balancing performance
        }
        
    async def initialize(self):
        """Initialize real system components"""
        logger.info("🚀 INITIALIZING REAL SYSTEM COMPONENTS...")
        
        # Connect to ClickHouse
        self.clickhouse_manager.connect()
        logger.info("✅ ClickHouse connected")
        
        # Create database
        self.clickhouse_manager.create_database()
        logger.info("✅ Database created/verified")
        
        # Create news_testing table for the test
        self.clickhouse_manager.create_news_testing_table()
        logger.info("✅ news_testing table created/verified")
        
        # Initialize sentiment service
        self.sentiment_service = LoadBalancedSentimentService()
        await self.sentiment_service.initialize()
        logger.info("✅ Sentiment service initialized")
        
        # Clear cache for fair testing
        await clear_sentiment_cache()
        logger.info("✅ Sentiment cache cleared")
        
    def get_real_articles_from_breaking_news(self, count: int = 30) -> List[Dict[str, Any]]:
        """Get real articles from breaking_news table to simulate WebSocket input"""
        try:
            logger.info(f"📖 READING {count} REAL ARTICLES from breaking_news table...")
            
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
                    'content_hash': f"real_hash_{row[0]}_{int(time.time())}_{hash(row[1]) % 10000}_{len(articles)}",  # Added index for uniqueness
                    'news_type': 'other',
                    'urgency_score': 5
                })
            
            logger.info(f"✅ Retrieved {len(articles)} real articles from breaking_news")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error retrieving real articles: {e}")
            return []
    
    async def process_article_batch_individually(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of articles with REAL individual processing and zero loss guarantee
        """
        logger.info(f"🚀 REAL SYSTEM INDIVIDUAL PROCESSING: Starting {len(articles)} articles")
        
        self.stats['articles_received'] += len(articles)
        start_time = time.time()
        
        # Create individual processing tasks
        processing_tasks = []
        for i, article in enumerate(articles):
            task = asyncio.create_task(self._process_single_article_real(article, i))
            processing_tasks.append(task)
        
        # Start all processing tasks concurrently
        await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Process retry queue until empty or max attempts reached
        await self._process_retry_queue_real()
        
        # Final verification - ensure zero loss
        await self._verify_zero_loss(articles)
        
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] = processing_time
        
        # Capture load balancing stats
        if self.sentiment_service:
            self.stats['load_balancing_stats'] = self.sentiment_service.get_load_balancing_stats()
        
        # Return comprehensive summary
        return {
            'processing_time': processing_time,
            'stats': self.stats.copy(),
            'retry_queue_final_size': len(self.retry_queue),
            'zero_loss_achieved': self.stats['zero_loss_guarantee']
        }
    
    async def _process_single_article_real(self, article: Dict[str, Any], index: int):
        """Process a single article with REAL sentiment analysis and immediate database insertion"""
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}_{index}")
        
        try:
            logger.info(f"🧠 #{index+1:2d} ANALYZING: {ticker} (REAL API CALL)")
            
            # REAL SENTIMENT ANALYSIS using Claude API
            analysis_start = time.time()
            analysis_result = await self.sentiment_service.analyze_article_sentiment(article)
            analysis_time = time.time() - analysis_start
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Immediate insertion to news_testing table
                await self._insert_article_to_news_testing_real(article, analysis_result, index)
                self.stats['articles_analyzed_successfully'] += 1
                self.stats['articles_inserted_immediately'] += 1
                self.processed_articles.add(content_hash)
                
                # Track timing
                current_time = time.time()
                if self.stats['first_insert_time'] is None:
                    self.stats['first_insert_time'] = current_time
                self.stats['last_insert_time'] = current_time
                
                logger.info(f"✅ #{index+1:2d} SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')} (inserted immediately in {analysis_time:.1f}s)")
                
            else:
                # FAILURE: Add to retry queue
                await self._add_to_retry_queue_real(article, analysis_result, index)
                self.stats['articles_failed_initial'] += 1
                
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                logger.warning(f"⚠️ #{index+1:2d} QUEUED FOR RETRY: {ticker} -> {error_msg}")
                
        except Exception as e:
            # EXCEPTION: Add to retry queue
            error_result = {'error': f'Exception: {str(e)}'}
            await self._add_to_retry_queue_real(article, error_result, index)
            self.stats['articles_failed_initial'] += 1
            
            logger.error(f"❌ #{index+1:2d} EXCEPTION: {ticker} -> {str(e)}")
    
    async def _insert_article_to_news_testing_real(self, article: Dict[str, Any], analysis_result: Dict[str, Any], index: int):
        """Insert article with sentiment data into news_testing table (REAL DATABASE OPERATION)"""
        # Update article with sentiment data
        article.update({
            'sentiment': analysis_result.get('sentiment', 'neutral'),
            'recommendation': analysis_result.get('recommendation', 'HOLD'),
            'confidence': analysis_result.get('confidence', 'low'),
            'explanation': analysis_result.get('explanation', 'No explanation'),
            'analysis_time_ms': analysis_result.get('analysis_time_ms', 0),
            'analyzed_at': analysis_result.get('analyzed_at', datetime.now())
        })
        
        try:
            # REAL DATABASE INSERT into news_testing table
            insert_start = time.time()
            
            # Use the same insert method as the live system
            inserted_count = self.clickhouse_manager.insert_articles_to_table([article], 'news_testing')
            
            insert_time = time.time() - insert_start
            
            if inserted_count > 0:
                logger.info(f"💾 #{index+1:2d} INSERTED: {article.get('ticker', 'UNKNOWN')} -> news_testing table ({insert_time:.2f}s)")
            else:
                logger.error(f"❌ #{index+1:2d} INSERT FAILED: {article.get('ticker', 'UNKNOWN')} -> news_testing table")
                
        except Exception as e:
            logger.error(f"❌ #{index+1:2d} DATABASE ERROR: {article.get('ticker', 'UNKNOWN')} -> {str(e)}")
            raise e
    
    async def _add_to_retry_queue_real(self, article: Dict[str, Any], analysis_result: Dict[str, Any], index: int):
        """Add failed article to retry queue with smart backoff calculation"""
        error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
        
        # Determine retry reason from error message
        retry_reason = RetryReason.UNKNOWN
        if 'HTTP 429' in error_msg or 'Rate limit' in error_msg:
            retry_reason = RetryReason.RATE_LIMIT
            self.stats['rate_limit_hits'] += 1
        elif 'HTTP 5' in error_msg or 'Internal server' in error_msg:
            retry_reason = RetryReason.API_ERROR
        elif 'timeout' in error_msg.lower():
            retry_reason = RetryReason.TIMEOUT
        elif 'JSON' in error_msg or 'parsing' in error_msg:
            retry_reason = RetryReason.PARSE_ERROR
        
        # Calculate next retry time based on failure type - AGGRESSIVE DELAYS
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
    
    async def _process_retry_queue_real(self):
        """Process the retry queue with REAL API calls until empty or max attempts reached"""
        if not self.retry_queue:
            return
            
        logger.info(f"🔄 PROCESSING RETRY QUEUE: {len(self.retry_queue)} articles to retry (REAL API CALLS)")
        
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
                # Stagger retries by 1s each to avoid hitting rate limits simultaneously
                stagger_delay = i * 1.0
                task = asyncio.create_task(self._retry_single_article_real_with_stagger(item, retry_round, stagger_delay))
                retry_tasks.append(task)
            
            await asyncio.gather(*retry_tasks, return_exceptions=True)
            
            retry_round += 1
        
        # Handle permanently failed articles
        await self._handle_permanent_failures_real()
    
    async def _retry_single_article_real_with_stagger(self, retry_item: RetryItem, retry_round: int, stagger_delay: float):
        """Retry a single article with staggered delay to avoid rate limit cascades"""
        if stagger_delay > 0:
            logger.info(f"⏳ STAGGER DELAY: {retry_item.article.get('ticker', 'UNKNOWN')} waiting {stagger_delay:.1f}s to avoid rate limit cascade")
            await asyncio.sleep(stagger_delay)
        
        await self._retry_single_article_real(retry_item, retry_round)
    
    async def _retry_single_article_real(self, retry_item: RetryItem, retry_round: int):
        """Retry a single article with REAL sentiment analysis and exponential backoff"""
        article = retry_item.article
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}")
        
        # Skip if already processed
        if content_hash in self.processed_articles:
            return
        
        try:
            logger.info(f"🔄 RETRY {retry_round}: {ticker} (attempt {retry_item.attempt_count + 1}) - REAL API CALL")
            
            self.stats['total_retry_attempts'] += 1
            
            # REAL SENTIMENT ANALYSIS retry
            analysis_start = time.time()
            analysis_result = await self.sentiment_service.analyze_article_sentiment(article)
            analysis_time = time.time() - analysis_start
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Insert immediately to news_testing table
                await self._insert_article_to_news_testing_real(article, analysis_result, 0)
                self.stats['articles_analyzed_successfully'] += 1
                self.stats['articles_recovered_on_retry'] += 1
                self.processed_articles.add(content_hash)
                
                # Track timing
                current_time = time.time()
                if self.stats['first_insert_time'] is None:
                    self.stats['first_insert_time'] = current_time
                self.stats['last_insert_time'] = current_time
                
                logger.info(f"✅ RETRY SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')} (recovered on retry in {analysis_time:.1f}s)")
                
            else:
                # STILL FAILING: Update retry item
                retry_item.attempt_count += 1
                retry_item.last_attempt_time = time.time()
                
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                
                # Update retry reason
                if 'HTTP 429' in error_msg or 'Rate limit' in error_msg:
                    retry_item.retry_reason = RetryReason.RATE_LIMIT
                    self.stats['rate_limit_hits'] += 1
                elif 'HTTP 5' in error_msg:
                    retry_item.retry_reason = RetryReason.API_ERROR
                elif 'timeout' in error_msg.lower():
                    retry_item.retry_reason = RetryReason.TIMEOUT
                
                # MORE AGGRESSIVE EXPONENTIAL BACKOFF
                base_backoff = 2 ** min(retry_item.attempt_count, 4)  # Cap at 16x multiplier
                
                # Different multipliers based on error type
                if retry_item.retry_reason == RetryReason.RATE_LIMIT:
                    # Extra aggressive for rate limits - these need time to reset
                    backoff_multiplier = base_backoff * 4  # 12s, 24s, 48s, 96s
                elif retry_item.retry_reason == RetryReason.API_ERROR:
                    # Aggressive for API errors - server issues need time
                    backoff_multiplier = base_backoff * 2  # 6s, 12s, 24s, 48s
                else:
                    # Standard exponential backoff for other errors
                    backoff_multiplier = base_backoff  # 3s, 6s, 12s, 24s
                
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
    
    async def _handle_permanent_failures_real(self):
        """Handle articles that permanently failed analysis - ZERO LOSS GUARANTEE with REAL database inserts"""
        if not self.retry_queue:
            return
        
        logger.warning(f"🚨 HANDLING {len(self.retry_queue)} PERMANENT FAILURES with default sentiment (REAL INSERTS)")
        
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
                'analysis_time_ms': 0,
                'analyzed_at': datetime.now()
            }
            
            await self._insert_article_to_news_testing_real(article, default_analysis, 0)
            self.processed_articles.add(content_hash)
            self.stats['articles_failed_permanently'] += 1
            
            logger.warning(f"🛡️ ZERO LOSS: {ticker} -> Inserted with default sentiment (HOLD) into news_testing")
        
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
            
            # Find missing articles - use actual content_hash values from articles
            expected_hashes = set()
            for article in original_articles:
                content_hash = article.get('content_hash')
                if content_hash:
                    expected_hashes.add(content_hash)
                else:
                    # Fallback if no content_hash (shouldn't happen in our test)
                    ticker = article.get('ticker', 'UNKNOWN')
                    expected_hashes.add(f"fallback_hash_{ticker}_{hash(str(article))}")
            
            missing_hashes = expected_hashes - self.processed_articles
            
            logger.error(f"🚨 MISSING ARTICLES: {missing_hashes}")
            logger.error(f"🔍 EXPECTED HASHES: {expected_hashes}")
            logger.error(f"🔍 PROCESSED HASHES: {self.processed_articles}")
            
            # Check for duplicate hashes (which could cause counting issues)
            if len(expected_hashes) != expected_count:
                logger.warning(f"⚠️ DUPLICATE HASH DETECTED: {expected_count} articles but only {len(expected_hashes)} unique hashes")
                logger.warning("⚠️ This could indicate duplicate content_hash values causing counting issues")
                
                # If we have duplicate hashes but all unique hashes were processed, consider it success
                if len(missing_hashes) == 0:
                    logger.info("✅ ZERO LOSS ACHIEVED: All unique content processed (despite duplicate hashes)")
                    self.stats['zero_loss_guarantee'] = True
                    return
            
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
        
        # Calculate timing advantages
        first_insert_time = None
        if self.stats['first_insert_time'] and self.stats['total_processing_time']:
            first_insert_time = self.stats['first_insert_time'] - (time.time() - self.stats['total_processing_time'])
        
        return {
            'total_articles': total_articles,
            'success_rate': f"{success_rate:.1f}%",
            'immediate_success_rate': f"{immediate_success_rate:.1f}%",
            'recovery_rate': f"{recovery_rate:.1f}%",
            'real_analysis_success_rate': f"{real_analysis_success_rate:.1f}%",  # KEY METRIC
            'default_insertion_rate': f"{default_insertion_rate:.1f}%",  # MINIMIZE THIS
            'zero_loss_achieved': self.stats['zero_loss_guarantee'],
            'total_processing_time': f"{self.stats['total_processing_time']:.2f}s",
            'first_insert_advantage': f"{first_insert_time:.2f}s" if first_insert_time else "N/A",
            'load_balancing_stats': self.stats.get('load_balancing_stats', {}),  # NEW: Load balancing performance
            'detailed_stats': self.stats
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        if self.clickhouse_manager:
            self.clickhouse_manager.close()

# Test function
async def test_real_system_individual_processing():
    """Test the REAL SYSTEM individual processing with actual API calls and database operations"""
    
    processor = RealSystemIndividualProcessor(max_retries=6, base_delay=3.0)
    
    try:
        # Initialize real system components
        await processor.initialize()
        
        # Get real articles from breaking_news table
        test_articles = processor.get_real_articles_from_breaking_news(count=30)
        
        if not test_articles:
            logger.error("❌ No test articles available from breaking_news table")
            return
        
        # Show articles being tested
        logger.info("📋 Real Articles from breaking_news:")
        for i, article in enumerate(test_articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')[:60] + "..." if len(article.get('headline', '')) > 60 else article.get('headline', '')
            logger.info(f"   {i:2d}. {ticker:6} | {headline}")
        
        logger.info("🚀 STARTING REAL SYSTEM INDIVIDUAL PROCESSING TEST")
        
        # Process with REAL individual processing
        result = await processor.process_article_batch_individually(test_articles)
        
        logger.info("📊 REAL SYSTEM PROCESSING COMPLETE")
        logger.info(f"Total Processing Time: {result['processing_time']:.2f}s")
        logger.info(f"Zero Loss Achieved: {result['zero_loss_achieved']}")
        
        summary = processor.get_summary()
        logger.info(f"🎯 REAL ANALYSIS SUCCESS: {summary['real_analysis_success_rate']} (KEY METRIC)")
        logger.info(f"⚠️ DEFAULT INSERTIONS: {summary['default_insertion_rate']} (MINIMIZE THIS)")
        logger.info(f"⚡ Immediate Success Rate: {summary['immediate_success_rate']}")
        logger.info(f"🔄 Recovery Rate: {summary['recovery_rate']}")
        logger.info(f"📊 Total Retry Attempts: {summary['detailed_stats']['total_retry_attempts']}")
        logger.info(f"🚨 Rate Limit Hits: {summary['detailed_stats']['rate_limit_hits']}")
        
        # Display load balancing statistics
        lb_stats = summary.get('load_balancing_stats', {})
        if lb_stats:
            logger.info(f"🔑 LOAD BALANCING: {lb_stats['keys_used']} keys used, {lb_stats['total_requests']} total requests")
            for key, stats in lb_stats.get('key_stats', {}).items():
                success_rate = (stats['successes'] / max(1, stats['requests']) * 100)
                logger.info(f"   {key}: {stats['requests']} requests, {success_rate:.1f}% success, {stats['rate_limits']} rate limits")
            
            logger.info("🔑 LOAD DISTRIBUTION:")
            for key, percentage in lb_stats.get('load_distribution', {}).items():
                logger.info(f"   {key}: {percentage}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Real system test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
        
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_real_system_individual_processing()) 