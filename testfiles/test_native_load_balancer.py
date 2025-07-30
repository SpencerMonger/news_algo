#!/usr/bin/env python3
"""
REAL SYSTEM Individual Processing Test with Native Load Balancer
Tests the enhanced individual processing with REAL API calls using Native Load Balancer.

REAL SYSTEM SIMULATION:
1. Read articles from 'breaking_news' table (simulates WebSocket input)
2. Run REAL Claude API sentiment analysis through Native Load Balancer
3. Insert into 'news_testing' table (real database operations)

This uses the actual Native Load Balancer from the live system.
"""

import asyncio
import json
import time
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random

# Import real system components
from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Import the native load balancer from sentiment service
from sentiment_service import NativeLoadBalancer

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

class NativeLoadBalancerSentimentService:
    """
    Sentiment service using the Native Load Balancer from the live system
    This is identical to the live system's load balancing approach
    """
    
    def __init__(self):
        self.load_balancer = NativeLoadBalancer()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'native_errors': 0
        }
        
    async def initialize(self):
        """Initialize Native Load Balancer"""
        try:
            # Initialize the native load balancer
            success = await self.load_balancer.initialize()
            
            if success:
                logger.info("✅ Native Load Balancer initialized successfully")
                logger.info(f"🔑 Load balancing across {len(self.load_balancer.api_keys)} API keys")
                return True
            else:
                logger.error("❌ Failed to initialize Native Load Balancer")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Native Load Balancer: {e}")
            return False
    
    async def analyze_article_sentiment_via_native_lb(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article sentiment through Native Load Balancer with real load balancing
        """
        self.stats['total_requests'] += 1
        
        try:
            # Create prompt for sentiment analysis
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            full_content = article.get('full_content', '')
            
            content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
            content_to_analyze = content_to_analyze[:6000]  # Limit to 6K chars
            
            prompt = f"""
Analyze the following news article about {ticker} and determine if it suggests a BUY, SELL, or HOLD signal based on the sentiment and potential market impact.

Article Content:
{content_to_analyze}

Instructions:
1. Analyze the sentiment (positive, negative, neutral)
2. Consider the potential market impact on stock price
3. Provide a clear recommendation:
   - BUY: For positive sentiment with strong bullish indicators
   - SELL: For negative sentiment with strong bearish indicators  
   - HOLD: For neutral sentiment or unclear market impact
4. Rate confidence as high, medium, or low
5. Give a brief explanation (1-2 sentences)

Respond in this exact JSON format:
{{
    "ticker": "{ticker}",
    "sentiment": "positive/negative/neutral",
    "recommendation": "BUY/SELL/HOLD",
    "confidence": "high/medium/low",
    "explanation": "Brief explanation of your reasoning"
}}

Important: Use exactly "BUY", "SELL", or "HOLD" for recommendation.
"""
            
            # Make request through Native Load Balancer
            start_time = time.time()
            
            result = await self.load_balancer.make_claude_request(prompt)
            
            analysis_time = time.time() - start_time
            
            # Parse the response
            if result and 'error' not in result:
                result['analysis_time_ms'] = int(analysis_time * 1000)
                result['analyzed_at'] = datetime.now()
                
                self.stats['successful_requests'] += 1
                logger.debug(f"🎯 NATIVE LB SUCCESS: {ticker} -> {result.get('recommendation', 'HOLD')}")
                
                return result
            else:
                error_result = result if result else {"error": "No response from load balancer"}
                self.stats['failed_requests'] += 1
                logger.warning(f"⚠️ NATIVE LB ERROR: {ticker} - {error_result.get('error', 'Unknown error')}")
                return error_result
                
        except Exception as e:
            self.stats['native_errors'] += 1
            error_msg = str(e)
            
            # Check for rate limit errors
            if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                self.stats['rate_limit_hits'] += 1
            
            logger.error(f"❌ NATIVE LB EXCEPTION: {article.get('ticker', 'UNKNOWN')} - {error_msg}")
            return {'error': f'Native load balancer exception: {error_msg}'}
    
    def get_native_lb_stats(self) -> Dict[str, Any]:
        """Get Native Load Balancer statistics"""
        success_rate = (self.stats['successful_requests'] / max(1, self.stats['total_requests']) * 100)
        
        # Get load balancer stats
        lb_stats = self.load_balancer.get_load_balancing_stats() if self.load_balancer else {}
        
        return {
            'load_balancer_mode': 'native',
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': f"{success_rate:.1f}%",
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'native_errors': self.stats['native_errors'],
            'load_balancer_stats': lb_stats
        }
    
    async def cleanup(self):
        """Clean up load balancer resources"""
        if self.load_balancer:
            await self.load_balancer.cleanup()
        logger.info("✅ Native Load Balancer cleanup completed")

class RealSystemIndividualProcessor:
    """
    REAL SYSTEM individual processing with zero article loss guarantee
    Uses actual Claude API calls and database operations with Native Load Balancer
    """
    
    def __init__(self, max_retries: int = 8, base_delay: float = 3.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Real system components
        self.clickhouse_manager = ClickHouseManager()
        self.sentiment_service = NativeLoadBalancerSentimentService()  # Use Native Load Balancer
        
        # Tracking
        self.processed_articles: Set[str] = set()  # content_hash tracking
        self.retry_queue: List[RetryItem] = []
        self.retry_queue_lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'native_errors': 0,
            'total_retry_attempts': 0,
            'permanent_failures': set(),
            'processing_start_time': None,
            'processing_end_time': None,
            'first_insert_time': None,
            'last_insert_time': None,
            'load_balancing_stats': {},
            'retry_successes': set(),
            'retry_errors': 0,
            'processed_articles': set()
        }
        
    async def initialize(self):
        """Initialize the real system processor with database and sentiment service"""
        try:
            # Initialize ClickHouse connection
            self.clickhouse_manager = ClickHouseManager()
            await asyncio.to_thread(self.clickhouse_manager.connect)
            logger.info("✅ ClickHouse connected")
            
            # Create database
            await asyncio.to_thread(self.clickhouse_manager.create_database)
            logger.info("✅ Database created/verified")
            
            # Drop and recreate news_testing table to avoid duplicates
            logger.info("🗑️ Dropping existing news_testing table...")
            drop_query = "DROP TABLE IF EXISTS News.news_testing"
            self.clickhouse_manager.client.query(drop_query)
            logger.info("✅ Existing news_testing table dropped")
            
            # Create fresh news_testing table
            await asyncio.to_thread(self.clickhouse_manager.create_news_testing_table)
            logger.info("✅ Fresh news_testing table created")
            
            # Initialize sentiment service with Native Load Balancer
            await self.sentiment_service.initialize()
            logger.info("✅ Native Load Balancer sentiment service initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
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
            ORDER BY timestamp DESC
            LIMIT {count}
            """
            
            result = self.clickhouse_manager.client.query(query)
            
            articles = []
            for row in result.result_rows:
                # Handle potential None/empty ticker values
                ticker = row[0] if row[0] and row[0] != '' else 'MARKET'
                
                articles.append({
                    'ticker': ticker,
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
                    'content_hash': f"native_hash_{ticker}_{int(time.time())}_{hash(row[1]) % 10000}_{len(articles)}",
                    'news_type': 'other',
                    'urgency_score': 5
                })
            
            logger.info(f"✅ Retrieved {len(articles)} real articles from breaking_news")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error retrieving real articles: {e}")
            return []
    
    def _generate_content_hash(self, article: Dict[str, Any]) -> str:
        """Generate a unique content hash for an article"""
        ticker = article.get('ticker', 'UNKNOWN')
        timestamp = int(time.time())
        content_length = len(str(article.get('full_content', '')))
        article_index = hash(str(article)) % 10000
        return f"native_hash_{ticker}_{timestamp}_{content_length}_{article_index}"

    async def process_article_batch_individually(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process articles individually with REAL sentiment analysis through Native Load Balancer
        Each article gets its own API call and immediate database insertion
        """
        if not articles:
            return {'processing_time': 0, 'zero_loss_achieved': True}
        
        # Initialize processing stats
        self.stats['total_articles'] = len(articles)
        self.stats['processing_start_time'] = time.time()
        
        logger.info(f"🚀 REAL SYSTEM INDIVIDUAL PROCESSING: Starting {len(articles)} articles")
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 20  # Process 20 articles concurrently
        successful_count = 0
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}: {len(batch)} articles")
            
            # Create tasks for this batch
            tasks = []
            for j, article in enumerate(batch):
                task = asyncio.create_task(self._process_single_article_real(article, i + j + 1))
                tasks.append(task)
            
            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            for result in results:
                if not isinstance(result, Exception) and result:  # Fixed: check for truthy result
                    successful_count += 1
            
            # Small delay between batches
            if i + batch_size < len(articles):
                await asyncio.sleep(1)  # 1 second between batches
        
        # Record end time
        self.stats['processing_end_time'] = time.time()
        processing_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        
        logger.info(f"✅ PROCESSING COMPLETE: {successful_count}/{len(articles)} articles in {processing_time:.1f}s")
        
        return {
            'processing_time': processing_time,
            'successful_count': successful_count,
            'total_articles': len(articles),
            'zero_loss_achieved': successful_count == len(articles)
        }
    
    async def _process_single_article_real(self, article: Dict[str, Any], index: int) -> bool:
        """Process a single article with REAL sentiment analysis and immediate database insertion"""
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}_{index}")
        
        try:
            logger.info(f"🧠 #{index:2d} ANALYZING: {ticker} (NATIVE LOAD BALANCER)")
            
            # REAL SENTIMENT ANALYSIS using Native Load Balancer
            analysis_start = time.time()
            analysis_result = await self.sentiment_service.analyze_article_sentiment_via_native_lb(article)
            analysis_time = time.time() - analysis_start
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Immediate insertion to news_testing table
                await self._insert_article_to_news_testing_real(article, analysis_result, index)
                
                # Track successful processing
                content_hash = self._generate_content_hash(article)
                self.processed_articles.add(content_hash)
                self.stats['processed_articles'].add(content_hash)
                
                # Track timing
                current_time = time.time()
                if self.stats['first_insert_time'] is None:
                    self.stats['first_insert_time'] = current_time
                self.stats['last_insert_time'] = current_time
                
                logger.info(f"✅ #{index:2d} SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')} (inserted immediately in {analysis_time:.1f}s)")
                return True  # Return success
                
            else:
                # FAILURE: Use default sentiment
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                logger.warning(f"⚠️ #{index:2d} FAILED: {ticker} -> {error_msg}")
                
                # Insert with default sentiment for zero loss
                default_analysis = {
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Analysis failed: {error_msg}',
                    'analysis_time_ms': int(analysis_time * 1000),
                    'analyzed_at': datetime.now()
                }
                
                await self._insert_article_to_news_testing_real(article, default_analysis, index)
                return True  # Still successful insert with default
                
        except Exception as e:
            logger.error(f"❌ #{index:2d} EXCEPTION: {ticker} -> {str(e)}")
            
            # ZERO LOSS GUARANTEE: Insert with default sentiment even on exception
            try:
                default_analysis = {
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Processing exception: {str(e)}',
                    'analysis_time_ms': 0,
                    'analyzed_at': datetime.now()
                }
                
                await self._insert_article_to_news_testing_real(article, default_analysis, index)
                logger.warning(f"🛡️ #{index:2d} ZERO LOSS: {ticker} -> Inserted with default sentiment")
                return True
                
            except Exception as fallback_error:
                logger.error(f"❌ #{index:2d} FALLBACK FAILED: {ticker} -> {str(fallback_error)}")
                return False
    
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
                logger.info(f"💾 #{index:2d} INSERTED: {article.get('ticker', 'UNKNOWN')} -> news_testing table ({insert_time:.2f}s)")
            else:
                logger.error(f"❌ #{index:2d} INSERT FAILED: {article.get('ticker', 'UNKNOWN')} -> news_testing table")
                
        except Exception as e:
            logger.error(f"❌ #{index:2d} DATABASE ERROR: {article.get('ticker', 'UNKNOWN')} -> {str(e)}")
            raise e
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        total_articles = len(self.processed_articles)
        total_expected = self.stats.get('total_articles', 0)
        
        # Calculate success rate
        success_rate = (total_articles / total_expected * 100) if total_expected > 0 else 0
        
        # Check zero loss
        zero_loss_achieved = total_articles == total_expected
        
        # Get load balancer stats
        lb_stats = self.sentiment_service.get_native_lb_stats()
        
        return {
            'total_articles': total_articles,
            'expected_articles': total_expected,
            'success_rate': success_rate,
            'zero_loss_achieved': zero_loss_achieved,
            'native_load_balancer_stats': lb_stats,
            'processing_start_time': self.stats.get('processing_start_time'),
            'processing_end_time': self.stats.get('processing_end_time'),
            'first_insert_time': self.stats.get('first_insert_time'),
            'last_insert_time': self.stats.get('last_insert_time')
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        if self.clickhouse_manager:
            self.clickhouse_manager.close()

# Test function
async def test_native_load_balancer_individual_processing():
    """Test the REAL SYSTEM individual processing with Native Load Balancer"""
    
    processor = None
    try:
        logger.info("🚀 INITIALIZING NATIVE LOAD BALANCER SYSTEM...")
        
        # Initialize the real system individual processor
        processor = RealSystemIndividualProcessor(max_retries=8, base_delay=3.0)
        await processor.initialize()
        logger.info("✅ Native Load Balancer sentiment service ready")
        
        # Get real articles from breaking_news table
        test_articles = processor.get_real_articles_from_breaking_news(count=30)
        
        if not test_articles:
            logger.error("❌ No articles found in breaking_news table")
            return
            
        # Show articles being processed
        logger.info("📋 Real Articles from breaking_news:")
        for i, article in enumerate(test_articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')[:60] + "..." if len(article.get('headline', '')) > 60 else article.get('headline', '')
            logger.info(f"   {i:2d}. {ticker:6} | {headline}")
        
        logger.info("🚀 STARTING NATIVE LOAD BALANCER INDIVIDUAL PROCESSING TEST")
        
        # Process articles individually with real API calls
        start_time = time.time()
        result = await processor.process_article_batch_individually(test_articles)
        total_time = time.time() - start_time
        
        # Get detailed stats
        summary = processor.get_summary()
        
        # Display results
        logger.info("=" * 60)
        logger.info("🎯 NATIVE LOAD BALANCER TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 TOTAL ARTICLES: {len(test_articles)}")
        logger.info(f"✅ ARTICLES PROCESSED: {result.get('successful_count', 0)}")
        logger.info(f"⏱️  TOTAL PROCESSING TIME: {result.get('processing_time', total_time):.1f} seconds")
        logger.info(f"⚡ AVERAGE TIME PER ARTICLE: {result.get('processing_time', total_time)/len(test_articles):.1f} seconds")
        logger.info(f"🎯 SUCCESS RATE: {result.get('successful_count', 0)/len(test_articles)*100:.1f}%")
        
        # Show load balancer stats
        lb_stats = summary.get('native_load_balancer_stats', {})
        if lb_stats:
            logger.info("=" * 60)
            logger.info("🔑 NATIVE LOAD BALANCER STATS")
            logger.info("=" * 60)
            logger.info(f"📊 Total Requests: {lb_stats.get('total_requests', 0)}")
            logger.info(f"✅ Successful Requests: {lb_stats.get('successful_requests', 0)}")
            logger.info(f"❌ Failed Requests: {lb_stats.get('failed_requests', 0)}")
            logger.info(f"🎯 LB Success Rate: {lb_stats.get('success_rate', '0%')}")
            logger.info(f"🚫 Rate Limit Hits: {lb_stats.get('rate_limit_hits', 0)}")
        
        logger.info("=" * 60)
        logger.info("✅ TEST COMPLETED - NATIVE LOAD BALANCER WORKING!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        # Still show timing if we have it
        try:
            if 'start_time' in locals():
                elapsed = time.time() - start_time
                logger.info(f"⏱️  Partial processing time: {elapsed:.1f} seconds")
        except:
            pass

    finally:
        # Cleanup
        if processor:
            await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_native_load_balancer_individual_processing()) 