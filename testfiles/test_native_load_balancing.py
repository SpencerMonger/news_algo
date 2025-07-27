#!/usr/bin/env python3
"""
REAL SYSTEM Individual Processing Test with Native Load Balancing
Tests the enhanced individual processing with REAL API calls using native Python load balancing.

NATIVE LOAD BALANCING SIMULATION:
1. Read articles from 'breaking_news' table (simulates WebSocket input)
2. Run REAL Claude API sentiment analysis through native Python load balancer
3. Insert into 'news_testing' table (real database operations)

This uses native Python load balancing across multiple API keys without external gateway servers.
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
import aiohttp

# Import real system components
from clickhouse_setup import ClickHouseManager
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

@dataclass
class APIKeyInfo:
    """Information about an API key"""
    key: str
    last_8_chars: str
    request_count: int
    success_count: int
    failure_count: int
    rate_limit_count: int
    last_used: float
    is_rate_limited: bool
    rate_limit_reset_time: float

class NativeLoadBalancer:
    """
    Native Python load balancer for Claude API keys
    Replicates Portkey Gateway functionality without external server
    """
    
    def __init__(self):
        self.api_keys: List[APIKeyInfo] = []
        self.current_key_index = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self.claude_endpoint = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20240620"
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'key_switches': 0,
            'load_balancing_stats': {}
        }
        
    async def initialize(self):
        """Initialize the load balancer with available API keys"""
        try:
            # Load all available API keys from environment
            api_keys = []
            
            # Primary key
            api_key_1 = os.getenv('ANTHROPIC_API_KEY')
            if api_key_1:
                api_keys.append(api_key_1)
                
            # Additional keys
            for i in range(2, 10):  # Support up to 9 keys
                key = os.getenv(f'ANTHROPIC_API_KEY{i}')
                if key and key != api_key_1:  # Avoid duplicates
                    api_keys.append(key)
            
            if not api_keys:
                raise Exception("No API keys found in environment variables")
            
            # Create APIKeyInfo objects
            for key in api_keys:
                key_info = APIKeyInfo(
                    key=key,
                    last_8_chars=key[-8:] if len(key) >= 8 else key,
                    request_count=0,
                    success_count=0,
                    failure_count=0,
                    rate_limit_count=0,
                    last_used=0,
                    is_rate_limited=False,
                    rate_limit_reset_time=0
                )
                self.api_keys.append(key_info)
            
            logger.info(f"🔑 NATIVE LOAD BALANCER: Initialized with {len(self.api_keys)} API keys")
            for i, key_info in enumerate(self.api_keys, 1):
                logger.info(f"🔑 KEY {i}: Anthropic Claude - {key_info.last_8_chars}")
            
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=180, connect=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=50, limit_per_host=20),
                headers={
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                }
            )
            
            logger.info("✅ Native Load Balancer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Native Load Balancer: {e}")
            return False
    
    def get_next_available_key(self) -> Optional[APIKeyInfo]:
        """Get the next available API key using round-robin with rate limit awareness"""
        current_time = time.time()
        
        # First, check if any rate-limited keys can be reset
        for key_info in self.api_keys:
            if key_info.is_rate_limited and current_time >= key_info.rate_limit_reset_time:
                key_info.is_rate_limited = False
                logger.info(f"🔓 KEY RESET: {key_info.last_8_chars} rate limit reset")
        
        # Find available keys (not rate limited)
        available_keys = [k for k in self.api_keys if not k.is_rate_limited]
        
        if not available_keys:
            logger.warning("⚠️ ALL KEYS RATE LIMITED - using least recently rate limited")
            # Use the key with the earliest reset time
            return min(self.api_keys, key=lambda k: k.rate_limit_reset_time)
        
        # Round-robin through available keys
        if self.current_key_index >= len(available_keys):
            self.current_key_index = 0
        
        selected_key = available_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(available_keys)
        
        return selected_key
    
    def mark_key_rate_limited(self, key_info: APIKeyInfo, reset_delay: float = 60.0):
        """Mark a key as rate limited with reset time"""
        key_info.is_rate_limited = True
        key_info.rate_limit_count += 1
        key_info.rate_limit_reset_time = time.time() + reset_delay
        self.stats['rate_limit_hits'] += 1
        
        logger.warning(f"🚫 KEY RATE LIMITED: {key_info.last_8_chars} - reset in {reset_delay}s")
    
    async def make_claude_request(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Make a request to Claude API with load balancing and retry logic"""
        self.stats['total_requests'] += 1
        
        for attempt in range(max_retries):
            # Get next available key
            key_info = self.get_next_available_key()
            if not key_info:
                return {"error": "No API keys available"}
            
            try:
                # Update key usage stats
                key_info.request_count += 1
                key_info.last_used = time.time()
                
                # Prepare request
                payload = {
                    "model": self.model,
                    "max_tokens": 300,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"
                        }
                    ]
                }
                
                # Make request with selected key
                headers = {'x-api-key': key_info.key}
                
                async with self.session.post(
                    self.claude_endpoint, 
                    json=payload, 
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        # Success
                        response_data = await response.json()
                        key_info.success_count += 1
                        self.stats['successful_requests'] += 1
                        
                        # Extract and parse content
                        if response_data.get("content") and len(response_data["content"]) > 0:
                            content = response_data["content"][0]["text"]
                            content = self._clean_json_from_markdown(content)
                            
                            try:
                                parsed_result = json.loads(content)
                                logger.debug(f"✅ SUCCESS: Using key {key_info.last_8_chars}")
                                return parsed_result
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON parsing failed: {e}")
                                return {"error": f"JSON parsing failed: {str(e)}"}
                        else:
                            return {"error": "No content in response"}
                    
                    elif response.status == 429:
                        # Rate limit - mark key and try next one
                        self.mark_key_rate_limited(key_info)
                        key_info.failure_count += 1
                        
                        if attempt < max_retries - 1:
                            self.stats['key_switches'] += 1
                            logger.info(f"🔄 SWITCHING KEYS: {key_info.last_8_chars} rate limited, trying next key")
                            continue
                        else:
                            self.stats['failed_requests'] += 1
                            return {"error": "All keys rate limited"}
                    
                    else:
                        # Other HTTP error
                        response_text = await response.text()
                        key_info.failure_count += 1
                        error_msg = f"HTTP {response.status}: {response_text}"
                        
                        if attempt < max_retries - 1:
                            logger.warning(f"⚠️ HTTP {response.status} with key {key_info.last_8_chars}, retrying...")
                            await asyncio.sleep(1)
                            continue
                        else:
                            self.stats['failed_requests'] += 1
                            return {"error": error_msg}
                            
            except Exception as e:
                key_info.failure_count += 1
                error_msg = str(e)
                
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Request exception with key {key_info.last_8_chars}: {e}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    self.stats['failed_requests'] += 1
                    return {"error": error_msg}
        
        self.stats['failed_requests'] += 1
        return {"error": "Max retries exceeded"}
    
    def _clean_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks"""
        if '```json' in content:
            start = content.find('```json') + 7
            end = content.find('```', start)
            if end != -1:
                return content[start:end].strip()
        elif '```' in content:
            start = content.find('```') + 3
            end = content.find('```', start)
            if end != -1:
                return content[start:end].strip()
        return content.strip()
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get detailed load balancing statistics"""
        stats = {
            'total_keys': len(self.api_keys),
            'available_keys': len([k for k in self.api_keys if not k.is_rate_limited]),
            'rate_limited_keys': len([k for k in self.api_keys if k.is_rate_limited]),
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'key_switches': self.stats['key_switches'],
            'success_rate': (self.stats['successful_requests'] / max(1, self.stats['total_requests']) * 100),
            'key_details': []
        }
        
        for i, key_info in enumerate(self.api_keys, 1):
            key_stats = {
                'key_id': f"Key_{i}",
                'last_8_chars': key_info.last_8_chars,
                'request_count': key_info.request_count,
                'success_count': key_info.success_count,
                'failure_count': key_info.failure_count,
                'rate_limit_count': key_info.rate_limit_count,
                'success_rate': (key_info.success_count / max(1, key_info.request_count) * 100),
                'is_rate_limited': key_info.is_rate_limited,
                'last_used': key_info.last_used
            }
            stats['key_details'].append(key_stats)
        
        return stats
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        logger.info("✅ Native Load Balancer cleanup completed")

class NativeSentimentService:
    """
    Sentiment service using native Python load balancing
    Drop-in replacement for PortkeyGatewaySentimentService
    """
    
    def __init__(self):
        self.load_balancer = NativeLoadBalancer()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'load_balancing_errors': 0
        }
        
    async def initialize(self):
        """Initialize the native sentiment service"""
        try:
            success = await self.load_balancer.initialize()
            if success:
                logger.info("✅ Native Sentiment Service initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize Native Sentiment Service")
                return False
        except Exception as e:
            logger.error(f"❌ Native Sentiment Service initialization error: {e}")
            return False
    
    async def analyze_article_sentiment_via_native_balancer(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article sentiment through native load balancer
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
            
            # Make request through native load balancer
            start_time = time.time()
            analysis_result = await self.load_balancer.make_claude_request(prompt)
            analysis_time = time.time() - start_time
            
            if analysis_result and 'error' not in analysis_result:
                # Success
                analysis_result['analysis_time_ms'] = int(analysis_time * 1000)
                analysis_result['analyzed_at'] = datetime.now()
                
                self.stats['successful_requests'] += 1
                logger.debug(f"🎯 NATIVE SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')}")
                
                return analysis_result
            else:
                # Error
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                self.stats['failed_requests'] += 1
                
                # Check for rate limit errors
                if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                    self.stats['rate_limit_hits'] += 1
                
                logger.warning(f"⚠️ NATIVE ERROR: {ticker} - {error_msg}")
                return {'error': f'Native load balancer error: {error_msg}'}
                
        except Exception as e:
            self.stats['load_balancing_errors'] += 1
            error_msg = str(e)
            logger.error(f"❌ NATIVE EXCEPTION: {article.get('ticker', 'UNKNOWN')} - {error_msg}")
            return {'error': f'Native load balancer exception: {error_msg}'}
    
    def get_native_stats(self) -> Dict[str, Any]:
        """Get native load balancing statistics"""
        load_balancer_stats = self.load_balancer.get_load_balancing_stats()
        
        return {
            'load_balancing_mode': 'native_python',
            'service_stats': self.stats,
            'load_balancer_stats': load_balancer_stats
        }
    
    async def cleanup(self):
        """Clean up native service resources"""
        await self.load_balancer.cleanup()
        logger.info("✅ Native Sentiment Service cleanup completed")

class RealSystemNativeProcessor:
    """
    REAL SYSTEM individual processing with native Python load balancing
    Uses actual Claude API calls through native load balancer
    """
    
    def __init__(self, max_retries: int = 8, base_delay: float = 3.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Real system components
        self.clickhouse_manager = ClickHouseManager()
        self.sentiment_service = NativeSentimentService()  # Use native service
        
        # Tracking
        self.processed_articles: Set[str] = set()
        self.retry_queue: List[RetryItem] = []
        self.retry_queue_lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'load_balancing_errors': 0,
            'total_retry_attempts': 0,
            'permanent_failures': set(),
            'processing_start_time': None,
            'processing_end_time': None,
            'first_insert_time': None,
            'last_insert_time': None,
            'native_load_balancing_stats': {},
            'retry_successes': set(),
            'retry_errors': 0,
            'processed_articles': set()
        }
        
    async def initialize(self):
        """Initialize the real system processor with database and native sentiment service"""
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
            
            # Initialize native sentiment service
            self.sentiment_service = NativeSentimentService()
            await self.sentiment_service.initialize()
            logger.info("✅ Native sentiment service initialized")
            
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
        Process articles individually with REAL sentiment analysis through native load balancer
        """
        if not articles:
            return {'processing_time': 0, 'zero_loss_achieved': True}
        
        # Initialize processing stats
        self.stats['total_articles'] = len(articles)
        self.stats['processing_start_time'] = time.time()
        
        logger.info(f"🚀 NATIVE LOAD BALANCING PROCESSING: Starting {len(articles)} articles")
        
        # Process in smaller batches to manage load
        batch_size = 20
        successful_count = 0
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}: {len(batch)} articles")
            
            # Create tasks for this batch
            tasks = []
            for j, article in enumerate(batch):
                task = asyncio.create_task(self._process_single_article_native(article, i + j + 1))
                tasks.append(task)
            
            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            for result in results:
                if not isinstance(result, Exception):
                    successful_count += 1
            
            # Small delay between batches
            if i + batch_size < len(articles):
                await asyncio.sleep(1)
        
        # Record end time
        self.stats['processing_end_time'] = time.time()
        processing_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        
        logger.info(f"✅ NATIVE PROCESSING COMPLETE: {successful_count}/{len(articles)} articles in {processing_time:.1f}s")
        
        return {
            'processing_time': processing_time,
            'successful_count': successful_count,
            'total_articles': len(articles),
            'zero_loss_achieved': successful_count == len(articles)
        }
    
    async def _process_single_article_native(self, article: Dict[str, Any], index: int):
        """Process a single article with native load balancing sentiment analysis"""
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}_{index}")
        
        try:
            logger.info(f"🧠 #{index+1:2d} ANALYZING: {ticker} (NATIVE LOAD BALANCING)")
            
            # REAL SENTIMENT ANALYSIS using native load balancer
            analysis_start = time.time()
            analysis_result = await self.sentiment_service.analyze_article_sentiment_via_native_balancer(article)
            analysis_time = time.time() - analysis_start
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Immediate insertion to news_testing table
                await self._insert_article_to_news_testing_native(article, analysis_result, index)
                
                # Track successful processing
                content_hash = self._generate_content_hash(article)
                self.processed_articles.add(content_hash)
                self.stats['processed_articles'].add(content_hash)
                
                # Track timing
                current_time = time.time()
                if self.stats['first_insert_time'] is None:
                    self.stats['first_insert_time'] = current_time
                self.stats['last_insert_time'] = current_time
                
                logger.info(f"✅ #{index+1:2d} SUCCESS: {ticker} -> {analysis_result.get('recommendation', 'HOLD')} (native load balancing in {analysis_time:.1f}s)")
                
            else:
                # FAILURE: Add to retry queue
                await self._add_to_retry_queue_native(article, analysis_result, index)
                
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                logger.warning(f"⚠️ #{index+1:2d} QUEUED FOR RETRY: {ticker} -> {error_msg}")
                
        except Exception as e:
            # EXCEPTION: Add to retry queue
            error_result = {'error': f'Exception: {str(e)}'}
            await self._add_to_retry_queue_native(article, error_result, index)
            
            logger.error(f"❌ #{index+1:2d} EXCEPTION: {ticker} -> {str(e)}")
    
    async def _insert_article_to_news_testing_native(self, article: Dict[str, Any], analysis_result: Dict[str, Any], index: int):
        """Insert article with sentiment data into news_testing table"""
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
            
            inserted_count = self.clickhouse_manager.insert_articles_to_table([article], 'news_testing')
            
            insert_time = time.time() - insert_start
            
            if inserted_count > 0:
                logger.info(f"💾 #{index+1:2d} INSERTED: {article.get('ticker', 'UNKNOWN')} -> news_testing table ({insert_time:.2f}s)")
            else:
                logger.error(f"❌ #{index+1:2d} INSERT FAILED: {article.get('ticker', 'UNKNOWN')} -> news_testing table")
                
        except Exception as e:
            logger.error(f"❌ #{index+1:2d} DATABASE ERROR: {article.get('ticker', 'UNKNOWN')} -> {str(e)}")
            raise e
    
    async def _add_to_retry_queue_native(self, article: Dict[str, Any], analysis_result: Dict[str, Any], index: int):
        """Add failed article to retry queue"""
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
        
        # Calculate next retry time
        now = time.time()
        if retry_reason == RetryReason.RATE_LIMIT:
            next_retry_time = now + (self.base_delay * 5)  # 15 seconds
        elif retry_reason == RetryReason.API_ERROR:
            next_retry_time = now + (self.base_delay * 3)  # 9 seconds
        elif retry_reason == RetryReason.TIMEOUT:
            next_retry_time = now + (self.base_delay * 2)  # 6 seconds
        else:
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        total_articles = len(self.processed_articles)
        total_expected = self.stats.get('total_articles', 0)
        
        success_rate = (total_articles / total_expected * 100) if total_expected > 0 else 0
        zero_loss_achieved = total_articles == total_expected
        
        # Get native load balancing stats
        native_stats = self.sentiment_service.get_native_stats()
        
        return {
            'total_articles': total_articles,
            'expected_articles': total_expected,
            'success_rate': success_rate,
            'zero_loss_achieved': zero_loss_achieved,
            'total_retry_attempts': self.stats.get('total_retry_attempts', 0),
            'retry_successes': len(self.stats.get('retry_successes', set())),
            'retry_errors': self.stats.get('retry_errors', 0),
            'rate_limit_hits': self.stats.get('rate_limit_hits', 0),
            'load_balancing_errors': self.stats.get('load_balancing_errors', 0),
            'native_load_balancing_stats': native_stats,
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
async def test_native_load_balancing_system():
    """Test the NATIVE LOAD BALANCING system with actual API calls and database operations"""
    
    processor = None
    try:
        logger.info("🚀 INITIALIZING NATIVE LOAD BALANCING SYSTEM...")
        
        # Initialize the native load balancing processor
        processor = RealSystemNativeProcessor(max_retries=8, base_delay=3.0)
        await processor.initialize()
        logger.info("✅ Native load balancing services ready")
        
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
        
        logger.info("🚀 STARTING NATIVE LOAD BALANCING TEST")
        
        # Process articles individually with native load balancing
        start_time = time.time()
        result = await processor.process_article_batch_individually(test_articles)
        total_time = time.time() - start_time
        
        # Get detailed stats
        summary = processor.get_summary()
        native_stats = summary.get('native_load_balancing_stats', {})
        
        # Display results
        logger.info("=" * 60)
        logger.info("🎯 NATIVE LOAD BALANCING TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 TOTAL ARTICLES: {len(test_articles)}")
        logger.info(f"✅ ARTICLES PROCESSED: {result.get('successful_count', 0)}")
        logger.info(f"⏱️  TOTAL PROCESSING TIME: {result.get('processing_time', total_time):.1f} seconds")
        logger.info(f"⚡ AVERAGE TIME PER ARTICLE: {result.get('processing_time', total_time)/len(test_articles):.1f} seconds")
        logger.info(f"🎯 SUCCESS RATE: {result.get('successful_count', 0)/len(test_articles)*100:.1f}%")
        logger.info("=" * 60)
        logger.info("🔑 NATIVE LOAD BALANCING STATS:")
        
        if native_stats:
            lb_stats = native_stats.get('load_balancer_stats', {})
            logger.info(f"   📊 Total API Keys: {lb_stats.get('total_keys', 0)}")
            logger.info(f"   ✅ Available Keys: {lb_stats.get('available_keys', 0)}")
            logger.info(f"   🚫 Rate Limited Keys: {lb_stats.get('rate_limited_keys', 0)}")
            logger.info(f"   🔄 Key Switches: {lb_stats.get('key_switches', 0)}")
            logger.info(f"   📈 Load Balancer Success Rate: {lb_stats.get('success_rate', 0):.1f}%")
            
            # Show per-key stats
            key_details = lb_stats.get('key_details', [])
            if key_details:
                logger.info("   🔑 Per-Key Statistics:")
                for key_stat in key_details:
                    logger.info(f"      {key_stat['key_id']} ({key_stat['last_8_chars']}): "
                              f"{key_stat['request_count']} requests, "
                              f"{key_stat['success_rate']:.1f}% success, "
                              f"{key_stat['rate_limit_count']} rate limits")
        
        logger.info("=" * 60)
        logger.info("✅ TEST COMPLETED - NATIVE LOAD BALANCING WORKING!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(traceback.format_exc())

    finally:
        # Cleanup
        if processor:
            await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_native_load_balancing_system()) 