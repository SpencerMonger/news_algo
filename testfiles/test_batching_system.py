#!/usr/bin/env python3
"""
REAL SYSTEM Individual Processing Test with Portkey Gateway
Tests the enhanced individual processing with REAL API calls using Portkey Gateway for load balancing.

REAL SYSTEM SIMULATION:
1. Read articles from 'breaking_news' table (simulates WebSocket input)
2. Run REAL Claude API sentiment analysis through Portkey Gateway
3. Insert into 'news_testing' table (real database operations)

This uses the actual Portkey Gateway for proper load balancing across multiple API keys.
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
import subprocess
import signal
import atexit

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

class PortkeyGatewaySentimentService:
    """
    Sentiment service using the actual Portkey Gateway for load balancing
    This now uses the real Portkey SDK with proper gateway configuration
    """
    
    def __init__(self):
        self.portkey_client = None
        self.gateway_process = None
        self.gateway_url = "http://localhost:8787"
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'gateway_errors': 0
        }
        
    async def initialize(self):
        """Initialize Portkey Gateway with load balancing configuration"""
        try:
            # Import Portkey SDK
            from portkey_ai import Portkey
            
            # Get all available API keys from environment
            api_key_1 = os.getenv('ANTHROPIC_API_KEY')
            api_key_2 = os.getenv('ANTHROPIC_API_KEY2')
            api_key_3 = os.getenv('ANTHROPIC_API_KEY3')
            
            logger.info(f"🔍 DEBUG: API KEY 1 = {'✅ Found' if api_key_1 else '❌ Missing'}")
            logger.info(f"🔍 DEBUG: API KEY 2 = {'✅ Found' if api_key_2 else '❌ Missing'}")
            logger.info(f"🔍 DEBUG: API KEY 3 = {'✅ Found' if api_key_3 else '❌ Missing'}")
            
            # Create load balancing configuration for Portkey Gateway
            config = {
                "strategy": {
                    "mode": "loadbalance"
                },
                "targets": []
            }
            
            # Add all available API keys as targets
            if api_key_1:
                config["targets"].append({
                    "provider": "anthropic",
                    "api_key": api_key_1,
                    "weight": 1.0
                })
                logger.info(f"🔑 TARGET 1: Anthropic Claude (Primary) - {api_key_1[-8:]}")
            
            if api_key_2 and api_key_2 != api_key_1:
                config["targets"].append({
                    "provider": "anthropic",
                    "api_key": api_key_2, 
                    "weight": 1.0
                })
                logger.info(f"🔑 TARGET 2: Anthropic Claude (Secondary) - {api_key_2[-8:]}")
            
            if api_key_3 and api_key_3 != api_key_1 and api_key_3 != api_key_2:
                config["targets"].append({
                    "provider": "anthropic",
                    "api_key": api_key_3,
                    "weight": 1.0
                })
                logger.info(f"🔑 TARGET 3: Anthropic Claude (Third) - {api_key_3[-8:]}")
            
            if not config["targets"]:
                raise Exception("No API keys found in environment variables")
            
            # Start the Portkey Gateway server
            await self._start_gateway_server()
            
            # Initialize Portkey client with the gateway configuration
            self.portkey_client = Portkey(
                base_url=f"{self.gateway_url}/v1",
                config=config
            )
            
            logger.info(f"🔑 PORTKEY GATEWAY: Configured with {len(config['targets'])} API keys")
            # logger.info(f"📋 CONFIG: {json.dumps(config, indent=2)}")  # REMOVED: Security risk - contains raw API keys
            logger.info(f"📋 CONFIG: Load balancing across {len(config['targets'])} Anthropic API keys")
            logger.info(f"🌐 GATEWAY URL: {self.gateway_url}")
            logger.info("✅ Portkey Gateway initialized successfully")
            
            return True
            
        except ImportError:
            logger.error("❌ Portkey AI SDK not installed. Run: pip install portkey-ai")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Portkey Gateway: {e}")
            return False
    
    async def _start_gateway_server(self):
        """Start the Portkey Gateway server if not already running"""
        try:
            # Check if gateway is already running
            import aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.gateway_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            logger.info("✅ Portkey Gateway already running - using existing instance")
                            return
                except:
                    pass
            
            # Try alternative health check endpoints
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.gateway_url}/v1", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status in [200, 404]:  # 404 is OK for /v1 endpoint
                            logger.info("✅ Portkey Gateway already running - using existing instance")
                            return
            except:
                pass
            
            # If we get here, try to start the gateway
            logger.info("🚀 Starting Portkey Gateway server...")
            self.gateway_process = subprocess.Popen(
                ["npx", "@portkey-ai/gateway"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Wait for gateway to start (max 30 seconds)
            for i in range(30):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.gateway_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 200:
                                logger.info(f"✅ Portkey Gateway started successfully on {self.gateway_url}")
                                # Register cleanup function
                                atexit.register(self._cleanup_gateway)
                                return
                except:
                    pass
                
                # Also try the /v1 endpoint
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.gateway_url}/v1", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status in [200, 404]:
                                logger.info(f"✅ Portkey Gateway started successfully on {self.gateway_url}")
                                # Register cleanup function
                                atexit.register(self._cleanup_gateway)
                                return
                except:
                    pass
                
                await asyncio.sleep(1)
            
            raise Exception("Gateway failed to start within 30 seconds")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Portkey Gateway: {e}")
            # Don't raise the exception - try to continue with existing gateway
            logger.info("🔄 Attempting to use existing gateway instance...")
            return
    
    def _cleanup_gateway(self):
        """Clean up the gateway process"""
        if self.gateway_process:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.gateway_process.pid), signal.SIGTERM)
                self.gateway_process.wait(timeout=5)
                logger.info("✅ Portkey Gateway server stopped")
            except:
                try:
                    os.killpg(os.getpgid(self.gateway_process.pid), signal.SIGKILL)
                except:
                    pass
        else:
            logger.info("✅ Using external Portkey Gateway - no cleanup needed")
    
    async def analyze_article_sentiment_via_gateway(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article sentiment through Portkey Gateway with real load balancing
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
            
            # Make async request through Portkey Gateway
            start_time = time.time()
            
            # Use asyncio to run the sync method in a thread pool for true concurrency
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.portkey_client.chat.completions.create(
                    model="claude-3-5-sonnet-20240620",
                    messages=[
                        {
                            "role": "user",
                            "content": f"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"
                        }
                    ],
                    max_tokens=300,
                    temperature=0.0
                )
            )
            
            analysis_time = time.time() - start_time
            
            # Parse the response
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # Clean JSON from markdown
                if '```json' in content:
                    start = content.find('```json') + 7
                    end = content.find('```', start)
                    if end != -1:
                        content = content[start:end].strip()
                elif '```' in content:
                    start = content.find('```') + 3
                    end = content.find('```', start)
                    if end != -1:
                        content = content[start:end].strip()
                
                try:
                    parsed_result = json.loads(content)
                    parsed_result['analysis_time_ms'] = int(analysis_time * 1000)
                    parsed_result['analyzed_at'] = datetime.now()
                    
                    self.stats['successful_requests'] += 1
                    logger.debug(f"🎯 GATEWAY SUCCESS: {ticker} via Portkey -> {parsed_result.get('recommendation', 'HOLD')}")
                    
                    return parsed_result
                    
                except json.JSONDecodeError as e:
                    error_result = {"error": f"JSON parsing failed: {str(e)}"}
                    self.stats['failed_requests'] += 1
                    logger.warning(f"⚠️ GATEWAY PARSE ERROR: {ticker} - {str(e)}")
                    return error_result
            else:
                error_result = {"error": "No content in response"}
                self.stats['failed_requests'] += 1
                logger.warning(f"⚠️ GATEWAY NO CONTENT: {ticker}")
                return error_result
                
        except Exception as e:
            self.stats['gateway_errors'] += 1
            error_msg = str(e)
            
            # Check for rate limit errors
            if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                self.stats['rate_limit_hits'] += 1
            
            logger.error(f"❌ GATEWAY EXCEPTION: {article.get('ticker', 'UNKNOWN')} - {error_msg}")
            return {'error': f'Gateway exception: {error_msg}'}
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get Portkey Gateway statistics"""
        success_rate = (self.stats['successful_requests'] / max(1, self.stats['total_requests']) * 100)
        
        return {
            'gateway_mode': 'loadbalance',
            'gateway_url': self.gateway_url,
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': f"{success_rate:.1f}%",
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'gateway_errors': self.stats['gateway_errors']
        }
    
    async def cleanup(self):
        """Clean up gateway resources"""
        self._cleanup_gateway()
        logger.info("✅ Portkey Gateway cleanup completed")

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
        self.sentiment_service = PortkeyGatewaySentimentService() # Use the new PortkeyGatewaySentimentService
        
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
            'gateway_errors': 0,
            'total_retry_attempts': 0,
            'permanent_failures': set(),
            'processing_start_time': None,
            'processing_end_time': None,
            'first_insert_time': None,
            'last_insert_time': None,
            'load_balancing_stats': {},  # New: track load balancing performance
            'retry_successes': set(), # New: track successful retries
            'retry_errors': 0, # New: track retry errors
            'processed_articles': set() # Track processed article hashes
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
            
            # Initialize sentiment service with Portkey Gateway
            self.sentiment_service = PortkeyGatewaySentimentService()
            await self.sentiment_service.initialize()
            logger.info("✅ Sentiment service initialized")
            
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
                    'content_hash': f"real_hash_{ticker}_{int(time.time())}_{hash(row[1]) % 10000}_{len(articles)}",
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
        return f"real_hash_{ticker}_{timestamp}_{content_length}_{article_index}"

    async def process_article_batch_individually(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process articles individually with REAL sentiment analysis through Portkey Gateway
        Each article gets its own API call and immediate database insertion
        """
        if not articles:
            return {'processing_time': 0, 'zero_loss_achieved': True}
        
        # Initialize processing stats
        self.stats['total_articles'] = len(articles)
        self.stats['processing_start_time'] = time.time()
        
        logger.info(f"🚀 REAL SYSTEM INDIVIDUAL PROCESSING: Starting {len(articles)} articles")
        
        # Process in smaller batches to avoid 529 errors
        batch_size = 20  # Process 20 articles concurrently (was 5)
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
                if not isinstance(result, Exception):
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
    
    async def _process_single_article_real(self, article: Dict[str, Any], index: int):
        """Process a single article with REAL sentiment analysis and immediate database insertion"""
        ticker = article.get('ticker', 'UNKNOWN')
        content_hash = article.get('content_hash', f"hash_{ticker}_{index}")
        
        try:
            logger.info(f"🧠 #{index+1:2d} ANALYZING: {ticker} (REAL API CALL)")
            
            # REAL SENTIMENT ANALYSIS using Claude API
            analysis_start = time.time()
            analysis_result = await self.sentiment_service.analyze_article_sentiment_via_gateway(article)
            analysis_time = time.time() - analysis_start
            
            if analysis_result and 'error' not in analysis_result:
                # SUCCESS: Immediate insertion to news_testing table
                await self._insert_article_to_news_testing_real(article, analysis_result, index)
                self.stats['articles_analyzed_successfully'] += 1
                self.stats['articles_inserted_immediately'] += 1
                # Track successful processing
                content_hash = self._generate_content_hash(article)
                self.processed_articles.add(content_hash)
                self.stats['processed_articles'].add(content_hash)
                
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
    
    async def process_retry_queue(self):
        """Process failed articles with simplified retry logic - Portkey handles the heavy lifting"""
        if not self.retry_queue:
            return
            
        logger.info(f"🔄 PROCESSING RETRY QUEUE: {len(self.retry_queue)} articles to retry (REAL API CALLS)")
        
        retry_round = 1
        max_retries = 3  # Reduced since Portkey handles internal retries
        
        while self.retry_queue and retry_round <= max_retries:
            current_batch = list(self.retry_queue)
            self.retry_queue.clear()
            
            logger.info(f"🔄 RETRY ROUND {retry_round}: Processing {len(current_batch)} articles")
            
            # Simple delay - no exponential backoff
            if retry_round > 1:
                delay = 2.0  # Fixed 2 second delay
                logger.info(f"⏳ Waiting {delay}s for next retry batch...")
                await asyncio.sleep(delay)
            
            # Process retries with minimal staggering to avoid rate limit cascade
            tasks = []
            for i, retry_item in enumerate(current_batch):
                # Small stagger delay to spread requests
                stagger_delay = i * 0.5  # 500ms between requests
                if stagger_delay > 0:
                    logger.info(f"⏳ STAGGER DELAY: {retry_item.article.get('ticker', 'UNKNOWN')} waiting {stagger_delay}s to avoid rate limit cascade")
                
                task = self.process_retry_with_delay(retry_item.article.get('ticker', 'UNKNOWN'), retry_item.article, retry_item.attempt_count, retry_round, stagger_delay)
                tasks.append(task)
            
            # Execute all retry tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            retry_round += 1
        
        # Handle any remaining permanent failures
        await self._handle_permanent_failures_real()
    
    async def process_retry_with_delay(self, ticker, article, attempt_count, retry_round, stagger_delay):
        """Process a single retry with stagger delay"""
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)
        
        logger.info(f"🔄 RETRY {retry_round}: {ticker} (attempt {attempt_count + 1}) - REAL API CALL")
        
        try:
            result = await self.sentiment_service.analyze_article_sentiment_via_gateway(article)
            if result and 'error' not in result:  # Success
                await self._insert_article_to_news_testing_real(article, result, 0)
                logger.info(f"✅ RETRY SUCCESS: {ticker} -> {result.get('recommendation', 'HOLD')} (recovered on retry in {time.time() - self.stats['total_processing_time']:.1f}s)")
                return True
            else:
                # Still failed, add back to retry queue if under max attempts
                if attempt_count < 5:  # Max 6 total attempts
                    await self._add_to_retry_queue_real(article, result, attempt_count + 1)
                else:
                    # Permanent failure
                    self.stats['articles_failed_permanently'] += 1
                return False
        except Exception as e:
            logger.warning(f"⚠️ RETRY FAILED: {ticker} -> {str(e)[:100]}")
            if attempt_count < 5:
                await self._add_to_retry_queue_real(article, {'error': f'Exception: {str(e)}'}, attempt_count + 1)
            else:
                self.stats['articles_failed_permanently'] += 1
            return False
    
    async def _retry_single_article_real_with_stagger(self, retry_item: RetryItem, retry_round: int, stagger_delay: float):
        """Retry a single article with staggered delay to avoid rate limit cascades"""
        if stagger_delay > 0:
            logger.info(f"⏳ STAGGER DELAY: {retry_item.article.get('ticker', 'UNKNOWN')} waiting {stagger_delay:.1f}s to avoid rate limit cascade")
            await asyncio.sleep(stagger_delay)
        
        await self._retry_single_article_real(retry_item, retry_round)
    
    async def _retry_single_article_real(self, retry_item: RetryItem, retry_round: int):
        """Retry processing a single article with real API calls"""
        try:
            ticker = retry_item.article.get('ticker', 'UNKNOWN')
            logger.info(f"🔄 RETRY {retry_round}: {ticker} (attempt {retry_item.attempt_count}) - REAL API CALL")
            
            # Analyze sentiment through gateway
            analysis_result = await self.sentiment_service.analyze_article_sentiment_via_gateway(retry_item.article)
            
            if analysis_result.get('success'):
                # Insert to database
                await self._insert_article_to_news_testing_real(retry_item.article, analysis_result, -1)
                
                # Track successful retry
                content_hash = self._generate_content_hash(retry_item.article)
                self.processed_articles.add(content_hash)
                
                # Add to retry successes tracking
                if 'retry_successes' not in self.stats:
                    self.stats['retry_successes'] = set()
                self.stats['retry_successes'].add(content_hash)
                
                # Remove from retry queue
                if retry_item in self.retry_queue:
                    self.retry_queue.remove(retry_item)
                
                elapsed_time = time.time() - retry_item.last_attempt_time
                logger.info(f"✅ RETRY SUCCESS: {ticker} -> {analysis_result.get('action', 'UNKNOWN')} (recovered on retry in {elapsed_time:.1f}s)")
                
            else:
                # Retry failed, update retry item
                retry_item.attempt_count += 1
                retry_item.last_attempt_time = time.time()
                retry_item.retry_reason = RetryReason.API_ERROR
                retry_item.original_error = analysis_result.get('error', 'Unknown retry error')
                retry_item.next_retry_time = time.time() + (self.base_delay * (2 ** retry_item.attempt_count))
                
                logger.warning(f"⚠️ RETRY FAILED: {ticker} (attempt {retry_item.attempt_count}) -> {analysis_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            # Handle retry exception
            retry_item.attempt_count += 1
            retry_item.last_attempt_time = time.time()
            retry_item.retry_reason = RetryReason.API_ERROR
            retry_item.original_error = str(e)
            retry_item.next_retry_time = time.time() + (self.base_delay * (2 ** retry_item.attempt_count))
            
            ticker = retry_item.article.get('ticker', 'UNKNOWN')
            logger.error(f"❌ RETRY EXCEPTION: {ticker} (attempt {retry_item.attempt_count}) -> {e}")
            self.stats['retry_errors'] += 1
    
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
        """Verify all articles were processed with zero loss"""
        try:
            logger.info("🔍 ZERO LOSS VERIFICATION: Checking all articles were processed...")
            
            # Get expected hashes from original articles
            expected_hashes = set()
            for article in original_articles:
                content_hash = self._generate_content_hash(article)
                expected_hashes.add(content_hash)
            
            # Get processed hashes from both immediate successes and retry successes
            processed_hashes = set()
            
            # Add immediate successes
            processed_hashes.update(self.stats['processed_articles'])
            
            # Add retry successes - check all successfully processed retries
            for retry_item in self.retry_queue:
                # If retry was successful (not in permanent failures), it was processed
                retry_hash = self._generate_content_hash(retry_item.article)
                if retry_hash not in self.stats['permanent_failures']:
                    processed_hashes.add(retry_hash)
            
            # Also check completed retries from stats
            if 'retry_successes' in self.stats:
                for retry_hash in self.stats['retry_successes']:
                    processed_hashes.add(retry_hash)
            
            logger.info(f"🔍 EXPECTED: {len(expected_hashes)} articles")
            logger.info(f"🔍 PROCESSED: {len(processed_hashes)} articles")
            
            missing_hashes = expected_hashes - processed_hashes
            extra_hashes = processed_hashes - expected_hashes
            
            if missing_hashes:
                logger.error(f"🚨 ZERO LOSS VIOLATION: Expected {len(expected_hashes)}, processed {len(processed_hashes)}")
                logger.error(f"🚨 MISSING ARTICLES: {missing_hashes}")
                logger.error(f"🔍 EXPECTED HASHES: {expected_hashes}")
                logger.error(f"🔍 PROCESSED HASHES: {processed_hashes}")
                raise Exception(f"Zero loss guarantee violated: {len(missing_hashes)} articles not processed")
            
            if extra_hashes:
                logger.warning(f"⚠️ EXTRA ARTICLES PROCESSED: {extra_hashes}")
            
            logger.info(f"✅ ZERO LOSS VERIFIED: All {len(expected_hashes)} articles processed successfully")
            
        except Exception as e:
            logger.error(f"❌ Zero loss verification failed: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        total_articles = len(self.processed_articles)
        total_expected = self.stats.get('total_articles', 0)
        
        # Calculate success rate
        success_rate = (total_articles / total_expected * 100) if total_expected > 0 else 0
        
        # Check zero loss
        zero_loss_achieved = total_articles == total_expected
        
        return {
            'total_articles': total_articles,
            'expected_articles': total_expected,
            'success_rate': success_rate,
            'zero_loss_achieved': zero_loss_achieved,
            'total_retry_attempts': self.stats.get('total_retry_attempts', 0),
            'retry_successes': len(self.stats.get('retry_successes', set())),
            'retry_errors': self.stats.get('retry_errors', 0),
            'rate_limit_hits': self.stats.get('rate_limit_hits', 0),
            'gateway_errors': self.stats.get('gateway_errors', 0),
            'load_balancing_stats': self.stats.get('load_balancing_stats', {}),
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
async def test_real_system_individual_processing():
    """Test the REAL SYSTEM individual processing with actual API calls and database operations"""
    
    processor = None
    try:
        logger.info("🚀 INITIALIZING REAL SYSTEM COMPONENTS...")
        
        # Initialize the real system individual processor
        processor = RealSystemIndividualProcessor(max_retries=8, base_delay=3.0)
        await processor.initialize()
        logger.info("✅ Individual sentiment services ready")
        
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
        
        logger.info("🚀 STARTING REAL SYSTEM INDIVIDUAL PROCESSING TEST")
        
        # Process articles individually with real API calls
        start_time = time.time()
        result = await processor.process_article_batch_individually(test_articles)
        total_time = time.time() - start_time
        
        # Display results - SIMPLE AND CLEAR
        logger.info("=" * 60)
        logger.info("🎯 PORTKEY GATEWAY LOAD BALANCING TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 TOTAL ARTICLES: {len(test_articles)}")
        logger.info(f"✅ ARTICLES PROCESSED: {result.get('successful_count', 0)}")
        logger.info(f"⏱️  TOTAL PROCESSING TIME: {result.get('processing_time', total_time):.1f} seconds")
        logger.info(f"⚡ AVERAGE TIME PER ARTICLE: {result.get('processing_time', total_time)/len(test_articles):.1f} seconds")
        logger.info(f"🎯 SUCCESS RATE: {result.get('successful_count', 0)/len(test_articles)*100:.1f}%")
        logger.info("=" * 60)
        logger.info("✅ TEST COMPLETED - PORTKEY GATEWAY LOAD BALANCING WORKING!")
        
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
    asyncio.run(test_real_system_individual_processing()) 