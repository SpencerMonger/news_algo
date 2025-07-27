#!/usr/bin/env python3
"""
Sentiment Analysis Service - Integrated into NewsHead Pipeline
Analyzes articles before database insertion using Claude API with native load balancing
"""

import asyncio
import logging
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIKeyInfo:
    """Information about an API key for load balancing"""
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
    Provides Portkey-like functionality without external dependencies
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
            'key_switches': 0
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
                
            # Additional keys (support up to 9 total keys)
            for i in range(2, 10):
                key = os.getenv(f'ANTHROPIC_API_KEY{i}')
                if key and key not in api_keys:  # Avoid duplicates
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
            
            # Create aiohttp session with generous timeouts for Claude API
            timeout = aiohttp.ClientTimeout(total=180, connect=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=50,
                    limit_per_host=20,
                    ttl_dns_cache=300
                ),
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

class SentimentService:
    """
    Sentiment analysis service for real-time article analysis
    Now includes native load balancing across multiple API keys
    Uses Claude API for sentiment analysis with automatic failover
    """
    
    def __init__(self, claude_api_key: str = None):
        # Native load balancer (replaces single key approach)
        self.load_balancer = NativeLoadBalancer()
        
        # Legacy support for single API key (backward compatibility)
        self.legacy_api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
        
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Sentiment cache to avoid re-analyzing identical content
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        
        # Country cache to avoid repeated database queries
        self.country_cache: Dict[str, str] = {}
        
        # Stats tracking (enhanced with load balancing stats)
        self.stats = {
            'total_analyzed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'start_time': time.time(),
            'load_balancing_enabled': False,
            'load_balancing_stats': {}
        }

    async def get_ticker_country(self, ticker: str) -> str:
        """
        Get the country for a ticker from the float_list table
        Returns 'UNKNOWN' if not found
        """
        # Check cache first
        if ticker in self.country_cache:
            return self.country_cache[ticker]
        
        try:
            # Import here to avoid circular imports
            from clickhouse_setup import ClickHouseManager
            
            ch_manager = ClickHouseManager()
            ch_manager.connect()
            
            query = """
            SELECT country FROM News.float_list 
            WHERE ticker = %s
            LIMIT 1
            """
            
            result = ch_manager.client.query(query, [ticker])
            
            if result.result_rows:
                country = result.result_rows[0][0] or 'UNKNOWN'
            else:
                country = 'UNKNOWN'
            
            # Cache the result
            self.country_cache[ticker] = country
            
            ch_manager.close()
            logger.debug(f"Ticker {ticker} country: {country}")
            return country
            
        except Exception as e:
            logger.warning(f"Error getting country for ticker {ticker}: {e}")
            # Cache unknown result to avoid repeated failures
            self.country_cache[ticker] = 'UNKNOWN'
            return 'UNKNOWN'

    async def get_batch_ticker_countries(self, tickers: List[str]) -> Dict[str, str]:
        """
        Get countries for multiple tickers in a single database query for efficiency
        """
        # Filter out tickers already in cache
        uncached_tickers = [t for t in tickers if t not in self.country_cache]
        
        if uncached_tickers:
            try:
                # Import here to avoid circular imports
                from clickhouse_setup import ClickHouseManager
                
                ch_manager = ClickHouseManager()
                ch_manager.connect()
                
                # Build query with placeholder for IN clause
                placeholders = ','.join(['%s'] * len(uncached_tickers))
                query = f"""
                SELECT ticker, country FROM News.float_list 
                WHERE ticker IN ({placeholders})
                """
                
                result = ch_manager.client.query(query, uncached_tickers)
                
                # Update cache with results
                for row in result.result_rows:
                    ticker, country = row[0], row[1] or 'UNKNOWN'
                    self.country_cache[ticker] = country
                
                # Set UNKNOWN for tickers not found in database
                for ticker in uncached_tickers:
                    if ticker not in self.country_cache:
                        self.country_cache[ticker] = 'UNKNOWN'
                
                ch_manager.close()
                logger.debug(f"Batch loaded countries for {len(result.result_rows)} tickers")
                
            except Exception as e:
                logger.warning(f"Error getting batch countries: {e}")
                # Set all uncached tickers to UNKNOWN
                for ticker in uncached_tickers:
                    self.country_cache[ticker] = 'UNKNOWN'
        
        # Return country mapping for requested tickers
        return {ticker: self.country_cache.get(ticker, 'UNKNOWN') for ticker in tickers}
    
    def scrape_article_content(self, url: str, max_chars: int = 6000) -> str:
        """Scrape article content from URL, limited to max_chars"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to extract article content specifically for Benzinga
            article_content = ""
            
            if 'benzinga.com' in url:
                # Target Benzinga's article paragraph structure
                article_paragraphs = soup.find_all('p')
                content_paragraphs = []
                
                for p in article_paragraphs:
                    # Check if paragraph is in article content container
                    if p.parent and p.parent.get('class'):
                        parent_classes = p.parent.get('class', [])
                        # Look for the specific classes that contain article content
                        if any('cAazyy' in str(cls) or 'dIYChw' in str(cls) for cls in parent_classes):
                            text = p.get_text().strip()
                            if len(text) > 20:  # Only substantial paragraphs
                                content_paragraphs.append(text)
                
                article_content = ' '.join(content_paragraphs)
            
            # Fallback to general content extraction if specific method fails
            if not article_content or len(article_content) < 100:
                # Try common article selectors
                selectors_to_try = [
                    'article',
                    '.article-content',
                    '.story-body',
                    '.post-content',
                    '.content',
                    '.article-body',
                    '[data-module="ArticleBody"]',
                    '.article-wrap',
                    '.entry-content'
                ]
                
                for selector in selectors_to_try:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0].get_text()
                        break
                
                # Final fallback to full page text (original method)
                if not article_content:
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    article_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Clean up the content
            lines = (line.strip() for line in article_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit to max_chars
            if len(clean_content) > max_chars:
                clean_content = clean_content[:max_chars]
                
            logger.info(f"Scraped content: {len(clean_content)} characters")
            return clean_content
            
        except Exception as e:
            logger.error(f"Error scraping content from {url}: {e}")
            return ""

    async def initialize(self):
        """Initialize the sentiment service with native load balancing"""
        try:
            # Try to initialize native load balancer first
            load_balancer_success = await self.load_balancer.initialize()
            
            if load_balancer_success:
                self.stats['load_balancing_enabled'] = True
                logger.info("✅ Sentiment Service initialized with NATIVE LOAD BALANCING")
                logger.info(f"🤖 Using model: {self.load_balancer.model}")
                logger.info(f"🔑 Load balancing across {len(self.load_balancer.api_keys)} API keys")
                return True
            else:
                # Fallback to legacy single-key mode
                if self.legacy_api_key:
                    logger.warning("⚠️ Load balancing failed, falling back to single API key mode")
                    logger.info("✅ Sentiment Service initialized with SINGLE API KEY")
                    logger.info(f"🤖 Using model: claude-3-5-sonnet-20240620")
                    return True
                else:
                    logger.error("❌ No API keys found in environment variables")
                    logger.error("❌ Please add ANTHROPIC_API_KEY=your_key_here to your .env file")
                    return False
            
        except Exception as e:
            logger.error(f"Error initializing sentiment service: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to Claude API"""
        try:
            logger.info("🔍 Testing Claude API connection...")
            
            if self.stats['load_balancing_enabled']:
                # Test using load balancer
                test_prompt = "Respond with just: {'status': 'connected'}"
                result = await self.load_balancer.make_claude_request(test_prompt)
                
                if result and 'error' not in result:
                    logger.info(f"✅ Claude API connection successful via load balancer")
                    return True
                else:
                    logger.error(f"❌ Claude API connection failed via load balancer: {result.get('error', 'Unknown error')}")
                    return False
            else:
                # Test using legacy single key mode
                # Create temporary session for testing
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    test_payload = {
                        "model": "claude-3-5-sonnet-20240620",
                        "max_tokens": 20,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Respond with just: {'status': 'connected'}"
                            }
                        ]
                    }
                    
                    headers = {
                        'anthropic-version': '2023-06-01',
                        'x-api-key': self.legacy_api_key,
                        'content-type': 'application/json'
                    }
                    
                    async with session.post(
                        "https://api.anthropic.com/v1/messages",
                        json=test_payload,
                        headers=headers
                    ) as response:
                        
                        if response.status == 200:
                            response_data = await response.json()
                            content = response_data.get('content', [{}])[0].get('text', '')
                            logger.info(f"✅ Claude API connection successful (single key) - Response: {content}")
                            return True
                        else:
                            response_text = await response.text()
                            logger.error(f"❌ Claude API connection failed (single key): {response.status} - {response_text}")
                            return False
                    
        except Exception as e:
            logger.error(f"❌ Claude API connection test failed: {e}")
            return False
    
    async def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create a prompt for sentiment analysis with country-specific considerations"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        full_content = article.get('full_content', '')
        article_url = article.get('article_url', '')
        
        # Get country information for this ticker
        country = await self.get_ticker_country(ticker)
        
        # Always scrape full content from URL if available
        if article_url:
            logger.info(f"Scraping full content from URL: {article_url}")
            scraped_content = self.scrape_article_content(article_url, max_chars=6000)
            if scraped_content:
                content_to_analyze = scraped_content
                logger.info(f"Using scraped content: {len(content_to_analyze)} characters")
            else:
                content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        else:
            content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        
        # Apply 6K character limit to prevent token overflow
        content_to_analyze = content_to_analyze[:6000] if content_to_analyze else f"{headline}\n\n{summary}"

        # Base prompt
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
5. Give a brief explanation (1-2 sentences)"""

        # Add Bitcoin/crypto consideration ONLY for USA tickers
        if country == 'USA':
            prompt += """

Special consideration: If the article discusses Bitcoin, cryptocurrency investments, or crypto-related business activities by the company, these should generally be viewed as high-confidence market movers. Bitcoin/crypto news often has significant immediate market impact on stock prices."""

        prompt += f"""

Respond in this exact JSON format:
{{
    "ticker": "{ticker}",
    "sentiment": "positive/negative/neutral",
    "recommendation": "BUY/SELL/HOLD",
    "confidence": "high/medium/low",
    "explanation": "Brief explanation of your reasoning"
}}

Important: Use exactly "BUY", "SELL", or "HOLD" for recommendation (not "NEUTRAL").
"""
        return prompt
    
    async def analyze_article_sentiment(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a single article using native load balancing or legacy mode
        Returns sentiment data to be added to the article
        """
        try:
            self.stats['total_analyzed'] += 1
            
            # Check cache first
            content_hash = article.get('content_hash', '')
            if content_hash and content_hash in self.sentiment_cache:
                self.stats['cache_hits'] += 1
                logger.debug(f"📋 Cache hit for sentiment analysis: {article.get('ticker', 'UNKNOWN')}")
                return self.sentiment_cache[content_hash]
            
            # Create prompt (now with country-specific logic)
            prompt = await self.create_sentiment_prompt(article)
            
            # Analyze with Claude API (load balanced or legacy)
            start_time = time.time()
            
            if self.stats['load_balancing_enabled']:
                # Use native load balancer
                analysis_result = await self.load_balancer.make_claude_request(prompt)
            else:
                # Use legacy single key approach
                analysis_result = await self.query_claude_api_legacy(prompt)
            
            analysis_time = time.time() - start_time
            
            if analysis_result and 'error' not in analysis_result:
                self.stats['successful_analyses'] += 1
                
                # Add timing information
                analysis_result['analysis_time_ms'] = int(analysis_time * 1000)
                analysis_result['analyzed_at'] = datetime.now()
                
                # Cache the result
                if content_hash:
                    self.sentiment_cache[content_hash] = analysis_result
                
                mode = "LOAD BALANCED" if self.stats['load_balancing_enabled'] else "SINGLE KEY"
                logger.info(f"✅ SENTIMENT ANALYSIS ({mode}): {article.get('ticker', 'UNKNOWN')} - {analysis_result.get('recommendation', 'UNKNOWN')} "
                           f"({analysis_result.get('confidence', 'unknown')} confidence) in {analysis_time:.2f}s")
                
                return analysis_result
            else:
                self.stats['failed_analyses'] += 1
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                logger.warning(f"❌ SENTIMENT ANALYSIS FAILED: {article.get('ticker', 'UNKNOWN')} - {error_msg}")
                
                # Return default sentiment for failed analysis
                return {
                    'ticker': article.get('ticker', 'UNKNOWN'),
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Analysis failed: {error_msg}',
                    'analysis_time_ms': int(analysis_time * 1000),
                    'analyzed_at': datetime.now(),
                    'error': error_msg
                }
                
        except Exception as e:
            self.stats['failed_analyses'] += 1
            logger.error(f"❌ Error analyzing article sentiment: {e}")
            
            # Return default sentiment for exceptions
            return {
                'ticker': article.get('ticker', 'UNKNOWN'),
                'sentiment': 'neutral',
                'recommendation': 'HOLD',
                'confidence': 'low',
                'explanation': f'Analysis error: {str(e)}',
                'analysis_time_ms': 0,
                'analyzed_at': datetime.now(),
                'error': str(e)
            }
    
    async def query_claude_api_legacy(self, prompt: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """Legacy single-key Claude API method (backward compatibility)"""
        if not self.load_balancer.session:
            # Create session if not exists
            timeout = aiohttp.ClientTimeout(total=180, connect=30)
            self.load_balancer.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=7, ttl_dns_cache=300),
                headers={
                    'anthropic-version': '2023-06-01',
                    'x-api-key': self.legacy_api_key,
                    'content-type': 'application/json'
                }
            )
        
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": "claude-3-5-sonnet-20240620",
                    "max_tokens": 300,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"
                        }
                    ]
                }
                
                async with self.load_balancer.session.post(
                    "https://api.anthropic.com/v1/messages", 
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if response_data.get("content") and len(response_data["content"]) > 0:
                            content = response_data["content"][0]["text"]
                            content = self.load_balancer._clean_json_from_markdown(content)
                            
                            try:
                                parsed_result = json.loads(content)
                                return parsed_result
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON parsing failed: {e}")
                                return {"error": f"JSON parsing failed: {str(e)}"}
                        else:
                            return {"error": "No content in response"}
                    
                    elif response.status == 429:
                        if attempt < max_retries:
                            wait_time = (2 ** attempt) * 2
                            logger.warning(f"⚠️ Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            response_text = await response.text()
                            return {"error": f"Rate limit exceeded: {response_text}"}
                    
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Claude API HTTP {response.status} error: {response_text}")
                        return {"error": f"HTTP {response.status}: {response_text}"}
                        
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 1
                    logger.warning(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Request failed after {max_retries + 1} attempts: {e}")
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    async def analyze_batch_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of articles with native load balancing
        Returns articles with sentiment data added
        """
        if not articles:
            return articles
        
        mode = "LOAD BALANCED" if self.stats['load_balancing_enabled'] else "SINGLE KEY"
        logger.info(f"🧠 SENTIMENT ANALYSIS ({mode}): Processing batch of {len(articles)} articles")
        
        # Pre-load country data for all tickers in batch for efficiency
        tickers = [article.get('ticker', '') for article in articles if article.get('ticker')]
        if tickers:
            await self.get_batch_ticker_countries(tickers)
            logger.debug(f"Pre-loaded country data for {len(tickers)} tickers")
        
        # Process articles in parallel (limited by ThreadPoolExecutor)
        analysis_tasks = [self.analyze_article_sentiment(article) for article in articles]
        
        try:
            # Wait for all analyses to complete
            sentiment_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Add sentiment data to articles
            enriched_articles = []
            successful_analyses = 0
            
            for i, (article, sentiment_result) in enumerate(zip(articles, sentiment_results)):
                try:
                    if isinstance(sentiment_result, Exception):
                        logger.error(f"Exception in sentiment analysis for article {i}: {sentiment_result}")
                        # Add default sentiment
                        sentiment_data = {
                            'sentiment': 'neutral',
                            'recommendation': 'HOLD',
                            'confidence': 'low',
                            'explanation': f'Analysis exception: {str(sentiment_result)}',
                            'analysis_time_ms': 0,
                            'analyzed_at': datetime.now(),
                            'error': str(sentiment_result)
                        }
                    else:
                        sentiment_data = sentiment_result
                        if 'error' not in sentiment_data:
                            successful_analyses += 1
                    
                    # Add sentiment fields to article
                    article.update({
                        'sentiment': sentiment_data.get('sentiment', 'neutral'),
                        'recommendation': sentiment_data.get('recommendation', 'HOLD'),
                        'confidence': sentiment_data.get('confidence', 'low'),
                        'explanation': sentiment_data.get('explanation', 'No explanation'),
                        'analysis_time_ms': sentiment_data.get('analysis_time_ms', 0),
                        'analyzed_at': sentiment_data.get('analyzed_at', datetime.now())
                    })
                    
                    enriched_articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error enriching article {i} with sentiment: {e}")
                    # Add article with default sentiment
                    article.update({
                        'sentiment': 'neutral',
                        'recommendation': 'HOLD',
                        'confidence': 'low',
                        'explanation': f'Enrichment error: {str(e)}',
                        'analysis_time_ms': 0,
                        'analyzed_at': datetime.now()
                    })
                    enriched_articles.append(article)
            
            logger.info(f"✅ SENTIMENT ANALYSIS COMPLETE ({mode}): {successful_analyses}/{len(articles)} successful analyses")
            return enriched_articles
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            # Return articles with default sentiment
            for article in articles:
                article.update({
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Batch analysis error: {str(e)}',
                    'analysis_time_ms': 0,
                    'analyzed_at': datetime.now()
                })
            return articles
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics including load balancing stats"""
        runtime = time.time() - self.stats['start_time']
        success_rate = (self.stats['successful_analyses'] / self.stats['total_analyzed'] * 100) if self.stats['total_analyzed'] > 0 else 0
        
        base_stats = {
            'runtime_seconds': runtime,
            'total_analyzed': self.stats['total_analyzed'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'cache_hits': self.stats['cache_hits'],
            'success_rate': success_rate,
            'cache_size': len(self.sentiment_cache),
            'load_balancing_enabled': self.stats['load_balancing_enabled']
        }
        
        # Add load balancing stats if enabled
        if self.stats['load_balancing_enabled']:
            base_stats['load_balancing_stats'] = self.load_balancer.get_load_balancing_stats()
        
        return base_stats
    
    def clear_cache(self):
        """Clear both sentiment and country caches for testing purposes"""
        sentiment_cache_size = len(self.sentiment_cache)
        country_cache_size = len(self.country_cache)
        
        self.sentiment_cache.clear()
        self.country_cache.clear()
        
        logger.info(f"🧹 CACHE CLEARED: {sentiment_cache_size} sentiment entries, {country_cache_size} country entries")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.load_balancer:
            await self.load_balancer.cleanup()
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("✅ Sentiment service cleanup completed")

# Global sentiment service instance
sentiment_service = None

async def get_sentiment_service() -> SentimentService:
    """Get or create global sentiment service instance"""
    global sentiment_service
    
    if sentiment_service is None:
        sentiment_service = SentimentService()
        await sentiment_service.initialize()
    
    return sentiment_service

async def clear_sentiment_cache():
    """Clear the global sentiment service cache for testing purposes"""
    global sentiment_service
    
    if sentiment_service is not None:
        sentiment_service.clear_cache()
    else:
        logger.info("🧹 CACHE CLEAR: No global sentiment service instance exists")

async def analyze_articles_with_sentiment(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to analyze articles with sentiment
    This is the main integration point for the news pipeline
    Now supports native load balancing automatically
    """
    if not articles:
        return articles
    
    try:
        service = await get_sentiment_service()
        return await service.analyze_batch_articles(articles)
    except Exception as e:
        logger.error(f"Error in sentiment analysis integration: {e}")
        # Return articles with default sentiment
        for article in articles:
            article.update({
                'sentiment': 'neutral',
                'recommendation': 'HOLD',
                'confidence': 'low',
                'explanation': f'Integration error: {str(e)}',
                'analysis_time_ms': 0,
                'analyzed_at': datetime.now()
            })
        return articles

if __name__ == "__main__":
    # Test the sentiment service
    async def test_sentiment_service():
        service = SentimentService()
        await service.initialize()
        
        # Test articles - simulate USA and non-USA tickers
        test_articles = [
            {
                'ticker': 'AAPL',  # Assuming USA ticker
                'headline': 'Apple Reports Strong Q4 Earnings, Announces Bitcoin Investment',
                'summary': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations and announced a major Bitcoin investment strategy.',
                'full_content': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue growth. The company also announced plans to invest $1 billion in Bitcoin.',
                'content_hash': 'test_hash_usa_123'
            },
            {
                'ticker': 'TSM',  # Assuming non-USA ticker (Taiwan)
                'headline': 'Taiwan Semiconductor Announces Bitcoin Mining Chip Development',
                'summary': 'Taiwan Semiconductor announced development of new chips optimized for Bitcoin mining.',
                'full_content': 'Taiwan Semiconductor announced development of new chips optimized for Bitcoin mining, targeting the growing cryptocurrency market.',
                'content_hash': 'test_hash_non_usa_456'
            }
        ]
        
        # Test load balancing functionality
        print("=== TESTING NATIVE LOAD BALANCING SENTIMENT SERVICE ===")
        
        # Show load balancing status
        stats = service.get_stats()
        if stats['load_balancing_enabled']:
            lb_stats = stats['load_balancing_stats']
            print(f"🔑 Load Balancing: ENABLED with {lb_stats['total_keys']} API keys")
            print(f"✅ Available Keys: {lb_stats['available_keys']}")
            print(f"🚫 Rate Limited Keys: {lb_stats['rate_limited_keys']}")
        else:
            print("🔑 Load Balancing: DISABLED (using single API key)")
        
        # Test batch analysis
        enriched_articles = await service.analyze_batch_articles(test_articles)
        
        print(f"\n📊 Analysis Results:")
        for article in enriched_articles:
            ticker = article['ticker']
            recommendation = article.get('recommendation', 'UNKNOWN')
            confidence = article.get('confidence', 'unknown')
            print(f"   {ticker}: {recommendation} ({confidence} confidence)")
        
        # Show final stats
        final_stats = service.get_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   Total Analyzed: {final_stats['total_analyzed']}")
        print(f"   Success Rate: {final_stats['success_rate']:.1f}%")
        print(f"   Cache Hits: {final_stats['cache_hits']}")
        
        if final_stats['load_balancing_enabled']:
            lb_stats = final_stats['load_balancing_stats']
            print(f"   Load Balancer Success Rate: {lb_stats['success_rate']:.1f}%")
            print(f"   Key Switches: {lb_stats['key_switches']}")
        
        print("\n✅ Test completed!")
        
        await service.cleanup()
    
    asyncio.run(test_sentiment_service()) 