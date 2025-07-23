#!/usr/bin/env python3
"""
Sentiment Analysis Service - Integrated into NewsHead Pipeline
Analyzes articles before database insertion using Claude API
"""

import asyncio
import logging
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
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

class SentimentService:
    """
    Sentiment analysis service for real-time article analysis
    Designed to be integrated into the news pipeline
    Uses Claude API for sentiment analysis
    """
    
    def __init__(self, claude_api_key: str = None):
        # Claude API configuration
        self.claude_endpoint = "https://api.anthropic.com/v1/messages"
        self.api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-3-5-sonnet-20240620"  # Updated model with higher rate limits
        
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=20)  # Increased from 7 to 20 for better concurrency
        
        # Sentiment cache to avoid re-analyzing identical content
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        
        # Country cache to avoid repeated database queries
        self.country_cache: Dict[str, str] = {}
        
        # Stats tracking
        self.stats = {
            'total_analyzed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'start_time': time.time()
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
        """Initialize the sentiment service"""
        try:
            # Check API key
            if not self.api_key:
                logger.error("❌ ANTHROPIC_API_KEY not found in environment variables")
                logger.error("❌ Please add ANTHROPIC_API_KEY=your_key_here to your .env file")
                return False
            
            # Create aiohttp session for async requests with generous timeouts for Claude API
            timeout = aiohttp.ClientTimeout(total=180, connect=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=20,  # Increased from 7 to match max_workers
                    ttl_dns_cache=300
                ),
                headers={
                    'anthropic-version': '2023-06-01',
                    'x-api-key': self.api_key,
                    'content-type': 'application/json'
                }
            )
            
            # Test connection
            is_connected = await self.test_connection()
            if not is_connected:
                logger.error("❌ Failed to connect to Claude API - sentiment analysis will be disabled")
                return False
            
            logger.info("✅ Claude API Sentiment Analysis Service initialized successfully")
            logger.info(f"🤖 Using model: {self.model}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing sentiment service: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to Claude API"""
        try:
            logger.info("🔍 Testing Claude API connection...")
            
            test_payload = {
                "model": self.model,
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": "Respond with just: {'status': 'connected'}"
                    }
                ]
            }
            
            async with self.session.post(
                self.claude_endpoint,
                json=test_payload
            ) as response:
                
                if response.status == 200:
                    response_data = await response.json()
                    content = response_data.get('content', [{}])[0].get('text', '')
                    logger.info(f"✅ Claude API connection successful - Response: {content}")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Claude API connection failed: {response.status} - {response_text}")
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
        Analyze sentiment of a single article
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
            
            # Analyze with Claude API
            start_time = time.time()
            analysis_result = await self.query_claude_api_async(prompt)
            analysis_time = time.time() - start_time
            
            if analysis_result and 'error' not in analysis_result:
                self.stats['successful_analyses'] += 1
                
                # Add timing information
                analysis_result['analysis_time_ms'] = int(analysis_time * 1000)
                analysis_result['analyzed_at'] = datetime.now()
                
                # Cache the result
                if content_hash:
                    self.sentiment_cache[content_hash] = analysis_result
                
                logger.info(f"✅ SENTIMENT ANALYSIS: {article.get('ticker', 'UNKNOWN')} - {analysis_result.get('recommendation', 'UNKNOWN')} "
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
    
    async def query_claude_api_async(self, prompt: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """Send async request to Claude API with rate limit handling"""
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "max_tokens": 300,
                    "temperature": 0.0,  # Consistent results
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"
                        }
                    ]
                }
                
                async with self.session.post(self.claude_endpoint, json=payload) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # Extract content from Claude response
                        if response_data.get("content") and len(response_data["content"]) > 0:
                            content = response_data["content"][0]["text"]
                            
                            # Check for empty content
                            if not content or content.strip() == "":
                                logger.warning(f"⚠️ Claude API returned empty content!")
                                return {"error": "Empty response from Claude API"}
                            
                            # Clean up JSON if wrapped in markdown
                            content = self.clean_json_from_markdown(content)
                            
                            # Parse JSON
                            try:
                                parsed_result = json.loads(content)
                                return parsed_result
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON parsing failed: {e}")
                                logger.error(f"❌ Raw content: {repr(content)}")
                                return {"error": f"JSON parsing failed: {str(e)}", "raw_response": content}
                        else:
                            logger.error(f"❌ No content in Claude response!")
                            return {"error": "No content in response"}
                    
                    elif response.status == 429:
                        # Rate limit error - retry with exponential backoff
                        if attempt < max_retries:
                            wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                            logger.warning(f"⚠️ Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            response_text = await response.text()
                            logger.error(f"❌ Rate limit exceeded after {max_retries + 1} attempts")
                            return {"error": f"Rate limit exceeded: {response_text}"}
                    
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Claude API HTTP {response.status} error: {response_text}")
                        return {"error": f"HTTP {response.status}: {response_text}"}
                        
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 1  # 1s, 2s, 4s
                    logger.warning(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Request failed after {max_retries + 1} attempts: {e}")
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def clean_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks"""
        # Remove markdown code blocks
        if '```json' in content:
            # Extract content between ```json and ```
            start = content.find('```json') + 7
            end = content.find('```', start)
            if end != -1:
                return content[start:end].strip()
        elif '```' in content:
            # Extract content between ``` and ```
            start = content.find('```') + 3
            end = content.find('```', start)
            if end != -1:
                return content[start:end].strip()
        
        return content.strip()
    
    async def analyze_batch_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of articles
        Returns articles with sentiment data added
        """
        if not articles:
            return articles
        
        logger.info(f"🧠 SENTIMENT ANALYSIS (CLAUDE API): Processing batch of {len(articles)} articles")
        
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
            
            logger.info(f"✅ SENTIMENT ANALYSIS COMPLETE: {successful_analyses}/{len(articles)} successful analyses")
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
        """Get sentiment analysis statistics"""
        runtime = time.time() - self.stats['start_time']
        success_rate = (self.stats['successful_analyses'] / self.stats['total_analyzed'] * 100) if self.stats['total_analyzed'] > 0 else 0
        
        return {
            'runtime_seconds': runtime,
            'total_analyzed': self.stats['total_analyzed'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'cache_hits': self.stats['cache_hits'],
            'success_rate': success_rate,
            'cache_size': len(self.sentiment_cache)
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
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

async def analyze_articles_with_sentiment(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to analyze articles with sentiment
    This is the main integration point for the news pipeline
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
        
        # Test prompt generation to see country-specific logic
        print("=== TESTING COUNTRY-SPECIFIC SENTIMENT PROMPTS ===")
        
        for article in test_articles:
            ticker = article['ticker']
            
            # Manually set country for testing (simulate database lookup)
            if ticker == 'AAPL':
                service.country_cache[ticker] = 'USA'
            else:
                service.country_cache[ticker] = 'Taiwan'
            
            print(f"\n🔍 Testing ticker: {ticker}")
            prompt = await service.create_sentiment_prompt(article)
            
            # Check if Bitcoin consideration is included
            has_bitcoin_consideration = "Bitcoin/crypto news often has significant immediate market impact" in prompt
            country = service.country_cache.get(ticker, 'UNKNOWN')
            
            print(f"📍 Country: {country}")
            print(f"🪙 Bitcoin consideration included: {has_bitcoin_consideration}")
            
            if country == 'USA' and has_bitcoin_consideration:
                print("✅ CORRECT: USA ticker has Bitcoin consideration")
            elif country != 'USA' and not has_bitcoin_consideration:
                print("✅ CORRECT: Non-USA ticker does NOT have Bitcoin consideration")
            else:
                print("❌ ERROR: Bitcoin consideration logic is incorrect")
            
            print("-" * 60)
        
        print("\n🧪 Test completed - Bitcoin consideration now only applies to USA tickers!")
        
        await service.cleanup()
    
    asyncio.run(test_sentiment_service()) 