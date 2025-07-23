#!/usr/bin/env python3
"""
Historical Sentiment Analysis for Backtesting
Analyzes sentiment of historical news articles scraped from Finviz
Uses Claude API for sentiment analysis with country-specific considerations
"""

import asyncio
import aiohttp
import logging
import json
import time
import os
import sys
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
from bs4 import BeautifulSoup
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalSentimentAnalyzer:
    """
    Historical sentiment analysis service for backtesting
    Analyzes sentiment of scraped news articles using Claude API
    """
    
    def __init__(self):
        # Claude API configuration
        self.claude_endpoint = "https://api.anthropic.com/v1/messages"
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-3-5-sonnet-20240620"
        
        self.ch_manager = None
        self.session = None
        
        # Country cache for ticker country lookups
        self.country_cache: Dict[str, str] = {}
        
        # Content scraping cache to avoid re-scraping
        self.content_cache: Dict[str, str] = {}
        
        # Stats tracking
        self.stats = {
            'articles_processed': 0,
            'articles_analyzed': 0,
            'articles_failed': 0,
            'content_scraped': 0,
            'cache_hits': 0,
            'start_time': time.time()
        }

    async def initialize(self):
        """Initialize the sentiment analyzer"""
        try:
            # Check API key
            if not self.api_key:
                logger.error("❌ ANTHROPIC_API_KEY not found in environment variables")
                return False
            
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create aiohttp session for async requests
            timeout = aiohttp.ClientTimeout(total=180, connect=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=20,
                    ttl_dns_cache=300
                ),
                headers={
                    'anthropic-version': '2023-06-01',
                    'x-api-key': self.api_key,
                    'content-type': 'application/json'
                }
            )
            
            # Test Claude API connection
            if not await self.test_claude_connection():
                logger.error("❌ Failed to connect to Claude API")
                return False
            
            logger.info("✅ Historical Sentiment Analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            return False

    async def test_claude_connection(self) -> bool:
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
                    logger.info(f"✅ Claude API connection successful")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Claude API connection failed: {response.status} - {response_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Claude API connection test failed: {e}")
            return False

    async def get_ticker_country(self, ticker: str) -> str:
        """Get the country for a ticker from the ticker_master_backtest table"""
        if ticker in self.country_cache:
            return self.country_cache[ticker]
        
        try:
            query = """
            SELECT country FROM News.ticker_master_backtest 
            WHERE ticker = %s
            LIMIT 1
            """
            
            result = self.ch_manager.client.query(query, [ticker])
            
            if result.result_rows:
                country = result.result_rows[0][0] or 'UNKNOWN'
            else:
                # Fallback to float_list table if not in master table
                fallback_query = """
                SELECT country FROM News.float_list 
                WHERE ticker = %s
                LIMIT 1
                """
                fallback_result = self.ch_manager.client.query(fallback_query, [ticker])
                
                if fallback_result.result_rows:
                    country = fallback_result.result_rows[0][0] or 'UNKNOWN'
                else:
                    country = 'UNKNOWN'
            
            # Cache the result
            self.country_cache[ticker] = country
            
            logger.debug(f"Ticker {ticker} country: {country}")
            return country
            
        except Exception as e:
            logger.warning(f"Error getting country for ticker {ticker}: {e}")
            self.country_cache[ticker] = 'UNKNOWN'
            return 'UNKNOWN'

    def scrape_finviz_article_content(self, url: str, max_chars: int = 6000) -> str:
        """Scrape article content from Finviz news URL"""
        try:
            # Check cache first
            if url in self.content_cache:
                self.stats['cache_hits'] += 1
                return self.content_cache[url]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            article_content = ""
            
            # Handle different URL types
            if 'finviz.com/news' in url:
                # Direct Finviz news page
                content_selectors = [
                    '.news-content',
                    '.article-content',
                    '.content',
                    'article',
                    '.news-body'
                ]
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0].get_text()
                        break
                
                # Fallback to main content area
                if not article_content:
                    # Look for the main content container
                    main_content = soup.find('td', {'class': 'fullview-news'})
                    if main_content:
                        article_content = main_content.get_text()
            
            elif 'globenewswire.com' in url:
                # GlobeNewswire content extraction
                content_selectors = [
                    '.article-body',
                    '.news-article-body',
                    '#article-body',
                    '.content'
                ]
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0].get_text()
                        break
                
                # Look for paragraph content
                if not article_content:
                    paragraphs = soup.find_all('p')
                    content_paragraphs = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 20:
                            content_paragraphs.append(text)
                    article_content = ' '.join(content_paragraphs)
            
            elif 'prnewswire.com' in url:
                # PRNewswire content extraction
                content_selectors = [
                    '.xn-newslines',
                    '.news-release-body',
                    '.release-body',
                    '.content'
                ]
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0].get_text()
                        break
            
            elif 'businesswire.com' in url:
                # BusinessWire content extraction
                content_selectors = [
                    '.bw-release-main',
                    '.release-body',
                    '.bw-release-body',
                    '.content'
                ]
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0].get_text()
                        break
            
            elif 'accesswire.com' in url:
                # Accesswire content extraction
                content_selectors = [
                    '.article-content',
                    '.press-release-content',
                    '.content',
                    '.release-body'
                ]
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0].get_text()
                        break
            
            # Final fallback - extract all text content
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
            
            # Cache the result
            self.content_cache[url] = clean_content
            self.stats['content_scraped'] += 1
            
            logger.debug(f"Scraped content: {len(clean_content)} characters from {url}")
            return clean_content
            
        except Exception as e:
            logger.error(f"Error scraping content from {url}: {e}")
            return ""

    async def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create a prompt for sentiment analysis with country-specific considerations"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        article_url = article.get('article_url', '')
        
        # Get country information for this ticker
        country = await self.get_ticker_country(ticker)
        
        # Scrape full content from URL
        content_to_analyze = ""
        if article_url:
            logger.debug(f"Scraping content from URL: {article_url}")
            scraped_content = self.scrape_finviz_article_content(article_url, max_chars=6000)
            if scraped_content:
                content_to_analyze = scraped_content
                logger.debug(f"Using scraped content: {len(content_to_analyze)} characters")
            else:
                content_to_analyze = headline
        else:
            content_to_analyze = headline
        
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

    async def query_claude_api_async(self, prompt: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """Send async request to Claude API with rate limit handling"""
        for attempt in range(max_retries + 1):
            try:
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
                
                async with self.session.post(self.claude_endpoint, json=payload) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if response_data.get("content") and len(response_data["content"]) > 0:
                            content = response_data["content"][0]["text"]
                            
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
                                return {"error": f"JSON parsing failed: {str(e)}", "raw_response": content}
                        else:
                            logger.error(f"❌ No content in Claude response!")
                            return {"error": "No content in response"}
                    
                    elif response.status == 429:
                        # Rate limit error - retry with exponential backoff
                        if attempt < max_retries:
                            wait_time = (2 ** attempt) * 2
                            logger.warning(f"⚠️ Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time}s...")
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
                    wait_time = (2 ** attempt) * 1
                    logger.warning(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Request failed after {max_retries + 1} attempts: {e}")
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}

    def clean_json_from_markdown(self, content: str) -> str:
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

    async def analyze_article_sentiment(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of a single article"""
        try:
            self.stats['articles_processed'] += 1
            
            # Create prompt
            prompt = await self.create_sentiment_prompt(article)
            
            # Analyze with Claude API
            start_time = time.time()
            analysis_result = await self.query_claude_api_async(prompt)
            analysis_time = time.time() - start_time
            
            if analysis_result and 'error' not in analysis_result:
                self.stats['articles_analyzed'] += 1
                
                # Add metadata
                analysis_result['analysis_time_ms'] = int(analysis_time * 1000)
                analysis_result['analyzed_at'] = datetime.now()
                analysis_result['content_hash'] = article.get('content_hash', '')
                analysis_result['country'] = await self.get_ticker_country(article.get('ticker', ''))
                
                logger.info(f"✅ SENTIMENT: {article.get('ticker', 'UNKNOWN')} - {analysis_result.get('recommendation', 'UNKNOWN')} "
                           f"({analysis_result.get('confidence', 'unknown')} confidence) in {analysis_time:.2f}s")
                
                return analysis_result
            else:
                self.stats['articles_failed'] += 1
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                logger.warning(f"❌ SENTIMENT FAILED: {article.get('ticker', 'UNKNOWN')} - {error_msg}")
                
                # Return default sentiment for failed analysis
                return {
                    'ticker': article.get('ticker', 'UNKNOWN'),
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Analysis failed: {error_msg}',
                    'analysis_time_ms': int(analysis_time * 1000),
                    'analyzed_at': datetime.now(),
                    'content_hash': article.get('content_hash', ''),
                    'country': await self.get_ticker_country(article.get('ticker', '')),
                    'error': error_msg
                }
                
        except Exception as e:
            self.stats['articles_failed'] += 1
            logger.error(f"❌ Error analyzing article sentiment: {e}")
            
            return {
                'ticker': article.get('ticker', 'UNKNOWN'),
                'sentiment': 'neutral',
                'recommendation': 'HOLD',
                'confidence': 'low',
                'explanation': f'Analysis error: {str(e)}',
                'analysis_time_ms': 0,
                'analyzed_at': datetime.now(),
                'content_hash': article.get('content_hash', ''),
                'country': 'UNKNOWN',
                'error': str(e)
            }

    async def get_articles_to_analyze(self, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Get articles from historical_news that need sentiment analysis"""
        try:
            # Get articles that don't have sentiment analysis yet
            query = """
            SELECT 
                hn.ticker, 
                hn.headline, 
                hn.article_url, 
                hn.published_utc,
                hn.content_hash
            FROM News.historical_news hn
            LEFT JOIN News.historical_sentiment hs 
                ON hn.content_hash = hs.content_hash
            WHERE hs.content_hash IS NULL
            AND hn.content_hash != ''
            ORDER BY hn.published_utc DESC
            LIMIT %s
            """
            
            result = self.ch_manager.client.query(query, [batch_size])
            
            articles = []
            for row in result.result_rows:
                ticker, headline, article_url, published_utc, content_hash = row
                articles.append({
                    'ticker': ticker,
                    'headline': headline,
                    'article_url': article_url,
                    'published_utc': published_utc,
                    'content_hash': content_hash
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles to analyze: {e}")
            return []

    async def store_sentiment_results(self, sentiment_results: List[Dict[str, Any]]):
        """Store sentiment analysis results in ClickHouse"""
        if not sentiment_results:
            return
        
        try:
            # Prepare data for insertion
            sentiment_data = []
            for result in sentiment_results:
                # Get original article data for the result
                original_article = None
                for article in self.current_batch:
                    if article.get('content_hash') == result.get('content_hash'):
                        original_article = article
                        break
                
                if not original_article:
                    continue
                
                sentiment_data.append((
                    result.get('ticker', 'UNKNOWN'),
                    original_article.get('headline', ''),
                    original_article.get('article_url', ''),
                    original_article.get('published_utc', datetime.now()),
                    result.get('sentiment', 'neutral'),
                    result.get('recommendation', 'HOLD'),
                    result.get('confidence', 'low'),
                    result.get('explanation', 'No explanation'),
                    result.get('analysis_time_ms', 0),
                    result.get('analyzed_at', datetime.now()),
                    result.get('content_hash', ''),
                    result.get('country', 'UNKNOWN')
                ))
            
            # Insert sentiment data
            self.ch_manager.client.insert(
                'News.historical_sentiment',
                sentiment_data,
                column_names=['ticker', 'headline', 'article_url', 'published_utc', 'sentiment', 'recommendation', 'confidence', 'explanation', 'analysis_time_ms', 'analyzed_at', 'content_hash', 'country']
            )
            
            logger.info(f"✅ Stored {len(sentiment_data)} sentiment analysis results")
            
        except Exception as e:
            logger.error(f"Error storing sentiment results: {e}")

    async def run_historical_sentiment_analysis(self, batch_size: int = 20):
        """Run the complete historical sentiment analysis process"""
        try:
            logger.info("🧠 Starting Historical Sentiment Analysis...")
            
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize sentiment analyzer")
                return False
            
            total_processed = 0
            batch_count = 0
            
            while True:
                batch_count += 1
                
                # Get next batch of articles to analyze
                logger.info(f"📊 BATCH {batch_count}: Getting articles to analyze...")
                articles = await self.get_articles_to_analyze(batch_size)
                
                if not articles:
                    logger.info("✅ No more articles to analyze")
                    break
                
                self.current_batch = articles  # Store for sentiment result processing
                logger.info(f"🧠 BATCH {batch_count}: Analyzing {len(articles)} articles...")
                
                # Process articles in parallel
                analysis_tasks = [self.analyze_article_sentiment(article) for article in articles]
                sentiment_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                # Filter out exceptions and prepare valid results
                valid_results = []
                for i, result in enumerate(sentiment_results):
                    if isinstance(result, Exception):
                        logger.error(f"Exception in sentiment analysis for article {i}: {result}")
                        continue
                    valid_results.append(result)
                
                # Store results in database
                if valid_results:
                    await self.store_sentiment_results(valid_results)
                
                total_processed += len(articles)
                
                # Progress logging
                logger.info(f"📈 BATCH {batch_count} COMPLETE: {len(valid_results)}/{len(articles)} successful analyses")
                logger.info(f"🔄 TOTAL PROGRESS: {total_processed} articles processed")
                
                # Rate limiting between batches
                await asyncio.sleep(2)
            
            # Final stats
            elapsed = time.time() - self.stats['start_time']
            logger.info("🎉 HISTORICAL SENTIMENT ANALYSIS COMPLETE!")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Articles processed: {self.stats['articles_processed']}")
            logger.info(f"  • Articles analyzed: {self.stats['articles_analyzed']}")
            logger.info(f"  • Articles failed: {self.stats['articles_failed']}")
            logger.info(f"  • Content scraped: {self.stats['content_scraped']}")
            logger.info(f"  • Cache hits: {self.stats['cache_hits']}")
            logger.info(f"  • Time elapsed: {elapsed/60:.1f} minutes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in historical sentiment analysis: {e}")
            return False
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("✅ Historical sentiment analyzer cleanup completed")

async def main():
    """Main function"""
    analyzer = HistoricalSentimentAnalyzer()
    success = await analyzer.run_historical_sentiment_analysis()
    
    if success:
        print("\n✅ Historical sentiment analysis completed successfully!")
    else:
        print("\n❌ Historical sentiment analysis failed!")

if __name__ == "__main__":
    asyncio.run(main()) 