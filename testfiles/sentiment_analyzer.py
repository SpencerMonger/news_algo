import asyncio
import logging
import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from clickhouse_setup import ClickHouseManager, setup_logging
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import aiohttp
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Initialize logging
logger = setup_logging()

class SentimentAnalyzer:
    """
    Standalone sentiment analysis tool that mirrors the live system logic
    Uses the same 4D timing and urgency prompt as the production system
    """
    
    def __init__(self):
        # Claude API configuration
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.claude_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.claude_endpoint = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20240620"
        self.session = None
        
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=180, connect=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=7,
                    ttl_dns_cache=300
                ),
                headers={
                    'anthropic-version': '2023-06-01',
                    'x-api-key': self.claude_api_key,
                    'content-type': 'application/json'
                }
            )
    
    def scrape_article_content(self, url: str, max_chars: int = 1500) -> str:
        """Scrape article content from URL, limited to max_chars - SYNCHRONOUS VERSION"""
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
            
            # Try to extract article content based on the URL domain
            article_content = ""
            
            if 'benzinga.com' in url:
                article_content = self._extract_benzinga_content(soup)
            elif 'businesswire.com' in url:
                article_content = self._extract_businesswire_content(soup)
            elif 'prnewswire.com' in url:
                article_content = self._extract_prnewswire_content(soup)
            elif 'finance.yahoo.com' in url:
                article_content = self._extract_yahoo_finance_content(soup)
            elif 'marketwatch.com' in url:
                article_content = self._extract_marketwatch_content(soup)
            else:
                article_content = self._extract_generic_content(soup)
            
            # Fallback to generic extraction if specific method fails
            if not article_content or len(article_content) < 100:
                article_content = self._extract_generic_content(soup)
            
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
    
    def _extract_benzinga_content(self, soup) -> str:
        """Extract content specifically from Benzinga articles"""
        article_paragraphs = soup.find_all('p')
        content_paragraphs = []
        
        for p in article_paragraphs:
            if p.parent and p.parent.get('class'):
                parent_classes = p.parent.get('class', [])
                if any('cAazyy' in str(cls) or 'dIYChw' in str(cls) for cls in parent_classes):
                    text = p.get_text().strip()
                    if len(text) > 20:
                        content_paragraphs.append(text)
        
        return ' '.join(content_paragraphs)
    
    def _extract_businesswire_content(self, soup) -> str:
        """Extract content specifically from BusinessWire articles"""
        # Try multiple BusinessWire-specific selectors
        selectors = [
            'div[data-module="ArticleBody"]',
            '.bw-release-main',
            '.bw-release-body',
            'div.bw-release-story',
            'div[id="releaseText"]',
            'div.release-body'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements[0].get_text()
        
        return ""
    
    def _extract_prnewswire_content(self, soup) -> str:
        """Extract content specifically from PR Newswire articles"""
        selectors = [
            'div[data-module="ArticleBody"]',
            '.release-body',
            'div.col-lg-10.col-md-10.col-sm-12.col-xs-12',
            'section.release-body',
            'div.row.release-body'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements[0].get_text()
        
        return ""
    
    def _extract_yahoo_finance_content(self, soup) -> str:
        """Extract content specifically from Yahoo Finance articles"""
        selectors = [
            'div[data-module="ArticleBody"]',
            '.caas-body',
            'div.caas-body',
            'div[data-module="FinanceArticleBody"]',
            '.finance-article-body'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements[0].get_text()
        
        return ""
    
    def _extract_marketwatch_content(self, soup) -> str:
        """Extract content specifically from MarketWatch articles"""
        selectors = [
            'div[data-module="ArticleBody"]',
            '.article__body',
            'div.article-body',
            'div.entry-content',
            'div.article-wrap'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements[0].get_text()
        
        return ""
    
    def _extract_generic_content(self, soup) -> str:
        """Generic content extraction for unknown sites"""
        # Try common article selectors
        selectors_to_try = [
            'article',
            'div[data-module="ArticleBody"]',
            '.article-content',
            '.story-body',
            '.post-content',
            '.content',
            '.article-body',
            '.article-wrap',
            '.entry-content',
            'main',
            '.main-content',
            '.content-body'
        ]
        
        for selector in selectors_to_try:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text()
                if len(content) > 200:  # Only use if substantial content
                    return content
        
        # Final fallback to full page text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)
    
    def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create the 4D timing and urgency prompt - EXACT COPY from live system"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        full_content = article.get('full_content', '')
        article_url = article.get('article_url', '')
        
        # Always scrape full content from URL if available
        if article_url:
            logger.info(f"Scraping full content from URL: {article_url}")
            scraped_content = self.scrape_article_content(article_url, max_chars=1500)
            if scraped_content:
                content_to_analyze = scraped_content
                logger.info(f"Using scraped content: {len(content_to_analyze)} characters")
            else:
                content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        else:
            content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        
        # Apply 1500 character limit to match 4D prompt strategy from live system
        content_to_analyze = content_to_analyze[:1500] if content_to_analyze else f"{headline}\n\n{summary}"

        # 4D Timing and Urgency Prompt - EXACT COPY from sentiment_service.py
        prompt = f"""Analyze this financial news for immediate market impact timing.

ARTICLE CONTENT:
{content_to_analyze}

TIMING ANALYSIS: Determine if this news will cause immediate explosive price action (hours/days) or delayed appreciation.

IMMEDIATE IMPACT CATALYSTS (BUY + high confidence):
- FDA approvals, merger announcements, major contract wins
- Earnings surprises with immediate market implications
- Breaking regulatory decisions or legal victories
- Emergency use authorizations or critical partnerships

DELAYED IMPACT NEWS (BUY + medium confidence):
- Product development milestones with future potential
- Strategic initiatives with 6-12 month timelines
- Market expansion plans requiring execution time
- Research results requiring further development

LOW IMPACT/SPECULATIVE (HOLD):
- Early-stage research or development updates
- Management commentary without concrete announcements
- Industry trend discussions without company-specific catalysts
- Vague future planning statements

Focus on: Will this move the stock price within 24-48 hours?

Respond with JSON:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Immediate impact timing assessment and catalyst urgency analysis"
}}"""
        return prompt
        
    async def query_claude(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send prompt to Claude API and get response"""
        # Ensure session is initialized
        await self._ensure_session()
        
        if not self.session:
            logger.error("Claude session not initialized. Cannot query API.")
            return {"error": "Claude session not initialized"}

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
                    
                    # Extract content from Claude response
                    if response_data.get("content") and len(response_data["content"]) > 0:
                        content = response_data["content"][0]["text"]
                        
                        # Clean up JSON if wrapped in markdown
                        if '```json' in content:
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif '```' in content:
                            content = content.split('```')[1].strip()
                        
                        # Try to parse JSON
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON response: {content}")
                            return {"error": "Invalid JSON response", "raw_response": content}
                    else:
                        logger.error(f"No content in Claude response!")
                        return {"error": "No content in response"}
                else:
                    response_text = await response.text()
                    logger.error(f"Claude API error: {response.status} - {response_text}")
                    return {"error": f"API error: {response.status}"}
                
        except aiohttp.ClientError as e:
            logger.error(f"Request to Claude failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error querying Claude: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_recent_articles(self, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent articles from ClickHouse"""
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
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            AND ticker != ''
            AND ticker != 'UNKNOWN'
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            
            result = self.ch_manager.client.query(query)
            
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
                    'published_utc': row[7]
                })
            
            logger.info(f"Retrieved {len(articles)} articles from the last {hours} hours")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            return []
    
    def analyze_articles(self, hours: int = 24, limit: int = 50):
        """Analyze recent articles and print results"""
        logger.info(f"Starting sentiment analysis for articles from the last {hours} hours")
        
        # Get articles from ClickHouse
        articles = self.get_recent_articles(hours, limit)
        
        if not articles:
            logger.info("No articles found to analyze")
            return
        
        print(f"\n{'='*80}")
        print(f"SENTIMENT ANALYSIS RESULTS - {len(articles)} Articles")
        print(f"{'='*80}")
        
        successful_analyses = 0
        failed_analyses = 0
        ticker_results = []  # Store results for summary
        
        for i, article in enumerate(articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')[:100] + "..." if len(article.get('headline', '')) > 100 else article.get('headline', '')
            timestamp = article.get('timestamp', '')
            
            print(f"\n{'-'*60}")
            print(f"Article {i}/{len(articles)}")
            print(f"Ticker: {ticker}")
            print(f"Time: {timestamp}")
            print(f"Headline: {headline}")
            print(f"Source: {article.get('source', 'Unknown')}")
            print(f"{'-'*60}")
            
            # Create prompt and analyze
            prompt = self.create_sentiment_prompt(article)
            
            # Run analysis in async context
            async def run_analysis():
                return await self.query_claude(prompt)
            
            analysis = self._run_async_safely(run_analysis())
            
            if analysis and 'error' not in analysis:
                successful_analyses += 1
                recommendation = analysis.get('recommendation', 'unknown')
                confidence = analysis.get('confidence', 'unknown')
                
                print(f"✅ ANALYSIS SUCCESSFUL:")
                print(f"   Recommendation: {recommendation}")
                print(f"   Confidence: {confidence}")
                print(f"   Reasoning: {analysis.get('reasoning', 'No reasoning provided')}")
                
                # Store result for summary
                ticker_results.append({
                    'ticker': ticker,
                    'recommendation': recommendation,
                    'confidence': confidence
                })
                
                # Log the analysis
                logger.info(f"ANALYSIS - {ticker}: {recommendation} "
                           f"({confidence} confidence)")
                
            else:
                failed_analyses += 1
                print(f"❌ ANALYSIS FAILED:")
                if analysis:
                    print(f"   Error: {analysis.get('error', 'Unknown error')}")
                    if 'raw_response' in analysis:
                        print(f"   Raw Response: {analysis['raw_response'][:200]}...")
                else:
                    print(f"   Error: No response from Claude")
                
                # Store failed result for summary
                ticker_results.append({
                    'ticker': ticker,
                    'recommendation': 'FAILED',
                    'confidence': 'N/A'
                })
                
                logger.warning(f"ANALYSIS FAILED - {ticker}: {analysis.get('error', 'Unknown') if analysis else 'No response'}")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total Articles: {len(articles)}")
        print(f"Successful Analyses: {successful_analyses}")
        print(f"Failed Analyses: {failed_analyses}")
        print(f"Success Rate: {(successful_analyses/len(articles)*100):.1f}%")
        
        # Add ticker sentiment summary
        if ticker_results:
            print(f"\n{'='*80}")
            print(f"TICKER SENTIMENT SUMMARY")
            print(f"{'='*80}")
            
            # Sort results alphabetically by ticker
            sorted_results = sorted(ticker_results, key=lambda x: x['ticker'])
            
            for result in sorted_results:
                ticker = result['ticker']
                recommendation = result['recommendation']
                confidence = result['confidence']
                
                if recommendation == 'FAILED':
                    print(f"{ticker} - FAILED")
                else:
                    # Format recommendation with confidence indicator
                    confidence_indicator = {
                        'high': '🔥',
                        'medium': '📊',
                        'low': '❓'
                    }.get(confidence.lower(), '')
                    
                    print(f"{ticker} - {recommendation} {confidence_indicator}")
        
        logger.info(f"Sentiment analysis completed: {successful_analyses}/{len(articles)} successful")
    
    def test_deterministic_analysis(self, test_runs: int = 5):
        """Test if sentiment analysis is deterministic by running the same prompt multiple times"""
        print(f"Testing deterministic behavior with {test_runs} runs...")
        
        # Create a test article
        test_article = {
            'ticker': 'TEST',
            'headline': 'Test Company Reports Strong Q4 Earnings',
            'summary': 'Test Company exceeded expectations with strong revenue growth.',
            'full_content': 'Test Company reported quarterly earnings that exceeded analyst expectations, driven by strong revenue growth and improved margins.',
            'article_url': ''  # No URL to avoid scraping variability
        }
        
        # Run analysis multiple times
        results = []
        for i in range(test_runs):
            print(f"Run {i+1}/{test_runs}...", end=" ")
            prompt = self.create_sentiment_prompt(test_article)
            
            # Run analysis in async context
            async def run_analysis():
                return await self.query_claude(prompt)
            
            analysis = self._run_async_safely(run_analysis())
            
            if analysis and 'error' not in analysis:
                result = {
                    'recommendation': analysis.get('recommendation', 'unknown'),
                    'confidence': analysis.get('confidence', 'unknown')
                }
                results.append(result)
                print(f"✅ {result['recommendation']} ({result['confidence']})")
            else:
                print("❌ FAILED")
                results.append({'error': 'failed'})
        
        # Check for consistency
        if results:
            first_result = results[0]
            all_same = all(r == first_result for r in results if 'error' not in r)
            
            print(f"\n{'='*60}")
            print(f"DETERMINISTIC TEST RESULTS")
            print(f"{'='*60}")
            print(f"Total runs: {test_runs}")
            print(f"Successful runs: {len([r for r in results if 'error' not in r])}")
            print(f"All results identical: {'✅ YES' if all_same else '❌ NO'}")
            
            if not all_same:
                print(f"\nVariations found:")
                unique_results = list({str(r): r for r in results if 'error' not in r}.values())
                for i, result in enumerate(unique_results, 1):
                    count = sum(1 for r in results if r == result)
                    print(f"  {i}. {result} (appeared {count} times)")
        
        return results
    
    def _run_async_safely(self, coro):
        """Safely run an async coroutine, handling event loop conflicts"""
        try:
            # Try to get the existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                # If no loop is running, use asyncio.run
                return asyncio.run(coro)
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(coro)

    async def test_claude_connection(self):
        """Test connection to Claude API - ASYNC VERSION"""
        print("Testing Claude API connection...")
        
        test_prompt = "Hello, please respond with a simple JSON object containing 'status': 'connected'"
        
        try:
            response = await self.query_claude(test_prompt)
            
            if response and 'error' not in response:
                print("✅ Claude API connection successful!")
                print(f"Response: {response}")
                return True
            else:
                print(f"❌ Claude API connection failed: {response.get('error', 'Unknown error') if response else 'No response'}")
                return False
                
        except Exception as e:
            print(f"❌ Claude API connection failed: {e}")
            return False
    
    def close(self):
        """Close connections and cleanup resources"""
        try:
            if self.session:
                # Schedule session closure in the event loop if it's running
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, schedule the cleanup
                        loop.create_task(self.session.close())
                    else:
                        # If loop is not running, run the cleanup synchronously
                        asyncio.run(self.session.close())
                except RuntimeError:
                    # Event loop is closed or not available, skip session cleanup
                    pass
                self.session = None
                
            if self.ch_manager:
                self.ch_manager.close()
                
            logger.info("SentimentAnalyzer cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def async_close(self):
        """Async version of close for proper cleanup"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
                
            if self.ch_manager:
                self.ch_manager.close()
                
            logger.info("SentimentAnalyzer async cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during async cleanup: {e}")
    
    async def debug_scraping(self, url: str):
        """Debug function to show exactly what content is being scraped"""
        print(f"\n{'='*80}")
        print(f"DEBUG: URL SCRAPING ANALYSIS")
        print(f"{'='*80}")
        print(f"URL: {url}")
        print(f"{'='*80}")
        
        try:
            # Use the new scrape_article_content method
            scraped_content = self.scrape_article_content(url, max_chars=6000)
            
            if scraped_content:
                print(f"✅ Scraped: {len(scraped_content)} characters")
                
                # Create a test article with the scraped content
                test_article = {
                    'ticker': 'DEBUG',
                    'headline': 'Debug Article',
                    'summary': 'Debug summary',
                    'full_content': scraped_content,
                    'article_url': url
                }
                
                # Create prompt and analyze
                prompt = self.create_sentiment_prompt(test_article)
                analysis = await self.query_claude(prompt)
                
                if analysis and 'error' not in analysis:
                    print(f"✅ SENTIMENT ANALYSIS SUCCESSFUL:")
                    print(f"   Recommendation: {analysis.get('recommendation', 'unknown')}")
                    print(f"   Confidence: {analysis.get('confidence', 'unknown')}")
                    
                    # Map 4D prompt "reasoning" field to "explanation" for display
                    reasoning = analysis.get('reasoning', analysis.get('explanation', 'No reasoning provided'))
                    print(f"   Reasoning: {reasoning}")
                    
                    # Add sentiment mapping based on recommendation (4D prompt doesn't return sentiment)
                    recommendation = analysis.get('recommendation', 'HOLD')
                    if recommendation == 'BUY':
                        sentiment = 'positive'
                    elif recommendation == 'SELL':
                        sentiment = 'negative'
                    else:  # HOLD
                        sentiment = 'neutral'
                    
                    print(f"   Derived Sentiment: {sentiment}")
                    
                else:
                    print(f"❌ SENTIMENT ANALYSIS FAILED:")
                    if analysis:
                        print(f"   Error: {analysis.get('error', 'Unknown error')}")
                        if 'raw_response' in analysis:
                            print(f"   Raw Response: {analysis['raw_response'][:200]}...")
                    else:
                        print(f"   Error: No response from Claude")
                
                print(f"{'='*80}")
                
                return scraped_content
            else:
                print(f"❌ Scraping failed for {url}")
                return ""
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            return ""
    
    async def debug_recent_article(self, ticker: str = None):
        """Debug scraping for a recent article"""
        try:
            if ticker:
                query = f"""
                SELECT ticker, headline, article_url, full_content
                FROM News.breaking_news 
                WHERE ticker = '{ticker}' 
                AND timestamp >= now() - INTERVAL 24 HOUR
                AND article_url != ''
                ORDER BY timestamp DESC
                LIMIT 1
                """
            else:
                query = """
                SELECT ticker, headline, article_url, full_content
                FROM News.breaking_news 
                WHERE timestamp >= now() - INTERVAL 24 HOUR
                AND article_url != ''
                ORDER BY timestamp DESC
                LIMIT 1
                """
            
            result = self.ch_manager.client.query(query)
            
            if result.result_rows:
                row = result.result_rows[0]
                ticker = row[0]
                headline = row[1]
                article_url = row[2]
                db_content = row[3]
                
                print(f"\n{'='*80}")
                print(f"COMPARING DATABASE vs SCRAPED CONTENT")
                print(f"{'='*80}")
                print(f"Ticker: {ticker}")
                print(f"Headline: {headline}")
                print(f"URL: {article_url}")
                print(f"{'='*80}")
                
                # Show database content
                print(f"\n📄 DATABASE CONTENT ({len(db_content)} characters):")
                print(f"{'='*50}")
                print(db_content[:500] + "..." if len(db_content) > 500 else db_content)
                
                # Show scraped content
                print(f"\n🌐 SCRAPED CONTENT:")
                scraped_content = await self.debug_scraping(article_url)
                
                # Compare
                print(f"\n🔍 COMPARISON:")
                print(f"{'='*50}")
                print(f"Database content length: {len(db_content)}")
                print(f"Scraped content length: {len(scraped_content)}")
                print(f"Content identical: {'✅ YES' if db_content == scraped_content else '❌ NO'}")
                
                if db_content != scraped_content:
                    print(f"\n⚠️ CONTENT DIFFERENCES DETECTED!")
                    print(f"This explains why you're getting different sentiment results.")
                
            else:
                print("❌ No recent articles found with URLs")
                
        except Exception as e:
            print(f"❌ Error in debug: {e}")

    async def analyze_single_url(self, url: str, ticker: str = "UNKNOWN"):
        """Analyze a single URL for sentiment using 4D prompt"""
        print(f"\n{'='*80}")
        print(f"ANALYZING SINGLE URL - 4D TIMING & URGENCY ANALYSIS")
        print(f"{'='*80}")
        print(f"URL: {url}")
        print(f"Ticker: {ticker}")
        print(f"Model: {self.model}")
        print(f"{'='*80}")
        
        try:
            # Ensure session is ready before starting
            await self._ensure_session()
            
            # Create a test article from the URL
            article = {
                'ticker': ticker,
                'headline': 'Single URL Analysis',
                'summary': 'Analyzing single URL provided by user',
                'full_content': '',
                'article_url': url,
                'source': 'User Input',
                'timestamp': datetime.now()
            }
            
            # Create 4D prompt and analyze
            prompt = self.create_sentiment_prompt(article)
            analysis = await self.query_claude(prompt)
            
            if analysis and 'error' not in analysis:
                print(f"✅ 4D ANALYSIS SUCCESSFUL:")
                print(f"   Recommendation: {analysis.get('recommendation', 'unknown')}")
                print(f"   Confidence: {analysis.get('confidence', 'unknown')}")
                
                # Map 4D prompt "reasoning" field to "explanation" for display
                reasoning = analysis.get('reasoning', analysis.get('explanation', 'No reasoning provided'))
                print(f"   Reasoning: {reasoning}")
                
                # Add sentiment mapping based on recommendation (4D prompt doesn't return sentiment)
                recommendation = analysis.get('recommendation', 'HOLD')
                if recommendation == 'BUY':
                    sentiment = 'positive'
                elif recommendation == 'SELL':
                    sentiment = 'negative'
                else:  # HOLD
                    sentiment = 'neutral'
                
                print(f"   Derived Sentiment: {sentiment}")
                
                logger.info(f"4D URL ANALYSIS - {ticker}: {analysis.get('recommendation', 'unknown')} "
                           f"({analysis.get('confidence', 'unknown')} confidence)")
                return analysis
            else:
                print(f"❌ 4D ANALYSIS FAILED:")
                if analysis:
                    print(f"   Error: {analysis.get('error', 'Unknown error')}")
                    if 'raw_response' in analysis:
                        print(f"   Raw Response: {analysis['raw_response'][:200]}...")
                else:
                    print(f"   Error: No response from Claude")
                
                logger.warning(f"4D URL ANALYSIS FAILED - {ticker}: {analysis.get('error', 'Unknown') if analysis else 'No response'}")
                return None
                
        except Exception as e:
            print(f"❌ ANALYSIS EXCEPTION: {str(e)}")
            logger.error(f"Exception in analyze_single_url: {e}")
            return None

    async def test_url_consistency(self, url: str, test_runs: int = 3):
        """Test if sentiment analysis is consistent for the same URL across multiple runs"""
        print(f"\n{'='*80}")
        print(f"TESTING URL CONSISTENCY ({test_runs} runs)")
        print(f"{'='*80}")
        print(f"URL: {url}")
        print(f"{'='*80}")
        
        results = []
        
        for i in range(test_runs):
            print(f"\n🔄 Run {i+1}/{test_runs}:")
            print(f"{'='*40}")
            
            # Scrape content
            scraped_content = self.scrape_article_content(url, max_chars=6000)
            
            if scraped_content:
                print(f"✅ Scraped: {len(scraped_content)} characters")
                
                # Create test article
                test_article = {
                    'ticker': 'TEST',
                    'headline': 'Test Article',
                    'summary': 'Test summary',
                    'full_content': scraped_content,
                    'article_url': url
                }
                
                # Analyze sentiment
                prompt = self.create_sentiment_prompt(test_article)
                analysis = await self.query_claude(prompt)
                
                if analysis and 'error' not in analysis:
                    result = {
                        'recommendation': analysis.get('recommendation', 'unknown'),
                        'confidence': analysis.get('confidence', 'unknown'),
                        'content_length': len(scraped_content)
                    }
                    results.append(result)
                    print(f"✅ Analysis: {result['recommendation']} ({result['confidence']} confidence)")
                else:
                    print(f"❌ Analysis failed: {analysis.get('error', 'Unknown') if analysis else 'No response'}")
                    results.append({'error': 'failed'})
            else:
                print(f"❌ Scraping failed")
                results.append({'error': 'scraping_failed'})
        
        # Check consistency
        print(f"\n{'='*80}")
        print(f"CONSISTENCY RESULTS")
        print(f"{'='*80}")
        
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            print(f"Successful runs: {len(successful_results)}/{test_runs}")
            
            # Check if all content lengths are identical
            content_lengths = [r['content_length'] for r in successful_results]
            content_consistent = len(set(content_lengths)) == 1
            print(f"Content length consistent: {'✅ YES' if content_consistent else '❌ NO'}")
            if content_consistent:
                print(f"Content length: {content_lengths[0]} characters")
            else:
                print(f"Content lengths: {content_lengths}")
            
            # Check if all sentiment results are identical
            sentiment_results = [(r['recommendation'], r['confidence']) for r in successful_results]
            sentiment_consistent = len(set(sentiment_results)) == 1
            print(f"Sentiment consistent: {'✅ YES' if sentiment_consistent else '❌ NO'}")
            
            if sentiment_consistent:
                result = successful_results[0]
                print(f"Consistent result: {result['recommendation']} ({result['confidence']} confidence)")
            else:
                print(f"\nVariations found:")
                unique_results = list({str(r): r for r in successful_results}.values())
                for i, result in enumerate(unique_results, 1):
                    count = sum(1 for r in successful_results if (r['recommendation'], r['confidence']) == (result['recommendation'], result['confidence']))
                    print(f"  {i}. {result['recommendation']} ({result['confidence']}) - appeared {count} times")
        else:
            print(f"❌ No successful runs to compare")
        
        return results


async def main_async():
    """Main async function with argument parsing for URL analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Sentiment Analyzer - 4D Timing & Urgency Analysis (matches live system)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific URL (ticker will be auto-detected or set to UNKNOWN)
  python sentiment_analyzer.py --url "https://www.benzinga.com/news/earnings/..."

  # Analyze a specific URL with explicit ticker
  python sentiment_analyzer.py --url "https://www.benzinga.com/news/earnings/..." --ticker AAPL

  # Run default database analysis
  python sentiment_analyzer.py

  # Test URL consistency
  python sentiment_analyzer.py --url "https://..." --test-consistency 5
        """
    )
    
    parser.add_argument('--url', type=str, help='Specific URL to analyze')
    parser.add_argument('--ticker', type=str, help='Ticker symbol (optional, defaults to UNKNOWN or auto-detected)')
    parser.add_argument('--test-consistency', type=int, metavar='N', help='Test URL consistency N times')
    parser.add_argument('--analyze-recent', action='store_true', help='Analyze recent articles from database')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    try:
        # Test Claude connection first
        if not await analyzer.test_claude_connection():
            print("❌ Failed to connect to Claude API. Please check your ANTHROPIC_API_KEY.")
            return
            
        print("✅ Claude API connection successful")
        print(f"🤖 Using model: {analyzer.model}")
        print(f"📊 Using 4D Timing & Urgency Analysis (matches live system)")
        
        # Handle different modes based on arguments
        if args.url:
            # Use provided ticker or default to UNKNOWN
            ticker = args.ticker if args.ticker else "UNKNOWN"
            
            if args.test_consistency:
                # Test URL consistency
                await analyzer.test_url_consistency(args.url, args.test_consistency)
            else:
                # Single URL analysis
                await analyzer.analyze_single_url(args.url, ticker)
        elif args.analyze_recent:
            # Analyze recent articles from database
            analyzer.analyze_articles(hours=24, limit=10)
        else:
            # Default: analyze recent articles
            print("\n🔍 No specific URL provided. Analyzing recent articles from database...")
            analyzer.analyze_articles(hours=24, limit=10)
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}")
    finally:
        await analyzer.async_close()
        print("\n✅ Analysis completed")

def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 