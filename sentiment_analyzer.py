import asyncio
import logging
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from clickhouse_setup import ClickHouseManager, setup_logging
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Initialize logging
logger = setup_logging()

class SentimentAnalyzer:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        
    def scrape_article_content(self, url: str, max_chars: int = 8000) -> str:
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
    
    def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create a prompt for sentiment analysis"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        full_content = article.get('full_content', '')
        article_url = article.get('article_url', '')
        
        # Always scrape full content from URL if available
        if article_url:
            logger.info(f"Scraping full content from URL: {article_url}")
            scraped_content = self.scrape_article_content(article_url, max_chars=8000)
            if scraped_content:
                content_to_analyze = scraped_content
                logger.info(f"Using scraped content: {len(content_to_analyze)} characters")
            else:
                content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        else:
            content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        
        # Apply 8K character limit to prevent token overflow
        content_to_analyze = content_to_analyze[:8000] if content_to_analyze else f"{headline}\n\n{summary}"
        
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

Important: Use exactly "BUY", "SELL", or "HOLD" for recommendation (not "NEUTRAL").
"""
        return prompt
        
    def query_lm_studio(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send prompt to LM Studio and get response"""
        try:
            payload = {
                "model": "local-model",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 300,
                "stream": False
            }
            
            response = requests.post(
                self.lm_studio_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Try to parse JSON from the response
                try:
                    # Clean up the response if it has markdown code blocks
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].strip()
                    
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response: {content}")
                    return {"error": "Invalid JSON response", "raw_response": content}
                    
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to LM Studio failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error querying LM Studio: {e}")
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
            analysis = self.query_lm_studio(prompt)
            
            if analysis and 'error' not in analysis:
                successful_analyses += 1
                recommendation = analysis.get('recommendation', 'unknown')
                confidence = analysis.get('confidence', 'unknown')
                
                print(f"✅ ANALYSIS SUCCESSFUL:")
                print(f"   Sentiment: {analysis.get('sentiment', 'unknown')}")
                print(f"   Recommendation: {recommendation}")
                print(f"   Confidence: {confidence}")
                print(f"   Explanation: {analysis.get('explanation', 'No explanation provided')}")
                
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
                    print(f"   Error: No response from LM Studio")
                
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
            analysis = self.query_lm_studio(prompt)
            
            if analysis and 'error' not in analysis:
                result = {
                    'sentiment': analysis.get('sentiment', 'unknown'),
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
    
    def test_lm_studio_connection(self):
        """Test connection to LM Studio"""
        print("Testing LM Studio connection...")
        
        test_prompt = "Hello, please respond with a simple JSON object containing 'status': 'connected'"
        
        try:
            payload = {
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": test_prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 50,
                "stream": False
            }
            
            response = requests.post(
                self.lm_studio_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ LM Studio connection successful!")
                response_data = response.json()
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"Response: {content}")
                return True
            else:
                print(f"❌ LM Studio connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ LM Studio connection failed: {e}")
            return False
    
    def close(self):
        """Close ClickHouse connection"""
        if self.ch_manager:
            self.ch_manager.close()

    def debug_scraping(self, url: str):
        """Debug function to show exactly what content is being scraped"""
        print(f"\n{'='*80}")
        print(f"DEBUG: URL SCRAPING ANALYSIS")
        print(f"{'='*80}")
        print(f"URL: {url}")
        print(f"{'='*80}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            print("🌐 Fetching URL...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"✅ Response Status: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            print(f"✅ HTML Parsed Successfully")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            print("✅ Scripts and styles removed")
            
            # Show what we're targeting for Benzinga
            if 'benzinga.com' in url:
                print(f"\n🎯 BENZINGA-SPECIFIC EXTRACTION:")
                print(f"{'='*50}")
                
                article_paragraphs = soup.find_all('p')
                print(f"Total paragraphs found: {len(article_paragraphs)}")
                
                content_paragraphs = []
                relevant_paragraphs = []
                
                for i, p in enumerate(article_paragraphs):
                    if p.parent and p.parent.get('class'):
                        parent_classes = p.parent.get('class', [])
                        text = p.get_text().strip()
                        
                        # Check if this is a target paragraph
                        is_target = any('cAazyy' in str(cls) or 'dIYChw' in str(cls) for cls in parent_classes)
                        
                        if is_target and len(text) > 20:
                            content_paragraphs.append(text)
                            relevant_paragraphs.append({
                                'index': i,
                                'parent_classes': parent_classes,
                                'text': text[:100] + "..." if len(text) > 100 else text
                            })
                
                print(f"Relevant paragraphs found: {len(relevant_paragraphs)}")
                
                if relevant_paragraphs:
                    print(f"\n📝 RELEVANT PARAGRAPHS:")
                    for para in relevant_paragraphs[:5]:  # Show first 5
                        print(f"  P{para['index']}: {para['text']}")
                        print(f"    Classes: {para['parent_classes']}")
                        print()
                
                article_content = ' '.join(content_paragraphs)
                print(f"✅ Benzinga extraction successful: {len(article_content)} characters")
                
            else:
                print(f"\n🎯 GENERIC EXTRACTION (Non-Benzinga):")
                article_content = ""
            
            # Show fallback attempts
            if not article_content or len(article_content) < 100:
                print(f"\n🔄 FALLBACK EXTRACTION:")
                print(f"Current content length: {len(article_content)}")
                
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
                        fallback_content = elements[0].get_text()
                        print(f"  ✅ {selector}: Found {len(fallback_content)} characters")
                        article_content = fallback_content
                        break
                    else:
                        print(f"  ❌ {selector}: Not found")
                
                if not article_content:
                    print(f"  🔄 Using full page text extraction...")
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    article_content = ' '.join(chunk for chunk in chunks if chunk)
                    print(f"  ✅ Full page extraction: {len(article_content)} characters")
            
            # Clean up the content
            lines = (line.strip() for line in article_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Apply 8K limit
            if len(clean_content) > 8000:
                clean_content = clean_content[:8000]
                print(f"✂️ Content truncated to 8,000 characters")
            
            print(f"\n{'='*80}")
            print(f"FINAL SCRAPED CONTENT ({len(clean_content)} characters):")
            print(f"{'='*80}")
            print(clean_content)
            print(f"{'='*80}")
            
            # NEW: Run sentiment analysis on the scraped content
            print(f"\n🧠 SENTIMENT ANALYSIS:")
            print(f"{'='*50}")
            
            # Create a test article with the scraped content
            test_article = {
                'ticker': 'DEBUG',
                'headline': 'Debug Article',
                'summary': 'Debug summary',
                'full_content': clean_content,
                'article_url': url
            }
            
            # Create prompt and analyze
            prompt = self.create_sentiment_prompt(test_article)
            analysis = self.query_lm_studio(prompt)
            
            if analysis and 'error' not in analysis:
                print(f"✅ SENTIMENT ANALYSIS SUCCESSFUL:")
                print(f"   Sentiment: {analysis.get('sentiment', 'unknown')}")
                print(f"   Recommendation: {analysis.get('recommendation', 'unknown')}")
                print(f"   Confidence: {analysis.get('confidence', 'unknown')}")
                print(f"   Explanation: {analysis.get('explanation', 'No explanation provided')}")
            else:
                print(f"❌ SENTIMENT ANALYSIS FAILED:")
                if analysis:
                    print(f"   Error: {analysis.get('error', 'Unknown error')}")
                    if 'raw_response' in analysis:
                        print(f"   Raw Response: {analysis['raw_response'][:200]}...")
                else:
                    print(f"   Error: No response from LM Studio")
            
            print(f"{'='*80}")
            
            return clean_content
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            return ""
    
    def debug_recent_article(self, ticker: str = None):
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
                scraped_content = self.debug_scraping(article_url)
                
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

    def test_url_consistency(self, url: str, test_runs: int = 3):
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
            scraped_content = self.scrape_article_content(url, max_chars=8000)
            
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
                analysis = self.query_lm_studio(prompt)
                
                if analysis and 'error' not in analysis:
                    result = {
                        'sentiment': analysis.get('sentiment', 'unknown'),
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
            sentiment_results = [(r['sentiment'], r['recommendation'], r['confidence']) for r in successful_results]
            sentiment_consistent = len(set(sentiment_results)) == 1
            print(f"Sentiment consistent: {'✅ YES' if sentiment_consistent else '❌ NO'}")
            
            if sentiment_consistent:
                result = successful_results[0]
                print(f"Consistent result: {result['recommendation']} ({result['confidence']} confidence)")
            else:
                print(f"\nVariations found:")
                unique_results = list({str(r): r for r in successful_results}.values())
                for i, result in enumerate(unique_results, 1):
                    count = sum(1 for r in successful_results if (r['sentiment'], r['recommendation'], r['confidence']) == (result['sentiment'], result['recommendation'], result['confidence']))
                    print(f"  {i}. {result['recommendation']} ({result['confidence']}) - appeared {count} times")
        else:
            print(f"❌ No successful runs to compare")
        
        return results


def main():
    """Main execution function"""
    import sys
    
    print("Starting News Sentiment Analysis with LM Studio")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    try:
        # Check for debug commands
        if len(sys.argv) > 1:
            if sys.argv[1] == '--debug-scraping':
                if len(sys.argv) > 2:
                    # Debug specific URL
                    url = sys.argv[2]
                    print(f"\n🔍 Debugging URL scraping...")
                    analyzer.debug_scraping(url)
                else:
                    # Debug recent article
                    print(f"\n🔍 Debugging recent article scraping...")
                    analyzer.debug_recent_article()
                return
            
            elif sys.argv[1] == '--debug-ticker':
                if len(sys.argv) > 2:
                    ticker = sys.argv[2]
                    print(f"\n🔍 Debugging scraping for ticker: {ticker}")
                    analyzer.debug_recent_article(ticker)
                else:
                    print("❌ Please provide a ticker symbol: --debug-ticker AAPL")
                return
            
            elif sys.argv[1] == '--test-deterministic':
                print("\n🧪 Running deterministic test...")
                analyzer.test_deterministic_analysis(test_runs=5)
                return
            
            elif sys.argv[1] == '--test-url-consistency':
                if len(sys.argv) > 2:
                    url = sys.argv[2]
                    test_runs = 3 # Default to 3 runs
                    if len(sys.argv) > 3:
                        try:
                            test_runs = int(sys.argv[3])
                        except ValueError:
                            print(f"Invalid number of test runs: {sys.argv[3]}. Using default: {test_runs}")
                    print(f"\n🧪 Testing URL consistency for {test_runs} runs on {url}...")
                    analyzer.test_url_consistency(url, test_runs=test_runs)
                else:
                    print("❌ Please provide a URL: --test-url-consistency http://example.com")
                return
            
            elif sys.argv[1] == '--help':
                print("\n📖 USAGE:")
                print("  python3 sentiment_analyzer.py                    # Run normal analysis")
                print("  python3 sentiment_analyzer.py --debug-scraping   # Debug recent article scraping")
                print("  python3 sentiment_analyzer.py --debug-scraping <URL>  # Debug specific URL")
                print("  python3 sentiment_analyzer.py --debug-ticker <TICKER>  # Debug specific ticker")
                print("  python3 sentiment_analyzer.py --test-deterministic     # Test AI consistency")
                print("  python3 sentiment_analyzer.py --test-url-consistency <URL> [N]  # Test URL consistency (N=3 default)")
                print("  python3 sentiment_analyzer.py --help                   # Show this help")
                return
        
        # Test LM Studio connection first
        if not analyzer.test_lm_studio_connection():
            print("\n❌ Cannot connect to LM Studio. Please ensure:")
            print("1. LM Studio is running")
            print("2. A model is loaded")
            print("3. The server is accessible at http://localhost:1234")
            return
        
        print("\n🚀 Starting sentiment analysis...")
        
        # Analyze articles (default: last 24 hours, limit 100 to get all articles)
        analyzer.analyze_articles(hours=24, limit=100)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        logger.info("Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        logger.error(f"Error during analysis: {e}")
    finally:
        analyzer.close()
        print("\n✅ Analysis completed")


if __name__ == "__main__":
    main() 