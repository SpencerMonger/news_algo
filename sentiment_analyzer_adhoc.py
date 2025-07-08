import asyncio
import logging
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from clickhouse_setup import ClickHouseManager, setup_logging
import re
from urllib.parse import urlparse

# Initialize logging
logger = setup_logging()

class AdHocSentimentAnalyzer:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.articles_file = "articles.txt"
        
    def load_articles_from_file(self) -> List[str]:
        """Load article URLs from articles.txt file"""
        try:
            with open(self.articles_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f.readlines() if line.strip()]
            
            logger.info(f"Loaded {len(urls)} URLs from {self.articles_file}")
            return urls
            
        except FileNotFoundError:
            logger.error(f"File {self.articles_file} not found")
            return []
        except Exception as e:
            logger.error(f"Error reading {self.articles_file}: {e}")
            return []
    
    def extract_ticker_from_url(self, url: str) -> str:
        """Extract potential ticker symbol from URL or return 'UNKNOWN'"""
        try:
            # Common patterns in news URLs
            ticker_patterns = [
                r'/([A-Z]{1,5})[-_]',  # Ticker followed by dash or underscore
                r'/([A-Z]{1,5})\.',    # Ticker followed by dot
                r'/([A-Z]{1,5})/',     # Ticker in path segment
                r'ticker[=:]([A-Z]{1,5})',  # ticker=AAPL or ticker:AAPL
                r'symbol[=:]([A-Z]{1,5})',  # symbol=AAPL
            ]
            
            for pattern in ticker_patterns:
                match = re.search(pattern, url)
                if match:
                    potential_ticker = match.group(1)
                    # Validate ticker (2-5 characters, all caps)
                    if 2 <= len(potential_ticker) <= 5:
                        return potential_ticker
            
            # If no ticker found in URL, return UNKNOWN
            return 'UNKNOWN'
            
        except Exception as e:
            logger.warning(f"Error extracting ticker from URL {url}: {e}")
            return 'UNKNOWN'
    
    def scrape_article_content(self, url: str) -> Dict[str, Any]:
        """Scrape article content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Basic content extraction (this could be enhanced with more sophisticated parsing)
            html = response.text
            
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            headline = title_match.group(1).strip() if title_match else 'No title found'
            
            # Clean up headline
            headline = re.sub(r'\s+', ' ', headline)  # Normalize whitespace
            headline = headline.replace(' | GlobeNewswire', '')  # Remove common suffixes
            headline = headline.replace(' | Business Wire', '')
            headline = headline.replace(' | PR Newswire', '')
            
            # Extract meta description for summary
            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            summary = desc_match.group(1).strip() if desc_match else ''
            
            # Try to extract main content (basic approach)
            # Remove HTML tags for content analysis
            content_clean = re.sub(r'<[^>]+>', ' ', html)
            content_clean = re.sub(r'\s+', ' ', content_clean).strip()
            
            # Take first 2000 characters as content sample
            content_sample = content_clean[:2000] if content_clean else headline
            
            # Extract ticker from URL
            ticker = self.extract_ticker_from_url(url)
            
            article = {
                'ticker': ticker,
                'headline': headline,
                'summary': summary,
                'full_content': content_sample,
                'source': urlparse(url).netloc,
                'timestamp': datetime.now(),
                'article_url': url,
                'published_utc': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully scraped article: {ticker} - {headline[:50]}...")
            return article
            
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return None
        
    def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create a prompt for sentiment analysis"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        full_content = article.get('full_content', '')
        
        # Use full content if available, otherwise use summary and headline
        content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
        
        prompt = f"""
Analyze the following news article about {ticker} and determine if it suggests a BUY or SELL signal based on the sentiment and potential market impact.

Article Content:
{content_to_analyze}

Instructions:
1. Analyze the sentiment (positive, negative, neutral)
2. Consider the potential market impact
3. Provide a clear BUY or SELL recommendation
4. Give a brief explanation (1-2 sentences)

Respond in the following JSON format:
{{
    "ticker": "{ticker}",
    "sentiment": "positive/negative/neutral",
    "recommendation": "BUY/SELL",
    "confidence": "high/medium/low",
    "explanation": "Brief explanation of your reasoning"
}}
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
                "temperature": 0.3,
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
    
    def analyze_articles_from_file(self):
        """Analyze articles from URLs in articles.txt file"""
        logger.info("Starting ad-hoc sentiment analysis from articles.txt")
        
        # Load URLs from file
        urls = self.load_articles_from_file()
        
        if not urls:
            print("❌ No URLs found in articles.txt")
            return
        
        print(f"\n{'='*80}")
        print(f"AD-HOC SENTIMENT ANALYSIS - {len(urls)} URLs")
        print(f"{'='*80}")
        
        successful_analyses = 0
        failed_analyses = 0
        scraping_failures = 0
        ticker_results = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n{'-'*60}")
            print(f"Article {i}/{len(urls)}")
            print(f"URL: {url}")
            print(f"{'-'*60}")
            
            # Scrape article content
            article = self.scrape_article_content(url)
            
            if not article:
                scraping_failures += 1
                print(f"❌ SCRAPING FAILED: Could not extract content from URL")
                ticker_results.append({
                    'ticker': 'UNKNOWN',
                    'recommendation': 'SCRAPING_FAILED',
                    'confidence': 'N/A'
                })
                continue
            
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')[:100] + "..." if len(article.get('headline', '')) > 100 else article.get('headline', '')
            
            print(f"✅ SCRAPING SUCCESSFUL:")
            print(f"   Ticker: {ticker}")
            print(f"   Headline: {headline}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Content Length: {len(article.get('full_content', ''))} chars")
            
            # Create prompt and analyze
            prompt = self.create_sentiment_prompt(article)
            analysis = self.query_lm_studio(prompt)
            
            if analysis and 'error' not in analysis:
                successful_analyses += 1
                recommendation = analysis.get('recommendation', 'unknown')
                confidence = analysis.get('confidence', 'unknown')
                
                print(f"✅ SENTIMENT ANALYSIS SUCCESSFUL:")
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
                print(f"❌ SENTIMENT ANALYSIS FAILED:")
                if analysis:
                    print(f"   Error: {analysis.get('error', 'Unknown error')}")
                    if 'raw_response' in analysis:
                        print(f"   Raw Response: {analysis['raw_response'][:200]}...")
                else:
                    print(f"   Error: No response from LM Studio")
                
                # Store failed result for summary
                ticker_results.append({
                    'ticker': ticker,
                    'recommendation': 'ANALYSIS_FAILED',
                    'confidence': 'N/A'
                })
                
                logger.warning(f"ANALYSIS FAILED - {ticker}: {analysis.get('error', 'Unknown') if analysis else 'No response'}")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total URLs: {len(urls)}")
        print(f"Scraping Failures: {scraping_failures}")
        print(f"Successful Analyses: {successful_analyses}")
        print(f"Failed Analyses: {failed_analyses}")
        print(f"Overall Success Rate: {(successful_analyses/len(urls)*100):.1f}%")
        
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
                
                if recommendation in ['SCRAPING_FAILED', 'ANALYSIS_FAILED']:
                    print(f"{ticker} - {recommendation}")
                else:
                    # Format recommendation with confidence indicator
                    confidence_indicator = {
                        'high': '🔥',
                        'medium': '📊',
                        'low': '❓'
                    }.get(confidence.lower(), '')
                    
                    print(f"{ticker} - {recommendation} {confidence_indicator}")
        
        logger.info(f"Ad-hoc sentiment analysis completed: {successful_analyses}/{len(urls)} successful")
    
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


def main():
    """Main execution function"""
    print("Starting Ad-Hoc News Sentiment Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AdHocSentimentAnalyzer()
    
    try:
        # Test LM Studio connection first
        if not analyzer.test_lm_studio_connection():
            print("\n❌ Cannot connect to LM Studio. Please ensure:")
            print("1. LM Studio is running")
            print("2. A model is loaded")
            print("3. The server is accessible at http://localhost:1234")
            return
        
        print(f"\n🚀 Starting analysis of URLs from {analyzer.articles_file}...")
        
        # Analyze articles from file
        analyzer.analyze_articles_from_file()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        logger.info("Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        logger.error(f"Error during analysis: {e}")
    finally:
        print("\n✅ Analysis completed")


if __name__ == "__main__":
    main() 