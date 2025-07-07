import asyncio
import logging
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from clickhouse_setup import ClickHouseManager, setup_logging

# Initialize logging
logger = setup_logging()

class SentimentAnalyzer:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        
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


def main():
    """Main execution function"""
    print("Starting News Sentiment Analysis with LM Studio")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    try:
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