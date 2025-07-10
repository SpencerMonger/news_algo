#!/usr/bin/env python3
"""
Test Single Batch Request for LM Studio Sentiment Analysis
Tests sending all articles in one request and parsing results back to individual rows
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from clickhouse_setup import ClickHouseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchSentimentTester:
    """Test class for single batch request sentiment analysis"""
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.session = None
        self.ch_manager = ClickHouseManager()
        self.test_articles = []
    
    async def initialize(self):
        """Initialize the test service"""
        try:
            # Create aiohttp session with timeouts - 60 second limit for batch processing
            timeout = aiohttp.ClientTimeout(total=60, connect=10)  # 60 seconds total, 10 seconds connect
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            )
            
            # Test connection
            is_connected = await self.test_connection()
            if not is_connected:
                logger.error("❌ Failed to connect to LM Studio")
                return False
            
            logger.info("✅ LM Studio connection successful")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing test service: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to LM Studio API"""
        try:
            test_payload = {
                "model": "local-model",
                "messages": [{"role": "user", "content": "Respond with just: {'status': 'connected'}"}],
                "temperature": 0.1,
                "max_tokens": 20,
                "stream": False
            }
            
            async with self.session.post(
                self.lm_studio_url,
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    logger.info("✅ LM Studio connection test successful")
                    return True
                else:
                    logger.error(f"❌ LM Studio connection failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ LM Studio connection test failed: {e}")
            return False
    
    def get_test_articles(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get real articles from the breaking news table"""
        try:
            self.ch_manager.connect()
            
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
            WHERE timestamp >= now() - INTERVAL 48 HOUR
            AND ticker != ''
            AND ticker != 'UNKNOWN'
            AND headline != ''
            ORDER BY timestamp DESC
            LIMIT {count}
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
            
            logger.info(f"Retrieved {len(articles)} real articles for testing")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving test articles: {e}")
            return []
        finally:
            self.ch_manager.close()
    
    def create_batch_sentiment_prompt(self, articles: List[Dict[str, Any]]) -> str:
        """Create a single batch prompt for all articles"""
        article_data = []
        
        for i, article in enumerate(articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            full_content = article.get('full_content', '')
            
            # Use same logic as live system
            content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
            content_to_analyze = content_to_analyze[:4000] if content_to_analyze else f"{headline}\n\n{summary}"
            
            article_data.append(f"""
Article {i}:
Ticker: {ticker}
Content: {content_to_analyze}
""")
        
        articles_text = "\n".join(article_data)
        
        prompt = f"""
Analyze the following {len(articles)} news articles and determine if each suggests a BUY, SELL, or HOLD signal based on the sentiment and potential market impact.

{articles_text}

Instructions:
1. For each article, analyze the sentiment (positive, negative, neutral)
2. Consider the potential market impact on stock price
3. Provide a clear recommendation:
   - BUY: For positive sentiment with strong bullish indicators
   - SELL: For negative sentiment with strong bearish indicators  
   - HOLD: For neutral sentiment or unclear market impact
4. Rate confidence as high, medium, or low
5. Give a brief explanation (1-2 sentences)

Respond with a JSON array containing exactly {len(articles)} objects in this format:
[
    {{
        "article_number": 1,
        "ticker": "TICKER1",
        "sentiment": "positive/negative/neutral",
        "recommendation": "BUY/SELL/HOLD",
        "confidence": "high/medium/low",
        "explanation": "Brief explanation of your reasoning"
    }},
    {{
        "article_number": 2,
        "ticker": "TICKER2",
        "sentiment": "positive/negative/neutral",
        "recommendation": "BUY/SELL/HOLD",
        "confidence": "high/medium/low",
        "explanation": "Brief explanation of your reasoning"
    }}
]

Important: 
- Use exactly "BUY", "SELL", or "HOLD" for recommendation (not "NEUTRAL")
- Return exactly {len(articles)} results in the same order as the articles
- Ensure valid JSON format
"""
        return prompt
    
    async def send_batch_request(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send single batch request to LM Studio API"""
        try:
            # Use much higher token limit for batch processing - need space for 20 detailed responses
            payload = {
                "model": "local-model",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON arrays when analyzing multiple articles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 4000,  # Increased significantly for batch response (20 articles * ~400 tokens each)
                "stream": False
            }
            
            logger.info(f"📤 Sending batch request to LM Studio...")
            logger.info(f"   Payload size: {len(json.dumps(payload))} characters")
            logger.info(f"   Prompt length: {len(prompt)} characters")
            logger.info(f"   Max tokens for response: {payload['max_tokens']}")
            
            async with self.session.post(
                self.lm_studio_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                logger.info(f"📥 Response status: {response.status}")
                
                if response.status == 200:
                    response_data = await response.json()
                    content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    logger.info(f"📥 Response content length: {len(content)} characters")
                    logger.info(f"📥 Raw response preview: {content[:200]}...")
                    
                    # Try to parse JSON from the response
                    try:
                        # Clean up the response if it has markdown code blocks
                        if '```json' in content:
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif '```' in content:
                            content = content.split('```')[1].strip()
                        
                        parsed_content = json.loads(content)
                        
                        # Validate that we got an array
                        if isinstance(parsed_content, list):
                            logger.info(f"✅ Successfully parsed JSON array with {len(parsed_content)} items")
                            return {"batch_results": parsed_content}
                        else:
                            logger.warning(f"❌ Expected array but got: {type(parsed_content)}")
                            return {"error": "Response is not an array", "raw_response": content}
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON response: {e}")
                        logger.error(f"Raw content: {content}")
                        return {"error": "Invalid JSON response", "raw_response": content}
                else:
                    error_text = await response.text()
                    logger.error(f"❌ LM Studio API error: {response.status} - {error_text}")
                    return {"error": f"API error: {response.status}", "details": error_text}
                    
        except asyncio.TimeoutError:
            logger.error(f"❌ Batch request timed out after 60 seconds - batch processing not viable")
            return {"error": "Request timed out after 60 seconds - batch processing too slow to be viable"}
        except Exception as e:
            logger.error(f"❌ Batch request to LM Studio failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
    
    def parse_batch_results(self, articles: List[Dict[str, Any]], batch_result: Dict[str, Any], total_time: float) -> List[Dict[str, Any]]:
        """Parse batch results and match them to articles"""
        enriched_articles = []
        batch_results = batch_result.get('batch_results', [])
        
        # Calculate per-article time
        per_article_time = total_time / len(articles) if articles else 0
        
        logger.info(f"🔄 Parsing batch results: {len(batch_results)} results for {len(articles)} articles")
        
        for i, article in enumerate(articles):
            article_copy = article.copy()
            
            # Try to find matching result
            matching_result = None
            if i < len(batch_results):
                matching_result = batch_results[i]
            
            if matching_result:
                # Use the batch result
                article_copy.update({
                    'sentiment': matching_result.get('sentiment', 'neutral'),
                    'recommendation': matching_result.get('recommendation', 'HOLD'),
                    'confidence': matching_result.get('confidence', 'low'),
                    'explanation': matching_result.get('explanation', 'No explanation'),
                    'analysis_time_ms': int(per_article_time * 1000),
                    'analyzed_at': datetime.now()
                })
                
                logger.info(f"✅ {i+1:2d}. {article.get('ticker', 'UNKNOWN'):6} - {matching_result.get('recommendation', 'UNKNOWN'):4} "
                           f"({matching_result.get('confidence', 'unknown'):6} confidence) - {matching_result.get('explanation', 'No explanation')[:50]}...")
            else:
                # No matching result found
                article_copy.update({
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': 'No result in batch response',
                    'analysis_time_ms': int(per_article_time * 1000),
                    'analyzed_at': datetime.now(),
                    'error': 'Missing from batch response'
                })
                
                logger.warning(f"❌ {i+1:2d}. {article.get('ticker', 'UNKNOWN'):6} - Missing from batch response")
            
            enriched_articles.append(article_copy)
        
        return enriched_articles
    
    async def run_batch_test(self, num_articles: int = 5):
        """Run the batch sentiment analysis test"""
        logger.info(f"🚀 STARTING BATCH SENTIMENT ANALYSIS TEST")
        logger.info(f"📊 Testing single batch request with {num_articles} articles")
        
        # Initialize
        if not await self.initialize():
            logger.error("❌ Failed to initialize - aborting test")
            return
        
        # Get test articles
        self.test_articles = self.get_test_articles(num_articles)
        if not self.test_articles:
            logger.error("❌ No test articles found - aborting test")
            return
        
        logger.info(f"📰 Loaded {len(self.test_articles)} test articles")
        
        # Show articles being tested
        logger.info("📋 Test Articles:")
        for i, article in enumerate(self.test_articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            headline = article.get('headline', '')[:60] + "..." if len(article.get('headline', '')) > 60 else article.get('headline', '')
            logger.info(f"   {i:2d}. {ticker:6} | {headline}")
        
        print(f"\n{'='*80}")
        print(f"🧠 BATCH SENTIMENT ANALYSIS TEST")
        print(f"{'='*80}")
        
        # Create batch prompt
        start_time = time.time()
        batch_prompt = self.create_batch_sentiment_prompt(self.test_articles)
        prompt_time = time.time() - start_time
        
        logger.info(f"📝 Created batch prompt in {prompt_time:.3f}s")
        logger.info(f"📝 Prompt length: {len(batch_prompt)} characters")
        
        # Send batch request
        request_start = time.time()
        batch_result = await self.send_batch_request(batch_prompt)
        request_time = time.time() - request_start
        
        logger.info(f"⏱️  Batch request completed in {request_time:.2f}s")
        
        # Parse results
        if batch_result and 'error' not in batch_result:
            parse_start = time.time()
            enriched_articles = self.parse_batch_results(self.test_articles, batch_result, request_time)
            parse_time = time.time() - parse_start
            
            total_time = time.time() - start_time
            
            logger.info(f"🔄 Parsed results in {parse_time:.3f}s")
            logger.info(f"⏱️  Total processing time: {total_time:.2f}s")
            
            # Print results
            self.print_results(enriched_articles, {
                'prompt_time': prompt_time,
                'request_time': request_time,
                'parse_time': parse_time,
                'total_time': total_time
            })
        else:
            # Handle batch failure
            error_msg = batch_result.get('error', 'Unknown error') if batch_result else 'No response'
            logger.error(f"❌ BATCH ANALYSIS FAILED: {error_msg}")
            
            if batch_result and 'raw_response' in batch_result:
                logger.error(f"Raw response: {batch_result['raw_response'][:500]}...")
            
            print(f"\n❌ BATCH TEST FAILED: {error_msg}")
    
    def print_results(self, enriched_articles: List[Dict[str, Any]], timing_stats: Dict[str, float]):
        """Print detailed test results"""
        successful_analyses = len([a for a in enriched_articles if 'error' not in a])
        failed_analyses = len([a for a in enriched_articles if 'error' in a])
        
        print(f"\n{'='*80}")
        print(f"📊 BATCH SENTIMENT ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        print(f"⏱️  TIMING BREAKDOWN:")
        print(f"   Prompt Creation: {timing_stats['prompt_time']:.3f}s")
        print(f"   LM Studio Request: {timing_stats['request_time']:.2f}s")
        print(f"   Result Parsing: {timing_stats['parse_time']:.3f}s")
        print(f"   Total Time: {timing_stats['total_time']:.2f}s")
        print(f"   Average per Article: {timing_stats['total_time'] / len(enriched_articles):.2f}s")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total Articles: {len(enriched_articles)}")
        print(f"   Successful Analyses: {successful_analyses}")
        print(f"   Failed Analyses: {failed_analyses}")
        print(f"   Success Rate: {(successful_analyses / len(enriched_articles) * 100):.1f}%")
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"{'#':<3} {'Ticker':<8} {'Recommendation':<12} {'Confidence':<10} {'Status'}")
        print(f"{'-'*55}")
        
        for i, article in enumerate(enriched_articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            recommendation = article.get('recommendation', 'UNKNOWN')
            confidence = article.get('confidence', 'unknown')
            status = "✅" if 'error' not in article else "❌"
            
            print(f"{i:<3} {ticker:<8} {recommendation:<12} {confidence:<10} {status}")
        
        # Show recommendation distribution
        recommendations = [a.get('recommendation', 'UNKNOWN') for a in enriched_articles if 'error' not in a]
        rec_counts = {}
        for rec in recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        print(f"\n📊 RECOMMENDATION DISTRIBUTION:")
        for rec, count in sorted(rec_counts.items()):
            percentage = (count / len(recommendations)) * 100 if recommendations else 0
            print(f"   {rec}: {count} ({percentage:.1f}%)")
        
        # Show any errors
        errors = [a for a in enriched_articles if 'error' in a]
        if errors:
            print(f"\n❌ ERRORS ({len(errors)} failed):")
            for error_article in errors:
                ticker = error_article.get('ticker', 'UNKNOWN')
                error_msg = error_article.get('error', 'Unknown error')
                print(f"   {ticker}: {error_msg}")
        
        print(f"\n{'='*80}")
        
        # Performance comparison
        if successful_analyses == len(enriched_articles):
            print(f"✅ EXCELLENT: 100% success rate with batch processing!")
            print(f"   Estimated vs Concurrent: {timing_stats['total_time']:.1f}s vs ~{len(enriched_articles) * 2.5:.0f}s sequential")
        else:
            print(f"⚠️  BATCH PROCESSING: {(successful_analyses / len(enriched_articles) * 100):.1f}% success rate")
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

async def main():
    """Main test function"""
    tester = BatchSentimentTester()
    
    try:
        await tester.run_batch_test(num_articles=5)
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 