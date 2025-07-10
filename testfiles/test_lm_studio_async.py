#!/usr/bin/env python3
"""
Test LM Studio Async Request Handling
Simulates exactly how the live system sends concurrent async requests to LM Studio
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

class LMStudioAsyncTester:
    """Test class that mimics exactly how sentiment_service.py sends requests"""
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.session = None
        self.ch_manager = ClickHouseManager()
        
        # Test configuration
        self.max_workers = 5  # Same as live system
        self.test_articles = []
        
        # Stats tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': None,
            'end_time': None,
            'individual_times': []
        }
    
    async def initialize(self):
        """Initialize the test service - exactly like sentiment_service.py"""
        try:
            # Create aiohttp session with INCREASED TIMEOUTS for LM Studio
            timeout = aiohttp.ClientTimeout(total=60, connect=30)  # Increased from total=30, connect=5
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=self.max_workers,  # Match live system
                    ttl_dns_cache=300
                )
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
        """Test connection to LM Studio API - same as live system"""
        try:
            test_payload = {
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "Respond with just: {'status': 'connected'}"
                    }
                ],
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
    
    def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create sentiment prompt - EXACTLY like sentiment_service.py"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        full_content = article.get('full_content', '')
        
        # Use same logic as live system
        content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
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

    async def analyze_article_sentiment(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze single article - EXACTLY like sentiment_service.py"""
        try:
            self.stats['total_requests'] += 1
            
            # Create prompt
            prompt = self.create_sentiment_prompt(article)
            
            # Time the request
            start_time = time.time()
            analysis_result = await self.query_lm_studio_async(prompt)
            analysis_time = time.time() - start_time
            
            # Track individual timing
            self.stats['individual_times'].append(analysis_time)
            
            if analysis_result and 'error' not in analysis_result:
                self.stats['successful_requests'] += 1
                
                # Add timing information
                analysis_result['analysis_time_ms'] = int(analysis_time * 1000)
                analysis_result['analyzed_at'] = datetime.now()
                
                logger.info(f"✅ SUCCESS: {article.get('ticker', 'UNKNOWN')} - {analysis_result.get('recommendation', 'UNKNOWN')} "
                           f"({analysis_result.get('confidence', 'unknown')} confidence) in {analysis_time:.2f}s")
                
                return analysis_result
            else:
                self.stats['failed_requests'] += 1
                error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
                logger.warning(f"❌ FAILED: {article.get('ticker', 'UNKNOWN')} - {error_msg} in {analysis_time:.2f}s")
                
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
            self.stats['failed_requests'] += 1
            logger.error(f"❌ EXCEPTION: {article.get('ticker', 'UNKNOWN')} - {str(e)}")
            
            return {
                'ticker': article.get('ticker', 'UNKNOWN'),
                'sentiment': 'neutral',
                'recommendation': 'HOLD',
                'confidence': 'low',
                'explanation': f'Analysis exception: {str(e)}',
                'analysis_time_ms': 0,
                'analyzed_at': datetime.now(),
                'error': str(e)
            }
    
    async def query_lm_studio_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send async request to LM Studio API - EXACTLY like sentiment_service.py"""
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
            
            async with self.session.post(
                self.lm_studio_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    response_data = await response.json()
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
                    logger.error(f"LM Studio API error: {response.status}")
                    return {"error": f"API error: {response.status}"}
                    
        except Exception as e:
            logger.error(f"Request to LM Studio failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
    
    async def analyze_batch_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze batch of articles - EXACTLY like sentiment_service.py"""
        logger.info(f"🧠 SENTIMENT ANALYSIS (CONCURRENT): Processing batch of {len(articles)} articles")
        
        self.stats['start_time'] = time.time()
        
        # Create analysis tasks - EXACTLY like live system
        analysis_tasks = [
            self.analyze_article_sentiment(article) 
            for article in articles
        ]
        
        # Execute all tasks concurrently with same error handling
        sentiment_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        self.stats['end_time'] = time.time()
        
        # Process results - EXACTLY like live system
        enriched_articles = []
        successful_analyses = 0
        
        for i, (article, sentiment_result) in enumerate(zip(articles, sentiment_results)):
            try:
                if isinstance(sentiment_result, Exception):
                    logger.error(f"Exception in sentiment analysis for article {i}: {sentiment_result}")
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
                logger.error(f"Error processing sentiment result for article {i}: {e}")
                continue
        
        total_time = self.stats['end_time'] - self.stats['start_time']
        logger.info(f"✅ SENTIMENT ANALYSIS COMPLETE: {successful_analyses}/{len(articles)} successful analyses in {total_time:.2f}s")
        
        return enriched_articles
    
    async def run_test(self, num_articles: int = 20):
        """Run the concurrent request test"""
        logger.info(f"🚀 STARTING LM STUDIO CONCURRENT REQUEST TEST")
        logger.info(f"📊 Testing concurrent requests with max_workers={self.max_workers}")
        
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
        print(f"🧠 CONCURRENT SENTIMENT ANALYSIS TEST")
        print(f"{'='*80}")
        
        # Run concurrent analysis
        enriched_articles = await self.analyze_batch_articles(self.test_articles)
        
        # Print results
        self.print_results(enriched_articles)
    
    def print_results(self, enriched_articles: List[Dict[str, Any]]):
        """Print detailed test results"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        print(f"\n{'='*80}")
        print(f"📊 LM STUDIO CONCURRENT REQUEST TEST RESULTS")
        print(f"{'='*80}")
        
        print(f"⏱️  TIMING BREAKDOWN:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Time per Article: {(total_time / len(self.test_articles)):.2f}s")
        
        if self.stats['individual_times']:
            individual_times = self.stats['individual_times']
            print(f"   Fastest Request: {min(individual_times):.2f}s")
            print(f"   Slowest Request: {max(individual_times):.2f}s")
            print(f"   Median Request Time: {sorted(individual_times)[len(individual_times)//2]:.2f}s")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total Articles: {len(self.test_articles)}")
        print(f"   Successful Requests: {self.stats['successful_requests']}")
        print(f"   Failed Requests: {self.stats['failed_requests']}")
        print(f"   Success Rate: {(self.stats['successful_requests'] / self.stats['total_requests'] * 100):.1f}%")
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"{'#':<3} {'Ticker':<8} {'Recommendation':<12} {'Confidence':<10} {'Time':<6} {'Status'}")
        print(f"{'-'*55}")
        
        for i, article in enumerate(enriched_articles, 1):
            ticker = article.get('ticker', 'UNKNOWN')
            recommendation = article.get('recommendation', 'UNKNOWN')
            confidence = article.get('confidence', 'unknown')
            analysis_time = article.get('analysis_time_ms', 0) / 1000
            status = "✅" if 'error' not in article else "❌"
            
            print(f"{i:<3} {ticker:<8} {recommendation:<12} {confidence:<10} {analysis_time:<6.2f} {status}")
        
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
        
        # Performance recommendations
        if self.stats['failed_requests'] > 0:
            failure_rate = (self.stats['failed_requests'] / self.stats['total_requests']) * 100
            print(f"⚠️  RECOMMENDATION: {failure_rate:.1f}% failure rate detected")
            
            if failure_rate > 50:
                print(f"   Consider reducing max_workers from {self.max_workers} to 1 for sequential processing")
                print(f"   Sequential processing would take ~{len(self.test_articles) * 2.5:.0f}s but with ~95% success rate")
            elif failure_rate > 10:
                print(f"   Consider reducing max_workers from {self.max_workers} to 2-3 for better reliability")
        else:
            print(f"✅ EXCELLENT: 100% success rate with concurrent processing!")
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

async def main():
    """Main test function"""
    tester = LMStudioAsyncTester()
    
    try:
        await tester.run_test(num_articles=20)
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 