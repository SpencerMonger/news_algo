#!/usr/bin/env python3
"""
Sentiment Analysis Service - Integrated into NewsHead Pipeline
Analyzes articles before database insertion using LM Studio
"""

import asyncio
import logging
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp
from concurrent.futures import ThreadPoolExecutor

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
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/chat/completions"):
        self.lm_studio_url = lm_studio_url
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=5)  # Limit concurrent requests
        
        # Sentiment cache to avoid re-analyzing identical content
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        
        # Stats tracking
        self.stats = {
            'total_analyzed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'start_time': time.time()
        }
    
    async def initialize(self):
        """Initialize the sentiment service"""
        try:
            # Create aiohttp session for async requests
            timeout = aiohttp.ClientTimeout(total=30, connect=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300
                )
            )
            
            # Test connection
            is_connected = await self.test_connection()
            if not is_connected:
                logger.error("❌ Failed to connect to LM Studio - sentiment analysis will be disabled")
                return False
            
            logger.info("✅ Sentiment Analysis Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing sentiment service: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to LM Studio API"""
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
                    logger.info("✅ LM Studio connection successful")
                    return True
                else:
                    logger.error(f"❌ LM Studio connection failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ LM Studio connection test failed: {e}")
            return False
    
    def create_sentiment_prompt(self, article: Dict[str, Any]) -> str:
        """Create a prompt for sentiment analysis"""
        ticker = article.get('ticker', 'UNKNOWN')
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        full_content = article.get('full_content', '')
        
        # Use the most comprehensive content available
        content_to_analyze = full_content if full_content and len(full_content) > len(headline) else f"{headline}\n\n{summary}"
        
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
            
            # Create prompt
            prompt = self.create_sentiment_prompt(article)
            
            # Analyze with LM Studio
            start_time = time.time()
            analysis_result = await self.query_lm_studio_async(prompt)
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
    
    async def query_lm_studio_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send async request to LM Studio API"""
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
        """
        Analyze sentiment for a batch of articles
        Returns articles with sentiment data added
        """
        if not articles:
            return articles
        
        logger.info(f"🧠 SENTIMENT ANALYSIS: Processing batch of {len(articles)} articles")
        
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
        
        # Test article
        test_article = {
            'ticker': 'AAPL',
            'headline': 'Apple Reports Strong Q4 Earnings, Beats Expectations',
            'summary': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales.',
            'full_content': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue growth.',
            'content_hash': 'test_hash_123'
        }
        
        result = await service.analyze_article_sentiment(test_article)
        print(f"Test result: {result}")
        
        stats = service.get_stats()
        print(f"Stats: {stats}")
        
        await service.cleanup()
    
    asyncio.run(test_sentiment_service()) 