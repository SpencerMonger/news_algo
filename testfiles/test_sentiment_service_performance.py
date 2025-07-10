#!/usr/bin/env python3
"""
Dedicated Test for Sentiment Service Performance
Tests the sentiment_service.py without web scraping to measure pure AI analysis performance
"""

import asyncio
import time
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path to import sentiment_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_service import SentimentService, analyze_articles_with_sentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentServicePerformanceTester:
    """Performance tester for sentiment service"""
    
    def __init__(self):
        self.service = None
        self.test_results = []
    
    def get_test_articles(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate test articles with NO URLs to avoid web scraping"""
        test_articles = []
        
        # Sample articles with different sentiments
        sample_articles = [
            {
                'ticker': 'AAPL',
                'headline': 'Apple Reports Strong Q4 Earnings, Beats Expectations',
                'summary': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales.',
                'full_content': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue growth. The company posted revenue of $94.9 billion, up 6% year-over-year. iPhone sales were particularly strong in international markets.',
                'content_hash': f'test_hash_positive_{1}',
                'article_url': ''  # NO URL to avoid scraping
            },
            {
                'ticker': 'TSLA',
                'headline': 'Tesla Faces Production Challenges, Misses Delivery Targets',
                'summary': 'Tesla reported lower than expected vehicle deliveries due to production issues.',
                'full_content': 'Tesla Inc. reported vehicle deliveries that fell short of analyst expectations, citing production challenges at its manufacturing facilities. The company delivered 435,000 vehicles in the quarter, below the expected 450,000. Production issues at the Shanghai factory contributed to the shortfall.',
                'content_hash': f'test_hash_negative_{2}',
                'article_url': ''  # NO URL to avoid scraping
            },
            {
                'ticker': 'GOOGL',
                'headline': 'Google Announces New AI Features for Search',
                'summary': 'Google unveiled new artificial intelligence capabilities for its search engine.',
                'full_content': 'Google announced new AI-powered features for its search engine, including enhanced natural language processing and improved search result relevance. The company expects these features to improve user experience and maintain its competitive edge in the search market.',
                'content_hash': f'test_hash_positive_{3}',
                'article_url': ''  # NO URL to avoid scraping
            },
            {
                'ticker': 'AMZN',
                'headline': 'Amazon Faces Regulatory Scrutiny Over Market Practices',
                'summary': 'Amazon is under investigation by regulators for potential anti-competitive practices.',
                'full_content': 'Amazon is facing increased regulatory scrutiny from federal authorities investigating potential anti-competitive practices in its marketplace operations. The investigation focuses on how Amazon treats third-party sellers and whether it uses their data to compete unfairly.',
                'content_hash': f'test_hash_negative_{4}',
                'article_url': ''  # NO URL to avoid scraping
            },
            {
                'ticker': 'MSFT',
                'headline': 'Microsoft Reports Steady Growth in Cloud Services',
                'summary': 'Microsoft Azure cloud services showed continued growth in the latest quarter.',
                'full_content': 'Microsoft Corporation reported steady growth in its Azure cloud services division, with revenue increasing 27% year-over-year. The company continues to gain market share in the competitive cloud computing market, driven by enterprise adoption of its services.',
                'content_hash': f'test_hash_positive_{5}',
                'article_url': ''  # NO URL to avoid scraping
            },
            {
                'ticker': 'META',
                'headline': 'Meta Announces Workforce Reduction Plans',
                'summary': 'Meta plans to reduce its workforce by 10% as part of efficiency measures.',
                'full_content': 'Meta Platforms Inc. announced plans to reduce its workforce by approximately 10% as part of ongoing efficiency measures. The company cited the need to streamline operations and reduce costs in a challenging economic environment. The layoffs will affect multiple divisions across the company.',
                'content_hash': f'test_hash_negative_{6}',
                'article_url': ''  # NO URL to avoid scraping
            }
        ]
        
        # Replicate articles to reach desired count
        for i in range(count):
            article = sample_articles[i % len(sample_articles)].copy()
            article['content_hash'] = f"test_hash_{i}"
            article['ticker'] = f"{article['ticker']}"  # Keep original ticker
            test_articles.append(article)
        
        return test_articles
    
    async def test_single_article_performance(self):
        """Test performance of single article analysis"""
        logger.info("🔍 Testing single article performance...")
        
        self.service = SentimentService()
        initialized = await self.service.initialize()
        
        if not initialized:
            logger.error("❌ Failed to initialize sentiment service")
            return
        
        # Test single article
        test_article = self.get_test_articles(1)[0]
        
        start_time = time.time()
        result = await self.service.analyze_article_sentiment(test_article)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        logger.info(f"✅ Single article analysis completed in {analysis_time:.2f}s")
        logger.info(f"   Ticker: {result.get('ticker', 'N/A')}")
        logger.info(f"   Sentiment: {result.get('sentiment', 'N/A')}")
        logger.info(f"   Recommendation: {result.get('recommendation', 'N/A')}")
        logger.info(f"   Confidence: {result.get('confidence', 'N/A')}")
        
        await self.service.cleanup()
        return analysis_time
    
    async def test_batch_performance(self, num_articles: int = 10):
        """Test performance of batch article analysis"""
        logger.info(f"🔍 Testing batch performance with {num_articles} articles...")
        
        self.service = SentimentService()
        initialized = await self.service.initialize()
        
        if not initialized:
            logger.error("❌ Failed to initialize sentiment service")
            return
        
        # Get test articles
        test_articles = self.get_test_articles(num_articles)
        
        logger.info(f"📰 Processing {len(test_articles)} articles...")
        
        start_time = time.time()
        enriched_articles = await self.service.analyze_batch_articles(test_articles)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_article = total_time / len(test_articles)
        
        # Count successful analyses
        successful_analyses = sum(1 for a in enriched_articles if a.get('recommendation') != 'HOLD' or a.get('confidence') != 'low')
        success_rate = (successful_analyses / len(enriched_articles)) * 100
        
        logger.info(f"✅ Batch analysis completed:")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average per article: {avg_time_per_article:.2f}s")
        logger.info(f"   Articles processed: {len(enriched_articles)}")
        logger.info(f"   Successful analyses: {successful_analyses}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        # Show individual results
        logger.info(f"📊 Individual results:")
        for i, article in enumerate(enriched_articles):
            logger.info(f"   {i+1}. {article.get('ticker', 'N/A')} - {article.get('recommendation', 'N/A')} ({article.get('confidence', 'N/A')})")
        
        # Get service stats
        stats = self.service.get_stats()
        logger.info(f"📈 Service statistics:")
        logger.info(f"   Total analyzed: {stats['total_analyzed']}")
        logger.info(f"   Successful: {stats['successful_analyses']}")
        logger.info(f"   Failed: {stats['failed_analyses']}")
        logger.info(f"   Cache hits: {stats['cache_hits']}")
        logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
        
        await self.service.cleanup()
        return total_time, avg_time_per_article, success_rate
    
    async def test_convenience_function(self, num_articles: int = 10):
        """Test the convenience function used by the main system"""
        logger.info(f"🔍 Testing convenience function with {num_articles} articles...")
        
        # Get test articles
        test_articles = self.get_test_articles(num_articles)
        
        logger.info(f"📰 Processing {len(test_articles)} articles via convenience function...")
        
        start_time = time.time()
        enriched_articles = await analyze_articles_with_sentiment(test_articles)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_article = total_time / len(test_articles)
        
        # Count successful analyses
        successful_analyses = sum(1 for a in enriched_articles if a.get('recommendation') != 'HOLD' or a.get('confidence') != 'low')
        success_rate = (successful_analyses / len(enriched_articles)) * 100
        
        logger.info(f"✅ Convenience function test completed:")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average per article: {avg_time_per_article:.2f}s")
        logger.info(f"   Articles processed: {len(enriched_articles)}")
        logger.info(f"   Successful analyses: {successful_analyses}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        return total_time, avg_time_per_article, success_rate
    
    async def run_comprehensive_test(self):
        """Run comprehensive performance tests"""
        logger.info("🚀 Starting comprehensive sentiment service performance test...")
        
        results = {}
        
        # Test 1: Single article
        try:
            single_time = await self.test_single_article_performance()
            results['single_article'] = single_time
        except Exception as e:
            logger.error(f"❌ Single article test failed: {e}")
            results['single_article'] = None
        
        logger.info("=" * 80)
        
        # Test 2: Small batch (5 articles)
        try:
            total_time, avg_time, success_rate = await self.test_batch_performance(5)
            results['batch_5'] = {'total': total_time, 'avg': avg_time, 'success_rate': success_rate}
        except Exception as e:
            logger.error(f"❌ Batch 5 test failed: {e}")
            results['batch_5'] = None
        
        logger.info("=" * 80)
        
        # Test 3: Medium batch (10 articles)
        try:
            total_time, avg_time, success_rate = await self.test_batch_performance(10)
            results['batch_10'] = {'total': total_time, 'avg': avg_time, 'success_rate': success_rate}
        except Exception as e:
            logger.error(f"❌ Batch 10 test failed: {e}")
            results['batch_10'] = None
        
        logger.info("=" * 80)
        
        # Test 4: Large batch (20 articles)
        try:
            total_time, avg_time, success_rate = await self.test_batch_performance(20)
            results['batch_20'] = {'total': total_time, 'avg': avg_time, 'success_rate': success_rate}
        except Exception as e:
            logger.error(f"❌ Batch 20 test failed: {e}")
            results['batch_20'] = None
        
        logger.info("=" * 80)
        
        # Test 5: Convenience function (10 articles)
        try:
            total_time, avg_time, success_rate = await self.test_convenience_function(10)
            results['convenience_10'] = {'total': total_time, 'avg': avg_time, 'success_rate': success_rate}
        except Exception as e:
            logger.error(f"❌ Convenience function test failed: {e}")
            results['convenience_10'] = None
        
        # Print summary
        logger.info("=" * 80)
        logger.info("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        if results['single_article']:
            logger.info(f"Single Article: {results['single_article']:.2f}s")
        
        if results['batch_5']:
            logger.info(f"Batch 5 Articles: {results['batch_5']['total']:.2f}s total, {results['batch_5']['avg']:.2f}s avg, {results['batch_5']['success_rate']:.1f}% success")
        
        if results['batch_10']:
            logger.info(f"Batch 10 Articles: {results['batch_10']['total']:.2f}s total, {results['batch_10']['avg']:.2f}s avg, {results['batch_10']['success_rate']:.1f}% success")
        
        if results['batch_20']:
            logger.info(f"Batch 20 Articles: {results['batch_20']['total']:.2f}s total, {results['batch_20']['avg']:.2f}s avg, {results['batch_20']['success_rate']:.1f}% success")
        
        if results['convenience_10']:
            logger.info(f"Convenience Function 10 Articles: {results['convenience_10']['total']:.2f}s total, {results['convenience_10']['avg']:.2f}s avg, {results['convenience_10']['success_rate']:.1f}% success")
        
        logger.info("=" * 80)
        
        return results

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Service Performance Test')
    parser.add_argument('--single', action='store_true', help='Test single article only')
    parser.add_argument('--batch', type=int, default=10, help='Test batch with N articles')
    parser.add_argument('--convenience', action='store_true', help='Test convenience function')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive tests')
    
    args = parser.parse_args()
    
    tester = SentimentServicePerformanceTester()
    
    if args.single:
        await tester.test_single_article_performance()
    elif args.convenience:
        await tester.test_convenience_function(args.batch)
    elif args.comprehensive:
        await tester.run_comprehensive_test()
    else:
        await tester.test_batch_performance(args.batch)

if __name__ == "__main__":
    asyncio.run(main()) 