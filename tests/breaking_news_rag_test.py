#!/usr/bin/env python3
"""
Breaking News RAG Sentiment Analysis Test

This script reads articles from the breaking_news table and analyzes them
using the RAG-enhanced sentiment analysis system.

Usage:
    python3 tests/breaking_news_rag_test.py --limit 20
    python3 tests/breaking_news_rag_test.py --ticker AAPL --limit 10
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from sentiment_service import get_sentiment_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BreakingNewsRAGTester:
    """Test RAG sentiment analysis on breaking news articles"""
    
    def __init__(self, buy_high_threshold: float = 0.8):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.buy_high_threshold = buy_high_threshold
        self.confidence_map = {'low': 0.55, 'medium': 0.7, 'high': 0.95}
        
    async def initialize(self):
        """Initialize the breaking news tester"""
        logger.info("🧪 Initializing Breaking News RAG Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize sentiment service
        self.sentiment_service = await get_sentiment_service()
        
        # Verify training vectors exist for RAG functionality
        await self.verify_training_vectors()
        
        logger.info("✅ Breaking news RAG test framework initialized successfully")
    
    async def verify_training_vectors(self):
        """Verify that training vectors exist for RAG functionality"""
        try:
            # First try the dedicated training vectors table
            count_query = 'SELECT COUNT(*) FROM News.rag_training_vectors'
            try:
                vector_count = self.ch_manager.client.query(count_query).result_rows[0][0]
                if vector_count > 0:
                    logger.info(f"✅ Found {vector_count} training vectors for RAG system")
                    self.vectors_table = 'News.rag_training_vectors'
                    return
            except:
                logger.info("ℹ️ Training vectors table not found, checking for existing vectors...")
            
            # Fallback: use existing vectors filtered by training set
            fallback_query = '''
            SELECT COUNT(*) 
            FROM News.rag_article_vectors v
            INNER JOIN News.rag_training_set t ON v.original_content_hash = t.original_content_hash
            '''
            try:
                vector_count = self.ch_manager.client.query(fallback_query).result_rows[0][0]
                if vector_count > 0:
                    logger.info(f"✅ Found {vector_count} existing vectors that match training set")
                    self.vectors_table = 'News.rag_article_vectors'
                    self.use_training_filter = True
                    return
            except:
                pass
            
            # No vectors found - warn but continue (will use traditional analysis only)
            logger.warning("⚠️ No training vectors found! RAG analysis will fall back to traditional sentiment analysis.")
            self.vectors_table = None
                
        except Exception as e:
            logger.error(f"Error verifying training vectors: {e}")
            self.vectors_table = None
    
    async def get_breaking_news_articles(self, limit: int = 20, ticker: Optional[str] = None, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recent breaking news articles"""
        try:
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            
            # Build query based on parameters
            where_conditions = [f"detected_at >= '{time_threshold.strftime('%Y-%m-%d %H:%M:%S')}'"]
            params = []
            
            if ticker:
                where_conditions.append("ticker = %s")
                params.append(ticker.upper())
            
            query = f"""
            SELECT 
                ticker,
                headline,
                full_content,
                detected_at,
                article_url
            FROM News.breaking_news
            WHERE {' AND '.join(where_conditions)}
            ORDER BY detected_at DESC
            LIMIT %s
            """
            params.append(limit)
            
            result = self.ch_manager.client.query(query, parameters=params)
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'full_content': row[2],
                    'detected_at': row[3],
                    'article_url': row[4],
                    'content': row[2] or f"{row[0]}: {row[1]}"  # Use full_content if available
                })
            
            logger.info(f"📄 Retrieved {len(articles)} breaking news articles from last {hours_back} hours")
            
            if ticker:
                logger.info(f"🎯 Filtered for ticker: {ticker}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving breaking news articles: {e}")
            return []
    
    async def get_similar_training_examples(self, query_content: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar examples from training set using RAG"""
        if not self.vectors_table:
            return []
            
        try:
            # Generate features for the query content
            features = await self.generate_query_features(query_content)
            if not features:
                return []
            
            # Choose query based on available vectors table
            if hasattr(self, 'use_training_filter') and self.use_training_filter:
                # Use existing vectors filtered by training set
                similarity_query = """
                SELECT 
                    v.ticker,
                    v.headline,
                    v.outcome_type,
                    v.price_increase_ratio,
                    cosineDistance(v.feature_vector, %s) as distance,
                    1 - cosineDistance(v.feature_vector, %s) as similarity
                FROM News.rag_article_vectors v
                INNER JOIN News.rag_training_set t ON v.original_content_hash = t.original_content_hash
                ORDER BY distance ASC
                LIMIT %s
                """
            else:
                # Use dedicated training vectors table
                similarity_query = """
                SELECT 
                    ticker,
                    headline,
                    outcome_type,
                    price_increase_ratio,
                    cosineDistance(feature_vector, %s) as distance,
                    1 - cosineDistance(feature_vector, %s) as similarity
                FROM News.rag_training_vectors
                ORDER BY distance ASC
                LIMIT %s
                """
            
            result = self.ch_manager.client.query(
                similarity_query, 
                parameters=[features, features, top_k]
            )
            
            similar_examples = []
            for row in result.result_rows:
                similar_examples.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'outcome_type': row[2],
                    'price_increase_ratio': float(row[3]),
                    'distance': float(row[4]),
                    'similarity': float(row[5])
                })
            
            return similar_examples
            
        except Exception as e:
            logger.error(f"Error finding similar examples: {e}")
            return []
    
    async def generate_query_features(self, content: str) -> List[float]:
        """Generate features for query content (same method as training)"""
        try:
            # Use the same feature extraction as in generate_vectors.py
            analysis_prompt = f"""
Analyze the following financial news article and extract key features for similarity comparison.
Focus on: sentiment, topic, urgency, market impact, company type, and news type.

Article: {content[:4500]}

Respond with a JSON object containing numerical scores (0.0 to 1.0) for these features:
{{
    "sentiment_score": 0.0-1.0,
    "bullish_score": 0.0-1.0, 
    "urgency_score": 0.0-1.0,
    "financial_impact_score": 0.0-1.0,
    "earnings_related": 0.0-1.0,
    "partnership_related": 0.0-1.0,
    "product_related": 0.0-1.0,
    "regulatory_related": 0.0-1.0,
    "clinical_trial_related": 0.0-1.0,
    "acquisition_related": 0.0-1.0,
    "market_general": 0.0-1.0,
    "biotech_pharma": 0.0-1.0,
    "tech_software": 0.0-1.0,
    "finance_banking": 0.0-1.0,
    "energy_commodities": 0.0-1.0,
    "retail_consumer": 0.0-1.0,
    "manufacturing": 0.0-1.0,
    "healthcare": 0.0-1.0,
    "real_estate": 0.0-1.0,
    "transportation": 0.0-1.0
}}
"""
            
            result = await self.sentiment_service.load_balancer.make_claude_request(analysis_prompt)
            
            if result and isinstance(result, dict):
                # Convert Claude's analysis to feature vector (same as generate_vectors.py)
                feature_vector = [
                    result.get('sentiment_score', 0.5),
                    result.get('bullish_score', 0.5),
                    result.get('urgency_score', 0.5),
                    result.get('financial_impact_score', 0.5),
                    result.get('earnings_related', 0.0),
                    result.get('partnership_related', 0.0),
                    result.get('product_related', 0.0),
                    result.get('regulatory_related', 0.0),
                    result.get('clinical_trial_related', 0.0),
                    result.get('acquisition_related', 0.0),
                    result.get('market_general', 0.0),
                    result.get('biotech_pharma', 0.0),
                    result.get('tech_software', 0.0),
                    result.get('finance_banking', 0.0),
                    result.get('energy_commodities', 0.0),
                    result.get('retail_consumer', 0.0),
                    result.get('manufacturing', 0.0),
                    result.get('healthcare', 0.0),
                    result.get('real_estate', 0.0),
                    result.get('transportation', 0.0)
                ]
                
                # Add derived features to reach 50 dimensions
                while len(feature_vector) < 50:
                    if len(feature_vector) < 30:
                        feature_vector.append(feature_vector[0] * feature_vector[1])  # sentiment * bullish
                    elif len(feature_vector) < 40:
                        feature_vector.append(feature_vector[2] * feature_vector[3])  # urgency * impact
                    else:
                        feature_vector.append(0.5)  # neutral padding
                
                return feature_vector
            else:
                return [0.5] * 50  # Neutral features on failure
                
        except Exception as e:
            logger.error(f"Query feature generation failed: {e}")
            return [0.5] * 50
    
    async def analyze_rag_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment using RAG-enhanced method"""
        start_time = datetime.now()
        
        try:
            # Get similar examples from training set
            similar_examples = await self.get_similar_training_examples(content, top_k=5)
            
            # Create RAG prompt with similar examples
            rag_context = self.create_rag_context(similar_examples)
            
            # Clean content for JSON safety
            clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
            
            # Analyze with RAG context
            rag_prompt = f"""Based on these similar historical examples and their outcomes:

{rag_context}

Now analyze this new article:
{clean_content}

IMPORTANT: Consider the historical patterns above. Look for opportunities that match successful patterns from TRUE_BULLISH examples. Don't be overly conservative - if the article shows strong positive signals similar to past winners, recommend BUY with appropriate confidence.

Key considerations:
- If similar to TRUE_BULLISH examples with high similarity: Consider BUY with high confidence
- If shows positive signals but uncertain: BUY with medium confidence is better than missing opportunities
- Only use HOLD if genuinely neutral or similar to FALSE_PUMP/NEUTRAL examples

Provide sentiment analysis considering the historical patterns shown above.
Respond with JSON: {{"action": "BUY/HOLD/SELL", "confidence": "high/medium/low", "reasoning": "explanation based on historical patterns"}}"""
            
            result = await self.sentiment_service.load_balancer.make_claude_request(rag_prompt)
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            if result and isinstance(result, dict):
                # Extract action and confidence
                action = result.get('action', 'HOLD')
                confidence_str = result.get('confidence', 'medium')
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                # Convert confidence string to float
                confidence = self.confidence_map.get(confidence_str, 0.5)
                
                return {
                    'action': action,
                    'confidence': confidence,
                    'confidence_str': confidence_str,
                    'reasoning': reasoning,
                    'analysis_time': analysis_time,
                    'similar_examples': similar_examples,
                    'method': 'RAG'
                }
            else:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'confidence_str': 'medium',
                    'reasoning': 'Analysis failed',
                    'analysis_time': analysis_time,
                    'similar_examples': similar_examples,
                    'method': 'RAG (fallback)'
                }
                
        except Exception as e:
            logger.error(f"RAG sentiment analysis failed: {e}")
            analysis_time = (datetime.now() - start_time).total_seconds()
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'confidence_str': 'medium',
                'reasoning': f'Error: {str(e)}',
                'analysis_time': analysis_time,
                'similar_examples': [],
                'method': 'RAG (error)'
            }
    
    def create_rag_context(self, similar_examples: List[Dict]) -> str:
        """Create context string from similar examples"""
        if not similar_examples:
            return "No similar historical examples found."
        
        context_parts = []
        for i, example in enumerate(similar_examples, 1):
            outcome_desc = {
                'TRUE_BULLISH': 'Led to 30%+ price increase',
                'FALSE_PUMP': 'Was a false pump (no real gains)',
                'NEUTRAL': 'Had minimal price impact'
            }.get(example['outcome_type'], 'Unknown outcome')
            
            context_parts.append(
                f"{i}. {example['ticker']}: {example['headline'][:100]}..."
                f"\n   Outcome: {outcome_desc}"
                f"\n   Similarity: {example['similarity']:.3f}"
            )
        
        return "\n\n".join(context_parts)
    
    async def analyze_breaking_news(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze breaking news articles with RAG sentiment"""
        logger.info(f"🚀 Starting RAG sentiment analysis on {len(articles)} breaking news articles...")
        
        results = []
        
        for i, article in enumerate(articles, 1):
            logger.info(f"📊 Analyzing article {i}/{len(articles)}: {article['ticker']}")
            
            # Analyze with RAG
            sentiment_result = await self.analyze_rag_sentiment(article['content'])
            
            # Combine article info with sentiment result
            result = {
                'ticker': article['ticker'],
                'headline': article['headline'],
                'detected_at': article['detected_at'],
                'article_url': article['article_url'],
                **sentiment_result
            }
            
            results.append(result)
            
            # Log result
            confidence_display = f"{sentiment_result['confidence_str']} ({sentiment_result['confidence']:.2f})"
            logger.info(f"  📈 Result: {sentiment_result['action']} - {confidence_display}")
            if sentiment_result['similar_examples']:
                logger.info(f"    🧠 Used {len(sentiment_result['similar_examples'])} similar examples")
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of sentiment analysis results"""
        print("\n" + "="*80)
        print("🔥 BREAKING NEWS RAG SENTIMENT ANALYSIS SUMMARY")
        print("="*80)
        
        # Group by ticker
        ticker_results = {}
        for result in results:
            ticker = result['ticker']
            if ticker not in ticker_results:
                ticker_results[ticker] = []
            ticker_results[ticker].append(result)
        
        # Print by ticker
        for ticker, ticker_articles in ticker_results.items():
            print(f"\n📊 {ticker} ({len(ticker_articles)} articles)")
            print("-" * 50)
            
            for article in ticker_articles:
                # Format confidence display
                confidence_icon = "🔥" if article['confidence'] >= self.buy_high_threshold else "⚡" if article['confidence'] >= 0.5 else "💤"
                action_color = article['action']
                
                print(f"  {confidence_icon} {action_color} ({article['confidence_str']}) - {article['headline'][:80]}...")
                print(f"    📅 {article['detected_at']} | ⏱️ {article['analysis_time']:.2f}s")
                if article['similar_examples']:
                    print(f"    🧠 Based on {len(article['similar_examples'])} similar examples")
                print(f"    💭 {article['reasoning'][:100]}...")
                print()
        
        # Overall stats
        buy_high_count = sum(1 for r in results if r['action'] == 'BUY' and r['confidence'] >= self.buy_high_threshold)
        buy_any_count = sum(1 for r in results if r['action'] == 'BUY')
        hold_count = sum(1 for r in results if r['action'] == 'HOLD')
        sell_count = sum(1 for r in results if r['action'] == 'SELL')
        
        print("\n📈 OVERALL SUMMARY")
        print("-" * 30)
        print(f"🔥 BUY+high: {buy_high_count} articles ({buy_high_count/len(results):.1%})")
        print(f"📈 BUY (any): {buy_any_count} articles ({buy_any_count/len(results):.1%})")
        print(f"⏸️ HOLD: {hold_count} articles ({hold_count/len(results):.1%})")
        print(f"📉 SELL: {sell_count} articles ({sell_count/len(results):.1%})")
        print(f"⏱️ Avg analysis time: {sum(r['analysis_time'] for r in results)/len(results):.2f}s")
        
        print("\n" + "="*80)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Breaking News RAG Sentiment Analysis')
    parser.add_argument('--limit', type=int, default=20, help='Number of articles to analyze (default: 20)')
    parser.add_argument('--ticker', type=str, help='Filter by specific ticker (optional)')
    parser.add_argument('--hours-back', type=int, default=24, help='Hours back to look for articles (default: 24)')
    parser.add_argument('--buy-high-threshold', type=float, default=0.8, help='Confidence threshold for BUY+high (default: 0.8)')
    
    args = parser.parse_args()
    
    tester = BreakingNewsRAGTester(buy_high_threshold=args.buy_high_threshold)
    
    try:
        # Initialize test framework
        await tester.initialize()
        
        # Get breaking news articles
        articles = await tester.get_breaking_news_articles(
            limit=args.limit, 
            ticker=args.ticker, 
            hours_back=args.hours_back
        )
        
        if not articles:
            logger.error("No breaking news articles found!")
            return
        
        # Analyze articles with RAG
        results = await tester.analyze_breaking_news(articles)
        
        # Print summary
        tester.print_summary(results)
        
        logger.info("✅ Breaking news RAG analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis execution failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 