#!/usr/bin/env python3
"""
Breaking News RAG Sentiment Analysis Test

This script reads articles from the breaking_news table and analyzes them
using the optimized RAG-enhanced sentiment analysis system with local embeddings.

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

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedBreakingNewsRAGTester:
    """Test optimized RAG sentiment analysis on breaking news articles"""
    
    def __init__(self, buy_high_threshold: float = 0.8):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.embedding_model = None
        self.buy_high_threshold = buy_high_threshold
        self.confidence_map = {'low': 0.55, 'medium': 0.7, 'high': 0.95}
        
    async def initialize(self):
        """Initialize the optimized breaking news tester"""
        logger.info("🚀 Initializing Optimized Breaking News RAG Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize sentiment service
        self.sentiment_service = await get_sentiment_service()
        
        # Initialize local embedding model
        await self.initialize_embedding_model()
        
        # Verify training vectors exist for RAG functionality
        await self.verify_training_vectors()
        
        logger.info("✅ Optimized breaking news RAG test framework initialized successfully")
    
    async def initialize_embedding_model(self):
        """Initialize local fin-E5 embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.error("❌ sentence-transformers required for local embeddings. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers package required")
        
        try:
            # Load embedding model for financial text embeddings
            logger.info("📥 Loading embedding model...")
            start_time = datetime.now()
            
            # Try E5 models that match existing vector dimensions
            try:
                # Use E5-large which produces 1024 dimensions to match existing vectors
                self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
                model_name = "e5-large-v2 (1024-dim)"
            except:
                try:
                    # Alternative: multilingual E5 large
                    self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
                    model_name = "multilingual-e5-large (1024-dim)"
                except:
                    # Final fallback - will need vector regeneration
                    self.embedding_model = SentenceTransformer('intfloat/e5-base-v2')
                    model_name = "e5-base-v2 (512-dim) - WARNING: Dimension mismatch!"
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Loaded {model_name} embedding model in {load_time:.2f}s")
            
            # Test embedding generation speed
            test_text = "Test article for embedding speed"
            start_time = datetime.now()
            test_embedding = self.embedding_model.encode([test_text])
            embedding_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"⚡ Embedding generation speed: {embedding_time*1000:.1f}ms per article")
            logger.info(f"📏 Embedding dimension: {len(test_embedding[0])}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
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
    
    async def get_similar_training_examples(self, query_content: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar examples from training set using optimized vector search"""
        if not self.vectors_table:
            return []
            
        try:
            # Generate embedding for the query content using local model
            embedding_start = datetime.now()
            
            # Prepare text for E5 model (add query prefix for better performance)
            query_text = f"query: {query_content[:1000]}"  # Limit to 1000 chars for speed
            
            # Generate embedding using local model
            query_embedding = self.embedding_model.encode([query_text])[0].tolist()
            embedding_time = (datetime.now() - embedding_start).total_seconds()
            
            if not query_embedding:
                return []
            
            # Choose query based on available vectors table
            if hasattr(self, 'use_training_filter') and self.use_training_filter:
                # Use existing vectors filtered by training set with similarity threshold
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
                WHERE cosineDistance(v.feature_vector, %s) < 0.5
                ORDER BY distance ASC
                LIMIT %s
                """
                params = [query_embedding, query_embedding, query_embedding, top_k]
            else:
                # Use dedicated training vectors table with similarity threshold
                similarity_query = """
                SELECT 
                    ticker,
                    headline,
                    outcome_type,
                    price_increase_ratio,
                    cosineDistance(feature_vector, %s) as distance,
                    1 - cosineDistance(feature_vector, %s) as similarity
                FROM News.rag_training_vectors
                WHERE cosineDistance(feature_vector, %s) < 0.5
                ORDER BY distance ASC
                LIMIT %s
                """
                params = [query_embedding, query_embedding, query_embedding, top_k]
            
            result = self.ch_manager.client.query(similarity_query, parameters=params)
            
            similar_examples = []
            for row in result.result_rows:
                # Only include examples with >50% similarity (distance < 0.5)
                similarity = float(row[5])
                if similarity > 0.5:
                    similar_examples.append({
                        'ticker': row[0],
                        'headline': row[1],
                        'outcome_type': row[2],
                        'price_increase_ratio': float(row[3]),
                        'distance': float(row[4]),
                        'similarity': similarity
                    })
            
            logger.debug(f"Found {len(similar_examples)} similar examples in {embedding_time*1000:.1f}ms")
            return similar_examples
            
        except Exception as e:
            logger.error(f"Error finding similar examples: {e}")
            return []
    
    async def analyze_rag_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment using optimized RAG method"""
        start_time = datetime.now()
        
        try:
            # Fast similarity search with local embeddings
            similar_examples = await self.get_similar_training_examples(content, top_k=3)
            
            # Create optimized RAG context
            rag_context = self.create_optimized_rag_context(similar_examples)
            
            # Single optimized LLM call
            llm_start = datetime.now()
            
            # Clean content for JSON safety
            clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
            
            # Optimized prompt - shorter and more focused
            if similar_examples:
                rag_prompt = f"""Based on these {len(similar_examples)} similar historical examples:

{rag_context}

Analyze this new article: {clean_content[:2000]}

TRADING DECISION RULES:
- If similar examples are mostly TRUE_BULLISH (led to 30%+ gains): Recommend BUY with HIGH confidence
- If mixed results but some TRUE_BULLISH: BUY with MEDIUM confidence  
- If mostly FALSE_PUMP/NEUTRAL: HOLD
- Be AGGRESSIVE on opportunities - missing TRUE_BULLISH is worse than catching FALSE_PUMP

Respond with JSON: {{"action": "BUY/HOLD/SELL", "confidence": "high/medium/low", "reasoning": "brief explanation"}}"""
            else:
                # Fallback to traditional-style analysis if no similar examples
                rag_prompt = f"""Analyze this financial news for trading potential: {clean_content[:2000]}

Focus on: partnerships, clinical trials, earnings, acquisitions, product launches.

Respond with JSON: {{"action": "BUY/HOLD/SELL", "confidence": "high/medium/low", "reasoning": "brief explanation"}}"""
            
            result = await self.sentiment_service.load_balancer.make_claude_request(rag_prompt)
            llm_time = (datetime.now() - llm_start).total_seconds()
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
                    'method': 'Optimized RAG',
                    'llm_time': llm_time,
                    'total_time': analysis_time
                }
            else:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'confidence_str': 'medium',
                    'reasoning': 'Analysis failed',
                    'analysis_time': analysis_time,
                    'similar_examples': similar_examples,
                    'method': 'Optimized RAG (fallback)',
                    'llm_time': llm_time,
                    'total_time': analysis_time
                }
                
        except Exception as e:
            logger.error(f"Optimized RAG sentiment analysis failed: {e}")
            analysis_time = (datetime.now() - start_time).total_seconds()
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'confidence_str': 'medium',
                'reasoning': f'Error: {str(e)}',
                'analysis_time': analysis_time,
                'similar_examples': [],
                'method': 'Optimized RAG (error)',
                'llm_time': 0.0,
                'total_time': analysis_time
            }
    
    def create_optimized_rag_context(self, similar_examples: List[Dict]) -> str:
        """Create optimized context string from similar examples"""
        if not similar_examples:
            return "No similar historical examples found."
        
        context_parts = []
        outcome_counts = {'TRUE_BULLISH': 0, 'FALSE_PUMP': 0, 'NEUTRAL': 0}
        
        for i, example in enumerate(similar_examples, 1):
            outcome = example['outcome_type']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            outcome_desc = {
                'TRUE_BULLISH': 'SUCCESS: 30%+ gain',
                'FALSE_PUMP': 'FAILED: False pump',
                'NEUTRAL': 'NEUTRAL: <5% move'
            }.get(outcome, 'Unknown')
            
            context_parts.append(
                f"{i}. {example['ticker']}: {example['headline'][:80]}... → {outcome_desc} (sim: {example['similarity']:.2f})"
            )
        
        # Add pattern summary
        total = len(similar_examples)
        if outcome_counts['TRUE_BULLISH'] >= total * 0.6:
            pattern = "STRONG BUY SIGNAL: Most similar examples succeeded"
        elif outcome_counts['TRUE_BULLISH'] >= total * 0.4:
            pattern = "MIXED SIGNAL: Some similar examples succeeded"
        else:
            pattern = "WEAK SIGNAL: Few similar examples succeeded"
        
        return f"PATTERN: {pattern}\n\n" + "\n".join(context_parts)
    
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
                if 'llm_time' in article and article['llm_time'] > 0:
                    print(f"    ⚡ Performance: LLM={article['llm_time']*1000:.0f}ms, Total={article['total_time']*1000:.0f}ms")
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
    
    tester = OptimizedBreakingNewsRAGTester(buy_high_threshold=args.buy_high_threshold)
    
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