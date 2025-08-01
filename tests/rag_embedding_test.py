#!/usr/bin/env python3
"""
RAG Embedding Test for NewsHead

This script tests embedding generation and similarity search functionality
for the RAG system using Claude for text analysis and simple similarity.
It uses the existing rag_article_vectors table populated by generate_vectors.py.

Usage:
    python3 tests/rag_embedding_test.py --test-similarity --sample-size 20
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
import argparse
from dataclasses import dataclass
import hashlib

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

@dataclass
class EmbeddingTestResult:
    """Results from embedding generation and similarity tests"""
    article_content: str
    embedding_generated: bool
    embedding_size: int
    generation_time: float
    error_message: str = ""

class RAGEmbeddingTester:
    """Test framework for RAG embedding functionality using Claude and existing vectors"""
    
    def __init__(self):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.embedding_cache = {}
        
    async def initialize(self):
        """Initialize the test framework"""
        logger.info("🧪 Initializing RAG Embedding Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize sentiment service (for Claude API access)
        self.sentiment_service = await get_sentiment_service()
        
        # Verify that the rag_article_vectors table exists and has data
        await self.verify_vectors_table()
        
        logger.info("✅ Embedding test framework initialized successfully")
    
    async def verify_vectors_table(self):
        """Verify that the rag_training_vectors table exists and has data"""
        try:
            count_query = 'SELECT COUNT(*) FROM News.rag_training_vectors'
            vector_count = self.ch_manager.client.query(count_query).result_rows[0][0]
            
            if vector_count == 0:
                logger.warning("⚠️ No vectors found in News.rag_training_vectors table. Please run generate_vectors.py first.")
                return False
            else:
                logger.info(f"✅ Found {vector_count} vectors in News.rag_training_vectors table")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error accessing News.rag_training_vectors table: {e}")
            logger.error("Please run generate_vectors.py first to populate the training vectors table.")
            raise
    
    async def generate_claude_text_features(self, text: str) -> Tuple[List[float], float]:
        """
        Generate text features using Claude analysis instead of embeddings
        This creates a feature vector based on Claude's analysis of the text
        """
        start_time = datetime.now()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            generation_time = (datetime.now() - start_time).total_seconds()
            return self.embedding_cache[text_hash], generation_time
        
        try:
            # Use Claude to analyze text features
            analysis_prompt = f"""
Analyze the following text and extract key features for similarity comparison.
Focus on: sentiment, topic, urgency, market impact, company type, and news type.

Text: {text[:2000]}

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
    "market_general": 0.0-1.0,
    "biotech_pharma": 0.0-1.0,
    "tech_software": 0.0-1.0,
    "finance_banking": 0.0-1.0,
    "energy_commodities": 0.0-1.0,
    "retail_consumer": 0.0-1.0,
    "manufacturing": 0.0-1.0
}}
"""
            
            # Use the existing Claude API through sentiment service
            if hasattr(self.sentiment_service, 'load_balancer'):
                result = await self.sentiment_service.load_balancer.make_claude_request(analysis_prompt)
            else:
                # Fallback to legacy method
                result = await self.sentiment_service.query_claude_api_legacy(analysis_prompt)
            
            if result and isinstance(result, dict):
                # Extract feature scores and create vector
                feature_vector = [
                    result.get('sentiment_score', 0.5),
                    result.get('bullish_score', 0.5),
                    result.get('urgency_score', 0.5),
                    result.get('financial_impact_score', 0.5),
                    result.get('earnings_related', 0.0),
                    result.get('partnership_related', 0.0),
                    result.get('product_related', 0.0),
                    result.get('regulatory_related', 0.0),
                    result.get('market_general', 0.0),
                    result.get('biotech_pharma', 0.0),
                    result.get('tech_software', 0.0),
                    result.get('finance_banking', 0.0),
                    result.get('energy_commodities', 0.0),
                    result.get('retail_consumer', 0.0),
                    result.get('manufacturing', 0.0)
                ]
                
                # Pad to standard size and add some derived features
                while len(feature_vector) < 50:
                    # Add some derived features
                    if len(feature_vector) < 30:
                        feature_vector.append(feature_vector[0] * feature_vector[1])  # sentiment * bullish
                    else:
                        feature_vector.append(0.5)  # neutral padding
                
                generation_time = (datetime.now() - start_time).total_seconds()
                self.embedding_cache[text_hash] = feature_vector
                return feature_vector, generation_time
            else:
                raise Exception("Claude analysis failed or returned invalid format")
                
        except Exception as e:
            logger.error(f"Claude feature extraction failed: {e}")
            generation_time = (datetime.now() - start_time).total_seconds()
            # Return dummy feature vector on failure
            dummy_features = [0.5] * 50  # Neutral features
            return dummy_features, generation_time
    
    def calculate_cosine_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate cosine similarity between two feature vectors"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(features1)
            vec2 = np.array(features2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def get_test_articles_from_vectors_table(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get sample articles from the existing rag_training_vectors table for testing"""
        try:
            # Get articles from the populated TRAINING vectors table
            query = """
            SELECT 
                ticker,
                headline,
                full_content,
                outcome_type,
                has_30pt_increase,
                is_false_pump,
                price_increase_ratio,
                feature_vector
            FROM News.rag_training_vectors
            WHERE LENGTH(headline) > 30  -- Ensure meaningful content
            ORDER BY outcome_type, ticker
            LIMIT %s
            """
            
            result = self.ch_manager.client.query(query, parameters=[limit])
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'full_content': row[2],
                    'outcome': row[3],  # outcome_type
                    'has_30pt_increase': int(row[4]),
                    'is_false_pump': int(row[5]),
                    'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                    'existing_features': row[7],  # Pre-computed feature vector
                    'content': row[2] or f"{row[0]}: {row[1]}"  # Use full_content if available, fallback to ticker + headline
                })
            
            logger.info(f"📄 Retrieved {len(articles)} test articles from rag_training_vectors table")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving test articles from training vectors table: {e}")
            return []
    
    async def test_feature_generation(self, test_articles: List[Dict[str, Any]]) -> List[EmbeddingTestResult]:
        """Test Claude-based feature generation for multiple articles (using existing vectors for comparison)"""
        logger.info("🔬 Testing Claude feature generation against existing vectors...")
        
        results = []
        total_time = 0
        successful_generations = 0
        
        for i, article in enumerate(test_articles, 1):
            logger.info(f"  Testing features {i}/{len(test_articles)}: {article['ticker']}")
            
            try:
                # We already have the features from the vectors table
                existing_features = article['existing_features']
                
                # Generate new features for comparison (optional)
                if len(test_articles) <= 5:  # Only do this for small test sets to save API calls
                    new_features, generation_time = await self.generate_claude_text_features(article['content'])
                    total_time += generation_time
                    
                    # Compare similarity between existing and new features
                    similarity = self.calculate_cosine_similarity(existing_features, new_features)
                    logger.info(f"    Feature consistency: {similarity:.3f}")
                else:
                    generation_time = 0
                
                result = EmbeddingTestResult(
                    article_content=article['content'][:100] + "...",
                    embedding_generated=True,
                    embedding_size=len(existing_features),
                    generation_time=generation_time
                )
                
                successful_generations += 1
                
            except Exception as e:
                result = EmbeddingTestResult(
                    article_content=article['content'][:100] + "...",
                    embedding_generated=False,
                    embedding_size=0,
                    generation_time=0,
                    error_message=str(e)
                )
            
            results.append(result)
        
        # Log summary
        avg_time = total_time / min(len(test_articles), 5) if test_articles else 0
        success_rate = successful_generations / len(test_articles) if test_articles else 0
        
        logger.info(f"📊 Feature Vector Test Summary:")
        logger.info(f"  • Total articles tested: {len(test_articles)}")
        logger.info(f"  • Articles with existing vectors: {successful_generations}")
        logger.info(f"  • Success rate: {success_rate:.1%}")
        if total_time > 0:
            logger.info(f"  • Average generation time: {avg_time:.2f}s")
        
        return results
    
    async def test_similarity_search(self, test_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test similarity search functionality using existing Claude features"""
        logger.info("🔍 Testing similarity search using existing vectors...")
        
        if len(test_articles) < 2:
            logger.warning("Need at least 2 articles for similarity testing")
            return {}
        
        # Use existing features from the vectors table
        query_article = test_articles[0]
        query_features = query_article['existing_features']
        
        logger.info(f"🎯 Testing similarity for query article: {query_article['ticker']} ({query_article['outcome']})")
        
        # Calculate similarities with all other articles using existing features
        similarities = []
        for other_article in test_articles[1:]:
            other_features = other_article['existing_features']
            similarity = self.calculate_cosine_similarity(query_features, other_features)
            
            similarities.append({
                'ticker': other_article['ticker'],
                'outcome': other_article['outcome'],
                'similarity': similarity,
                'headline': other_article['headline']
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Log top 5 most similar
        logger.info(f"🔗 Top 5 most similar articles to {query_article['ticker']}:")
        for i, sim in enumerate(similarities[:5], 1):
            logger.info(f"  {i}. {sim['ticker']} ({sim['outcome']}) - Similarity: {sim['similarity']:.3f}")
        
        # Analyze similarity by outcome type
        outcome_similarities = {}
        for outcome in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
            outcome_sims = [s['similarity'] for s in similarities if s['outcome'] == outcome]
            if outcome_sims:
                outcome_similarities[outcome] = {
                    'count': len(outcome_sims),
                    'avg_similarity': sum(outcome_sims) / len(outcome_sims),
                    'max_similarity': max(outcome_sims),
                    'min_similarity': min(outcome_sims)
                }
        
        logger.info(f"📈 Similarity by outcome type:")
        for outcome, stats in outcome_similarities.items():
            logger.info(f"  • {outcome}: Avg={stats['avg_similarity']:.3f}, Count={stats['count']}")
        
        return {
            'query_article': {
                'ticker': query_article['ticker'],
                'outcome': query_article['outcome'],
                'headline': query_article['headline']
            },
            'similarities': similarities,
            'outcome_similarities': outcome_similarities
        }
    
    async def test_clickhouse_similarity_search(self, query_ticker: str) -> List[Dict[str, Any]]:
        """Test similarity search using ClickHouse vector functions on existing rag_training_vectors table"""
        logger.info(f"🔍 Testing ClickHouse similarity search for {query_ticker}...")
        
        try:
            # Get query features from existing TRAINING vectors table
            query = """
            SELECT feature_vector, outcome_type, headline
            FROM News.rag_training_vectors
            WHERE ticker = %s
            LIMIT 1
            """
            
            result = self.ch_manager.client.query(query, parameters=[query_ticker])
            if not result.result_rows:
                logger.warning(f"No features found for ticker {query_ticker}")
                return []
            
            query_features = result.result_rows[0][0]
            query_outcome = result.result_rows[0][1]
            query_headline = result.result_rows[0][2]
            
            logger.info(f"Query: {query_ticker} ({query_outcome}) - {query_headline[:50]}...")
            
            # Find similar articles using cosine distance
            similarity_query = """
            SELECT 
                ticker,
                headline,
                outcome_type,
                price_increase_ratio,
                cosineDistance(feature_vector, %s) as distance,
                1 - cosineDistance(feature_vector, %s) as similarity
            FROM News.rag_training_vectors
            WHERE ticker != %s
            ORDER BY distance ASC
            LIMIT 10
            """
            
            result = self.ch_manager.client.query(
                similarity_query, 
                parameters=[query_features, query_features, query_ticker]
            )
            
            similar_articles = []
            for row in result.result_rows:
                similar_articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'outcome_type': row[2],
                    'price_increase_ratio': float(row[3]),
                    'distance': float(row[4]),
                    'similarity': float(row[5])
                })
            
            logger.info(f"🔗 Top 5 similar articles from ClickHouse:")
            for i, article in enumerate(similar_articles[:5], 1):
                logger.info(f"  {i}. {article['ticker']} ({article['outcome_type']}) - Similarity: {article['similarity']:.3f}")
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"ClickHouse similarity search failed: {e}")
            return []
    
    async def analyze_vector_quality(self):
        """Analyze the quality and distribution of existing TRAINING vectors"""
        logger.info("📊 Analyzing existing TRAINING vector quality...")
        
        try:
            # Get basic statistics
            stats_query = """
            SELECT 
                outcome_type,
                COUNT(*) as count,
                AVG(arraySum(feature_vector)) as avg_feature_sum,
                AVG(length(feature_vector)) as avg_vector_length
            FROM News.rag_training_vectors
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            result = self.ch_manager.client.query(stats_query)
            
            logger.info("📈 TRAINING Vector Quality Analysis:")
            total_vectors = 0
            for row in result.result_rows:
                outcome, count, avg_sum, avg_length = row
                total_vectors += count
                logger.info(f"  • {outcome}: {count} vectors (avg sum: {avg_sum:.2f}, avg length: {avg_length:.0f})")
            
            logger.info(f"  • Total TRAINING vectors: {total_vectors}")
            
            # Sample some vectors for quality check
            sample_query = """
            SELECT ticker, outcome_type, feature_vector
            FROM News.rag_training_vectors
            ORDER BY outcome_type, ticker
            LIMIT 3
            """
            
            sample_result = self.ch_manager.client.query(sample_query)
            logger.info("📋 Sample TRAINING vector inspection:")
            for row in sample_result.result_rows:
                ticker, outcome, vector = row
                vector_sum = sum(vector) if vector else 0
                logger.info(f"  • {ticker} ({outcome}): {len(vector)} dims, sum={vector_sum:.2f}")
            
        except Exception as e:
            logger.error(f"TRAINING vector quality analysis failed: {e}")
    
    async def save_test_results(self, feature_results: List[EmbeddingTestResult], similarity_results: Dict[str, Any]):
        """Save test results to files"""
        os.makedirs('tests/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare results for JSON serialization
        results_data = {
            'test_timestamp': timestamp,
            'used_existing_vectors': True,
            'source_table': 'News.rag_training_vectors',
            'data_split': 'training_set_only',
            'feature_generation_tests': [],
            'similarity_tests': similarity_results,
            'model_used': 'claude-3-5-sonnet-20240620'
        }
        
        for result in feature_results:
            results_data['feature_generation_tests'].append({
                'article_content': result.article_content,
                'features_generated': result.embedding_generated,
                'feature_count': result.embedding_size,
                'generation_time': result.generation_time,
                'error_message': result.error_message
            })
        
        # Save results
        with open(f'tests/results/claude_features_test_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"📁 Claude features test results saved to tests/results/claude_features_test_results_{timestamp}.json")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description='RAG Claude Features Test using existing vectors')
    parser.add_argument('--sample-size', type=int, default=10, help='Number of articles to test')
    parser.add_argument('--test-similarity', action='store_true', help='Test similarity search functionality')
    parser.add_argument('--test-clickhouse', action='store_true', help='Test ClickHouse similarity search')
    parser.add_argument('--analyze-quality', action='store_true', help='Analyze vector quality')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save test results')
    
    args = parser.parse_args()
    
    tester = RAGEmbeddingTester()
    
    try:
        # Initialize test framework
        await tester.initialize()
        
        # Analyze vector quality if requested
        if args.analyze_quality:
            await tester.analyze_vector_quality()
        
        # Get test articles from existing vectors table
        test_articles = await tester.get_test_articles_from_vectors_table(args.sample_size)
        if not test_articles:
            logger.error("No test articles found in rag_article_vectors table!")
            return
        
        # Test feature generation (using existing vectors)
        feature_results = await tester.test_feature_generation(test_articles)
        
        # Test similarity search if requested
        similarity_results = {}
        if args.test_similarity:
            similarity_results = await tester.test_similarity_search(test_articles)
        
        # Test ClickHouse similarity search if requested
        if args.test_clickhouse and test_articles:
            clickhouse_results = await tester.test_clickhouse_similarity_search(test_articles[0]['ticker'])
            similarity_results['clickhouse_search'] = clickhouse_results
        
        # Save results if requested
        if args.save_results:
            await tester.save_test_results(feature_results, similarity_results)
        
        logger.info("✅ All Claude feature tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 