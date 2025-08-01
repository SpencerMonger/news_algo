#!/usr/bin/env python3
"""
RAG Similarity Search Test for NewsHead

This script tests the similarity search functionality using the labeled data
from price_movement_analysis to validate that similar articles with known outcomes
are properly retrieved for RAG context enhancement.

Usage:
    python3 tests/rag_similarity_test.py --test-outcome-correlation --validate-precision
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimilarityTestResult:
    """Results from similarity search testing"""
    query_ticker: str
    query_outcome: str
    query_headline: str
    similar_articles: List[Dict[str, Any]]
    outcome_precision: Dict[str, float]  # Precision for each outcome type
    top_k_accuracy: float  # How many of top K are same outcome
    cross_outcome_similarity: Dict[str, float]  # Avg similarity across different outcomes

class RAGSimilarityTester:
    """Test framework for RAG similarity search functionality"""
    
    def __init__(self):
        self.ch_manager = ClickHouseManager()
        self.test_results: List[SimilarityTestResult] = []
    
    async def initialize(self):
        """Initialize the test framework"""
        logger.info("🧪 Initializing RAG Similarity Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        logger.info("✅ Similarity test framework initialized successfully")
    
    async def get_labeled_articles_by_outcome(self, outcome_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get articles filtered by specific outcome type"""
        logger.info(f"📄 Fetching {limit} articles with outcome: {outcome_type}")
        
        try:
            # Use the balanced training dataset instead of the imbalanced original
            query = f"""
            SELECT 
                ticker,
                headline,
                article_url,
                has_30pt_increase,
                is_false_pump,
                price_increase_ratio,
                max_price_ratio,
                published_est
            FROM News.rag_training_dataset
            WHERE outcome_type = %s
            AND LENGTH(headline) > 30  -- Ensure meaningful content
            ORDER BY selection_priority DESC  -- Best examples first
            LIMIT %s
            """
            
            result = self.ch_manager.client.query(query, parameters=[outcome_type, limit])
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'article_url': row[2] or '',
                    'has_30pt_increase': bool(row[3]),
                    'is_false_pump': bool(row[4]),
                    'price_increase_ratio': float(row[5]) if row[5] else 0.0,
                    'max_price_ratio': float(row[6]) if row[6] else 0.0,
                    'published_est': row[7],
                    'outcome_type': outcome_type,
                    'content': f"{row[0]}: {row[1]}"  # Simple content for testing
                })
            
            logger.info(f"  Retrieved {len(articles)} {outcome_type} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching {outcome_type} articles: {e}")
            return []
    
    async def find_similar_articles_claude(self, query_article: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar articles using Claude feature extraction and cosine similarity
        """
        try:
            # Import the embedding tester to use its Claude feature extraction
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from rag_embedding_test import RAGEmbeddingTester
            
            # Initialize Claude feature extractor
            feature_extractor = RAGEmbeddingTester()
            await feature_extractor.initialize()
            
            query_ticker = query_article['ticker']
            query_content = query_article['content']
            
            logger.info(f"🔬 Generating Claude features for query article: {query_ticker}")
            
            # Generate features for the query article using Claude
            query_features, _ = await feature_extractor.generate_claude_text_features(query_content)
            
            # Get all other articles from the balanced dataset (excluding the query article)
            all_articles_query = """
            SELECT 
                ticker,
                headline,
                has_30pt_increase,
                is_false_pump,
                price_increase_ratio,
                max_price_ratio,
                published_est,
                outcome_type,
                full_content
            FROM News.rag_training_dataset
            WHERE ticker != %s
            ORDER BY selection_priority DESC
            LIMIT 50
            """
            
            result = self.ch_manager.client.query(all_articles_query, parameters=[query_ticker])
            
            logger.info(f"🔍 Comparing against {len(result.result_rows)} candidate articles...")
            
            similar_articles = []
            
            # Generate features and calculate similarity for each candidate article
            for i, row in enumerate(result.result_rows):
                if i % 10 == 0:
                    logger.info(f"  Processing article {i+1}/{len(result.result_rows)}...")
                
                candidate_ticker = row[0]
                candidate_headline = row[1]
                candidate_content = f"{candidate_ticker}: {candidate_headline}"
                outcome_type = row[7]
                
                # Generate Claude features for candidate article
                candidate_features, _ = await feature_extractor.generate_claude_text_features(candidate_content)
                
                # Calculate cosine similarity between query and candidate features
                similarity_score = feature_extractor.calculate_cosine_similarity(query_features, candidate_features)
                
                similar_articles.append({
                    'ticker': candidate_ticker,
                    'headline': candidate_headline,
                    'outcome_type': outcome_type,
                    'has_30pt_increase': bool(row[2]),
                    'is_false_pump': bool(row[3]),
                    'price_increase_ratio': float(row[4]) if row[4] else 0.0,
                    'max_price_ratio': float(row[5]) if row[5] else 0.0,
                    'published_est': row[6],
                    'similarity_score': similarity_score
                })
            
            # Sort by similarity score (descending) and return top results
            similar_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Cleanup
            await feature_extractor.cleanup()
            
            logger.info(f"✅ Found {len(similar_articles)} similar articles, returning top {limit}")
            return similar_articles[:limit]
            
        except Exception as e:
            logger.error(f"Error in Claude-based similarity search: {e}")
            # Fallback to simple keyword matching
            return await self.find_similar_articles_simple(query_article, limit)
    
    async def find_similar_articles_simple(self, query_article: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar articles using simple keyword matching
        (Placeholder for actual embedding-based similarity)
        """
        try:
            query_ticker = query_article['ticker']
            query_headline = query_article['headline'].lower()
            
            # Extract keywords from headline (simple approach)
            keywords = [word for word in query_headline.split() if len(word) > 3][:5]
            
            # Build search conditions
            keyword_conditions = []
            parameters = []
            
            for keyword in keywords:
                keyword_conditions.append("position(lower(headline), %s) > 0")
                parameters.append(keyword)
            
            # Combine conditions
            if keyword_conditions:
                keyword_condition = " OR ".join(keyword_conditions)
            else:
                keyword_condition = "1=1"  # Fallback
            
            # Use balanced training dataset
            query = f"""
            SELECT 
                ticker,
                headline,
                has_30pt_increase,
                is_false_pump,
                price_increase_ratio,
                max_price_ratio,
                published_est,
                outcome_type
            FROM News.rag_training_dataset
            WHERE ticker != %s AND ({keyword_condition})
            ORDER BY selection_priority DESC  -- Best examples first
            LIMIT %s
            """
            
            parameters = [query_ticker] + parameters + [limit]
            result = self.ch_manager.client.query(query, parameters=parameters)
            
            similar_articles = []
            for row in result.result_rows:
                # Get outcome type directly from the balanced dataset
                outcome_type = row[7]  # outcome_type column
                
                # Calculate simple similarity score (number of matching keywords)
                similarity_score = sum(1 for keyword in keywords if keyword in row[1].lower()) / len(keywords) if keywords else 0
                
                similar_articles.append({
                    'ticker': row[0],
                    'headline': row[1], 
                    'outcome_type': outcome_type,
                    'has_30pt_increase': bool(row[2]),
                    'is_false_pump': bool(row[3]),
                    'price_increase_ratio': float(row[4]) if row[4] else 0.0,
                    'max_price_ratio': float(row[5]) if row[5] else 0.0,
                    'published_est': row[6],
                    'similarity_score': similarity_score
                })
            
            # Sort by similarity score (descending)
            similar_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {e}")
            return []
    
    async def test_similarity_for_outcome_type(self, outcome_type: str, sample_size: int = 10) -> List[SimilarityTestResult]:
        """Test similarity search for articles of a specific outcome type"""
        logger.info(f"🔍 Testing similarity search for outcome type: {outcome_type}")
        
        # Get sample articles of this outcome type
        query_articles = await self.get_labeled_articles_by_outcome(outcome_type, sample_size)
        if not query_articles:
            logger.warning(f"No articles found for outcome type: {outcome_type}")
            return []
        
        results = []
        
        for i, query_article in enumerate(query_articles, 1):
            logger.info(f"  Testing article {i}/{len(query_articles)}: {query_article['ticker']} - {query_article['headline'][:50]}...")
            
            try:
                # Find similar articles
                similar_articles = await self.find_similar_articles_claude(query_article, limit=10)
                
                if not similar_articles:
                    logger.warning(f"No similar articles found for {query_article['ticker']}")
                    continue
                
                # Calculate outcome precision (how many similar articles have same outcome)
                outcome_counts = {'TRUE_BULLISH': 0, 'FALSE_PUMP': 0, 'NEUTRAL': 0}
                for article in similar_articles:
                    outcome_counts[article['outcome_type']] += 1
                
                total_similar = len(similar_articles)
                outcome_precision = {
                    outcome: count / total_similar if total_similar > 0 else 0.0 
                    for outcome, count in outcome_counts.items()
                }
                
                # Calculate top-K accuracy (top 3 articles with same outcome)
                top_k = 3
                top_k_same_outcome = sum(1 for article in similar_articles[:top_k] if article['outcome_type'] == outcome_type)
                top_k_accuracy = top_k_same_outcome / min(top_k, total_similar) if total_similar > 0 else 0.0
                
                # Calculate cross-outcome similarity scores
                cross_outcome_similarity = {}
                for outcome in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
                    outcome_articles = [a for a in similar_articles if a['outcome_type'] == outcome]
                    if outcome_articles:
                        avg_similarity = sum(a['similarity_score'] for a in outcome_articles) / len(outcome_articles)
                        cross_outcome_similarity[outcome] = avg_similarity
                    else:
                        cross_outcome_similarity[outcome] = 0.0
                
                # Create test result
                test_result = SimilarityTestResult(
                    query_ticker=query_article['ticker'],
                    query_outcome=outcome_type,
                    query_headline=query_article['headline'],
                    similar_articles=similar_articles,
                    outcome_precision=outcome_precision,
                    top_k_accuracy=top_k_accuracy,
                    cross_outcome_similarity=cross_outcome_similarity
                )
                
                results.append(test_result)
                
                # Log key metrics
                logger.info(f"    Same outcome precision: {outcome_precision[outcome_type]:.2f}")
                logger.info(f"    Top-{top_k} accuracy: {top_k_accuracy:.2f}")
                
            except Exception as e:
                logger.error(f"Error testing similarity for {query_article['ticker']}: {e}")
                continue
        
        return results
    
    async def test_outcome_correlation(self) -> Dict[str, Any]:
        """Test how well similarity search correlates with outcome types"""
        logger.info("📊 Testing outcome correlation in similarity search...")
        
        all_results = []
        
        # Test each outcome type
        for outcome_type in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
            outcome_results = await self.test_similarity_for_outcome_type(outcome_type, sample_size=5)
            all_results.extend(outcome_results)
        
        if not all_results:
            logger.error("No test results generated!")
            return {}
        
        # Analyze overall correlation
        outcome_analysis = {}
        
        for outcome_type in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
            outcome_results = [r for r in all_results if r.query_outcome == outcome_type]
            
            if outcome_results:
                # Calculate average metrics for this outcome type
                avg_same_outcome_precision = sum(r.outcome_precision[outcome_type] for r in outcome_results) / len(outcome_results)
                avg_top_k_accuracy = sum(r.top_k_accuracy for r in outcome_results) / len(outcome_results)
                
                # Calculate cross-outcome contamination
                other_outcomes = [ot for ot in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL'] if ot != outcome_type]
                avg_contamination = sum(
                    sum(r.outcome_precision[other_outcome] for other_outcome in other_outcomes) 
                    for r in outcome_results
                ) / len(outcome_results)
                
                outcome_analysis[outcome_type] = {
                    'sample_size': len(outcome_results),
                    'avg_same_outcome_precision': avg_same_outcome_precision,
                    'avg_top_k_accuracy': avg_top_k_accuracy,
                    'avg_contamination': avg_contamination,
                    'purity_score': avg_same_outcome_precision / (avg_same_outcome_precision + avg_contamination) if (avg_same_outcome_precision + avg_contamination) > 0 else 0
                }
                
                logger.info(f"📈 {outcome_type} Analysis:")
                logger.info(f"  • Same outcome precision: {avg_same_outcome_precision:.3f}")
                logger.info(f"  • Top-K accuracy: {avg_top_k_accuracy:.3f}")
                logger.info(f"  • Contamination rate: {avg_contamination:.3f}")
                logger.info(f"  • Purity score: {outcome_analysis[outcome_type]['purity_score']:.3f}")
        
        # Overall system metrics
        overall_precision = sum(r.outcome_precision[r.query_outcome] for r in all_results) / len(all_results)
        overall_top_k_accuracy = sum(r.top_k_accuracy for r in all_results) / len(all_results)
        
        logger.info(f"🎯 Overall System Performance:")
        logger.info(f"  • Overall precision: {overall_precision:.3f}")
        logger.info(f"  • Overall top-K accuracy: {overall_top_k_accuracy:.3f}")
        
        return {
            'overall_metrics': {
                'total_tests': len(all_results),
                'overall_precision': overall_precision,
                'overall_top_k_accuracy': overall_top_k_accuracy
            },
            'outcome_analysis': outcome_analysis,
            'detailed_results': [
                {
                    'query_ticker': r.query_ticker,
                    'query_outcome': r.query_outcome,  
                    'query_headline': r.query_headline,
                    'outcome_precision': r.outcome_precision,
                    'top_k_accuracy': r.top_k_accuracy,
                    'cross_outcome_similarity': r.cross_outcome_similarity,
                    'similar_articles_count': len(r.similar_articles)
                }
                for r in all_results
            ]
        }
    
    async def validate_rag_precision(self) -> Dict[str, Any]:
        """Validate that RAG will provide high-quality historical context"""
        logger.info("✅ Validating RAG precision for trading decisions...")
        
        # Get articles that should produce BUY+high recommendations (TRUE_BULLISH)
        true_bullish_articles = await self.get_labeled_articles_by_outcome('TRUE_BULLISH', limit=10)
        
        # Get articles that should be avoided (FALSE_PUMP)
        false_pump_articles = await self.get_labeled_articles_by_outcome('FALSE_PUMP', limit=10)
        
        validation_results = {
            'true_bullish_validation': [],
            'false_pump_validation': [],
            'cross_contamination_analysis': {}
        }
        
        # Test TRUE_BULLISH articles - should find similar TRUE_BULLISH examples
        logger.info("🎯 Testing TRUE_BULLISH article similarity...")
        for article in true_bullish_articles[:5]:  # Test first 5
            similar = await self.find_similar_articles_claude(article, limit=5)
            
            # Count how many similar articles are also TRUE_BULLISH
            true_bullish_similar = [a for a in similar if a['outcome_type'] == 'TRUE_BULLISH']
            false_pump_similar = [a for a in similar if a['outcome_type'] == 'FALSE_PUMP']
            
            precision = len(true_bullish_similar) / len(similar) if similar else 0
            contamination = len(false_pump_similar) / len(similar) if similar else 0
            
            validation_results['true_bullish_validation'].append({
                'ticker': article['ticker'],
                'headline': article['headline'][:50] + "...",
                'similar_count': len(similar),
                'true_bullish_similar': len(true_bullish_similar),
                'false_pump_contamination': len(false_pump_similar),
                'precision': precision,
                'contamination_rate': contamination
            })
            
            logger.info(f"  {article['ticker']}: {len(true_bullish_similar)}/{len(similar)} similar are TRUE_BULLISH (precision: {precision:.2f})")
        
        # Test FALSE_PUMP articles - should find patterns that help avoid them
        logger.info("⚠️ Testing FALSE_PUMP article similarity...")
        for article in false_pump_articles[:5]:  # Test first 5
            similar = await self.find_similar_articles_claude(article, limit=5)
            
            # Count similar FALSE_PUMP articles (good for learning what to avoid)
            false_pump_similar = [a for a in similar if a['outcome_type'] == 'FALSE_PUMP']
            true_bullish_similar = [a for a in similar if a['outcome_type'] == 'TRUE_BULLISH']
            
            precision = len(false_pump_similar) / len(similar) if similar else 0
            contamination = len(true_bullish_similar) / len(similar) if similar else 0
            
            validation_results['false_pump_validation'].append({
                'ticker': article['ticker'],
                'headline': article['headline'][:50] + "...",
                'similar_count': len(similar),
                'false_pump_similar': len(false_pump_similar),
                'true_bullish_contamination': len(true_bullish_similar),
                'precision': precision,
                'contamination_rate': contamination
            })
            
            logger.info(f"  {article['ticker']}: {len(false_pump_similar)}/{len(similar)} similar are FALSE_PUMP (precision: {precision:.2f})")
        
        # Calculate overall validation metrics
        if validation_results['true_bullish_validation']:
            avg_tb_precision = sum(r['precision'] for r in validation_results['true_bullish_validation']) / len(validation_results['true_bullish_validation'])
            avg_tb_contamination = sum(r['contamination_rate'] for r in validation_results['true_bullish_validation']) / len(validation_results['true_bullish_validation'])
        else:
            avg_tb_precision = avg_tb_contamination = 0
        
        if validation_results['false_pump_validation']:
            avg_fp_precision = sum(r['precision'] for r in validation_results['false_pump_validation']) / len(validation_results['false_pump_validation'])
            avg_fp_contamination = sum(r['contamination_rate'] for r in validation_results['false_pump_validation']) / len(validation_results['false_pump_validation'])
        else:
            avg_fp_precision = avg_fp_contamination = 0
        
        validation_results['cross_contamination_analysis'] = {
            'true_bullish_avg_precision': avg_tb_precision,
            'true_bullish_avg_contamination': avg_tb_contamination,
            'false_pump_avg_precision': avg_fp_precision,
            'false_pump_avg_contamination': avg_fp_contamination,
            'overall_separation_quality': (avg_tb_precision + avg_fp_precision) / 2,
            'overall_contamination_risk': (avg_tb_contamination + avg_fp_contamination) / 2
        }
        
        logger.info(f"📊 RAG Precision Validation Results:")
        logger.info(f"  • TRUE_BULLISH precision: {avg_tb_precision:.3f}")
        logger.info(f"  • FALSE_PUMP precision: {avg_fp_precision:.3f}")
        logger.info(f"  • Overall separation quality: {validation_results['cross_contamination_analysis']['overall_separation_quality']:.3f}")
        logger.info(f"  • Overall contamination risk: {validation_results['cross_contamination_analysis']['overall_contamination_risk']:.3f}")
        
        return validation_results
    
    async def save_test_results(self, outcome_correlation: Dict[str, Any], rag_validation: Dict[str, Any]):
        """Save similarity test results to files"""
        os.makedirs('tests/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_data = {
            'test_timestamp': timestamp,
            'outcome_correlation_test': outcome_correlation,
            'rag_precision_validation': rag_validation
        }
        
        # Save results
        with open(f'tests/results/similarity_test_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"📁 Similarity test results saved to tests/results/similarity_test_results_{timestamp}.json")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description='RAG Similarity Search Test')
    parser.add_argument('--test-outcome-correlation', action='store_true', default=True, 
                       help='Test correlation between similarity and outcome types')
    parser.add_argument('--validate-precision', action='store_true', default=True,
                       help='Validate RAG precision for trading decisions')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save test results')
    
    args = parser.parse_args()
    
    tester = RAGSimilarityTester()
    
    try:
        # Initialize test framework
        await tester.initialize()
        
        # Test outcome correlation
        outcome_correlation = {}
        if args.test_outcome_correlation:
            outcome_correlation = await tester.test_outcome_correlation()
        
        # Validate RAG precision
        rag_validation = {}
        if args.validate_precision:
            rag_validation = await tester.validate_rag_precision()
        
        # Save results if requested
        if args.save_results:
            await tester.save_test_results(outcome_correlation, rag_validation)
        
        # Print summary
        if outcome_correlation:
            overall = outcome_correlation.get('overall_metrics', {})
            logger.info("\n" + "="*80)
            logger.info("🎯 SIMILARITY TEST SUMMARY")
            logger.info("="*80)
            logger.info(f"📊 Total tests: {overall.get('total_tests', 0)}")
            logger.info(f"🔍 Overall precision: {overall.get('overall_precision', 0):.3f}")
            logger.info(f"🎯 Top-K accuracy: {overall.get('overall_top_k_accuracy', 0):.3f}")
        
        if rag_validation:
            analysis = rag_validation.get('cross_contamination_analysis', {})
            logger.info(f"✅ RAG separation quality: {analysis.get('overall_separation_quality', 0):.3f}")
            logger.info(f"⚠️ RAG contamination risk: {analysis.get('overall_contamination_risk', 0):.3f}")
            logger.info("="*80)
        
        logger.info("✅ All similarity tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 