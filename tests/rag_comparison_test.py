#!/usr/bin/env python3
"""
RAG vs Traditional Sentiment Analysis Comparison Test

This script compares traditional sentiment analysis against RAG-enhanced analysis
using proper train/test split to avoid data leakage.

- Training vectors: Generated from News.rag_training_set (420 articles)
- Test evaluation: Performed on News.rag_test_set (102 unseen articles)

Usage:
    python3 tests/rag_comparison_test.py --sample-size 50 --test-mode parallel
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
import numpy as np

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
class SentimentResult:
    """Results from sentiment analysis"""
    ticker: str
    headline: str
    traditional_sentiment: str
    traditional_confidence: float
    rag_sentiment: str = ""
    rag_confidence: float = 0.0
    actual_outcome: str = ""
    analysis_time_traditional: float = 0.0
    analysis_time_rag: float = 0.0
    similar_examples: List[Dict] = None

class RAGComparisonTester:
    """Compare traditional vs RAG-enhanced sentiment analysis"""
    
    def __init__(self, buy_high_threshold: float = 0.8, buy_medium_threshold: float = 0.5):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.buy_high_threshold = buy_high_threshold  # Configurable threshold for BUY+high
        self.buy_medium_threshold = buy_medium_threshold  # Configurable threshold for BUY+medium
        self.confidence_map = {'low': 0.55, 'medium': 0.7, 'high': 0.95}  # Standard confidence mapping
        
    async def initialize(self):
        """Initialize the comparison tester"""
        logger.info("🧪 Initializing RAG Comparison Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize sentiment service
        self.sentiment_service = await get_sentiment_service()
        
        # Verify training vectors exist
        await self.verify_training_vectors()
        
        logger.info("✅ RAG comparison test framework initialized successfully")
    
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
            
            # No vectors found
            logger.error("❌ No training vectors found! Please run generate_vectors.py first.")
            raise Exception("Training vectors not found")
                
        except Exception as e:
            logger.error(f"Error verifying training vectors: {e}")
            raise
    
    async def get_test_articles(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get balanced articles from the TEST SET for evaluation (unseen by RAG system)"""
        try:
            # Get balanced sample from each outcome type
            per_outcome = max(1, limit // 3)  # Divide by 3 outcome types
            
            all_articles = []
            
            for outcome_type in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
                query = """
                SELECT 
                    ticker,
                    headline,
                    full_content,
                    outcome_type,
                    has_30pt_increase,
                    is_false_pump,
                    price_increase_ratio,
                    original_content_hash
                FROM News.rag_test_set
                WHERE outcome_type = %s AND LENGTH(headline) > 30
                ORDER BY ticker
                LIMIT %s
                """
                
                result = self.ch_manager.client.query(query, parameters=[outcome_type, per_outcome])
                
                for row in result.result_rows:
                    all_articles.append({
                        'ticker': row[0],
                        'headline': row[1],
                        'full_content': row[2],
                        'outcome_type': row[3],
                        'has_30pt_increase': int(row[4]),
                        'is_false_pump': int(row[5]),
                        'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                        'original_content_hash': row[7],
                        'content': row[2] or f"{row[0]}: {row[1]}"  # Use full_content if available
                    })
            
            # Shuffle for random order
            import random
            random.shuffle(all_articles)
            
            logger.info(f"📄 Retrieved {len(all_articles)} balanced TEST articles for evaluation")
            
            # Log distribution
            outcome_counts = {}
            for article in all_articles:
                outcome = article['outcome_type']
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            logger.info("📊 Test set distribution:")
            for outcome, count in outcome_counts.items():
                logger.info(f"  • {outcome}: {count} articles")
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error retrieving test articles: {e}")
            return []
    
    async def get_similar_training_examples(self, query_content: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar examples from training set using RAG"""
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
    
    async def analyze_traditional_sentiment(self, content: str) -> Tuple[str, float, float]:
        """Analyze sentiment using traditional method only"""
        start_time = datetime.now()
        
        try:
            # Create article structure for sentiment service
            article = {
                'ticker': 'TEST',
                'headline': content[:200],  # First 200 chars as headline
                'full_content': content,
                'content': content
            }
            
            # Use existing sentiment analysis
            result = await self.sentiment_service.analyze_article_sentiment(article)
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            if result:
                # Extract action and confidence from result
                action = result.get('recommendation', 'HOLD')
                confidence_str = result.get('confidence', 'medium')
                
                # For traditional analysis, keep the original confidence string format
                # Only convert to float for threshold comparison in metrics calculation
                return action, confidence_str, analysis_time
            else:
                return 'HOLD', 'medium', analysis_time
                
        except Exception as e:
            logger.error(f"Traditional sentiment analysis failed: {e}")
            analysis_time = (datetime.now() - start_time).total_seconds()
            return 'HOLD', 'medium', analysis_time
    
    async def analyze_rag_sentiment(self, content: str) -> Tuple[str, float, float, List[Dict]]:
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
                # Extract action and confidence, converting to same format as traditional
                action = result.get('action', 'HOLD')
                confidence_str = result.get('confidence', 'medium')
                
                # Convert confidence string to float (same as traditional)
                confidence = self.confidence_map.get(confidence_str, 0.5)
                
                return (
                    action, 
                    confidence, 
                    analysis_time,
                    similar_examples
                )
            else:
                return 'HOLD', 0.5, analysis_time, similar_examples
                
        except Exception as e:
            logger.error(f"RAG sentiment analysis failed: {e}")
            analysis_time = (datetime.now() - start_time).total_seconds()
            return 'HOLD', 0.5, analysis_time, []
    
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
    
    async def run_comparison_test(self, test_articles: List[Dict[str, Any]], test_mode: str = "parallel") -> List[SentimentResult]:
        """Run comparison test between traditional and RAG methods"""
        logger.info(f"🚀 Starting {test_mode} comparison test with {len(test_articles)} articles...")
        
        results = []
        
        for i, article in enumerate(test_articles, 1):
            logger.info(f"📊 Analyzing article {i}/{len(test_articles)}: {article['ticker']} ({article['outcome_type']})")
            
            result = SentimentResult(
                ticker=article['ticker'],
                headline=article['headline'],
                traditional_sentiment="",
                traditional_confidence=0.0,
                actual_outcome=article['outcome_type']
            )
            
            if test_mode in ["traditional", "parallel"]:
                # Traditional analysis
                trad_sentiment, trad_conf, trad_time = await self.analyze_traditional_sentiment(article['content'])
                result.traditional_sentiment = trad_sentiment
                result.traditional_confidence = trad_conf
                result.analysis_time_traditional = trad_time
                
                logger.info(f"  🔍 Traditional: {trad_sentiment} ({trad_conf}) in {trad_time:.2f}s")
            
            if test_mode in ["rag", "parallel"]:
                # RAG analysis
                rag_sentiment, rag_conf, rag_time, similar_examples = await self.analyze_rag_sentiment(article['content'])
                result.rag_sentiment = rag_sentiment
                result.rag_confidence = rag_conf
                result.analysis_time_rag = rag_time
                result.similar_examples = similar_examples
                
                logger.info(f"  🧠 RAG: {rag_sentiment} ({rag_conf:.2f}) in {rag_time:.2f}s")
                if similar_examples:
                    logger.info(f"    📋 Used {len(similar_examples)} similar examples")
            
            results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Calculate performance metrics with focus on BUY+high precision"""
        metrics = {
            'total_articles': len(results),
            'traditional_metrics': {},
            'rag_metrics': {},
            'performance_comparison': {},
            'recall_metrics': {}  # Add recall metrics section
        }
        
        # Calculate recall metrics first (needed for both traditional and RAG analysis)
        true_bullish_articles = [r for r in results if r.actual_outcome == "TRUE_BULLISH"]
        total_true_bullish = len(true_bullish_articles)
        
        metrics['recall_metrics'] = {
            'total_true_bullish_articles': total_true_bullish,
            'traditional_buy_high_recall': 0.0,
            'rag_buy_high_recall': 0.0,
            'traditional_buy_any_recall': 0.0,
            'rag_buy_any_recall': 0.0
        }

        # Traditional metrics
        if any(r.traditional_sentiment for r in results):
            trad_correct = 0
            trad_buy_high_correct = 0
            trad_buy_high_total = 0
            trad_buy_medium_correct = 0
            trad_buy_medium_total = 0
            trad_buy_high_recall_count = 0
            trad_buy_any_recall_count = 0
            
            for result in results:
                # Check if traditional prediction matches actual outcome
                expected_action = self.outcome_to_expected_action(result.actual_outcome)
                if result.traditional_sentiment == expected_action:
                    trad_correct += 1
                
                # Check BUY+high precision (most critical metric)
                if result.traditional_sentiment == "BUY":
                    # Handle string confidence levels for traditional analysis
                    if isinstance(result.traditional_confidence, str):
                        if result.traditional_confidence == "high":
                            trad_buy_high_total += 1
                            if result.actual_outcome == "TRUE_BULLISH":
                                trad_buy_high_correct += 1
                        elif result.traditional_confidence == "medium":
                            trad_buy_medium_total += 1
                            if result.actual_outcome == "TRUE_BULLISH":
                                trad_buy_medium_correct += 1
                    else:
                        # Handle numeric confidence (fallback)
                        if result.traditional_confidence >= self.buy_high_threshold:
                            trad_buy_high_total += 1
                            if result.actual_outcome == "TRUE_BULLISH":
                                trad_buy_high_correct += 1
                        elif self.buy_medium_threshold <= result.traditional_confidence < self.buy_high_threshold:
                            trad_buy_medium_total += 1
                            if result.actual_outcome == "TRUE_BULLISH":
                                trad_buy_medium_correct += 1
                
                # Calculate recall for TRUE_BULLISH articles
                if result.actual_outcome == "TRUE_BULLISH":
                    if result.traditional_sentiment == "BUY":
                        trad_buy_any_recall_count += 1
                        # Check if it's BUY+high
                        if isinstance(result.traditional_confidence, str):
                            if result.traditional_confidence == "high":
                                trad_buy_high_recall_count += 1
                        else:
                            if result.traditional_confidence >= self.buy_high_threshold:
                                trad_buy_high_recall_count += 1
            
            # Calculate recall percentages
            if total_true_bullish > 0:
                metrics['recall_metrics']['traditional_buy_high_recall'] = trad_buy_high_recall_count / total_true_bullish
                metrics['recall_metrics']['traditional_buy_any_recall'] = trad_buy_any_recall_count / total_true_bullish
            
            metrics['traditional_metrics'] = {
                'accuracy': trad_correct / len(results),
                'buy_high_precision': trad_buy_high_correct / max(1, trad_buy_high_total),
                'buy_high_count': trad_buy_high_total,
                'buy_medium_precision': trad_buy_medium_correct / max(1, trad_buy_medium_total),
                'buy_medium_count': trad_buy_medium_total,
                'avg_analysis_time': sum(r.analysis_time_traditional for r in results) / len(results),
                'buy_high_recall': metrics['recall_metrics']['traditional_buy_high_recall'],
                'buy_any_recall': metrics['recall_metrics']['traditional_buy_any_recall']
            }
        
        # RAG metrics
        if any(r.rag_sentiment for r in results):
            rag_correct = 0
            rag_buy_high_correct = 0
            rag_buy_high_total = 0
            rag_buy_medium_correct = 0
            rag_buy_medium_total = 0
            rag_buy_high_recall_count = 0
            rag_buy_any_recall_count = 0
            
            for result in results:
                # Check if RAG prediction matches actual outcome
                expected_action = self.outcome_to_expected_action(result.actual_outcome)
                if result.rag_sentiment == expected_action:
                    rag_correct += 1
                
                # Check BUY+high precision (most critical metric)
                if result.rag_sentiment == "BUY" and result.rag_confidence >= self.buy_high_threshold:
                    rag_buy_high_total += 1
                    if result.actual_outcome == "TRUE_BULLISH":
                        rag_buy_high_correct += 1
                
                # Check BUY+medium precision (should be more cautious)
                if result.rag_sentiment == "BUY" and self.buy_medium_threshold <= result.rag_confidence < self.buy_high_threshold:
                    rag_buy_medium_total += 1
                    if result.actual_outcome == "TRUE_BULLISH":
                        rag_buy_medium_correct += 1
                
                # Calculate recall for TRUE_BULLISH articles
                if result.actual_outcome == "TRUE_BULLISH":
                    if result.rag_sentiment == "BUY":
                        rag_buy_any_recall_count += 1
                        if result.rag_confidence >= self.buy_high_threshold:
                            rag_buy_high_recall_count += 1
            
            # Calculate recall percentages
            if total_true_bullish > 0:
                metrics['recall_metrics']['rag_buy_high_recall'] = rag_buy_high_recall_count / total_true_bullish
                metrics['recall_metrics']['rag_buy_any_recall'] = rag_buy_any_recall_count / total_true_bullish
            
            metrics['rag_metrics'] = {
                'accuracy': rag_correct / len(results),
                'buy_high_precision': rag_buy_high_correct / max(1, rag_buy_high_total),
                'buy_high_count': rag_buy_high_total,
                'buy_medium_precision': rag_buy_medium_correct / max(1, rag_buy_medium_total),
                'buy_medium_count': rag_buy_medium_total,
                'avg_analysis_time': sum(r.analysis_time_rag for r in results) / len(results),
                'buy_high_recall': metrics['recall_metrics']['rag_buy_high_recall'],
                'buy_any_recall': metrics['recall_metrics']['rag_buy_any_recall']
            }
        
        # Performance comparison
        if metrics['traditional_metrics'] and metrics['rag_metrics']:
            metrics['performance_comparison'] = {
                'accuracy_improvement': metrics['rag_metrics']['accuracy'] - metrics['traditional_metrics']['accuracy'],
                'buy_high_precision_improvement': metrics['rag_metrics']['buy_high_precision'] - metrics['traditional_metrics']['buy_high_precision'],
                'buy_medium_precision_improvement': metrics['rag_metrics']['buy_medium_precision'] - metrics['traditional_metrics']['buy_medium_precision'],
                'buy_high_recall_improvement': metrics['rag_metrics']['buy_high_recall'] - metrics['traditional_metrics']['buy_high_recall'],
                'buy_any_recall_improvement': metrics['rag_metrics']['buy_any_recall'] - metrics['traditional_metrics']['buy_any_recall'],
                'time_overhead': metrics['rag_metrics']['avg_analysis_time'] - metrics['traditional_metrics']['avg_analysis_time']
            }
        
        return metrics

    def generate_detailed_analysis(self, results: List[SentimentResult], metrics: Dict[str, Any]) -> str:
        """Generate detailed markdown analysis of BUY+high performance"""
        
        # Get BUY+high predictions for both models
        traditional_buy_high = []
        rag_buy_high = []
        
        for result in results:
            # Traditional BUY+high
            is_trad_buy_high = False
            if result.traditional_sentiment == "BUY":
                if isinstance(result.traditional_confidence, str):
                    is_trad_buy_high = result.traditional_confidence == "high"
                else:
                    is_trad_buy_high = result.traditional_confidence >= self.buy_high_threshold
            
            if is_trad_buy_high:
                traditional_buy_high.append({
                    'ticker': result.ticker,
                    'headline': result.headline[:100] + "..." if len(result.headline) > 100 else result.headline,
                    'actual_outcome': result.actual_outcome,
                    'confidence': result.traditional_confidence,
                    'correct': result.actual_outcome == "TRUE_BULLISH"
                })
            
            # RAG BUY+high
            if result.rag_sentiment == "BUY" and result.rag_confidence >= self.buy_high_threshold:
                rag_buy_high.append({
                    'ticker': result.ticker,
                    'headline': result.headline[:100] + "..." if len(result.headline) > 100 else result.headline,
                    'actual_outcome': result.actual_outcome,
                    'confidence': result.rag_confidence,
                    'correct': result.actual_outcome == "TRUE_BULLISH",
                    'similar_examples_count': len(result.similar_examples) if result.similar_examples else 0
                })
        
        # Generate markdown report
        md_content = f"""# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Articles Analyzed**: {metrics['total_articles']}
- **TRUE_BULLISH Articles in Test Set**: {metrics['recall_metrics']['total_true_bullish_articles']}

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: {metrics['traditional_metrics']['accuracy']:.1%}
- **BUY+High Precision**: {metrics['traditional_metrics']['buy_high_precision']:.1%} ({metrics['traditional_metrics']['buy_high_count']} signals)
- **BUY+High Recall**: {metrics['traditional_metrics']['buy_high_recall']:.1%}
- **BUY (Any) Recall**: {metrics['traditional_metrics']['buy_any_recall']:.1%}

### RAG Model
- **Overall Accuracy**: {metrics['rag_metrics']['accuracy']:.1%}
- **BUY+High Precision**: {metrics['rag_metrics']['buy_high_precision']:.1%} ({metrics['rag_metrics']['buy_high_count']} signals)
- **BUY+High Recall**: {metrics['rag_metrics']['buy_high_recall']:.1%}
- **BUY (Any) Recall**: {metrics['rag_metrics']['buy_any_recall']:.1%}

### Performance Improvements
- **Accuracy**: {metrics['performance_comparison']['accuracy_improvement']:+.1%}
- **BUY+High Precision**: {metrics['performance_comparison']['buy_high_precision_improvement']:+.1%}
- **BUY+High Recall**: {metrics['performance_comparison']['buy_high_recall_improvement']:+.1%}
- **BUY (Any) Recall**: {metrics['performance_comparison']['buy_any_recall_improvement']:+.1%}

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions ({len(traditional_buy_high)} total)

"""
        
        # Traditional BUY+high breakdown
        if traditional_buy_high:
            correct_traditional = [p for p in traditional_buy_high if p['correct']]
            incorrect_traditional = [p for p in traditional_buy_high if not p['correct']]
            
            md_content += f"""
#### ✅ Correct Predictions ({len(correct_traditional)}/{len(traditional_buy_high)} = {len(correct_traditional)/len(traditional_buy_high):.1%})
"""
            for pred in correct_traditional:
                md_content += f"- **{pred['ticker']}**: {pred['headline']} (Confidence: {pred['confidence']})\n"
            
            md_content += f"""
#### ❌ Incorrect Predictions ({len(incorrect_traditional)}/{len(traditional_buy_high)} = {len(incorrect_traditional)/len(traditional_buy_high):.1%})
"""
            for pred in incorrect_traditional:
                outcome_desc = "False Pump" if pred['actual_outcome'] == "FALSE_PUMP" else "Neutral"
                md_content += f"- **{pred['ticker']}**: {pred['headline']} (Confidence: {pred['confidence']}, Actual: {outcome_desc})\n"
        else:
            md_content += "No BUY+high predictions made by traditional model.\n"
        
        # RAG BUY+high breakdown
        md_content += f"""
### RAG Model BUY+High Predictions ({len(rag_buy_high)} total)

"""
        
        if rag_buy_high:
            correct_rag = [p for p in rag_buy_high if p['correct']]
            incorrect_rag = [p for p in rag_buy_high if not p['correct']]
            
            md_content += f"""
#### ✅ Correct Predictions ({len(correct_rag)}/{len(rag_buy_high)} = {len(correct_rag)/len(rag_buy_high):.1%})
"""
            for pred in correct_rag:
                md_content += f"- **{pred['ticker']}**: {pred['headline']} (Confidence: {pred['confidence']:.2f}, Similar Examples: {pred['similar_examples_count']})\n"
            
            md_content += f"""
#### ❌ Incorrect Predictions ({len(incorrect_rag)}/{len(rag_buy_high)} = {len(incorrect_rag)/len(rag_buy_high):.1%})
"""
            for pred in incorrect_rag:
                outcome_desc = "False Pump" if pred['actual_outcome'] == "FALSE_PUMP" else "Neutral"
                md_content += f"- **{pred['ticker']}**: {pred['headline']} (Confidence: {pred['confidence']:.2f}, Actual: {outcome_desc}, Similar Examples: {pred['similar_examples_count']})\n"
        else:
            md_content += "No BUY+high predictions made by RAG model.\n"
        
        # Missed opportunities analysis
        missed_opportunities = [r for r in results if r.actual_outcome == "TRUE_BULLISH"]
        missed_by_traditional = []
        missed_by_rag = []
        
        for result in missed_opportunities:
            # Check if traditional missed it (didn't give BUY+high)
            is_trad_buy_high = False
            if result.traditional_sentiment == "BUY":
                if isinstance(result.traditional_confidence, str):
                    is_trad_buy_high = result.traditional_confidence == "high"
                else:
                    is_trad_buy_high = result.traditional_confidence >= self.buy_high_threshold
            
            if not is_trad_buy_high:
                missed_by_traditional.append({
                    'ticker': result.ticker,
                    'headline': result.headline[:100] + "..." if len(result.headline) > 100 else result.headline,
                    'prediction': result.traditional_sentiment,
                    'confidence': result.traditional_confidence
                })
            
            # Check if RAG missed it
            if not (result.rag_sentiment == "BUY" and result.rag_confidence >= self.buy_high_threshold):
                missed_by_rag.append({
                    'ticker': result.ticker,
                    'headline': result.headline[:100] + "..." if len(result.headline) > 100 else result.headline,
                    'prediction': result.rag_sentiment,
                    'confidence': result.rag_confidence
                })
        
        md_content += f"""
---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model ({len(missed_by_traditional)} missed)
"""
        
        for missed in missed_by_traditional:
            md_content += f"- **{missed['ticker']}**: {missed['headline']} (Predicted: {missed['prediction']}, Confidence: {missed['confidence']})\n"
        
        md_content += f"""
### TRUE_BULLISH Articles Missed by RAG Model ({len(missed_by_rag)} missed)
"""
        
        for missed in missed_by_rag:
            md_content += f"- **{missed['ticker']}**: {missed['headline']} (Predicted: {missed['prediction']}, Confidence: {missed['confidence']:.2f})\n"
        
        # Success criteria assessment
        traditional_precision = metrics['traditional_metrics']['buy_high_precision']
        rag_precision = metrics['rag_metrics']['buy_high_precision']
        traditional_recall = metrics['traditional_metrics']['buy_high_recall']
        rag_recall = metrics['rag_metrics']['buy_high_recall']
        
        md_content += f"""
---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: {traditional_precision:.1%} {'✅' if traditional_precision >= 0.8 else '❌'}
  - RAG: {rag_precision:.1%} {'✅' if rag_precision >= 0.8 else '❌'}

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: {metrics['traditional_metrics']['buy_any_recall']:.1%} {'✅' if metrics['traditional_metrics']['buy_any_recall'] >= 0.9 else '❌'}
  - RAG: {metrics['rag_metrics']['buy_any_recall']:.1%} {'✅' if metrics['rag_metrics']['buy_any_recall'] >= 0.9 else '❌'}

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: {traditional_recall:.1%}
  - RAG: {rag_recall:.1%}

### Integration Recommendation
"""
        
        # Generate recommendation based on results
        precision_improvement = metrics['performance_comparison']['buy_high_precision_improvement']
        recall_improvement = metrics['performance_comparison']['buy_high_recall_improvement']
        time_overhead = metrics['performance_comparison']['time_overhead']
        
        if precision_improvement >= 0.15 and time_overhead < 0.3:  # 15% precision improvement, <300ms overhead
            recommendation = "✅ **RECOMMEND INTEGRATION** - RAG shows significant precision improvement with acceptable overhead"
        elif precision_improvement >= 0.05 and recall_improvement >= 0.0:  # Any precision improvement without recall loss
            recommendation = "⚠️ **CONDITIONAL INTEGRATION** - RAG shows improvement but may need tuning"
        else:
            recommendation = "❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria"
        
        md_content += f"{recommendation}\n\n"
        md_content += f"**Analysis Time Overhead**: {time_overhead:.2f}s per article {'✅' if time_overhead < 0.3 else '❌' if time_overhead < 0.5 else '🚫'}\n\n"
        
        return md_content
    
    def outcome_to_expected_action(self, outcome: str) -> str:
        """Convert outcome type to expected trading action"""
        mapping = {
            'TRUE_BULLISH': 'BUY',
            'FALSE_PUMP': 'HOLD',  # Should avoid BUY for false pumps
            'NEUTRAL': 'HOLD'
        }
        return mapping.get(outcome, 'HOLD')
    
    async def save_results(self, results: List[SentimentResult], metrics: Dict[str, Any]):
        """Save test results to files"""
        os.makedirs('tests/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare results for JSON serialization
        results_data = {
            'test_timestamp': timestamp,
            'test_type': 'rag_vs_traditional_comparison',
            'data_split': 'proper_train_test_split',
            'training_vectors_used': 'News.rag_training_vectors',
            'test_articles_from': 'News.rag_test_set',
            'metrics': metrics,
            'detailed_results': []
        }
        
        for result in results:
            results_data['detailed_results'].append({
                'ticker': result.ticker,
                'headline': result.headline,
                'actual_outcome': result.actual_outcome,
                'traditional_sentiment': result.traditional_sentiment,
                'traditional_confidence': result.traditional_confidence,
                'rag_sentiment': result.rag_sentiment,
                'rag_confidence': result.rag_confidence,
                'analysis_time_traditional': result.analysis_time_traditional,
                'analysis_time_rag': result.analysis_time_rag,
                'similar_examples_count': len(result.similar_examples) if result.similar_examples else 0
            })
        
        # Save JSON results
        with open(f'tests/results/rag_comparison_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Generate and save detailed markdown analysis
        detailed_analysis = self.generate_detailed_analysis(results, metrics)
        with open(f'tests/results/rag_detailed_analysis_{timestamp}.md', 'w') as f:
            f.write(detailed_analysis)
        
        logger.info(f"📁 RAG comparison results saved to tests/results/rag_comparison_results_{timestamp}.json")
        logger.info(f"📊 Detailed BUY+high analysis saved to tests/results/rag_detailed_analysis_{timestamp}.md")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description='RAG vs Traditional Sentiment Analysis Comparison')
    parser.add_argument('--sample-size', type=int, default=30, help='Number of test articles to analyze')
    parser.add_argument('--test-mode', choices=['traditional', 'rag', 'parallel'], default='parallel', 
                        help='Test mode: traditional only, rag only, or both in parallel')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save test results')
    parser.add_argument('--buy-high-threshold', type=float, default=0.8, help='Confidence threshold for BUY+high (default: 0.8)')
    parser.add_argument('--buy-medium-threshold', type=float, default=0.5, help='Confidence threshold for BUY+medium (default: 0.5)')
    
    args = parser.parse_args()
    
    tester = RAGComparisonTester(buy_high_threshold=args.buy_high_threshold, buy_medium_threshold=args.buy_medium_threshold)
    
    try:
        # Initialize test framework
        await tester.initialize()
        
        # Get test articles (from unseen test set)
        test_articles = await tester.get_test_articles(args.sample_size)
        if not test_articles:
            logger.error("No test articles found!")
            return
        
        # Run comparison test
        results = await tester.run_comparison_test(test_articles, args.test_mode)
        
        # Calculate metrics
        metrics = tester.calculate_metrics(results)
        
        # Log summary
        logger.info("📊 Test Results Summary:")
        if metrics.get('traditional_metrics'):
            trad = metrics['traditional_metrics']
            logger.info(f"  🔍 Traditional: {trad['accuracy']:.1%} accuracy")
            logger.info(f"    • BUY+high: {trad['buy_high_count']} signals, {trad['buy_high_precision']:.1%} precision, {trad['buy_high_recall']:.1%} recall")
            logger.info(f"    • BUY+medium: {trad['buy_medium_count']} signals, {trad['buy_medium_precision']:.1%} precision")
            logger.info(f"    • BUY (any): {trad['buy_any_recall']:.1%} recall of TRUE_BULLISH articles")
        
        if metrics.get('rag_metrics'):
            rag = metrics['rag_metrics']
            logger.info(f"  🧠 RAG: {rag['accuracy']:.1%} accuracy")
            logger.info(f"    • BUY+high: {rag['buy_high_count']} signals, {rag['buy_high_precision']:.1%} precision, {rag['buy_high_recall']:.1%} recall")
            logger.info(f"    • BUY+medium: {rag['buy_medium_count']} signals, {rag['buy_medium_precision']:.1%} precision")
            logger.info(f"    • BUY (any): {rag['buy_any_recall']:.1%} recall of TRUE_BULLISH articles")
        
        if metrics.get('performance_comparison'):
            comp = metrics['performance_comparison']
            logger.info(f"  📈 Improvement: {comp['accuracy_improvement']:+.1%} accuracy")
            logger.info(f"    • BUY+high precision: {comp['buy_high_precision_improvement']:+.1%}")
            logger.info(f"    • BUY+high recall: {comp['buy_high_recall_improvement']:+.1%}")
            logger.info(f"    • BUY+medium precision: {comp['buy_medium_precision_improvement']:+.1%}")
            logger.info(f"    • BUY (any) recall: {comp['buy_any_recall_improvement']:+.1%}")
            logger.info(f"  ⏱️ Time overhead: {comp['time_overhead']:+.2f}s per article")
        
        if metrics.get('recall_metrics'):
            recall = metrics['recall_metrics']
            logger.info(f"  📋 Recall Analysis: {recall['total_true_bullish_articles']} TRUE_BULLISH articles in test set")
            logger.info(f"    • Traditional captured {recall['traditional_buy_high_recall']:.1%} with BUY+high")
            logger.info(f"    • RAG captured {recall['rag_buy_high_recall']:.1%} with BUY+high")
        
        # Save results if requested
        if args.save_results:
            await tester.save_results(results, metrics)
        
        logger.info("✅ RAG comparison test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 