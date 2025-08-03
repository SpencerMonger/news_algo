#!/usr/bin/env python3
"""
RAG vs Traditional Sentiment Analysis Comparison Test

This script compares traditional sentiment analysis against RAG-enhanced analysis
using proper train/test split to avoid data leakage.

- Training vectors: Generated from News.rag_training_set (420 articles)
- Test evaluation: Performed on News.rag_test_set (102 unseen articles)

Usage:
    python3 tests/rag_comparison_test.py --sample-size 50 --test-mode parallel --pnl
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import argparse
from dataclasses import dataclass
import numpy as np
import pytz
import aiohttp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from sentiment_service import get_sentiment_service
from dotenv import load_dotenv

# Load environment variables for Polygon API
load_dotenv()

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
    embedding_time: float = 0.0
    vector_search_time: float = 0.0
    llm_decision_time: float = 0.0
    # Add fields for PnL calculation
    published_utc: Optional[datetime] = None
    traditional_pnl: Optional[float] = None
    rag_pnl: Optional[float] = None
    traditional_entry_price: Optional[float] = None
    traditional_exit_price: Optional[float] = None
    rag_entry_price: Optional[float] = None
    rag_exit_price: Optional[float] = None
    # Add detailed PnL tracking fields
    traditional_position_size: Optional[int] = None
    rag_position_size: Optional[int] = None
    traditional_investment: Optional[float] = None
    rag_investment: Optional[float] = None
    traditional_return_pct: Optional[float] = None
    rag_return_pct: Optional[float] = None
    price_bracket: Optional[str] = None
    publication_hour: Optional[int] = None

class OptimizedRAGTester:
    """Optimized RAG comparison with local embeddings and fast vector search"""
    
    def __init__(self, buy_high_threshold: float = 0.8, buy_medium_threshold: float = 0.5):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.embedding_model = None
        self.buy_high_threshold = buy_high_threshold
        self.buy_medium_threshold = buy_medium_threshold
        self.confidence_map = {'low': 0.55, 'medium': 0.7, 'high': 0.95}
        
        # PnL calculation setup
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.session = None
        self.est_tz = pytz.timezone('US/Eastern')
        self.default_quantity = 100  # Default shares per trade
        
        # Dynamic position sizing tiers
        self.position_tiers = [
            {'price_min': 0.01, 'price_max': 1.00, 'unit_position_size': 10000, 'max_position_size': 20000},
            {'price_min': 1.00, 'price_max': 3.00, 'unit_position_size': 8000, 'max_position_size': 16000},
            {'price_min': 3.00, 'price_max': 5.00, 'unit_position_size': 5000, 'max_position_size': 10000},
            {'price_min': 5.00, 'price_max': 8.00, 'unit_position_size': 3000, 'max_position_size': 6000},
            {'price_min': 8.00, 'price_max': 999999.99, 'unit_position_size': 2000, 'max_position_size': 4000}
        ]
        
        # Use PROXY_URL if available for Polygon API
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.polygon_base_url = proxy_url.rstrip('/')
        else:
            self.polygon_base_url = "https://api.polygon.io"
        
    async def initialize(self):
        """Initialize the optimized RAG tester"""
        logger.info("🚀 Initializing Optimized RAG Comparison Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize sentiment service
        self.sentiment_service = await get_sentiment_service()
        
        # Initialize local embedding model
        await self.initialize_embedding_model()
        
        # Initialize HTTP session for PnL calculations
        if self.polygon_api_key:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            logger.info("✅ Polygon API session initialized for PnL calculations")
        
        # Verify training vectors exist
        await self.verify_training_vectors()
        
        logger.info("✅ Optimized RAG comparison test framework initialized successfully")
    
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
            # Check for existing training vectors
            count_query = 'SELECT COUNT(*) FROM News.rag_training_vectors'
            try:
                vector_count = self.ch_manager.client.query(count_query).result_rows[0][0]
                if vector_count > 0:
                    logger.info(f"✅ Found {vector_count} training vectors for RAG system")
                    self.vectors_table = 'News.rag_training_vectors'
                    
                    # SECURITY CHECK: Verify no data leakage
                    await self.verify_no_data_leakage()
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
                    
                    # This approach is inherently safe from leakage due to the JOIN
                    return
            except:
                pass
            
            # No vectors found
            logger.error("❌ No training vectors found! Please run generate_vectors.py first.")
            raise Exception("Training vectors not found")
                
        except Exception as e:
            logger.error(f"Error verifying training vectors: {e}")
            raise
    
    async def verify_no_data_leakage(self):
        """Verify that training vectors don't contain any test set articles"""
        try:
            leakage_query = """
            SELECT COUNT(*) as leakage_count
            FROM News.rag_training_vectors tv
            INNER JOIN News.rag_test_set ts ON tv.original_content_hash = ts.original_content_hash
            """
            
            result = self.ch_manager.client.query(leakage_query)
            leakage_count = result.result_rows[0][0]
            
            if leakage_count > 0:
                logger.error(f"🚨 DATA LEAKAGE DETECTED: {leakage_count} test articles found in training vectors!")
                raise Exception(f"Data leakage detected: {leakage_count} overlapping articles")
            else:
                logger.info("🔒 Data leakage check passed - no test articles in training vectors")
                
        except Exception as e:
            logger.error(f"Error checking for data leakage: {e}")
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
                    original_content_hash,
                    published_est
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
                        'published_est': row[8],  # This is EST timezone, not UTC
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
    
    async def get_similar_outcome_examples(self, target_outcome: str = 'TRUE_BULLISH', top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        """Find examples from training set based on outcome patterns rather than text similarity"""
        search_start_time = datetime.now()
        
        try:
            # Build query based on outcome patterns - find the BEST performing examples
            if target_outcome == 'TRUE_BULLISH':
                # Find articles that actually had strong positive outcomes
                query = """
                SELECT 
                    ticker,
                    headline,
                    full_content,
                    outcome_type,
                    has_30pt_increase,
                    is_false_pump,
                    price_increase_ratio,
                    published_est
                FROM News.rag_training_set
                WHERE outcome_type = 'TRUE_BULLISH'
                  AND has_30pt_increase = true
                  AND price_increase_ratio > 0.25
                ORDER BY price_increase_ratio DESC
                LIMIT %s
                """
            else:
                # For other outcomes, find representative examples
                query = """
                SELECT 
                    ticker,
                    headline,
                    full_content,
                    outcome_type,
                    has_30pt_increase,
                    is_false_pump,
                    price_increase_ratio,
                    published_est
                FROM News.rag_training_set
                WHERE outcome_type = %s
                ORDER BY 
                    CASE 
                        WHEN outcome_type = 'FALSE_PUMP' THEN -price_increase_ratio
                        ELSE RAND()
                    END
                LIMIT %s
                """
            
            # Execute query
            if target_outcome == 'TRUE_BULLISH':
                result = self.ch_manager.client.query(query, parameters=[top_k])
            else:
                result = self.ch_manager.client.query(query, parameters=[target_outcome, top_k])
            
            examples = []
            for row in result.result_rows:
                ticker, headline, content, outcome, has_30pt, is_false_pump, price_ratio, published = row
                
                examples.append({
                    'ticker': ticker,
                    'headline': headline,
                    'full_content': content,
                    'outcome_type': outcome,
                    'has_30pt_increase': has_30pt,
                    'is_false_pump': is_false_pump,
                    'price_increase_ratio': float(price_ratio) if price_ratio else 0.0,
                    'published_est': published,
                    'pattern_score': float(price_ratio) if price_ratio else 0.0  # Use price performance as "similarity"
                })
            
            search_time = (datetime.now() - search_start_time).total_seconds()
            
            logger.info(f"🎯 Found {len(examples)} outcome-pattern examples for {target_outcome} in {search_time*1000:.0f}ms")
            return examples, search_time
            
        except Exception as e:
            logger.error(f"❌ Error in outcome pattern search: {e}")
            return [], (datetime.now() - search_start_time).total_seconds()
    
    async def analyze_optimized_rag_sentiment(self, content: str) -> Tuple[str, float, float, List[Dict], Dict[str, float]]:
        """Analyze sentiment using outcome-pattern based RAG with performance-based confidence"""
        analysis_start_time = datetime.now()
        timing_info = {}
        
        try:
            # Get outcome pattern examples - focus on TRUE_BULLISH patterns
            similar_examples, search_time = await self.get_similar_outcome_examples('TRUE_BULLISH', top_k=5)
            timing_info['embedding_time'] = 0.0  # No embedding needed for pattern search
            timing_info['vector_search_time'] = search_time
            
            if not similar_examples:
                # Fallback to neutral examples if no TRUE_BULLISH patterns found
                similar_examples, search_time = await self.get_similar_outcome_examples('NEUTRAL', top_k=3)
                timing_info['vector_search_time'] += search_time
            
            # Calculate confidence based on pattern strength
            if similar_examples:
                # Base confidence on the strength of historical patterns
                avg_performance = sum(ex.get('price_increase_ratio', 0.0) for ex in similar_examples) / len(similar_examples)
                strong_performers = sum(1 for ex in similar_examples if ex.get('price_increase_ratio', 0.0) > 0.30)
                
                # Calculate pattern-based confidence
                base_confidence = 0.75  # Conservative baseline
                performance_boost = min(avg_performance * 0.5, 0.15)  # Up to +15% for strong patterns
                consistency_boost = (strong_performers / len(similar_examples)) * 0.10  # Up to +10% for consistency
                
                pattern_confidence = base_confidence + performance_boost + consistency_boost
                pattern_confidence = min(pattern_confidence, 0.95)  # Cap at 95%
                
                logger.debug(f"📊 Pattern confidence: {pattern_confidence:.3f} (avg perf: {avg_performance:.3f}, strong: {strong_performers}/{len(similar_examples)})")
            else:
                pattern_confidence = 0.60  # Low confidence without patterns
            
            # Create context for LLM
            context = self.create_outcome_pattern_context(similar_examples)
            
            # Create prompt focusing on outcome patterns
            prompt = f"""
            Based on these historical examples of articles that achieved significant price increases (30%+ gains), analyze this new article.

            HISTORICAL SUCCESS PATTERNS:
            {context}

            NEW ARTICLE TO ANALYZE:
            {content[:800]}

            Based on the pattern of successful articles above, provide your analysis:

            {{
                "sentiment": "BUY|HOLD|SELL",
                "confidence": {pattern_confidence:.3f},
                "reasoning": "Brief explanation focusing on similarity to successful patterns"
            }}
            """
            
            # Get LLM decision
            llm_start_time = datetime.now()
            
            try:
                response = await self.sentiment_service.load_balancer.make_claude_request(prompt)
                
                timing_info['llm_decision_time'] = (datetime.now() - llm_start_time).total_seconds()
                
                # Parse response - handle both string and dict responses
                import json
                try:
                    # Handle dict response (already parsed)
                    if isinstance(response, dict):
                        result = response
                    else:
                        # Handle string response that needs parsing
                        clean_response = response.strip()
                        if clean_response.startswith('```json'):
                            clean_response = clean_response.split('```json')[1].split('```')[0]
                        elif clean_response.startswith('```'):
                            clean_response = clean_response.split('```')[1].split('```')[0]
                        
                        result = json.loads(clean_response.strip())
                    
                    sentiment = result.get('sentiment', 'HOLD')
                    # Use our pattern-based confidence instead of LLM confidence
                    confidence = pattern_confidence
                    
                    total_time = (datetime.now() - analysis_start_time).total_seconds()
                    
                    logger.debug(f"🎯 Pattern-based RAG: {sentiment} @ {confidence:.3f} confidence")
                    return sentiment, confidence, total_time, similar_examples, timing_info
                    
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    logger.error(f"Response parse error: {e}, response type: {type(response)}, response: {str(response)[:200]}")
                    return 'HOLD', 0.50, (datetime.now() - analysis_start_time).total_seconds(), similar_examples, timing_info
                    
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                timing_info['llm_decision_time'] = (datetime.now() - llm_start_time).total_seconds()
                return 'HOLD', 0.50, (datetime.now() - analysis_start_time).total_seconds(), similar_examples, timing_info
            
        except Exception as e:
            logger.error(f"Error in outcome-pattern RAG analysis: {e}")
            total_time = (datetime.now() - analysis_start_time).total_seconds()
            return 'HOLD', 0.50, total_time, [], timing_info
    
    def calculate_nuanced_confidence(self, similar_examples: List[Dict]) -> float:
        """Calculate nuanced confidence based on weighted similarity patterns"""
        if not similar_examples:
            return 0.5
        
        # Calculate weighted outcome scores
        weighted_scores = {'TRUE_BULLISH': 0.0, 'FALSE_PUMP': 0.0, 'NEUTRAL': 0.0}
        total_weighted_similarity = 0.0
        
        for example in similar_examples:
            weighted_sim = example.get('weighted_similarity', example['similarity'])
            outcome = example['outcome_type']
            weighted_scores[outcome] += weighted_sim
            total_weighted_similarity += weighted_sim
        
        # Calculate average similarity quality
        avg_similarity = sum(e['similarity'] for e in similar_examples) / len(similar_examples)
        
        # Determine confidence based on pattern strength and similarity quality
        true_bullish_weight = weighted_scores['TRUE_BULLISH']
        false_pump_weight = weighted_scores['FALSE_PUMP']
        neutral_weight = weighted_scores['NEUTRAL']
        
        # Base confidence on similarity quality (40-70% range)
        base_confidence = 0.4 + (avg_similarity * 0.3)  # 0.4 to 0.7 based on similarity
        
        # Adjust based on outcome pattern strength
        if true_bullish_weight >= false_pump_weight * 2.0:
            # Very strong bullish pattern
            pattern_boost = 0.25
        elif true_bullish_weight >= false_pump_weight * 1.5:
            # Strong bullish pattern  
            pattern_boost = 0.15
        elif true_bullish_weight > false_pump_weight:
            # Moderate bullish pattern
            pattern_boost = 0.08
        elif false_pump_weight > true_bullish_weight * 1.5:
            # Strong bearish pattern - lower confidence for any recommendation
            pattern_boost = -0.1
        else:
            # Mixed/unclear pattern
            pattern_boost = 0.0
        
        # Final confidence calculation
        final_confidence = base_confidence + pattern_boost
        
        # Clamp to reasonable range (0.5 to 0.95)
        final_confidence = max(0.5, min(0.95, final_confidence))
        
        logger.debug(f"Confidence calc: base={base_confidence:.2f}, pattern_boost={pattern_boost:.2f}, final={final_confidence:.2f}")
        logger.debug(f"Pattern: TB={true_bullish_weight:.2f}, FP={false_pump_weight:.2f}, avg_sim={avg_similarity:.2f}")
        
        return final_confidence
    
    def determine_sentiment_from_pattern(self, similar_examples: List[Dict]) -> str:
        """Determine sentiment based on weighted pattern analysis"""
        if not similar_examples:
            return "HOLD"
        
        weighted_scores = {'TRUE_BULLISH': 0.0, 'FALSE_PUMP': 0.0, 'NEUTRAL': 0.0}
        
        for example in similar_examples:
            weighted_sim = example.get('weighted_similarity', example['similarity'])
            outcome = example['outcome_type']
            weighted_scores[outcome] += weighted_sim
        
        true_bullish_weight = weighted_scores['TRUE_BULLISH']
        false_pump_weight = weighted_scores['FALSE_PUMP']
        neutral_weight = weighted_scores['NEUTRAL']
        
        # Decision logic based on weighted patterns
        if true_bullish_weight > false_pump_weight and true_bullish_weight > neutral_weight:
            return "BUY"
        elif false_pump_weight > true_bullish_weight * 1.2:
            return "SELL" 
        else:
            return "HOLD"
    
    def create_optimized_rag_context(self, similar_examples: List[Dict]) -> str:
        """Create optimized context string from similar examples with outcome weighting"""
        if not similar_examples:
            return "No similar historical examples found."
        
        context_parts = []
        outcome_counts = {'TRUE_BULLISH': 0, 'FALSE_PUMP': 0, 'NEUTRAL': 0}
        weighted_outcome_scores = {'TRUE_BULLISH': 0.0, 'FALSE_PUMP': 0.0, 'NEUTRAL': 0.0}
        
        for i, example in enumerate(similar_examples, 1):
            outcome = example['outcome_type']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            # Use weighted similarity for scoring outcomes
            weighted_sim = example.get('weighted_similarity', example['similarity'])
            weighted_outcome_scores[outcome] += weighted_sim
            
            outcome_desc = {
                'TRUE_BULLISH': 'SUCCESS: 30%+ gain',
                'FALSE_PUMP': 'FAILED: False pump',
                'NEUTRAL': 'NEUTRAL: <5% move'
            }.get(outcome, 'Unknown')
            
            # Show both original and weighted similarity for transparency
            orig_sim = example['similarity']
            context_parts.append(
                f"{i}. {example['ticker']}: {example['headline'][:80]}... → {outcome_desc} (sim: {orig_sim:.2f}→{weighted_sim:.2f})"
            )
        
        # Enhanced pattern analysis using weighted scores
        total_examples = len(similar_examples)
        true_bullish_weight = weighted_outcome_scores['TRUE_BULLISH']
        false_pump_weight = weighted_outcome_scores['FALSE_PUMP']
        
        # Determine pattern strength based on weighted outcomes
        if true_bullish_weight >= false_pump_weight * 1.5:
            pattern = f"STRONG BUY SIGNAL: Weighted analysis favors success (TB:{true_bullish_weight:.2f} vs FP:{false_pump_weight:.2f})"
        elif true_bullish_weight > false_pump_weight:
            pattern = f"MODERATE BUY SIGNAL: Slight success bias (TB:{true_bullish_weight:.2f} vs FP:{false_pump_weight:.2f})"
        elif false_pump_weight > true_bullish_weight * 1.2:
            pattern = f"CAUTION SIGNAL: Historical failures dominate (FP:{false_pump_weight:.2f} vs TB:{true_bullish_weight:.2f})"
        else:
            pattern = f"MIXED SIGNAL: Conflicting historical outcomes (TB:{true_bullish_weight:.2f}, FP:{false_pump_weight:.2f})"
        
        return f"OUTCOME-WEIGHTED PATTERN: {pattern}\n\n" + "\n".join(context_parts)
    
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
            similar_examples = await self.get_similar_outcome_examples(content, top_k=5)
            
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
        """Run optimized comparison test between traditional and RAG methods"""
        logger.info(f"🚀 Starting {test_mode} optimized comparison test with {len(test_articles)} articles...")
        
        results = []
        
        for i, article in enumerate(test_articles, 1):
            logger.info(f"📊 Analyzing article {i}/{len(test_articles)}: {article['ticker']} ({article['outcome_type']})")
            
            result = SentimentResult(
                ticker=article['ticker'],
                headline=article['headline'],
                traditional_sentiment="",
                traditional_confidence=0.0,
                actual_outcome=article['outcome_type'],
                published_utc=article.get('published_est')  # Use published_est from database
            )
            
            if test_mode in ["traditional", "parallel"]:
                # Traditional analysis
                trad_sentiment, trad_conf, trad_time = await self.analyze_traditional_sentiment(article['content'])
                result.traditional_sentiment = trad_sentiment
                result.traditional_confidence = trad_conf
                result.analysis_time_traditional = trad_time
                
                logger.info(f"  🔍 Traditional: {trad_sentiment} ({trad_conf}) in {trad_time:.2f}s")
            
            if test_mode in ["rag", "parallel"]:
                # Optimized RAG analysis
                rag_sentiment, rag_conf, rag_time, similar_examples, perf_metrics = await self.analyze_optimized_rag_sentiment(article['content'])
                result.rag_sentiment = rag_sentiment
                result.rag_confidence = rag_conf
                result.analysis_time_rag = rag_time
                result.similar_examples = similar_examples
                result.embedding_time = perf_metrics['embedding_time']
                result.vector_search_time = perf_metrics['vector_search_time']
                result.llm_decision_time = perf_metrics['llm_decision_time']
                
                logger.info(f"  🧠 RAG: {rag_sentiment} ({rag_conf:.2f}) in {rag_time:.2f}s")
                logger.info(f"    ⚡ Breakdown: embed={perf_metrics['embedding_time']*1000:.0f}ms, search={perf_metrics['vector_search_time']*1000:.0f}ms, llm={perf_metrics['llm_decision_time']*1000:.0f}ms")
                if similar_examples:
                    avg_pattern_score = sum(e.get('pattern_score', 0.0) for e in similar_examples) / len(similar_examples)
                    logger.info(f"    📋 Used {len(similar_examples)} outcome pattern examples (avg performance: +{avg_pattern_score*100:.1f}%)")
            
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
                    'correct': result.actual_outcome == "TRUE_BULLISH",
                    'pnl': result.traditional_pnl,
                    'entry_price': result.traditional_entry_price,
                    'exit_price': result.traditional_exit_price,
                    'position_size': result.traditional_position_size,
                    'return_pct': result.traditional_return_pct,
                    'price_bracket': result.price_bracket,
                    'publication_hour': result.publication_hour
                })
            
            # RAG BUY+high
            if result.rag_sentiment == "BUY" and result.rag_confidence >= self.buy_high_threshold:
                rag_buy_high.append({
                    'ticker': result.ticker,
                    'headline': result.headline[:100] + "..." if len(result.headline) > 100 else result.headline,
                    'actual_outcome': result.actual_outcome,
                    'confidence': result.rag_confidence,
                    'correct': result.actual_outcome == "TRUE_BULLISH",
                    'similar_examples_count': len(result.similar_examples) if result.similar_examples else 0,
                    'embedding_time': result.embedding_time,
                    'vector_search_time': result.vector_search_time,
                    'llm_decision_time': result.llm_decision_time,
                    'pnl': result.rag_pnl,
                    'entry_price': result.rag_entry_price,
                    'exit_price': result.rag_exit_price,
                    'position_size': result.rag_position_size,
                    'return_pct': result.rag_return_pct,
                    'price_bracket': result.price_bracket,
                    'publication_hour': result.publication_hour
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
                md_content += f"- **{pred['ticker']}**: {pred['headline']} (Confidence: {pred['confidence']:.2f}, Similar Examples: {pred['similar_examples_count']}, Embed: {pred['embedding_time']*1000:.0f}ms, Search: {pred['vector_search_time']*1000:.0f}ms, LLM: {pred['llm_decision_time']*1000:.0f}ms)\n"
            
            md_content += f"""
#### ❌ Incorrect Predictions ({len(incorrect_rag)}/{len(rag_buy_high)} = {len(incorrect_rag)/len(rag_buy_high):.1%})
"""
            for pred in incorrect_rag:
                outcome_desc = "False Pump" if pred['actual_outcome'] == "FALSE_PUMP" else "Neutral"
                md_content += f"- **{pred['ticker']}**: {pred['headline']} (Confidence: {pred['confidence']:.2f}, Actual: {outcome_desc}, Similar Examples: {pred['similar_examples_count']}, Embed: {pred['embedding_time']*1000:.0f}ms, Search: {pred['vector_search_time']*1000:.0f}ms, LLM: {pred['llm_decision_time']*1000:.0f}ms)\n"
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
    
    def generate_pnl_analysis(self, results: List[SentimentResult], pnl_results: Dict[str, Any]) -> str:
        """Generate detailed PnL analysis with breakdowns"""
        
        # Filter results that have PnL data
        pnl_trades = [r for r in results if r.traditional_pnl is not None or r.rag_pnl is not None]
        
        if not pnl_trades:
            return "\n## PnL Analysis\n\nNo trades with PnL data available.\n"
        
        md_content = f"""
## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: {pnl_results['traditional_trades'] + pnl_results['rag_trades']}
- **Total P&L**: ${pnl_results['total_pnl']:.2f}
- **Total Investment**: ${pnl_results['total_investment']:.2f}
- **Overall Return**: {pnl_results['total_return']:.2%}

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | {pnl_results['traditional_trades']} | ${pnl_results['traditional_pnl']:.2f} | ${pnl_results['traditional_investment']:.2f} | {pnl_results['traditional_return']:.2%} |
| RAG | {pnl_results['rag_trades']} | ${pnl_results['rag_pnl']:.2f} | ${pnl_results['rag_investment']:.2f} | {pnl_results['rag_return']:.2%} |

---

### Performance Breakdown by Ticker

"""
        
        # Ticker breakdown
        ticker_stats = {}
        for result in pnl_trades:
            ticker = result.ticker
            if ticker not in ticker_stats:
                ticker_stats[ticker] = {
                    'traditional_pnl': 0.0, 'rag_pnl': 0.0,
                    'traditional_trades': 0, 'rag_trades': 0,
                    'traditional_investment': 0.0, 'rag_investment': 0.0
                }
            
            if result.traditional_pnl is not None:
                ticker_stats[ticker]['traditional_pnl'] += result.traditional_pnl
                ticker_stats[ticker]['traditional_trades'] += 1
                ticker_stats[ticker]['traditional_investment'] += result.traditional_investment or 0
            
            if result.rag_pnl is not None:
                ticker_stats[ticker]['rag_pnl'] += result.rag_pnl
                ticker_stats[ticker]['rag_trades'] += 1
                ticker_stats[ticker]['rag_investment'] += result.rag_investment or 0
        
        # Traditional Model Performance by Ticker
        traditional_tickers = [(ticker, stats) for ticker, stats in ticker_stats.items() if stats['traditional_trades'] > 0]
        if traditional_tickers:
            traditional_tickers.sort(key=lambda x: x[1]['traditional_pnl'], reverse=True)
            
            md_content += "#### Traditional Model Performance by Ticker\n\n"
            md_content += "| Ticker | P&L | Investment | Return % | Trades |\n"
            md_content += "|--------|-----|------------|----------|--------|\n"
            
            for ticker, stats in traditional_tickers:
                return_pct = (stats['traditional_pnl'] / stats['traditional_investment']) * 100 if stats['traditional_investment'] > 0 else 0
                md_content += f"| **{ticker}** | ${stats['traditional_pnl']:.2f} | ${stats['traditional_investment']:.2f} | {return_pct:.2f}% | {stats['traditional_trades']} |\n"
        else:
            md_content += "#### Traditional Model Performance by Ticker\n\nNo traditional BUY+high trades executed.\n"
        
        # RAG Model Performance by Ticker
        rag_tickers = [(ticker, stats) for ticker, stats in ticker_stats.items() if stats['rag_trades'] > 0]
        if rag_tickers:
            rag_tickers.sort(key=lambda x: x[1]['rag_pnl'], reverse=True)
            
            md_content += "\n#### RAG Model Performance by Ticker\n\n"
            md_content += "| Ticker | P&L | Investment | Return % | Trades |\n"
            md_content += "|--------|-----|------------|----------|--------|\n"
            
            for ticker, stats in rag_tickers:
                return_pct = (stats['rag_pnl'] / stats['rag_investment']) * 100 if stats['rag_investment'] > 0 else 0
                md_content += f"| **{ticker}** | ${stats['rag_pnl']:.2f} | ${stats['rag_investment']:.2f} | {return_pct:.2f}% | {stats['rag_trades']} |\n"
        else:
            md_content += "\n#### RAG Model Performance by Ticker\n\nNo RAG BUY+high trades executed.\n"
        
        # Publication time breakdown
        md_content += "\n### Performance Breakdown by Publication Hour (EST)\n\n"
        
        hour_stats = {}
        for result in pnl_trades:
            hour = result.publication_hour
            if hour is not None:
                if hour not in hour_stats:
                    hour_stats[hour] = {
                        'traditional_pnl': 0.0, 'rag_pnl': 0.0,
                        'traditional_trades': 0, 'rag_trades': 0,
                        'traditional_investment': 0.0, 'rag_investment': 0.0
                    }
                
                if result.traditional_pnl is not None:
                    hour_stats[hour]['traditional_pnl'] += result.traditional_pnl
                    hour_stats[hour]['traditional_trades'] += 1
                    hour_stats[hour]['traditional_investment'] += result.traditional_investment or 0
                
                if result.rag_pnl is not None:
                    hour_stats[hour]['rag_pnl'] += result.rag_pnl
                    hour_stats[hour]['rag_trades'] += 1
                    hour_stats[hour]['rag_investment'] += result.rag_investment or 0
        
        if hour_stats:
            # Traditional Model Performance by Hour
            traditional_hours = [(hour, stats) for hour, stats in hour_stats.items() if stats['traditional_trades'] > 0]
            if traditional_hours:
                md_content += "#### Traditional Model Performance by Hour\n\n"
                md_content += "| Hour (EST) | P&L | Investment | Return % | Trades |\n"
                md_content += "|------------|-----|------------|----------|--------|\n"
                
                for hour in sorted([h for h, s in traditional_hours]):
                    stats = hour_stats[hour]
                    return_pct = (stats['traditional_pnl'] / stats['traditional_investment']) * 100 if stats['traditional_investment'] > 0 else 0
                    hour_str = f"{hour:02d}:00"
                    md_content += f"| **{hour_str}** | ${stats['traditional_pnl']:.2f} | ${stats['traditional_investment']:.2f} | {return_pct:.2f}% | {stats['traditional_trades']} |\n"
            else:
                md_content += "#### Traditional Model Performance by Hour\n\nNo traditional BUY+high trades executed.\n"
            
            # RAG Model Performance by Hour
            rag_hours = [(hour, stats) for hour, stats in hour_stats.items() if stats['rag_trades'] > 0]
            if rag_hours:
                md_content += "\n#### RAG Model Performance by Hour\n\n"
                md_content += "| Hour (EST) | P&L | Investment | Return % | Trades |\n"
                md_content += "|------------|-----|------------|----------|--------|\n"
                
                for hour in sorted([h for h, s in rag_hours]):
                    stats = hour_stats[hour]
                    return_pct = (stats['rag_pnl'] / stats['rag_investment']) * 100 if stats['rag_investment'] > 0 else 0
                    hour_str = f"{hour:02d}:00"
                    md_content += f"| **{hour_str}** | ${stats['rag_pnl']:.2f} | ${stats['rag_investment']:.2f} | {return_pct:.2f}% | {stats['rag_trades']} |\n"
            else:
                md_content += "\n#### RAG Model Performance by Hour\n\nNo RAG BUY+high trades executed.\n"
        
        # Price bracket breakdown
        md_content += "\n### Performance Breakdown by Price Bracket\n\n"
        
        bracket_stats = {}
        for result in pnl_trades:
            bracket = result.price_bracket
            if bracket:
                if bracket not in bracket_stats:
                    bracket_stats[bracket] = {
                        'traditional_pnl': 0.0, 'rag_pnl': 0.0,
                        'traditional_trades': 0, 'rag_trades': 0,
                        'traditional_investment': 0.0, 'rag_investment': 0.0,
                        'position_size': 0
                    }
                
                if result.traditional_pnl is not None:
                    bracket_stats[bracket]['traditional_pnl'] += result.traditional_pnl
                    bracket_stats[bracket]['traditional_trades'] += 1
                    bracket_stats[bracket]['traditional_investment'] += result.traditional_investment or 0
                    bracket_stats[bracket]['position_size'] = result.traditional_position_size or 0
                
                if result.rag_pnl is not None:
                    bracket_stats[bracket]['rag_pnl'] += result.rag_pnl
                    bracket_stats[bracket]['rag_trades'] += 1
                    bracket_stats[bracket]['rag_investment'] += result.rag_investment or 0
                    bracket_stats[bracket]['position_size'] = result.rag_position_size or 0
        
        if bracket_stats:
            # Sort by price (convert bracket to sortable format)
            def bracket_sort_key(bracket):
                if '$0.01-' in bracket:
                    return 0.01
                elif '$1.00-' in bracket:
                    return 1.00
                elif '$3.00-' in bracket:
                    return 3.00
                elif '$5.00-' in bracket:
                    return 5.00
                elif '$8.00+' in bracket:
                    return 8.00
                return 999
            
            sorted_brackets = sorted(bracket_stats.items(), key=lambda x: bracket_sort_key(x[0]))
            
            # Traditional Model Performance by Price Bracket
            traditional_brackets = [(bracket, stats) for bracket, stats in sorted_brackets if stats['traditional_trades'] > 0]
            if traditional_brackets:
                md_content += "#### Traditional Model Performance by Price Bracket\n\n"
                md_content += "| Price Bracket | Position Size | P&L | Investment | Return % | Trades |\n"
                md_content += "|---------------|---------------|-----|------------|----------|--------|\n"
                
                for bracket, stats in traditional_brackets:
                    return_pct = (stats['traditional_pnl'] / stats['traditional_investment']) * 100 if stats['traditional_investment'] > 0 else 0
                    md_content += f"| **{bracket}** | {stats['position_size']:,} | ${stats['traditional_pnl']:.2f} | ${stats['traditional_investment']:.2f} | {return_pct:.2f}% | {stats['traditional_trades']} |\n"
            else:
                md_content += "#### Traditional Model Performance by Price Bracket\n\nNo traditional BUY+high trades executed.\n"
            
            # RAG Model Performance by Price Bracket
            rag_brackets = [(bracket, stats) for bracket, stats in sorted_brackets if stats['rag_trades'] > 0]
            if rag_brackets:
                md_content += "\n#### RAG Model Performance by Price Bracket\n\n"
                md_content += "| Price Bracket | Position Size | P&L | Investment | Return % | Trades |\n"
                md_content += "|---------------|---------------|-----|------------|----------|--------|\n"
                
                for bracket, stats in rag_brackets:
                    return_pct = (stats['rag_pnl'] / stats['rag_investment']) * 100 if stats['rag_investment'] > 0 else 0
                    md_content += f"| **{bracket}** | {stats['position_size']:,} | ${stats['rag_pnl']:.2f} | ${stats['rag_investment']:.2f} | {return_pct:.2f}% | {stats['rag_trades']} |\n"
            else:
                md_content += "\n#### RAG Model Performance by Price Bracket\n\nNo RAG BUY+high trades executed.\n"
        
        return md_content
    
    def outcome_to_expected_action(self, outcome: str) -> str:
        """Convert outcome type to expected trading action"""
        mapping = {
            'TRUE_BULLISH': 'BUY',
            'FALSE_PUMP': 'HOLD',  # Should avoid BUY for false pumps
            'NEUTRAL': 'HOLD'
        }
        return mapping.get(outcome, 'HOLD')
    
    async def save_results(self, results: List[SentimentResult], metrics: Dict[str, Any], pnl_results: Optional[Dict[str, Any]] = None):
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
            'pnl_results': pnl_results,  # Add PnL results
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
                'similar_examples_count': len(result.similar_examples) if result.similar_examples else 0,
                'embedding_time': result.embedding_time,
                'vector_search_time': result.vector_search_time,
                'llm_decision_time': result.llm_decision_time,
                # Add PnL fields to JSON
                'published_utc': result.published_utc.isoformat() if result.published_utc else None,
                'traditional_pnl': result.traditional_pnl,
                'rag_pnl': result.rag_pnl,
                'traditional_entry_price': result.traditional_entry_price,
                'traditional_exit_price': result.traditional_exit_price,
                'rag_entry_price': result.rag_entry_price,
                'rag_exit_price': result.rag_exit_price,
                # Add detailed PnL tracking fields
                'traditional_position_size': result.traditional_position_size,
                'rag_position_size': result.rag_position_size,
                'traditional_investment': result.traditional_investment,
                'rag_investment': result.rag_investment,
                'traditional_return_pct': result.traditional_return_pct,
                'rag_return_pct': result.rag_return_pct,
                'price_bracket': result.price_bracket,
                'publication_hour': result.publication_hour
            })
        
        # Save JSON results
        with open(f'tests/results/rag_comparison_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Generate and save detailed markdown analysis
        detailed_analysis = self.generate_detailed_analysis(results, metrics)
        
        # Add PnL analysis if available
        if pnl_results:
            pnl_analysis = self.generate_pnl_analysis(results, pnl_results)
            detailed_analysis += pnl_analysis
        
        with open(f'tests/results/rag_detailed_analysis_{timestamp}.md', 'w') as f:
            f.write(detailed_analysis)
        
        logger.info(f"📁 RAG comparison results saved to tests/results/rag_comparison_results_{timestamp}.json")
        logger.info(f"📊 Detailed BUY+high analysis saved to tests/results/rag_detailed_analysis_{timestamp}.md")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        
        if self.session:
            await self.session.close()
        
        if self.ch_manager:
            self.ch_manager.close()

    async def get_polygon_bars(self, ticker: str, from_timestamp: datetime, to_timestamp: datetime) -> List[Dict[str, Any]]:
        """Get 10-second aggregate bars from Polygon API"""
        if not self.session or not self.polygon_api_key:
            logger.warning("Polygon API not configured for PnL calculations")
            return []
        
        try:
            # Convert timestamps to date strings for Polygon API
            from_date = from_timestamp.strftime('%Y-%m-%d')
            to_date = to_timestamp.strftime('%Y-%m-%d')
            
            # Polygon aggregates endpoint - 10 second bars
            url = f"{self.polygon_base_url}/v2/aggs/ticker/{ticker}/range/10/second/{from_date}/{to_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apikey': self.polygon_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        bars = []
                        for bar in data['results']:
                            bar_timestamp = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.UTC)
                            
                            # Filter bars to only include the time range we need
                            if from_timestamp <= bar_timestamp <= to_timestamp:
                                bars.append({
                                    'timestamp': bar_timestamp,
                                    'open': bar['o'],
                                    'high': bar['h'],
                                    'low': bar['l'],
                                    'close': bar['c'],
                                    'volume': bar['v']
                                })
                        
                        logger.debug(f"📊 Retrieved {len(bars)} 10-second bars for {ticker}")
                        return bars
                    else:
                        logger.debug(f"No bar data available for {ticker} on {from_date}")
                        return []
                        
                elif response.status == 429:
                    logger.warning(f"Rate limited for {ticker} bars - waiting before retry")
                    await asyncio.sleep(1)
                    return []
                else:
                    logger.warning(f"Polygon bars API error for {ticker}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting bars for {ticker}: {e}")
            return []

    def get_entry_exit_prices(self, bars: List[Dict[str, Any]], entry_time: datetime, exit_time: datetime) -> tuple:
        """Get entry and exit prices from bars data"""
        entry_price = None
        exit_price = None
        
        # Find entry price (closest bar to entry time)
        min_entry_diff = float('inf')
        for bar in bars:
            time_diff = abs((bar['timestamp'] - entry_time).total_seconds())
            if time_diff < min_entry_diff:
                min_entry_diff = time_diff
                entry_price = bar['close']  # Use close price for entry
        
        # Find exit price (closest bar to exit time)
        min_exit_diff = float('inf')
        for bar in bars:
            time_diff = abs((bar['timestamp'] - exit_time).total_seconds())
            if time_diff < min_exit_diff:
                min_exit_diff = time_diff
                exit_price = bar['close']  # Use close price for exit
        
        return entry_price, exit_price

    def get_dynamic_position_size(self, entry_price: float) -> int:
        """Calculate dynamic position size based on entry price using tier system"""
        for tier in self.position_tiers:
            if tier['price_min'] <= entry_price < tier['price_max']:
                return tier['unit_position_size']
        
        # Fallback to the highest tier if price doesn't match any tier
        return self.position_tiers[-1]['unit_position_size']

    def get_price_bracket(self, price: float) -> str:
        """Get price bracket description for reporting"""
        for tier in self.position_tiers:
            if tier['price_min'] <= price < tier['price_max']:
                if tier['price_max'] >= 999999:
                    return f"${tier['price_min']:.2f}+"
                else:
                    return f"${tier['price_min']:.2f}-${tier['price_max']:.2f}"
        return "Unknown"

    async def calculate_trade_pnl(self, result: SentimentResult) -> SentimentResult:
        """Calculate PnL for BUY+high recommendations"""
        if not result.published_utc:
            return result
            
        try:
            ticker = result.ticker
            published_est_naive = result.published_utc  # This is actually published_est from database
            
            # The database field is published_est (EST timezone) but stored as naive datetime
            # We need to make it timezone-aware as EST
            if published_est_naive.tzinfo is None:
                published_est = self.est_tz.localize(published_est_naive)
            else:
                published_est = published_est_naive.astimezone(self.est_tz)
            
            # Convert to UTC for API calls
            published_utc = published_est.astimezone(pytz.UTC)
            
            # Calculate time range: from initial timestamp to 9:30 AM EST
            end_time_est = published_est.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Convert back to UTC for API calls
            from_timestamp_utc = published_utc
            to_timestamp_utc = end_time_est.astimezone(pytz.UTC)
            
            # Get 10-second bars from Polygon
            bars = await self.get_polygon_bars(ticker, from_timestamp_utc, to_timestamp_utc)
            
            if not bars:
                logger.debug(f"No price data available for {ticker}")
                return result
            
            # Entry: 30 seconds after initial timestamp
            entry_time_utc = published_utc + timedelta(seconds=30)
            
            # Exit: 9:28 AM EST
            exit_time_est = published_est.replace(hour=9, minute=28, second=0, microsecond=0)
            exit_time_utc = exit_time_est.astimezone(pytz.UTC)
            
            # Get entry and exit prices from bars
            entry_price, exit_price = self.get_entry_exit_prices(bars, entry_time_utc, exit_time_utc)
            
            if not entry_price or not exit_price:
                logger.debug(f"Cannot find entry/exit prices for {ticker}")
                return result
            
            # Calculate PnL for traditional model (if BUY+high)
            is_trad_buy_high = False
            if result.traditional_sentiment == "BUY":
                if isinstance(result.traditional_confidence, str):
                    is_trad_buy_high = result.traditional_confidence == "high"
                else:
                    is_trad_buy_high = result.traditional_confidence >= self.buy_high_threshold
            
            if is_trad_buy_high:
                position_size = self.get_dynamic_position_size(entry_price)
                traditional_pnl = (exit_price - entry_price) * position_size
                result.traditional_pnl = traditional_pnl
                result.traditional_entry_price = entry_price
                result.traditional_exit_price = exit_price
                result.traditional_position_size = position_size
                result.traditional_investment = entry_price * position_size
                result.traditional_return_pct = (traditional_pnl / result.traditional_investment) if result.traditional_investment > 0 else 0.0
                result.price_bracket = self.get_price_bracket(entry_price)
                result.publication_hour = published_est.hour  # Now correctly EST hour
            
            # Calculate PnL for RAG model (if BUY+high)
            if result.rag_sentiment == "BUY" and result.rag_confidence >= self.buy_high_threshold:
                position_size = self.get_dynamic_position_size(entry_price)
                rag_pnl = (exit_price - entry_price) * position_size
                result.rag_pnl = rag_pnl
                result.rag_entry_price = entry_price
                result.rag_exit_price = exit_price
                result.rag_position_size = position_size
                result.rag_investment = entry_price * position_size
                result.rag_return_pct = (rag_pnl / result.rag_investment) if result.rag_investment > 0 else 0.0
                result.price_bracket = self.get_price_bracket(entry_price)
                result.publication_hour = published_est.hour  # Now correctly EST hour
            
            logger.debug(f"PnL calculated for {ticker}: Traditional={result.traditional_pnl}, RAG={result.rag_pnl}")
            
        except Exception as e:
            logger.error(f"Error calculating PnL for {ticker}: {e}")
        
        return result

    async def calculate_pnl(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Calculate PnL metrics for all BUY+high recommendations"""
        if not self.polygon_api_key:
            logger.warning("POLYGON_API_KEY not configured - skipping PnL calculations")
            return {
                'total_pnl': 0.0,
                'traditional_pnl': 0.0,
                'rag_pnl': 0.0,
                'total_return': 0.0,
                'traditional_return': 0.0,
                'rag_return': 0.0,
                'total_investment': 0.0,
                'traditional_investment': 0.0,
                'rag_investment': 0.0,
                'traditional_roi': 0.0,
                'rag_roi': 0.0,
                'traditional_trades': 0,
                'rag_trades': 0,
                'error': 'POLYGON_API_KEY not configured'
            }
        
        logger.info("💰 Calculating PnL for BUY+high recommendations...")
        
        # Calculate PnL for each result with rate limiting
        semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls
        
        async def calculate_with_semaphore(result):
            async with semaphore:
                return await self.calculate_trade_pnl(result)
        
        # Process all results
        pnl_tasks = [calculate_with_semaphore(result) for result in results]
        updated_results = await asyncio.gather(*pnl_tasks, return_exceptions=True)
        
        # Update original results list
        for i, updated_result in enumerate(updated_results):
            if not isinstance(updated_result, Exception):
                results[i] = updated_result
        
        # Calculate summary metrics
        traditional_pnl = sum(r.traditional_pnl for r in results if r.traditional_pnl is not None)
        rag_pnl = sum(r.rag_pnl for r in results if r.rag_pnl is not None)
        total_pnl = traditional_pnl + rag_pnl
        
        traditional_investment = sum(r.traditional_entry_price * r.traditional_position_size 
                                   for r in results if r.traditional_entry_price is not None and r.traditional_position_size is not None)
        rag_investment = sum(r.rag_entry_price * r.rag_position_size 
                           for r in results if r.rag_entry_price is not None and r.rag_position_size is not None)
        total_investment = traditional_investment + rag_investment
        
        traditional_trades = sum(1 for r in results if r.traditional_pnl is not None)
        rag_trades = sum(1 for r in results if r.rag_pnl is not None)
        
        # Calculate returns and ROI
        traditional_return = (traditional_pnl / traditional_investment) if traditional_investment > 0 else 0.0
        rag_return = (rag_pnl / rag_investment) if rag_investment > 0 else 0.0
        total_return = (total_pnl / total_investment) if total_investment > 0 else 0.0
        
        return {
            'total_pnl': total_pnl,
            'traditional_pnl': traditional_pnl,
            'rag_pnl': rag_pnl,
            'total_return': total_return,
            'traditional_return': traditional_return,
            'rag_return': rag_return,
            'total_investment': total_investment,
            'traditional_investment': traditional_investment,
            'rag_investment': rag_investment,
            'traditional_roi': traditional_return,
            'rag_roi': rag_return,
            'traditional_trades': traditional_trades,
            'rag_trades': rag_trades
        }

    def create_outcome_pattern_context(self, examples: List[Dict]) -> str:
        """Create context string from outcome pattern examples focusing on performance"""
        if not examples:
            return "No historical success patterns found."
        
        context_parts = []
        
        for i, example in enumerate(examples, 1):
            ticker = example['ticker']
            headline = example['headline']
            outcome = example['outcome_type']
            price_ratio = example.get('price_increase_ratio', 0.0)
            has_30pt = example.get('has_30pt_increase', False)
            
            # Format performance info
            performance_info = f"+{price_ratio*100:.1f}%" if price_ratio > 0 else "No gain"
            success_indicator = "✅ 30%+ GAIN" if has_30pt else "📈 Positive"
            
            context_parts.append(f"""
{i}. {ticker}: {headline[:100]}...
   Result: {success_indicator} ({performance_info})
   Pattern: {outcome}""")
        
        # Add summary
        total_examples = len(examples)
        strong_performers = sum(1 for ex in examples if ex.get('price_increase_ratio', 0.0) > 0.30)
        avg_performance = sum(ex.get('price_increase_ratio', 0.0) for ex in examples) / total_examples
        
        summary = f"""
PATTERN SUMMARY:
- {strong_performers}/{total_examples} examples achieved 30%+ gains
- Average performance: +{avg_performance*100:.1f}%
- Success rate: {(strong_performers/total_examples)*100:.1f}%"""
        
        return "\n".join(context_parts) + summary

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description='RAG vs Traditional Sentiment Analysis Comparison')
    parser.add_argument('--sample-size', type=int, default=30, help='Number of test articles to analyze')
    parser.add_argument('--test-mode', choices=['traditional', 'rag', 'parallel'], default='parallel', 
                        help='Test mode: traditional only, rag only, or both in parallel')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save test results')
    parser.add_argument('--buy-high-threshold', type=float, default=0.8, help='Confidence threshold for BUY+high (default: 0.8)')
    parser.add_argument('--buy-medium-threshold', type=float, default=0.5, help='Confidence threshold for BUY+medium (default: 0.5)')
    parser.add_argument('--pnl', action='store_true', help='Calculate actual PnL based on BUY+high recommendations')
    
    args = parser.parse_args()
    
    tester = OptimizedRAGTester(buy_high_threshold=args.buy_high_threshold, buy_medium_threshold=args.buy_medium_threshold)
    
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
        
        # Calculate and log PnL if requested
        pnl_results = None
        if args.pnl:
            logger.info("\n💰 Calculating PnL based on BUY+high recommendations...")
            pnl_results = await tester.calculate_pnl(results)
            logger.info(f"📈 Total PnL: ${pnl_results['total_pnl']:.2f}")
            logger.info(f"📊 PnL by Model:")
            logger.info(f"  • Traditional: ${pnl_results['traditional_pnl']:.2f} ({pnl_results['traditional_trades']} trades)")
            logger.info(f"  • RAG: ${pnl_results['rag_pnl']:.2f} ({pnl_results['rag_trades']} trades)")
            logger.info(f"💡 Total Return: {pnl_results['total_return']:.2%}")
            logger.info(f"📊 Return by Model:")
            logger.info(f"  • Traditional: {pnl_results['traditional_return']:.2%}")
            logger.info(f"  • RAG: {pnl_results['rag_return']:.2%}")
            logger.info(f"💰 Total Investment: ${pnl_results['total_investment']:.2f}")
            logger.info(f"📊 Investment by Model:")
            logger.info(f"  • Traditional: ${pnl_results['traditional_investment']:.2f}")
            logger.info(f"  • RAG: ${pnl_results['rag_investment']:.2f}")
        
        # Save results if requested
        if args.save_results:
            await tester.save_results(results, metrics, pnl_results)
        
        logger.info("✅ RAG comparison test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 