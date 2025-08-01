#!/usr/bin/env python3
"""
Generate Claude Feature Vectors for RAG Training Dataset

This script pre-computes Claude feature vectors for all articles in the 
rag_training_dataset and stores them in rag_article_vectors for fast similarity search.

Usage:
    python3 tests/generate_vectors.py --batch-size 50
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
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

class VectorGenerator:
    """Generate and store Claude feature vectors for RAG training dataset"""
    
    def __init__(self, batch_size: int = 50):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.batch_size = batch_size
        self.feature_cache = {}
        
    async def initialize(self):
        """Initialize the vector generator"""
        logger.info("🧪 Initializing Claude Vector Generator...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize sentiment service (for Claude API access)
        self.sentiment_service = await get_sentiment_service()
        
        logger.info("✅ Vector generator initialized successfully")
    
    async def generate_claude_features(self, content: str) -> List[float]:
        """Generate Claude feature vector for given content"""
        
        # Use content hash for caching
        content_hash = str(hash(content))
        if content_hash in self.feature_cache:
            return self.feature_cache[content_hash]
        
        try:
            # Claude feature extraction prompt
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
            
            # Use existing Claude API through sentiment service
            result = await self.sentiment_service.load_balancer.make_claude_request(analysis_prompt)
            
            if result and isinstance(result, dict):
                # Convert Claude's analysis to feature vector
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
                    # Add some composite features
                    if len(feature_vector) < 30:
                        feature_vector.append(feature_vector[0] * feature_vector[1])  # sentiment * bullish
                    elif len(feature_vector) < 40:
                        feature_vector.append(feature_vector[2] * feature_vector[3])  # urgency * impact
                    else:
                        feature_vector.append(0.5)  # neutral padding
                
                # Cache the result
                self.feature_cache[content_hash] = feature_vector
                return feature_vector
            else:
                # Return neutral feature vector on failure
                return [0.5] * 50
                
        except Exception as e:
            logger.error(f"Claude feature extraction failed: {e}")
            # Return neutral feature vector on failure
            return [0.5] * 50
    
    async def get_training_articles(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get batch of articles from training dataset"""
        try:
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
            FROM News.rag_training_dataset
            ORDER BY outcome_type, ticker
            LIMIT %s OFFSET %s
            """
            
            result = self.ch_manager.client.query(query, parameters=[limit, offset])
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'full_content': row[2] or row[1],  # Use headline if full_content is empty
                    'outcome_type': row[3],
                    'has_30pt_increase': int(row[4]),
                    'is_false_pump': int(row[5]),
                    'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                    'original_content_hash': row[7],
                    'published_est': row[8],
                    'content': f"{row[0]}: {row[1]}"  # Ticker + headline for analysis
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching training articles: {e}")
            return []
    
    async def get_training_articles_balanced(self, batch_size: int = 25, processed_counts: dict = None) -> List[Dict[str, Any]]:
        """Get balanced batch of articles from each outcome type FROM TRAINING SET ONLY"""
        if processed_counts is None:
            processed_counts = {'TRUE_BULLISH': 0, 'FALSE_PUMP': 0, 'NEUTRAL': 0}
            
        try:
            # Get approximately equal numbers from each outcome type
            per_outcome = max(1, batch_size // 3)  # Divide batch size by 3 outcome types
            
            all_articles = []
            
            # Get articles from each outcome type with proper offset - FROM TRAINING SET ONLY
            for outcome_type in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
                offset = processed_counts.get(outcome_type, 0)
                
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
                FROM News.rag_training_set
                WHERE outcome_type = %s
                ORDER BY selection_priority DESC, ticker
                LIMIT %s OFFSET %s
                """
                
                result = self.ch_manager.client.query(query, parameters=[outcome_type, per_outcome, offset])
                
                batch_articles = []
                for row in result.result_rows:
                    article = {
                        'ticker': row[0],
                        'headline': row[1],
                        'full_content': row[2] or row[1],  # Use headline if full_content is empty
                        'outcome_type': row[3],
                        'has_30pt_increase': int(row[4]),
                        'is_false_pump': int(row[5]),
                        'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                        'original_content_hash': row[7],
                        'published_est': row[8],
                        'content': row[2] or f"{row[0]}: {row[1]}"  # Use full_content if available, fallback to ticker + headline
                    }
                    batch_articles.append(article)
                    all_articles.append(article)
                
                # Update processed count for this outcome type
                processed_counts[outcome_type] += len(batch_articles)
                logger.info(f"  📊 Got {len(batch_articles)} {outcome_type} articles (total processed: {processed_counts[outcome_type]})")
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error fetching balanced training articles: {e}")
            return []
    
    async def store_vectors(self, articles_with_vectors: List[Dict[str, Any]]):
        """Store articles with their feature vectors in the TRAINING vectors database"""
        try:
            # Prepare data for batch insert (only the columns we want to insert)
            data_rows = []
            for article in articles_with_vectors:
                data_rows.append([
                    article['ticker'],
                    article['headline'],
                    article['full_content'],
                    article['feature_vector'],
                    'claude-3-5-sonnet-20240620',  # feature_model
                    article['outcome_type'],
                    article['has_30pt_increase'],
                    article['is_false_pump'],
                    article['price_increase_ratio'],
                    article['original_content_hash'],
                    article['published_est']
                ])
            
            # Specify the columns we're inserting (excluding id and created_at which have defaults)
            columns = [
                'ticker', 'headline', 'full_content', 'feature_vector', 'feature_model',
                'outcome_type', 'has_30pt_increase', 'is_false_pump', 'price_increase_ratio',
                'original_content_hash', 'published_est'
            ]
            
            # Batch insert with explicit column specification - use TRAINING vectors table
            self.ch_manager.client.insert('News.rag_training_vectors', data_rows, column_names=columns)
            logger.info(f"✅ Stored {len(data_rows)} vectors in TRAINING database")
            
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise
    
    async def generate_all_vectors(self):
        """Generate feature vectors for all articles in the TRAINING dataset"""
        logger.info("🚀 Starting vector generation for TRAINING articles only...")
        
        # Create training vectors table if it doesn't exist
        await self.create_training_vectors_table()
        
        # Get total count from TRAINING SET
        total_query = 'SELECT COUNT(*) FROM News.rag_training_set'
        total_articles = self.ch_manager.client.query(total_query).result_rows[0][0]
        logger.info(f"📊 Total TRAINING articles to process: {total_articles}")
        
        processed = 0
        start_time = datetime.now()
        
        # Track processed articles by outcome type (training set counts)
        processed_counts = {'TRUE_BULLISH': 0, 'FALSE_PUMP': 0, 'NEUTRAL': 0}
        remaining_by_outcome = {
            'TRUE_BULLISH': 140,  # From train/test split
            'FALSE_PUMP': 140,
            'NEUTRAL': 140
        }
        
        batch_num = 1
        while any(count > 0 for count in remaining_by_outcome.values()):
            batch_start = datetime.now()
            logger.info(f"📦 Processing batch {batch_num}")
            logger.info(f"  📊 Remaining: TRUE_BULLISH={remaining_by_outcome['TRUE_BULLISH']}, FALSE_PUMP={remaining_by_outcome['FALSE_PUMP']}, NEUTRAL={remaining_by_outcome['NEUTRAL']}")
            
            # Get balanced batch with proper offset
            articles = await self.get_training_articles_balanced(self.batch_size, processed_counts)
            if not articles:
                logger.info("  ⚠️ No more articles to process")
                break
            
            # Generate vectors for this batch
            articles_with_vectors = []
            for i, article in enumerate(articles, 1):
                logger.info(f"  🔬 Generating features for {article['ticker']} ({article['outcome_type']}) ({i}/{len(articles)})")
                
                # Generate Claude features
                feature_vector = await self.generate_claude_features(article['content'])
                
                # Add vector to article data
                article['feature_vector'] = feature_vector
                articles_with_vectors.append(article)
                
                # Update remaining count
                remaining_by_outcome[article['outcome_type']] -= 1
            
            # Store this batch
            await self.store_vectors(articles_with_vectors)
            
            processed += len(articles)
            batch_time = (datetime.now() - batch_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Batch {batch_num} completed in {batch_time:.1f}s | Progress: {processed}/{total_articles} ({processed/total_articles*100:.1f}%)")
            logger.info(f"⏱️ Total time: {total_time:.1f}s")
            
            batch_num += 1
        
        logger.info(f"🎉 TRAINING vector generation completed! Processed {processed} articles in {total_time:.1f}s")
    
    async def create_training_vectors_table(self):
        """Create the training vectors table"""
        try:
            # Drop existing training vectors table if it exists
            self.ch_manager.client.command("DROP TABLE IF EXISTS News.rag_training_vectors")
            
            # Create new training vectors table (same schema as rag_article_vectors)
            create_table_sql = """
            CREATE TABLE News.rag_training_vectors (
                id UUID DEFAULT generateUUIDv4(),
                ticker String,
                headline String,
                full_content String,
                
                -- Claude-generated features
                feature_vector Array(Float32),
                feature_model String DEFAULT 'claude-3-5-sonnet-20240620',
                
                -- Outcome labels
                outcome_type String,
                has_30pt_increase UInt8,
                is_false_pump UInt8,
                price_increase_ratio Float64,
                
                -- Content identification
                original_content_hash String,
                published_est DateTime,
                
                -- Metadata
                created_at DateTime DEFAULT now(),
                
                -- Indexing
                INDEX idx_outcome_type (outcome_type) TYPE set(10) GRANULARITY 1,
                INDEX idx_ticker (ticker) TYPE set(1000) GRANULARITY 1
            ) ENGINE = ReplacingMergeTree(created_at)
            ORDER BY (ticker, original_content_hash)
            """
            
            self.ch_manager.client.command(create_table_sql)
            logger.info("✅ TRAINING vectors table created successfully")
            
        except Exception as e:
            logger.error(f"Error creating training vectors table: {e}")
            raise
    
    async def verify_vectors(self):
        """Verify that TRAINING vectors were stored correctly"""
        logger.info("🔍 Verifying stored TRAINING vectors...")
        
        # Check total count
        count_query = 'SELECT COUNT(*) FROM News.rag_training_vectors'
        vector_count = self.ch_manager.client.query(count_query).result_rows[0][0]
        
        # Check distribution by outcome
        dist_query = '''
        SELECT 
            outcome_type,
            COUNT(*) as count,
            AVG(arraySum(feature_vector)) as avg_feature_sum
        FROM News.rag_training_vectors
        GROUP BY outcome_type
        ORDER BY outcome_type
        '''
        
        result = self.ch_manager.client.query(dist_query)
        
        logger.info(f"📊 TRAINING Vector Storage Verification:")
        logger.info(f"  • Total vectors: {vector_count}")
        
        for row in result.result_rows:
            outcome, count, avg_sum = row
            logger.info(f"  • {outcome}: {count} vectors (avg feature sum: {avg_sum:.2f})")
        
        # Sample a few vectors
        sample_query = '''
        SELECT ticker, outcome_type, arraySum(feature_vector) as feature_sum
        FROM News.rag_training_vectors
        ORDER BY outcome_type, ticker
        LIMIT 5
        '''
        
        sample_result = self.ch_manager.client.query(sample_query)
        logger.info("📋 Sample TRAINING vectors:")
        for row in sample_result.result_rows:
            ticker, outcome, feature_sum = row
            logger.info(f"  • {ticker} ({outcome}): feature sum = {feature_sum:.2f}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sentiment_service:
            await self.sentiment_service.cleanup()
        
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate Claude Feature Vectors for RAG')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size for processing')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing vectors')
    
    args = parser.parse_args()
    
    generator = VectorGenerator(batch_size=args.batch_size)
    
    try:
        await generator.initialize()
        
        if args.verify_only:
            await generator.verify_vectors()
        else:
            await generator.generate_all_vectors()
            await generator.verify_vectors()
        
    except Exception as e:
        logger.error(f"Vector generation failed: {e}")
        raise
    finally:
        await generator.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 