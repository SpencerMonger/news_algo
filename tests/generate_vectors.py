#!/usr/bin/env python3
"""
Generate E5 Embedding Vectors for RAG Training Dataset

This script pre-computes E5 embedding vectors for all articles in the 
rag_training_dataset and stores them in rag_training_vectors for fast similarity search.

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

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.error("sentence-transformers not available. Install with: pip install sentence-transformers")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class E5VectorGenerator:
    """Generate and store E5 embedding vectors for RAG training dataset"""
    
    def __init__(self, batch_size: int = 50):
        self.ch_manager = ClickHouseManager()
        self.sentiment_service = None
        self.embedding_model = None
        self.batch_size = batch_size
        
    async def initialize(self):
        """Initialize the vector generator"""
        logger.info("🚀 Initializing E5 Vector Generator...")
        
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers package required. Install with: pip install sentence-transformers")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize embedding model
        await self.initialize_embedding_model()
        
        logger.info("✅ E5 vector generator initialized successfully")
    
    async def initialize_embedding_model(self):
        """Initialize E5 embedding model"""
        try:
            logger.info("📥 Loading E5 embedding model...")
            start_time = datetime.now()
            
            # Use E5-large for high-quality embeddings
            try:
                self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
                model_name = "e5-large-v2"
            except:
                try:
                    # Fallback to multilingual E5
                    self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
                    model_name = "multilingual-e5-large"
                except:
                    # Final fallback
                    self.embedding_model = SentenceTransformer('intfloat/e5-base-v2')
                    model_name = "e5-base-v2"
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Loaded {model_name} embedding model in {load_time:.2f}s")
            
            # Test embedding generation
            test_text = "Test article for embedding"
            start_time = datetime.now()
            test_embedding = self.embedding_model.encode([test_text])
            embedding_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"⚡ Embedding generation speed: {embedding_time*1000:.1f}ms per article")
            logger.info(f"📏 Embedding dimension: {len(test_embedding[0])}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def generate_e5_embedding(self, content: str) -> List[float]:
        """Generate E5 embedding for article content"""
        try:
            # Prepare text for E5 model (add passage prefix for better performance)
            passage_text = f"passage: {content[:1000]}"  # Limit to 1000 chars for efficiency
            
            # Generate embedding using local model
            start_time = datetime.now()
            embedding = self.embedding_model.encode([passage_text])[0].tolist()
            generation_time = (datetime.now() - start_time).total_seconds()
            
            logger.debug(f"Generated E5 embedding in {generation_time*1000:.1f}ms")
            
            return embedding
            
        except Exception as e:
            logger.error(f"E5 embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024  # Assuming e5-large dimensions
    
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
            FROM News.rag_training_set
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
    
    async def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of articles and generate E5 embeddings"""
        logger.info(f"📊 Processing batch of {len(articles)} articles...")
        
        articles_with_vectors = []
        
        for i, article in enumerate(articles, 1):
            try:
                logger.info(f"  🔄 Processing article {i}/{len(articles)}: {article['ticker']}")
                
                # Generate E5 embedding for the article content
                content = article.get('full_content') or article.get('headline', '')
                if not content:
                    logger.warning(f"⚠️ No content for {article['ticker']}, skipping...")
                    continue
                
                # Generate embedding
                feature_vector = await self.generate_e5_embedding(content)
                
                if not feature_vector or all(x == 0.0 for x in feature_vector):
                    logger.warning(f"⚠️ Failed to generate embedding for {article['ticker']}, skipping...")
                    continue
                
                # Add embedding to article data
                article['feature_vector'] = feature_vector
                articles_with_vectors.append(article)
                
                logger.info(f"  ✅ Generated {len(feature_vector)}-dim embedding for {article['ticker']}")
                
            except Exception as e:
                logger.error(f"Error processing article {article.get('ticker', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {len(articles_with_vectors)}/{len(articles)} articles")
        return articles_with_vectors
    
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
                    'e5-large-v2',  # feature_model - updated for E5
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
        """Generate E5 embedding vectors for all articles in the TRAINING dataset"""
        logger.info("🚀 Starting E5 vector generation for TRAINING articles...")
        
        # Create the training vectors table
        await self.create_training_vectors_table()
        
        # Process articles in balanced batches to ensure even coverage
        batch_num = 1
        total_processed = 0
        start_time = datetime.now()
        processed_counts = {'TRUE_BULLISH': 0, 'FALSE_PUMP': 0, 'NEUTRAL': 0}
        
        while True:
            logger.info(f"📦 Processing balanced batch {batch_num}...")
            
            # Get next balanced batch of training articles (uses correct rag_training_set)
            articles = await self.get_training_articles_balanced(
                batch_size=self.batch_size, 
                processed_counts=processed_counts
            )
            
            if not articles:
                logger.info("✅ No more articles to process")
                break
            
            logger.info(f"📄 Retrieved {len(articles)} articles for balanced batch {batch_num}")
            
            # Process the batch and generate E5 embeddings
            articles_with_vectors = await self.process_articles_batch(articles)
            
            if articles_with_vectors:
                # Store this batch
                await self.store_vectors(articles_with_vectors)
                total_processed += len(articles_with_vectors)
                logger.info(f"✅ Batch {batch_num} completed: {len(articles_with_vectors)} vectors stored")
            else:
                logger.warning(f"⚠️ Batch {batch_num} produced no valid vectors")
            
            # Update for next batch
            batch_num += 1
            
            # Progress update
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⏱️ Progress: {total_processed} articles processed in {elapsed_time:.1f}s")
            logger.info(f"📊 Processed counts: {processed_counts}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 E5 vector generation completed! Processed {total_processed} articles in {total_time:.1f}s")
        
        # Verify the results
        await self.verify_vectors()
    
    async def create_training_vectors_table(self):
        """Create the training vectors table"""
        try:
            # Drop existing training vectors table if it exists
            self.ch_manager.client.command("DROP TABLE IF EXISTS News.rag_training_vectors")
            
            # Create new training vectors table (optimized for E5 embeddings)
            create_table_sql = """
            CREATE TABLE News.rag_training_vectors (
                id UUID DEFAULT generateUUIDv4(),
                ticker String,
                headline String,
                full_content String,
                
                -- E5-generated embeddings
                feature_vector Array(Float64),
                feature_model String DEFAULT 'e5-large-v2',
                
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
                
                -- Indexing for fast similarity search
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
        try:
            # Check total count
            count_query = "SELECT COUNT(*) FROM News.rag_training_vectors"
            count_result = self.ch_manager.client.query(count_query)
            total_vectors = count_result.result_rows[0][0]
            
            # Check distribution by outcome type
            distribution_query = """
            SELECT outcome_type, COUNT(*) as count
            FROM News.rag_training_vectors
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            dist_result = self.ch_manager.client.query(distribution_query)
            
            logger.info("📊 TRAINING Vector Generation Summary:")
            logger.info(f"  • Total vectors stored: {total_vectors}")
            logger.info("  • Distribution by outcome:")
            
            for row in dist_result.result_rows:
                outcome, count = row
                logger.info(f"    - {outcome}: {count}")
            
            # SECURITY CHECK: Verify no test data leaked into training vectors
            await self.verify_no_test_data_leakage()
            
            logger.info("✅ Training vector verification completed successfully")
            
        except Exception as e:
            logger.error(f"Error verifying vectors: {e}")
            raise
    
    async def verify_no_test_data_leakage(self):
        """Verify that no test set articles were included in training vectors"""
        try:
            # Check if test set exists
            test_check_query = "SELECT COUNT(*) FROM News.rag_test_set"
            try:
                test_result = self.ch_manager.client.query(test_check_query)
                test_count = test_result.result_rows[0][0]
                
                if test_count == 0:
                    logger.warning("⚠️ No test set found - skipping leakage check")
                    return
                    
                logger.info(f"🔍 Checking for data leakage against {test_count} test articles...")
                
            except Exception:
                logger.warning("⚠️ Test set table not found - skipping leakage check")
                return
            
            # Check for overlapping articles between training vectors and test set
            leakage_query = """
            SELECT COUNT(*) as leakage_count
            FROM News.rag_training_vectors tv
            INNER JOIN News.rag_test_set ts ON tv.original_content_hash = ts.original_content_hash
            """
            
            result = self.ch_manager.client.query(leakage_query)
            leakage_count = result.result_rows[0][0]
            
            if leakage_count > 0:
                logger.error(f"🚨 CRITICAL: DATA LEAKAGE DETECTED!")
                logger.error(f"   {leakage_count} test articles found in training vectors!")
                logger.error("   This will invalidate test results - regenerate vectors from training set only")
                raise Exception(f"Data leakage detected: {leakage_count} overlapping articles")
            else:
                logger.info("🔒 Security check passed - no test data in training vectors")
                
        except Exception as e:
            logger.error(f"Error checking for data leakage: {e}")
            raise
    
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
    
    generator = E5VectorGenerator(batch_size=args.batch_size)
    
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