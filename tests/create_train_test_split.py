#!/usr/bin/env python3
"""
Create Train/Test Split for RAG System

This script properly splits the rag_training_dataset into training and testing sets
to avoid data leakage and ensure proper evaluation of the RAG system.

Usage:
    python3 tests/create_train_test_split.py --test-ratio 0.2
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import argparse
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainTestSplitter:
    """Create proper train/test split for RAG evaluation"""
    
    def __init__(self, test_ratio: float = 0.2):
        self.ch_manager = ClickHouseManager()
        self.test_ratio = test_ratio
        
    def initialize(self):
        """Initialize the splitter"""
        logger.info("🔀 Initializing Train/Test Splitter...")
        self.ch_manager.connect()
        logger.info("✅ Splitter initialized successfully")
    
    def get_balanced_articles_by_outcome(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all articles grouped by outcome type for balanced splitting"""
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
                published_est,
                selection_priority
            FROM News.rag_training_dataset
            ORDER BY outcome_type, selection_priority DESC, ticker
            """
            
            result = self.ch_manager.client.query(query)
            
            # Group by outcome type
            articles_by_outcome = {
                'TRUE_BULLISH': [],
                'FALSE_PUMP': [],
                'NEUTRAL': []
            }
            
            for row in result.result_rows:
                article = {
                    'ticker': row[0],
                    'headline': row[1],
                    'full_content': row[2],
                    'outcome_type': row[3],
                    'has_30pt_increase': int(row[4]),
                    'is_false_pump': int(row[5]),
                    'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                    'original_content_hash': row[7],
                    'published_est': row[8],
                    'selection_priority': float(row[9]) if row[9] else 0.0
                }
                
                articles_by_outcome[article['outcome_type']].append(article)
            
            # Log distribution
            logger.info("📊 Dataset distribution by outcome:")
            for outcome, articles in articles_by_outcome.items():
                logger.info(f"  • {outcome}: {len(articles)} articles")
            
            return articles_by_outcome
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            return {}
    
    def create_balanced_split(self, articles_by_outcome: Dict[str, List[Dict[str, Any]]]) -> tuple:
        """Create balanced train/test split maintaining outcome proportions"""
        
        train_articles = []
        test_articles = []
        
        # Set random seed for reproducibility
        random.seed(42)
        
        for outcome_type, articles in articles_by_outcome.items():
            # Shuffle articles within each outcome type
            shuffled_articles = articles.copy()
            random.shuffle(shuffled_articles)
            
            # Calculate split point
            test_count = max(1, int(len(shuffled_articles) * self.test_ratio))
            train_count = len(shuffled_articles) - test_count
            
            # Split
            test_split = shuffled_articles[:test_count]
            train_split = shuffled_articles[test_count:]
            
            train_articles.extend(train_split)
            test_articles.extend(test_split)
            
            logger.info(f"📋 {outcome_type} split:")
            logger.info(f"  • Training: {len(train_split)} articles")
            logger.info(f"  • Testing: {len(test_split)} articles")
        
        logger.info(f"📊 Overall split:")
        logger.info(f"  • Training set: {len(train_articles)} articles ({(1-self.test_ratio)*100:.0f}%)")
        logger.info(f"  • Test set: {len(test_articles)} articles ({self.test_ratio*100:.0f}%)")
        
        return train_articles, test_articles
    
    def create_train_test_tables(self):
        """Create separate tables for training and testing data"""
        try:
            # Drop existing tables if they exist
            self.ch_manager.client.command("DROP TABLE IF EXISTS News.rag_training_set")
            self.ch_manager.client.command("DROP TABLE IF EXISTS News.rag_test_set")
            
            # Create training table (same schema as original)
            create_train_table_sql = """
            CREATE TABLE News.rag_training_set (
                id UUID DEFAULT generateUUIDv4(),
                ticker String,
                headline String,
                full_content String,
                outcome_type String,
                has_30pt_increase UInt8,
                is_false_pump UInt8,
                price_increase_ratio Float64,
                original_content_hash String,
                published_est DateTime,
                selection_priority Float64,
                created_at DateTime DEFAULT now(),
                
                -- Indexing
                INDEX idx_outcome_type (outcome_type) TYPE set(10) GRANULARITY 1,
                INDEX idx_ticker (ticker) TYPE set(1000) GRANULARITY 1
            ) ENGINE = ReplacingMergeTree(created_at)
            ORDER BY (ticker, original_content_hash)
            """
            
            # Create test table (same schema)
            create_test_table_sql = """
            CREATE TABLE News.rag_test_set (
                id UUID DEFAULT generateUUIDv4(),
                ticker String,
                headline String,
                full_content String,
                outcome_type String,
                has_30pt_increase UInt8,
                is_false_pump UInt8,
                price_increase_ratio Float64,
                original_content_hash String,
                published_est DateTime,
                selection_priority Float64,
                created_at DateTime DEFAULT now(),
                
                -- Indexing
                INDEX idx_outcome_type (outcome_type) TYPE set(10) GRANULARITY 1,
                INDEX idx_ticker (ticker) TYPE set(1000) GRANULARITY 1
            ) ENGINE = ReplacingMergeTree(created_at)
            ORDER BY (ticker, original_content_hash)
            """
            
            self.ch_manager.client.command(create_train_table_sql)
            self.ch_manager.client.command(create_test_table_sql)
            
            logger.info("✅ Train/test tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating train/test tables: {e}")
            raise
    
    def populate_train_test_tables(self, train_articles: List[Dict[str, Any]], test_articles: List[Dict[str, Any]]):
        """Populate the train and test tables with split data"""
        try:
            # Prepare data for insertion
            def prepare_data_rows(articles):
                return [[
                    article['ticker'],
                    article['headline'],
                    article['full_content'],
                    article['outcome_type'],
                    article['has_30pt_increase'],
                    article['is_false_pump'],
                    article['price_increase_ratio'],
                    article['original_content_hash'],
                    article['published_est'],
                    article['selection_priority']
                ] for article in articles]
            
            columns = [
                'ticker', 'headline', 'full_content', 'outcome_type',
                'has_30pt_increase', 'is_false_pump', 'price_increase_ratio',
                'original_content_hash', 'published_est', 'selection_priority'
            ]
            
            # Insert training data
            train_data = prepare_data_rows(train_articles)
            self.ch_manager.client.insert('News.rag_training_set', train_data, column_names=columns)
            logger.info(f"✅ Inserted {len(train_data)} articles into training set")
            
            # Insert test data
            test_data = prepare_data_rows(test_articles)
            self.ch_manager.client.insert('News.rag_test_set', test_data, column_names=columns)
            logger.info(f"✅ Inserted {len(test_data)} articles into test set")
            
        except Exception as e:
            logger.error(f"Error populating train/test tables: {e}")
            raise
    
    def verify_split(self):
        """Verify the train/test split was created correctly"""
        try:
            # Check training set
            train_query = """
            SELECT outcome_type, COUNT(*) as count
            FROM News.rag_training_set
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            test_query = """
            SELECT outcome_type, COUNT(*) as count
            FROM News.rag_test_set
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            train_result = self.ch_manager.client.query(train_query)
            test_result = self.ch_manager.client.query(test_query)
            
            logger.info("📊 Final Train/Test Split Verification:")
            logger.info("  Training Set:")
            train_total = 0
            for row in train_result.result_rows:
                outcome, count = row
                train_total += count
                logger.info(f"    • {outcome}: {count}")
            logger.info(f"    • Total: {train_total}")
            
            logger.info("  Test Set:")
            test_total = 0
            for row in test_result.result_rows:
                outcome, count = row
                test_total += count
                logger.info(f"    • {outcome}: {count}")
            logger.info(f"    • Total: {test_total}")
            
            # Check for data leakage (should be 0)
            leakage_query = """
            SELECT COUNT(*) as leakage_count
            FROM News.rag_training_set t1
            INNER JOIN News.rag_test_set t2 
            ON t1.original_content_hash = t2.original_content_hash
            """
            
            leakage_result = self.ch_manager.client.query(leakage_query)
            leakage_count = leakage_result.result_rows[0][0]
            
            if leakage_count == 0:
                logger.info("✅ No data leakage detected - training and test sets are properly separated")
            else:
                logger.error(f"❌ Data leakage detected: {leakage_count} articles appear in both sets")
            
        except Exception as e:
            logger.error(f"Error verifying split: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.ch_manager:
            self.ch_manager.close()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Create Train/Test Split for RAG System')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of data to use for testing (default: 0.2)')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing split')
    
    args = parser.parse_args()
    
    if args.test_ratio <= 0 or args.test_ratio >= 1:
        logger.error("Test ratio must be between 0 and 1")
        return
    
    splitter = TrainTestSplitter(test_ratio=args.test_ratio)
    
    try:
        splitter.initialize()
        
        if args.verify_only:
            splitter.verify_split()
        else:
            # Get all articles grouped by outcome
            articles_by_outcome = splitter.get_balanced_articles_by_outcome()
            if not articles_by_outcome:
                logger.error("No articles found!")
                return
            
            # Create balanced split
            train_articles, test_articles = splitter.create_balanced_split(articles_by_outcome)
            
            # Create tables
            splitter.create_train_test_tables()
            
            # Populate tables
            splitter.populate_train_test_tables(train_articles, test_articles)
            
            # Verify split
            splitter.verify_split()
            
            logger.info("🎉 Train/test split completed successfully!")
            logger.info("📋 Next steps:")
            logger.info("  1. Run generate_vectors.py on TRAINING SET ONLY")
            logger.info("  2. Test RAG system on TEST SET to measure real performance")
        
    except Exception as e:
        logger.error(f"Split creation failed: {e}")
        raise
    finally:
        splitter.cleanup()

if __name__ == "__main__":
    main() 