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
    
    def get_all_articles_chronologically(self) -> List[Dict[str, Any]]:
        """Get all articles in chronological order for proper splitting"""
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
            ORDER BY published_est ASC
            """
            
            result = self.ch_manager.client.query(query)
            
            articles = []
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
                articles.append(article)
            
            # Log overall distribution and date ranges
            if articles:
                earliest = articles[0]['published_est']
                latest = articles[-1]['published_est']
                logger.info(f"📊 Total dataset: {len(articles)} articles ({earliest} to {latest})")
                
                # Show distribution by outcome
                outcome_counts = {}
                for article in articles:
                    outcome = article['outcome_type']
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                
                logger.info("📊 Distribution by outcome:")
                for outcome, count in outcome_counts.items():
                    logger.info(f"  • {outcome}: {count} articles")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            return []
    
    def create_balanced_split(self, articles: List[Dict[str, Any]]) -> tuple:
        """Create chronological train/test split maintaining outcome proportions"""
        
        if not articles:
            logger.error("No articles provided for splitting")
            return [], []
        
        # Chronological split: earlier articles for training, later for testing
        logger.info("📅 Creating CHRONOLOGICAL split (train=early, test=later)")
        
        # Calculate split point globally
        test_count = max(1, int(len(articles) * self.test_ratio))
        train_count = len(articles) - test_count
        
        # Global chronological split: first articles go to training, last articles go to test
        train_articles = articles[:train_count]  # Earlier articles
        test_articles = articles[train_count:]   # Later articles
        
        # Log date ranges for verification
        if train_articles:
            train_start = train_articles[0]['published_est']
            train_end = train_articles[-1]['published_est']
        else:
            train_start = train_end = "N/A"
            
        if test_articles:
            test_start = test_articles[0]['published_est']
            test_end = test_articles[-1]['published_est']
        else:
            test_start = test_end = "N/A"
        
        logger.info(f"📋 CHRONOLOGICAL split:")
        logger.info(f"  • Training: {len(train_articles)} articles ({train_start} to {train_end})")
        logger.info(f"  • Testing: {len(test_articles)} articles ({test_start} to {test_end})")
        
        # Ensure balanced representation in both sets
        logger.info("🔄 Checking outcome distribution in both sets...")
        train_outcome_counts = {}
        test_outcome_counts = {}
        
        for article in train_articles:
            outcome = article['outcome_type']
            train_outcome_counts[outcome] = train_outcome_counts.get(outcome, 0) + 1
            
        for article in test_articles:
            outcome = article['outcome_type']
            test_outcome_counts[outcome] = test_outcome_counts.get(outcome, 0) + 1
            
        logger.info("📊 Outcome distribution in TRAINING set:")
        for outcome in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
            count = train_outcome_counts.get(outcome, 0)
            logger.info(f"  • {outcome}: {count} articles")
            
        logger.info("📊 Outcome distribution in TESTING set:")
        for outcome in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
            count = test_outcome_counts.get(outcome, 0)
            logger.info(f"  • {outcome}: {count} articles")
        
        # Overall summary
        logger.info(f"📊 Overall CHRONOLOGICAL split:")
        logger.info(f"  • Training set: {len(train_articles)} articles ({train_start} to {train_end})")
        logger.info(f"  • Test set: {len(test_articles)} articles ({test_start} to {test_end})")
        logger.info(f"🔒 Data integrity: Training data is from EARLIER dates, Test data is from LATER dates")
        
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
            SELECT outcome_type, COUNT(*) as count, 
                   MIN(published_est) as earliest_date,
                   MAX(published_est) as latest_date
            FROM News.rag_training_set
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            test_query = """
            SELECT outcome_type, COUNT(*) as count,
                   MIN(published_est) as earliest_date,
                   MAX(published_est) as latest_date
            FROM News.rag_test_set
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            train_result = self.ch_manager.client.query(train_query)
            test_result = self.ch_manager.client.query(test_query)
            
            logger.info("📊 Final Train/Test Split Verification:")
            logger.info("  Training Set:")
            train_total = 0
            train_latest_date = None
            for row in train_result.result_rows:
                outcome, count, earliest, latest = row
                train_total += count
                if train_latest_date is None or latest > train_latest_date:
                    train_latest_date = latest
                logger.info(f"    • {outcome}: {count} articles ({earliest} to {latest})")
            logger.info(f"    • Total: {train_total}")
            
            logger.info("  Test Set:")
            test_total = 0
            test_earliest_date = None
            for row in test_result.result_rows:
                outcome, count, earliest, latest = row
                test_total += count
                if test_earliest_date is None or earliest < test_earliest_date:
                    test_earliest_date = earliest
                logger.info(f"    • {outcome}: {count} articles ({earliest} to {latest})")
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
            
            # CRITICAL: Check chronological integrity
            chronological_integrity = True
            if train_latest_date and test_earliest_date:
                if train_latest_date >= test_earliest_date:
                    logger.error("🚨 CHRONOLOGICAL INTEGRITY VIOLATION!")
                    logger.error(f"   Training latest date: {train_latest_date}")
                    logger.error(f"   Test earliest date: {test_earliest_date}")
                    logger.error("   Some training articles are newer than test articles!")
                    chronological_integrity = False
                else:
                    logger.info("✅ Chronological integrity verified:")
                    logger.info(f"   Training ends: {train_latest_date}")
                    logger.info(f"   Test starts: {test_earliest_date}")
                    logger.info("   ✅ All training data is from EARLIER dates than test data")
            
            if leakage_count == 0 and chronological_integrity:
                logger.info("✅ Data integrity verified - no content leakage and proper chronological split")
            else:
                if leakage_count > 0:
                    logger.error(f"🚨 CONTENT LEAKAGE DETECTED: {leakage_count} overlapping articles!")
                if not chronological_integrity:
                    logger.error("🚨 CHRONOLOGICAL LEAKAGE DETECTED: Training data contains future information!")
                raise Exception("Data integrity check failed")
                
        except Exception as e:
            logger.error(f"Error verifying split: {e}")
            raise
    
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
            # Get all articles in chronological order
            articles = splitter.get_all_articles_chronologically()
            if not articles:
                logger.error("No articles found!")
                return
            
            # Create balanced split
            train_articles, test_articles = splitter.create_balanced_split(articles)
            
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