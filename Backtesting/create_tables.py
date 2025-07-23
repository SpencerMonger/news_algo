#!/usr/bin/env python3
"""
Create ClickHouse Tables for Backtesting System
Standalone script to create all necessary tables for historical news backtesting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_backtesting_tables():
    """Create all ClickHouse tables needed for backtesting"""
    try:
        # Initialize ClickHouse connection
        ch_manager = ClickHouseManager()
        ch_manager.connect()
        
        logger.info("Creating ClickHouse tables for backtesting system...")
        
        # 1. Historical News Table
        historical_news_sql = """
        CREATE TABLE IF NOT EXISTS News.historical_news (
            ticker String,
            headline String,
            article_url String,
            published_utc DateTime,
            scraped_at DateTime DEFAULT now(),
            source String DEFAULT 'finviz',
            newswire_type String,
            article_content String DEFAULT '',
            content_hash String DEFAULT ''
        ) ENGINE = MergeTree()
        ORDER BY (ticker, published_utc)
        PARTITION BY toYYYYMM(published_utc)
        """
        
        ch_manager.client.command(historical_news_sql)
        logger.info("✅ Created historical_news table")
        
        # 2. Historical Sentiment Table
        historical_sentiment_sql = """
        CREATE TABLE IF NOT EXISTS News.historical_sentiment (
            ticker String,
            headline String,
            article_url String,
            published_utc DateTime,
            sentiment String,
            recommendation String,
            confidence String,
            explanation String,
            analysis_time_ms UInt32,
            analyzed_at DateTime DEFAULT now(),
            content_hash String DEFAULT '',
            country String DEFAULT 'UNKNOWN'
        ) ENGINE = MergeTree()
        ORDER BY (ticker, published_utc)
        PARTITION BY toYYYYMM(published_utc)
        """
        
        ch_manager.client.command(historical_sentiment_sql)
        logger.info("✅ Created historical_sentiment table")
        
        # 3. Backtest Trades Table
        backtest_trades_sql = """
        CREATE TABLE IF NOT EXISTS News.backtest_trades (
            trade_id String,
            ticker String,
            article_url String,
            published_utc DateTime,
            entry_time DateTime,
            exit_time DateTime,
            entry_price Float64,
            exit_price Float64,
            quantity UInt32 DEFAULT 100,
            entry_type String DEFAULT 'BUY',
            exit_type String DEFAULT 'SELL',
            pnl Float64,
            pnl_percent Float64,
            trade_duration_seconds UInt32,
            sentiment String,
            recommendation String,
            confidence String,
            explanation String,
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (ticker, published_utc)
        PARTITION BY toYYYYMM(published_utc)
        """
        
        ch_manager.client.command(backtest_trades_sql)
        logger.info("✅ Created backtest_trades table")
        
        # 4. Ticker Master List Table (for backtesting)
        ticker_master_sql = """
        CREATE TABLE IF NOT EXISTS News.ticker_master_backtest (
            ticker String,
            company_name String DEFAULT '',
            sector String DEFAULT '',
            industry String DEFAULT '',
            country String DEFAULT '',
            market_cap Float64 DEFAULT 0,
            price Float64 DEFAULT 0,
            volume UInt64 DEFAULT 0,
            float_shares Float64 DEFAULT 0,
            scraped_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY ticker
        """
        
        ch_manager.client.command(ticker_master_sql)
        logger.info("✅ Created ticker_master_backtest table")
        
        logger.info("🎉 All backtesting tables created successfully!")
        
        # Display table info
        logger.info("\n📊 BACKTESTING TABLES SUMMARY:")
        logger.info("  • historical_news: Store scraped Finviz news articles")
        logger.info("  • historical_sentiment: Store LLM sentiment analysis results")
        logger.info("  • backtest_trades: Store simulated trade results")
        logger.info("  • ticker_master_backtest: Store master ticker list from Finviz")
        
        ch_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating backtesting tables: {e}")
        return False

if __name__ == "__main__":
    success = create_backtesting_tables()
    if success:
        print("\n✅ Backtesting tables created successfully!")
        print("You can now run the backtesting system.")
    else:
        print("\n❌ Failed to create backtesting tables!")
        sys.exit(1) 