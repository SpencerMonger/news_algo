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
        
        # 1. Historical News Table - DROP AND RECREATE for clean data
        logger.info("🗑️ Dropping existing historical_news table...")
        ch_manager.client.command("DROP TABLE IF EXISTS News.historical_news")
        
        historical_news_sql = """
        CREATE TABLE News.historical_news (
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
        logger.info("✅ Created fresh historical_news table")
        
        # 2. Historical Sentiment Table - Also drop and recreate for consistency
        logger.info("🗑️ Dropping existing historical_sentiment table...")
        ch_manager.client.command("DROP TABLE IF EXISTS News.historical_sentiment")
        
        historical_sentiment_sql = """
        CREATE TABLE News.historical_sentiment (
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
        logger.info("✅ Created fresh historical_sentiment table")
        
        # 3. Backtest Trades Table - Also drop and recreate for consistency
        logger.info("🗑️ Dropping existing backtest_trades table...")
        ch_manager.client.command("DROP TABLE IF EXISTS News.backtest_trades")
        
        backtest_trades_sql = """
        CREATE TABLE News.backtest_trades (
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
        logger.info("✅ Created fresh backtest_trades table")
        
        # 4. Ticker Master Backtest Table - Keep existing logic (truncate)
        ticker_master_sql = """
        CREATE TABLE IF NOT EXISTS News.ticker_master_backtest (
            ticker String,
            company_name String,
            sector String,
            industry String,
            country String,
            market_cap Float64,
            price Float64,
            volume UInt64,
            float_shares Float64,
            scraped_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY ticker
        """
        
        ch_manager.client.command(ticker_master_sql)
        logger.info("✅ Created/verified ticker_master_backtest table")
        
        # Close connection
        ch_manager.close()
        
        logger.info("🎉 All backtesting tables created successfully!")
        logger.info("📊 Tables created:")
        logger.info("  • historical_news: Store scraped Finviz news articles (FRESH)")
        logger.info("  • historical_sentiment: Store Claude sentiment analysis results (FRESH)")
        logger.info("  • backtest_trades: Store simulated trade results (FRESH)")
        logger.info("  • ticker_master_backtest: Store ticker metadata")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating backtesting tables: {e}")
        return False

if __name__ == "__main__":
    success = create_backtesting_tables()
    if success:
        print("\n✅ Backtesting tables created successfully!")
    else:
        print("\n❌ Failed to create backtesting tables!")
        sys.exit(1) 