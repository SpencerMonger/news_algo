import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import clickhouse_connect
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClickHouseManager:
    def __init__(self):
        self.host = os.getenv('CLICKHOUSE_HOST', 'localhost')
        self.port = int(os.getenv('CLICKHOUSE_PORT', 8123))
        self.username = os.getenv('CLICKHOUSE_USER', 'default')
        self.password = os.getenv('CLICKHOUSE_PASSWORD', '')
        self.database = 'News'
        self.client = None
        
    def connect(self):
        """Establish connection to ClickHouse"""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                compress=True,
                send_receive_timeout=30
            )
            logger.info(f"Connected to ClickHouse at {self.host}:{self.port}")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    def create_database(self):
        """Create the News database if it doesn't exist"""
        try:
            self.client.command("CREATE DATABASE IF NOT EXISTS News")
            logger.info("News database created/verified")
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise

    def create_breaking_news_table(self):
        """Create the breaking_news table with optimized schema"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS News.breaking_news (
            id UUID DEFAULT generateUUIDv4(),
            timestamp DateTime64(3) DEFAULT now64(),
            source String,
            ticker String,
            headline String,
            published_utc DateTime64(3),
            article_url String,
            summary String,
            full_content String,
            detected_at DateTime64(3) DEFAULT now64(),
            processing_latency_ms UInt32,
            market_relevant UInt8 DEFAULT 0,
            
            -- Performance tracking
            source_check_time DateTime64(3),
            content_hash String,
            
            -- Classification fields
            news_type Enum8('earnings' = 1, 'clinical_trial' = 2, 'merger' = 3, 'fda_approval' = 4, 'partnership' = 5, 'other' = 6) DEFAULT 'other',
            urgency_score UInt8 DEFAULT 0,
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 3,
            INDEX idx_source (source) TYPE set(100) GRANULARITY 1,
            INDEX idx_content_hash (content_hash) TYPE bloom_filter GRANULARITY 1
        ) 
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, ticker)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("breaking_news table created/verified")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise

    def insert_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Insert articles in batch for better performance"""
        if not articles:
            return 0
            
        try:
            # Prepare data for insertion
            data_rows = []
            for article in articles:
                data_rows.append([
                    article.get('timestamp', datetime.now()),
                    article.get('source', ''),
                    article.get('ticker', ''),
                    article.get('headline', ''),
                    article.get('published_utc', datetime.now()),
                    article.get('article_url', ''),
                    article.get('summary', ''),
                    article.get('full_content', ''),
                    article.get('detected_at', datetime.now()),
                    article.get('processing_latency_ms', 0),
                    article.get('market_relevant', 0),
                    article.get('source_check_time', datetime.now()),
                    article.get('content_hash', ''),
                    article.get('news_type', 'other'),
                    article.get('urgency_score', 0)
                ])
            
            # Column names for insertion
            columns = [
                'timestamp', 'source', 'ticker', 'headline', 'published_utc',
                'article_url', 'summary', 'full_content', 'detected_at',
                'processing_latency_ms', 'market_relevant', 'source_check_time',
                'content_hash', 'news_type', 'urgency_score'
            ]
            
            result = self.client.insert(
                'News.breaking_news',
                data_rows,
                column_names=columns
            )
            
            logger.info(f"Inserted {len(articles)} articles into ClickHouse")
            return len(articles)
            
        except Exception as e:
            logger.error(f"Error inserting articles: {e}")
            raise

    def get_recent_articles(self, ticker: str = None, hours: int = 24) -> List[Dict]:
        """Query recent articles for analysis"""
        try:
            base_query = """
            SELECT * FROM News.breaking_news 
            WHERE timestamp >= now() - INTERVAL {} HOUR
            """.format(hours)
            
            if ticker:
                base_query += f" AND ticker = '{ticker}'"
                
            base_query += " ORDER BY timestamp DESC LIMIT 1000"
            
            result = self.client.query(base_query)
            return result.result_rows
            
        except Exception as e:
            logger.error(f"Error querying articles: {e}")
            return []

    def check_duplicate(self, content_hash: str) -> bool:
        """Check if article already exists based on content hash"""
        try:
            query = f"""
            SELECT COUNT(*) FROM News.breaking_news 
            WHERE content_hash = '{content_hash}'
            """
            result = self.client.query(query)
            count = result.result_rows[0][0] if result.result_rows else 0
            return count > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            query = """
            SELECT 
                source,
                COUNT(*) as article_count,
                AVG(processing_latency_ms) as avg_latency_ms,
                MAX(processing_latency_ms) as max_latency_ms,
                MIN(timestamp) as earliest_article,
                MAX(timestamp) as latest_article
            FROM News.breaking_news 
            WHERE timestamp >= now() - INTERVAL 24 HOUR
            GROUP BY source
            ORDER BY article_count DESC
            """
            result = self.client.query(query)
            return result.result_rows
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}

    def create_float_list_table(self):
        """Create the float_list table for dynamic ticker management"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS News.float_list (
            id UUID DEFAULT generateUUIDv4(),
            ticker String,
            company_name String,
            sector String,
            industry String,
            market_cap Float64,
            float_shares Float64,
            price Float64,
            volume UInt64,
            last_updated DateTime64(3) DEFAULT now64(),
            
            -- Additional Finviz data
            pe_ratio Float64,
            eps Float64,
            analyst_rating String,
            insider_ownership Float64,
            institutional_ownership Float64,
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_sector (sector) TYPE set(50) GRANULARITY 1,
            INDEX idx_last_updated (last_updated) TYPE minmax GRANULARITY 3
        ) 
        ENGINE = ReplacingMergeTree(last_updated)
        ORDER BY ticker
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("float_list table created/verified")
        except Exception as e:
            logger.error(f"Error creating float_list table: {e}")
            raise

    def insert_tickers(self, ticker_data: List[Dict[str, Any]]) -> int:
        """Insert/update ticker data in batch"""
        if not ticker_data:
            return 0
            
        try:
            # Prepare data for insertion
            data_rows = []
            for ticker in ticker_data:
                data_rows.append([
                    ticker.get('ticker', ''),
                    ticker.get('company_name', ''),
                    ticker.get('sector', ''),
                    ticker.get('industry', ''),
                    ticker.get('market_cap', 0.0),
                    ticker.get('float_shares', 0.0),
                    ticker.get('price', 0.0),
                    ticker.get('volume', 0),
                    datetime.now(),
                    ticker.get('pe_ratio', 0.0),
                    ticker.get('eps', 0.0),
                    ticker.get('analyst_rating', ''),
                    ticker.get('insider_ownership', 0.0),
                    ticker.get('institutional_ownership', 0.0)
                ])
            
            # Column names for insertion
            columns = [
                'ticker', 'company_name', 'sector', 'industry', 'market_cap',
                'float_shares', 'price', 'volume', 'last_updated',
                'pe_ratio', 'eps', 'analyst_rating', 'insider_ownership', 'institutional_ownership'
            ]
            
            result = self.client.insert(
                'News.float_list',
                data_rows,
                column_names=columns
            )
            
            logger.info(f"Inserted/updated {len(ticker_data)} tickers in ClickHouse")
            return len(ticker_data)
            
        except Exception as e:
            logger.error(f"Error inserting tickers: {e}")
            raise

    def get_active_tickers(self) -> List[str]:
        """Get current active ticker list"""
        try:
            query = """
            SELECT ticker FROM News.float_list 
            WHERE last_updated >= now() - INTERVAL 24 HOUR
            ORDER BY ticker
            """
            result = self.client.query(query)
            tickers = [row[0] for row in result.result_rows]
            logger.info(f"Retrieved {len(tickers)} active tickers from database")
            return tickers
        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            return []

    def close(self):
        """Close the connection"""
        if self.client:
            self.client.close()
            logger.info("ClickHouse connection closed")

    def drop_float_list_table(self):
        """Drop the float_list table to refresh ticker data"""
        try:
            self.client.command("DROP TABLE IF EXISTS News.float_list")
            logger.info("Dropped float_list table for refresh")
        except Exception as e:
            logger.error(f"Error dropping float_list table: {e}")
            raise

    def create_price_move_table(self):
        """Create the price_move table for tickers that pass price checks"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS News.price_move (
            id UUID DEFAULT generateUUIDv4(),
            timestamp DateTime64(3) DEFAULT now64(),
            ticker String,
            news_headline String,
            news_published_utc DateTime64(3),
            news_article_url String,
            
            -- Price data
            current_price Float64,
            current_price_timestamp DateTime64(3),
            minute_30_high Float64,
            minute_30_low Float64,
            minute_30_open Float64,
            minute_30_close Float64,
            minute_30_volume UInt64,
            minute_30_timestamp DateTime64(3),
            
            -- Price move metrics
            price_move_percentage Float64,
            price_above_30min_high UInt8 DEFAULT 1,
            
            -- Performance tracking
            price_check_latency_ms UInt32,
            news_to_price_check_delay_ms UInt32,
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 3,
            INDEX idx_price_move_percentage (price_move_percentage) TYPE minmax GRANULARITY 3
        ) 
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, ticker)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("price_move table created/verified")
        except Exception as e:
            logger.error(f"Error creating price_move table: {e}")
            raise

    def insert_price_moves(self, price_moves: List[Dict[str, Any]]) -> int:
        """Insert price move records in batch"""
        if not price_moves:
            return 0
            
        try:
            # Prepare data for insertion
            data_rows = []
            for move in price_moves:
                data_rows.append([
                    move.get('timestamp', datetime.now()),
                    move.get('ticker', ''),
                    move.get('news_headline', ''),
                    move.get('news_published_utc', datetime.now()),
                    move.get('news_article_url', ''),
                    move.get('current_price', 0.0),
                    move.get('current_price_timestamp', datetime.now()),
                    move.get('minute_30_high', 0.0),
                    move.get('minute_30_low', 0.0),
                    move.get('minute_30_open', 0.0),
                    move.get('minute_30_close', 0.0),
                    move.get('minute_30_volume', 0),
                    move.get('minute_30_timestamp', datetime.now()),
                    move.get('price_move_percentage', 0.0),
                    move.get('price_above_30min_high', 1),
                    move.get('price_check_latency_ms', 0),
                    move.get('news_to_price_check_delay_ms', 0)
                ])
            
            # Column names for insertion
            columns = [
                'timestamp', 'ticker', 'news_headline', 'news_published_utc', 'news_article_url',
                'current_price', 'current_price_timestamp', 'minute_30_high', 'minute_30_low',
                'minute_30_open', 'minute_30_close', 'minute_30_volume', 'minute_30_timestamp',
                'price_move_percentage', 'price_above_30min_high', 'price_check_latency_ms',
                'news_to_price_check_delay_ms'
            ]
            
            result = self.client.insert(
                'News.price_move',
                data_rows,
                column_names=columns
            )
            
            logger.info(f"Inserted {len(price_moves)} price moves into ClickHouse")
            return len(price_moves)
            
        except Exception as e:
            logger.error(f"Error inserting price moves: {e}")
            raise

    def create_tables(self):
        """Create all required tables"""
        try:
            # Create database
            self.client.command("CREATE DATABASE IF NOT EXISTS News")
            logger.info("News database created/verified")
            
            # Breaking news table
            breaking_news_sql = """
            CREATE TABLE IF NOT EXISTS News.breaking_news (
                timestamp DateTime DEFAULT now(),
                source String,
                ticker String,
                headline String,
                published_utc DateTime,
                article_url String,
                summary String,
                full_content String,
                detected_at DateTime DEFAULT now(),
                processing_latency_ms UInt32,
                market_relevant UInt8 DEFAULT 1,
                source_check_time DateTime,
                content_hash String,
                news_type String DEFAULT 'other',
                urgency_score UInt8 DEFAULT 5
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 30 DAY
            """
            self.client.command(breaking_news_sql)
            logger.info("breaking_news table created/verified")

            # Float list table
            float_list_sql = """
            CREATE TABLE IF NOT EXISTS News.float_list (
                ticker String,
                float_value Float64,
                market_cap Float64,
                price Float64,
                volume Int64,
                last_updated DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(last_updated)
            ORDER BY ticker
            """
            self.client.command(float_list_sql)
            logger.info("float_list table created/verified")

            # Price move table  
            price_move_sql = """
            CREATE TABLE IF NOT EXISTS News.price_move (
                timestamp DateTime DEFAULT now(),
                ticker String,
                headline String,
                published_utc DateTime,
                article_url String,
                latest_price Float64,
                previous_close Float64,
                price_change_percentage Float64,
                volume_change_percentage Int32,
                detected_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 90 DAY
            """
            self.client.command(price_move_sql)
            logger.info("price_move table created/verified")

            # Monitored tickers table
            monitored_tickers_sql = """
            CREATE TABLE IF NOT EXISTS News.monitored_tickers (
                ticker String,
                first_seen DateTime DEFAULT now(),
                news_headline String,
                news_url String,
                active UInt8 DEFAULT 1,
                last_updated DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(last_updated)
            ORDER BY ticker
            """
            self.client.command(monitored_tickers_sql)
            logger.info("monitored_tickers table created/verified")

            # Price tracking table
            price_tracking_sql = """
            CREATE TABLE IF NOT EXISTS News.price_tracking (
                timestamp DateTime DEFAULT now(),
                ticker String,
                price Float64,
                volume UInt64,
                source String DEFAULT 'polygon'
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 7 DAY
            """
            self.client.command(price_tracking_sql)
            logger.info("price_tracking table created/verified")

            # News alert table
            news_alert_sql = """
            CREATE TABLE IF NOT EXISTS News.news_alert (
                ticker String,
                timestamp DateTime DEFAULT now(),
                alert UInt8 DEFAULT 1
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 30 DAY
            """
            self.client.command(news_alert_sql)
            logger.info("news_alert table created/verified")

            return True
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False

def setup_clickhouse_database():
    """Initialize ClickHouse database and tables"""
    ch_manager = ClickHouseManager()
    
    try:
        # Connect to ClickHouse
        ch_manager.connect()
        
        # Create database
        ch_manager.create_database()
        
        # Create tables
        ch_manager.create_breaking_news_table()
        ch_manager.create_float_list_table()
        ch_manager.create_price_move_table()
        ch_manager.create_tables()
        
        logger.info("ClickHouse setup completed successfully")
        return ch_manager
        
    except Exception as e:
        logger.error(f"ClickHouse setup failed: {e}")
        raise

if __name__ == "__main__":
    # Test the setup
    manager = setup_clickhouse_database()
    
    # Test insert
    test_article = [{
        'source': 'Test',
        'ticker': 'TEST',
        'headline': 'Test Article',
        'published_utc': datetime.now(),
        'article_url': 'https://test.com',
        'summary': 'Test summary',
        'full_content': 'Test content',
        'processing_latency_ms': 100,
        'content_hash': 'test_hash_123',
        'news_type': 'other',
        'urgency_score': 5
    }]
    
    manager.insert_articles(test_article)
    
    # Test query
    stats = manager.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    manager.close() 