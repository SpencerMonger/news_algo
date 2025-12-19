import asyncio
import logging
import logging.handlers
from datetime import datetime
from typing import List, Dict, Any, Optional
import clickhouse_connect
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Create logs directory structure
os.makedirs('logs', exist_ok=True)
os.makedirs('logs/articles', exist_ok=True)

# Configure enhanced logging system
def setup_logging():
    """Setup comprehensive logging with file handlers"""
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Daily rotating file handler for general logs
    from logging.handlers import TimedRotatingFileHandler
    general_file_handler = TimedRotatingFileHandler(
        'logs/clickhouse_operations.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    general_file_handler.setLevel(logging.INFO)
    general_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    general_file_handler.setFormatter(general_formatter)
    
    # Dedicated article tracking file handler
    article_file_handler = TimedRotatingFileHandler(
        'logs/articles/article_tracking.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    article_file_handler.setLevel(logging.INFO)
    article_formatter = logging.Formatter('%(asctime)s - ARTICLE - %(message)s')
    article_file_handler.setFormatter(article_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(general_file_handler)
    logger.addHandler(article_file_handler)
    
    return logger

# Initialize enhanced logging
logger = setup_logging()

# Create a separate logger for article events
article_logger = logging.getLogger('article_events')
article_logger.setLevel(logging.INFO)
article_file_handler = logging.handlers.TimedRotatingFileHandler(
    'logs/articles/article_events.log',
    when='midnight',
    interval=1,
    backupCount=30,
    encoding='utf-8'
)
article_formatter = logging.Formatter('%(asctime)s - %(message)s')
article_file_handler.setFormatter(article_formatter)
article_logger.addHandler(article_file_handler)

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
            published_utc String,
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
            news_type String DEFAULT 'other',
            urgency_score UInt8 DEFAULT 0,
            
            -- Sentiment analysis fields
            sentiment String DEFAULT 'neutral',
            recommendation String DEFAULT 'HOLD',
            confidence String DEFAULT 'low',
            explanation String DEFAULT '',
            analysis_time_ms UInt32 DEFAULT 0,
            analyzed_at DateTime64(3) DEFAULT now64(),
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 3,
            INDEX idx_source (source) TYPE set(100) GRANULARITY 1,
            INDEX idx_content_hash (content_hash) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_sentiment (sentiment) TYPE set(10) GRANULARITY 1,
            INDEX idx_recommendation (recommendation) TYPE set(10) GRANULARITY 1
        ) 
        ENGINE = ReplacingMergeTree(detected_at)
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (content_hash, article_url)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("breaking_news table created/verified with sentiment analysis columns")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise

    def create_news_testing_table(self):
        """Create the news_testing table with same schema as breaking_news for testing"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS News.news_testing (
            id UUID DEFAULT generateUUIDv4(),
            timestamp DateTime64(3) DEFAULT now64(),
            source String,
            ticker String,
            headline String,
            published_utc String,
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
            news_type String DEFAULT 'other',
            urgency_score UInt8 DEFAULT 0,
            
            -- Sentiment analysis fields
            sentiment String DEFAULT 'neutral',
            recommendation String DEFAULT 'HOLD',
            confidence String DEFAULT 'low',
            explanation String DEFAULT '',
            analysis_time_ms UInt32 DEFAULT 0,
            analyzed_at DateTime64(3) DEFAULT now64(),
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 3,
            INDEX idx_source (source) TYPE set(100) GRANULARITY 1,
            INDEX idx_content_hash (content_hash) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_sentiment (sentiment) TYPE set(10) GRANULARITY 1,
            INDEX idx_recommendation (recommendation) TYPE set(10) GRANULARITY 1
        ) 
        ENGINE = ReplacingMergeTree(detected_at)
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (content_hash, article_url)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("news_testing table created/verified with sentiment analysis columns")
        except Exception as e:
            logger.error(f"Error creating news_testing table: {e}")
            raise

    def insert_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Insert articles in batch for better performance"""
        return self.insert_articles_to_table(articles, 'breaking_news')

    def insert_articles_to_table(self, articles: List[Dict[str, Any]], table_name: str) -> int:
        """Insert articles into specified table in batch for better performance"""
        if not articles:
            return 0
            
        try:
            logger.info(f"Processing batch of {len(articles)} articles for insertion into {table_name}")
            article_logger.info(f"BATCH_START: Processing {len(articles)} articles into {table_name}")
            
            # Only do ticker notifications and duplicate checking for breaking_news table
            new_tickers_for_notification = []
            if table_name == 'breaking_news':
                # Get recently seen tickers to avoid duplicate notifications
                recent_tickers_query = """
                SELECT DISTINCT ticker
                FROM News.breaking_news
                WHERE detected_at >= now() - INTERVAL 10 MINUTE
                AND ticker != ''
                """
                recent_result = self.client.query(recent_tickers_query)
                recently_seen_tickers = {row[0] for row in recent_result.result_rows}
            else:
                recently_seen_tickers = set()
            
            # Prepare data for insertion
            data_rows = []
            for i, article in enumerate(articles, 1):
                ticker = article.get('ticker', 'UNKNOWN')
                headline = article.get('headline', '')[:100]  # Truncate for logging
                source = article.get('source', '')
                content_hash = article.get('content_hash', '')
                
                # Check if this is a duplicate by content hash (only for breaking_news)
                is_duplicate = False
                if table_name == 'breaking_news' and content_hash:
                    try:
                        duplicate_check = self.client.query(
                            f"SELECT COUNT(*) FROM News.breaking_news WHERE content_hash = '{content_hash}'"
                        )
                        is_duplicate = duplicate_check.result_rows[0][0] > 0 if duplicate_check.result_rows else False
                    except:
                        is_duplicate = False
                
                if is_duplicate:
                    article_logger.warning(f"DUPLICATE_DETECTED: {ticker} | {content_hash} | Skipping to preserve original timestamp")
                    logger.info(f"Skipping duplicate article for {ticker}")
                    continue  # Skip duplicate articles entirely
                
                # FIXED: Only track TRULY NEW tickers for immediate notification (only for breaking_news)
                if table_name == 'breaking_news' and ticker and ticker != 'UNKNOWN' and ticker not in recently_seen_tickers:
                    new_tickers_for_notification.append({
                        'ticker': ticker,
                        'timestamp': article.get('timestamp', datetime.now())
                    })
                    logger.info(f"🔥 TRULY NEW TICKER DETECTED: {ticker} - WILL TRIGGER IMMEDIATE NOTIFICATION!")
                    recently_seen_tickers.add(ticker)  # Add to set to avoid duplicate notifications in same batch
                elif table_name == 'breaking_news' and ticker and ticker != 'UNKNOWN':
                    logger.debug(f"📝 EXISTING TICKER: {ticker} - no immediate notification needed")
                
                # Debug timestamp issue for NEW articles only
                if 'timestamp' not in article:
                    timestamp_val = datetime.now()
                    logger.warning(f"Missing timestamp for ticker {ticker}, using current time: {timestamp_val}")
                    article_logger.warning(f"MISSING_TIMESTAMP: {ticker} | {headline}")
                else:
                    timestamp_val = article.get('timestamp')
                    logger.info(f"Using existing timestamp for ticker {ticker}: {timestamp_val}")
                
                # Log detailed article information
                article_logger.info(f"ARTICLE_{i:02d}: {ticker} | {timestamp_val} | {source} | {headline}")
                
                # Check for potential duplicates by content hash
                if content_hash:
                    article_logger.info(f"CONTENT_HASH: {ticker} | {content_hash}")
                
                # Log potential timing issues
                current_time = datetime.now()
                if isinstance(timestamp_val, datetime):
                    try:
                        # Handle timezone-aware vs timezone-naive datetime comparison
                        if timestamp_val.tzinfo is not None and current_time.tzinfo is None:
                            # timestamp_val is timezone-aware, current_time is naive
                            # Convert timestamp_val to naive for comparison
                            timestamp_val_naive = timestamp_val.replace(tzinfo=None)
                            time_diff = (current_time - timestamp_val_naive).total_seconds()
                        elif timestamp_val.tzinfo is None and current_time.tzinfo is not None:
                            # timestamp_val is naive, current_time is timezone-aware
                            # Convert current_time to naive for comparison
                            current_time_naive = current_time.replace(tzinfo=None)
                            time_diff = (current_time_naive - timestamp_val).total_seconds()
                        else:
                            # Both are the same type (both naive or both aware)
                            time_diff = (current_time - timestamp_val).total_seconds()
                        
                        if abs(time_diff) > 60:  # More than 1 minute difference
                            article_logger.warning(f"TIME_DIFF: {ticker} | Article timestamp: {timestamp_val} | Current: {current_time} | Diff: {time_diff:.1f}s")
                            article_logger.warning(f"POTENTIAL_DUPLICATE: {ticker} | This may be a duplicate article with updated timestamp")
                    except Exception as e:
                        # If there's any issue with time comparison, just log and continue
                        logger.debug(f"Could not compare timestamps for {ticker}: {e}")
                
                data_rows.append([
                    timestamp_val,
                    article.get('source', ''),
                    article.get('ticker', ''),
                    article.get('headline', ''),
                    article.get('published_utc', 'NO_TIME'),
                    article.get('article_url', ''),
                    article.get('summary', ''),
                    article.get('full_content', ''),
                    article.get('detected_at', datetime.now()),
                    article.get('processing_latency_ms', 0),
                    article.get('market_relevant', 0),
                    article.get('source_check_time', datetime.now()),
                    article.get('content_hash', ''),
                    article.get('news_type', 'other'),
                    article.get('urgency_score', 0),
                    article.get('sentiment', 'neutral'),
                    article.get('recommendation', 'HOLD'),
                    article.get('confidence', 'low'),
                    article.get('explanation', ''),
                    article.get('analysis_time_ms', 0),
                    article.get('analyzed_at', datetime.now())
                ])
            
            # Skip if no new articles to insert
            if not data_rows:
                article_logger.info("No new articles to insert (all were duplicates)")
                return -1  # Return -1 to indicate duplicates were skipped, not failure
            
            # Column names for insertion
            columns = [
                'timestamp', 'source', 'ticker', 'headline', 'published_utc',
                'article_url', 'summary', 'full_content', 'detected_at',
                'processing_latency_ms', 'market_relevant', 'source_check_time',
                'content_hash', 'news_type', 'urgency_score', 'sentiment',
                'recommendation', 'confidence', 'explanation', 'analysis_time_ms', 'analyzed_at'
            ]
            
            # Log insertion attempt
            insertion_time = datetime.now()
            article_logger.info(f"DB_INSERT_START: {insertion_time} | {len(data_rows)} articles | Table: {table_name}")
            
            result = self.client.insert(
                f'News.{table_name}',
                data_rows,
                column_names=columns
            )
            
            completion_time = datetime.now()
            insert_duration = (completion_time - insertion_time).total_seconds()
            
            logger.info(f"Inserted {len(data_rows)} articles into {table_name} in {insert_duration:.2f}s")
            article_logger.info(f"DB_INSERT_COMPLETE: {completion_time} | {len(data_rows)} articles | Duration: {insert_duration:.2f}s | Table: {table_name}")
            
            # Force merge for immediate deduplication (only for breaking_news)
            if table_name == 'breaking_news':
                self.force_merge_breaking_news()
            
            # IMMEDIATE TICKER NOTIFICATIONS - ELIMINATES POLLING LAG (only for breaking_news)
            if table_name == 'breaking_news' and new_tickers_for_notification:
                logger.info(f"🚨 TRULY NEW TICKERS DETECTED: {len(new_tickers_for_notification)} tickers - SENDING IMMEDIATE NOTIFICATIONS!")
                
                # ZERO-LAG SOLUTION: Trigger immediate price checking directly
                logger.info(f"⚡ ZERO-LAG: Triggering IMMEDIATE price checks for {len(new_tickers_for_notification)} tickers!")
                
                # SIMPLE FILE-BASED TRIGGER: Create trigger files for immediate processing
                try:
                    # Create triggers directory if it doesn't exist
                    trigger_dir = "triggers"
                    os.makedirs(trigger_dir, exist_ok=True)
                    
                    # Create immediate trigger files
                    for ticker_info in new_tickers_for_notification:
                        ticker = ticker_info['ticker']
                        timestamp = ticker_info['timestamp']
                        
                        # Create trigger file with timestamp
                        trigger_file = os.path.join(trigger_dir, f"immediate_{ticker}_{int(datetime.now().timestamp())}.json")
                        trigger_data = {
                            "ticker": ticker,
                            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            "trigger_type": "immediate_price_check",
                            "created_at": datetime.now().isoformat()
                        }
                        
                        with open(trigger_file, 'w') as f:
                            json.dump(trigger_data, f)
                        
                        logger.info(f"🚀 IMMEDIATE TRIGGER: Created {trigger_file} for {ticker}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not create trigger files: {e}")
                
                for ticker_info in new_tickers_for_notification:
                    ticker = ticker_info['ticker']
                    timestamp = ticker_info['timestamp']
                    logger.info(f"📢 IMMEDIATE NOTIFICATION QUEUED: {ticker} at {timestamp}")
                    
            elif table_name == 'breaking_news':
                logger.info("📝 No truly new tickers detected - all tickers were recently seen")
                
            if table_name == 'breaking_news':
                logger.info(f"✅ IMMEDIATE NOTIFICATIONS: Processed {len(new_tickers_for_notification)} truly new tickers")
            
            return len(data_rows)
            
        except Exception as e:
            error_time = datetime.now()
            logger.error(f"Error inserting articles into {table_name}: {e}")
            article_logger.error(f"DB_INSERT_ERROR: {error_time} | {e} | Table: {table_name}")
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
            country String,
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
            
            -- Reverse split detection
            recent_split UInt8 DEFAULT 0,
            
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

    def create_float_list_detailed_table(self):
        """Create the float_list_detailed table for comprehensive stock statistics from StockAnalysis.com"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS News.float_list_detailed (
            id UUID DEFAULT generateUUIDv4(),
            ticker String,
            scraped_at DateTime64(3) DEFAULT now64(),
            source_url String,
            
            -- Total Valuation
            market_cap Nullable(Float64),
            enterprise_value Nullable(Float64),
            
            -- Important Dates
            earnings_date String,
            ex_dividend_date String,
            
            -- Stock Price Statistics
            beta_5y Nullable(Float64),
            `52_week_high` Nullable(Float64),
            `52_week_low` Nullable(Float64),
            `52_week_change` Nullable(Float64),
            `50_day_ma` Nullable(Float64),
            `200_day_ma` Nullable(Float64),
            relative_strength_index Nullable(Float64),
            average_volume_20d Nullable(Float64),
            
            -- Share Statistics
            current_share_class Nullable(Float64),
            shares_outstanding Nullable(Float64),
            shares_change_yoy Nullable(Float64),
            shares_change_qoq Nullable(Float64),
            percent_insiders Nullable(Float64),
            percent_institutions Nullable(Float64),
            shares_float Nullable(Float64),
            
            -- Short Selling Information
            short_interest Nullable(Float64),
            short_previous_month Nullable(Float64),
            short_percent_shares_out Nullable(Float64),
            short_percent_float Nullable(Float64),
            short_ratio Nullable(Float64),
            
            -- Valuation Ratios
            pe_ratio Nullable(Float64),
            forward_pe Nullable(Float64),
            ps_ratio Nullable(Float64),
            forward_ps Nullable(Float64),
            pb_ratio Nullable(Float64),
            p_tbv_ratio Nullable(Float64),
            p_fcf_ratio Nullable(Float64),
            p_ocf_ratio Nullable(Float64),
            peg_ratio Nullable(Float64),
            
            -- Enterprise Valuation
            ev_to_earnings Nullable(Float64),
            ev_to_sales Nullable(Float64),
            ev_to_ebitda Nullable(Float64),
            ev_to_ebit Nullable(Float64),
            ev_to_fcf Nullable(Float64),
            
            -- Financial Position
            current_ratio Nullable(Float64),
            quick_ratio Nullable(Float64),
            debt_to_equity Nullable(Float64),
            debt_to_ebitda Nullable(Float64),
            debt_to_fcf Nullable(Float64),
            interest_coverage Nullable(Float64),
            
            -- Financial Efficiency
            return_on_equity Nullable(Float64),
            return_on_assets Nullable(Float64),
            return_on_invested_capital Nullable(Float64),
            return_on_capital_employed Nullable(Float64),
            revenue_per_employee Nullable(Float64),
            profits_per_employee Nullable(Float64),
            employee_count Nullable(Int32),
            asset_turnover Nullable(Float64),
            inventory_turnover Nullable(Float64),
            
            -- Taxes
            income_tax Nullable(Float64),
            effective_tax_rate Nullable(Float64),
            
            -- Income Statement
            revenue Nullable(Float64),
            gross_profit Nullable(Float64),
            operating_income Nullable(Float64),
            pretax_income Nullable(Float64),
            net_income Nullable(Float64),
            ebitda Nullable(Float64),
            ebit Nullable(Float64),
            earnings_per_share Nullable(Float64),
            
            -- Balance Sheet
            cash_and_equivalents Nullable(Float64),
            total_debt Nullable(Float64),
            net_cash Nullable(Float64),
            net_cash_per_share Nullable(Float64),
            equity_book_value Nullable(Float64),
            book_value_per_share Nullable(Float64),
            working_capital Nullable(Float64),
            
            -- Cash Flow
            operating_cash_flow Nullable(Float64),
            capital_expenditures Nullable(Float64),
            free_cash_flow Nullable(Float64),
            fcf_per_share Nullable(Float64),
            
            -- Margins
            gross_margin Nullable(Float64),
            operating_margin Nullable(Float64),
            pretax_margin Nullable(Float64),
            profit_margin Nullable(Float64),
            ebitda_margin Nullable(Float64),
            ebit_margin Nullable(Float64),
            fcf_margin Nullable(Float64),
            
            -- Dividends & Yields
            dividend_per_share Nullable(Float64),
            dividend_yield Nullable(Float64),
            dividend_growth_yoy Nullable(Float64),
            years_dividend_growth Nullable(Float64),
            payout_ratio Nullable(Float64),
            buyback_yield Nullable(Float64),
            shareholder_yield Nullable(Float64),
            earnings_yield Nullable(Float64),
            fcf_yield Nullable(Float64),
            
            -- Stock Splits
            last_split_date String,
            split_type String,
            split_ratio String,
            
            -- Scores
            altman_z_score Nullable(Float64),
            piotroski_f_score Nullable(Int32),
            
            -- Strength Analysis (generated by stock_strength_analyzer.py)
            strength_score Nullable(Float64),
            analysis_timestamp Nullable(DateTime64(3)),
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_scraped_at (scraped_at) TYPE minmax GRANULARITY 3
        ) 
        ENGINE = ReplacingMergeTree(scraped_at)
        ORDER BY (ticker, scraped_at)
        PARTITION BY toYYYYMM(scraped_at)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("float_list_detailed table created/verified with complete statistics schema")
        except Exception as e:
            logger.error(f"Error creating float_list_detailed table: {e}")
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
                    ticker.get('country', ''),
                    ticker.get('market_cap', 0.0),
                    ticker.get('float_shares', 0.0),
                    ticker.get('price', 0.0),
                    ticker.get('volume', 0),
                    datetime.now(),
                    ticker.get('pe_ratio', 0.0),
                    ticker.get('eps', 0.0),
                    ticker.get('analyst_rating', ''),
                    ticker.get('insider_ownership', 0.0),
                    ticker.get('institutional_ownership', 0.0),
                    ticker.get('recent_split', 0)
                ])
            
            # Column names for insertion
            columns = [
                'ticker', 'company_name', 'sector', 'industry', 'country', 'market_cap',
                'float_shares', 'price', 'volume', 'last_updated',
                'pe_ratio', 'eps', 'analyst_rating', 'insider_ownership', 'institutional_ownership',
                'recent_split'
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

    def create_float_list_detailed_dedup_table(self):
        """Create the float_list_detailed_dedup table for deduplicated stock statistics from StockAnalysis.com"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS News.float_list_detailed_dedup (
            ticker String,
            scraped_at DateTime64(3) DEFAULT now64(),
            source_url String,
            
            -- Total Valuation
            market_cap Nullable(Float64),
            enterprise_value Nullable(Float64),
            
            -- Important Dates
            earnings_date String,
            ex_dividend_date String,
            
            -- Stock Price Statistics
            beta_5y Nullable(Float64),
            `52_week_high` Nullable(Float64),
            `52_week_low` Nullable(Float64),
            `52_week_change` Nullable(Float64),
            `50_day_ma` Nullable(Float64),
            `200_day_ma` Nullable(Float64),
            relative_strength_index Nullable(Float64),
            average_volume_20d Nullable(Float64),
            
            -- Share Statistics
            current_share_class Nullable(Float64),
            shares_outstanding Nullable(Float64),
            shares_change_yoy Nullable(Float64),
            shares_change_qoq Nullable(Float64),
            percent_insiders Nullable(Float64),
            percent_institutions Nullable(Float64),
            shares_float Nullable(Float64),
            
            -- Short Selling Information
            short_interest Nullable(Float64),
            short_previous_month Nullable(Float64),
            short_percent_shares_out Nullable(Float64),
            short_percent_float Nullable(Float64),
            short_ratio Nullable(Float64),
            
            -- Valuation Ratios
            pe_ratio Nullable(Float64),
            forward_pe Nullable(Float64),
            ps_ratio Nullable(Float64),
            forward_ps Nullable(Float64),
            pb_ratio Nullable(Float64),
            p_tbv_ratio Nullable(Float64),
            p_fcf_ratio Nullable(Float64),
            p_ocf_ratio Nullable(Float64),
            peg_ratio Nullable(Float64),
            
            -- Enterprise Valuation
            ev_to_earnings Nullable(Float64),
            ev_to_sales Nullable(Float64),
            ev_to_ebitda Nullable(Float64),
            ev_to_ebit Nullable(Float64),
            ev_to_fcf Nullable(Float64),
            
            -- Financial Position
            current_ratio Nullable(Float64),
            quick_ratio Nullable(Float64),
            debt_to_equity Nullable(Float64),
            debt_to_ebitda Nullable(Float64),
            debt_to_fcf Nullable(Float64),
            interest_coverage Nullable(Float64),
            
            -- Financial Efficiency
            return_on_equity Nullable(Float64),
            return_on_assets Nullable(Float64),
            return_on_invested_capital Nullable(Float64),
            return_on_capital_employed Nullable(Float64),
            revenue_per_employee Nullable(Float64),
            profits_per_employee Nullable(Float64),
            employee_count Nullable(Int32),
            asset_turnover Nullable(Float64),
            inventory_turnover Nullable(Float64),
            
            -- Taxes
            income_tax Nullable(Float64),
            effective_tax_rate Nullable(Float64),
            
            -- Income Statement
            revenue Nullable(Float64),
            gross_profit Nullable(Float64),
            operating_income Nullable(Float64),
            pretax_income Nullable(Float64),
            net_income Nullable(Float64),
            ebitda Nullable(Float64),
            ebit Nullable(Float64),
            earnings_per_share Nullable(Float64),
            
            -- Balance Sheet
            cash_and_equivalents Nullable(Float64),
            total_debt Nullable(Float64),
            net_cash Nullable(Float64),
            net_cash_per_share Nullable(Float64),
            equity_book_value Nullable(Float64),
            book_value_per_share Nullable(Float64),
            working_capital Nullable(Float64),
            
            -- Cash Flow
            operating_cash_flow Nullable(Float64),
            capital_expenditures Nullable(Float64),
            free_cash_flow Nullable(Float64),
            fcf_per_share Nullable(Float64),
            
            -- Margins
            gross_margin Nullable(Float64),
            operating_margin Nullable(Float64),
            pretax_margin Nullable(Float64),
            profit_margin Nullable(Float64),
            ebitda_margin Nullable(Float64),
            ebit_margin Nullable(Float64),
            fcf_margin Nullable(Float64),
            
            -- Dividends & Yields
            dividend_per_share Nullable(Float64),
            dividend_yield Nullable(Float64),
            dividend_growth_yoy Nullable(Float64),
            years_dividend_growth Nullable(Float64),
            payout_ratio Nullable(Float64),
            buyback_yield Nullable(Float64),
            shareholder_yield Nullable(Float64),
            earnings_yield Nullable(Float64),
            fcf_yield Nullable(Float64),
            
            -- Stock Splits
            last_split_date String,
            split_type String,
            split_ratio String,
            
            -- Scores
            altman_z_score Nullable(Float64),
            piotroski_f_score Nullable(Int32),
            
            -- Strength Analysis (generated by stock_strength_analyzer.py)
            strength_score Nullable(Float64),
            analysis_timestamp Nullable(DateTime64(3)),
            
            INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
            INDEX idx_scraped_at (scraped_at) TYPE minmax GRANULARITY 3
        ) 
        ENGINE = MergeTree()
        ORDER BY ticker
        PARTITION BY toYYYYMM(scraped_at)
        SETTINGS index_granularity = 8192
        """
        
        try:
            self.client.command(create_table_sql)
            logger.info("float_list_detailed_dedup table created/verified (simple MergeTree)")
        except Exception as e:
            logger.error(f"Error creating float_list_detailed_dedup table: {e}")
            raise

    def insert_float_list_detailed(self, stats_data: List[Dict[str, Any]]) -> int:
        """Insert detailed stock statistics into float_list_detailed table"""
        if not stats_data:
            return 0
            
        try:
            # Prepare data for insertion
            data_rows = []
            for stats in stats_data:
                data_rows.append([
                    stats.get('ticker', ''),
                    stats.get('scraped_at', datetime.now()),
                    stats.get('source_url', ''),
                    # Total Valuation
                    stats.get('market_cap', None),
                    stats.get('enterprise_value', None),
                    # Important Dates (strings - use empty string for NULL)
                    stats.get('earnings_date', '') or '',
                    stats.get('ex_dividend_date', '') or '',
                    # Stock Price Statistics
                    stats.get('beta_5y', None),
                    stats.get('52_week_high', None),
                    stats.get('52_week_low', None),
                    stats.get('52_week_change', None),
                    stats.get('50_day_ma', None),
                    stats.get('200_day_ma', None),
                    stats.get('relative_strength_index', None),
                    stats.get('average_volume_20d', None),
                    # Share Statistics
                    stats.get('current_share_class', None),
                    stats.get('shares_outstanding', None),
                    stats.get('shares_change_yoy', None),
                    stats.get('shares_change_qoq', None),
                    stats.get('percent_insiders', None),
                    stats.get('percent_institutions', None),
                    stats.get('shares_float', None),
                    # Short Selling Information
                    stats.get('short_interest', None),
                    stats.get('short_previous_month', None),
                    stats.get('short_percent_shares_out', None),
                    stats.get('short_percent_float', None),
                    stats.get('short_ratio', None),
                    # Valuation Ratios
                    stats.get('pe_ratio', None),
                    stats.get('forward_pe', None),
                    stats.get('ps_ratio', None),
                    stats.get('forward_ps', None),
                    stats.get('pb_ratio', None),
                    stats.get('p_tbv_ratio', None),
                    stats.get('p_fcf_ratio', None),
                    stats.get('p_ocf_ratio', None),
                    stats.get('peg_ratio', None),
                    # Enterprise Valuation
                    stats.get('ev_to_earnings', None),
                    stats.get('ev_to_sales', None),
                    stats.get('ev_to_ebitda', None),
                    stats.get('ev_to_ebit', None),
                    stats.get('ev_to_fcf', None),
                    # Financial Position
                    stats.get('current_ratio', None),
                    stats.get('quick_ratio', None),
                    stats.get('debt_to_equity', None),
                    stats.get('debt_to_ebitda', None),
                    stats.get('debt_to_fcf', None),
                    stats.get('interest_coverage', None),
                    # Financial Efficiency
                    stats.get('return_on_equity', None),
                    stats.get('return_on_assets', None),
                    stats.get('return_on_invested_capital', None),
                    stats.get('return_on_capital_employed', None),
                    stats.get('revenue_per_employee', None),
                    stats.get('profits_per_employee', None),
                    stats.get('employee_count', None),
                    stats.get('asset_turnover', None),
                    stats.get('inventory_turnover', None),
                    # Taxes
                    stats.get('income_tax', None),
                    stats.get('effective_tax_rate', None),
                    # Income Statement
                    stats.get('revenue', None),
                    stats.get('gross_profit', None),
                    stats.get('operating_income', None),
                    stats.get('pretax_income', None),
                    stats.get('net_income', None),
                    stats.get('ebitda', None),
                    stats.get('ebit', None),
                    stats.get('earnings_per_share', None),
                    # Balance Sheet
                    stats.get('cash_and_equivalents', None),
                    stats.get('total_debt', None),
                    stats.get('net_cash', None),
                    stats.get('net_cash_per_share', None),
                    stats.get('equity_book_value', None),
                    stats.get('book_value_per_share', None),
                    stats.get('working_capital', None),
                    # Cash Flow
                    stats.get('operating_cash_flow', None),
                    stats.get('capital_expenditures', None),
                    stats.get('free_cash_flow', None),
                    stats.get('fcf_per_share', None),
                    # Margins
                    stats.get('gross_margin', None),
                    stats.get('operating_margin', None),
                    stats.get('pretax_margin', None),
                    stats.get('profit_margin', None),
                    stats.get('ebitda_margin', None),
                    stats.get('ebit_margin', None),
                    stats.get('fcf_margin', None),
                    # Dividends & Yields
                    stats.get('dividend_per_share', None),
                    stats.get('dividend_yield', None),
                    stats.get('dividend_growth_yoy', None),
                    stats.get('years_dividend_growth', None),
                    stats.get('payout_ratio', None),
                    stats.get('buyback_yield', None),
                    stats.get('shareholder_yield', None),
                    stats.get('earnings_yield', None),
                    stats.get('fcf_yield', None),
                    # Stock Splits (strings - use empty string for NULL)
                    stats.get('last_split_date', '') or '',
                    stats.get('split_type', '') or '',
                    stats.get('split_ratio', '') or '',
                    # Scores
                    stats.get('altman_z_score', None),
                    stats.get('piotroski_f_score', None),
                    # Strength Analysis
                    stats.get('strength_score', None),
                    stats.get('analysis_timestamp', None),
                ])
            
            # Column names for insertion
            columns = [
                'ticker', 'scraped_at', 'source_url',
                # Total Valuation
                'market_cap', 'enterprise_value',
                # Important Dates
                'earnings_date', 'ex_dividend_date',
                # Stock Price Statistics
                'beta_5y', '52_week_high', '52_week_low', '52_week_change',
                '50_day_ma', '200_day_ma', 'relative_strength_index', 'average_volume_20d',
                # Share Statistics
                'current_share_class', 'shares_outstanding', 'shares_change_yoy', 'shares_change_qoq',
                'percent_insiders', 'percent_institutions', 'shares_float',
                # Short Selling Information
                'short_interest', 'short_previous_month', 'short_percent_shares_out',
                'short_percent_float', 'short_ratio',
                # Valuation Ratios
                'pe_ratio', 'forward_pe', 'ps_ratio', 'forward_ps', 'pb_ratio',
                'p_tbv_ratio', 'p_fcf_ratio', 'p_ocf_ratio', 'peg_ratio',
                # Enterprise Valuation
                'ev_to_earnings', 'ev_to_sales', 'ev_to_ebitda', 'ev_to_ebit', 'ev_to_fcf',
                # Financial Position
                'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_ebitda',
                'debt_to_fcf', 'interest_coverage',
                # Financial Efficiency
                'return_on_equity', 'return_on_assets', 'return_on_invested_capital',
                'return_on_capital_employed', 'revenue_per_employee', 'profits_per_employee',
                'employee_count', 'asset_turnover', 'inventory_turnover',
                # Taxes
                'income_tax', 'effective_tax_rate',
                # Income Statement
                'revenue', 'gross_profit', 'operating_income', 'pretax_income',
                'net_income', 'ebitda', 'ebit', 'earnings_per_share',
                # Balance Sheet
                'cash_and_equivalents', 'total_debt', 'net_cash', 'net_cash_per_share',
                'equity_book_value', 'book_value_per_share', 'working_capital',
                # Cash Flow
                'operating_cash_flow', 'capital_expenditures', 'free_cash_flow', 'fcf_per_share',
                # Margins
                'gross_margin', 'operating_margin', 'pretax_margin', 'profit_margin',
                'ebitda_margin', 'ebit_margin', 'fcf_margin',
                # Dividends & Yields
                'dividend_per_share', 'dividend_yield', 'dividend_growth_yoy',
                'years_dividend_growth', 'payout_ratio', 'buyback_yield',
                'shareholder_yield', 'earnings_yield', 'fcf_yield',
                # Stock Splits
                'last_split_date', 'split_type', 'split_ratio',
                # Scores
                'altman_z_score', 'piotroski_f_score',
                # Strength Analysis
                'strength_score', 'analysis_timestamp'
            ]
            
            result = self.client.insert(
                'News.float_list_detailed',
                data_rows,
                column_names=columns
            )
            
            logger.info(f"Inserted {len(stats_data)} detailed statistics records into ClickHouse")
            return len(stats_data)
            
        except Exception as e:
            logger.error(f"Error inserting detailed statistics: {e}")
            raise

    def insert_float_list_detailed_dedup(self, stats_data: List[Dict[str, Any]]) -> int:
        """Insert detailed stock statistics into float_list_detailed_dedup table (fresh table each run)"""
        if not stats_data:
            return 0
            
        try:
            # Prepare data for insertion (same structure as float_list_detailed but without id)
            data_rows = []
            for stats in stats_data:
                data_rows.append([
                    stats.get('ticker', ''),
                    stats.get('scraped_at', datetime.now()),
                    stats.get('source_url', ''),
                    # Total Valuation
                    stats.get('market_cap', None),
                    stats.get('enterprise_value', None),
                    # Important Dates (strings - use empty string for NULL)
                    stats.get('earnings_date', '') or '',
                    stats.get('ex_dividend_date', '') or '',
                    # Stock Price Statistics
                    stats.get('beta_5y', None),
                    stats.get('52_week_high', None),
                    stats.get('52_week_low', None),
                    stats.get('52_week_change', None),
                    stats.get('50_day_ma', None),
                    stats.get('200_day_ma', None),
                    stats.get('relative_strength_index', None),
                    stats.get('average_volume_20d', None),
                    # Share Statistics
                    stats.get('current_share_class', None),
                    stats.get('shares_outstanding', None),
                    stats.get('shares_change_yoy', None),
                    stats.get('shares_change_qoq', None),
                    stats.get('percent_insiders', None),
                    stats.get('percent_institutions', None),
                    stats.get('shares_float', None),
                    # Short Selling Information
                    stats.get('short_interest', None),
                    stats.get('short_previous_month', None),
                    stats.get('short_percent_shares_out', None),
                    stats.get('short_percent_float', None),
                    stats.get('short_ratio', None),
                    # Valuation Ratios
                    stats.get('pe_ratio', None),
                    stats.get('forward_pe', None),
                    stats.get('ps_ratio', None),
                    stats.get('forward_ps', None),
                    stats.get('pb_ratio', None),
                    stats.get('p_tbv_ratio', None),
                    stats.get('p_fcf_ratio', None),
                    stats.get('p_ocf_ratio', None),
                    stats.get('peg_ratio', None),
                    # Enterprise Valuation
                    stats.get('ev_to_earnings', None),
                    stats.get('ev_to_sales', None),
                    stats.get('ev_to_ebitda', None),
                    stats.get('ev_to_ebit', None),
                    stats.get('ev_to_fcf', None),
                    # Financial Position
                    stats.get('current_ratio', None),
                    stats.get('quick_ratio', None),
                    stats.get('debt_to_equity', None),
                    stats.get('debt_to_ebitda', None),
                    stats.get('debt_to_fcf', None),
                    stats.get('interest_coverage', None),
                    # Financial Efficiency
                    stats.get('return_on_equity', None),
                    stats.get('return_on_assets', None),
                    stats.get('return_on_invested_capital', None),
                    stats.get('return_on_capital_employed', None),
                    stats.get('revenue_per_employee', None),
                    stats.get('profits_per_employee', None),
                    stats.get('employee_count', None),
                    stats.get('asset_turnover', None),
                    stats.get('inventory_turnover', None),
                    # Taxes
                    stats.get('income_tax', None),
                    stats.get('effective_tax_rate', None),
                    # Income Statement
                    stats.get('revenue', None),
                    stats.get('gross_profit', None),
                    stats.get('operating_income', None),
                    stats.get('pretax_income', None),
                    stats.get('net_income', None),
                    stats.get('ebitda', None),
                    stats.get('ebit', None),
                    stats.get('earnings_per_share', None),
                    # Balance Sheet
                    stats.get('cash_and_equivalents', None),
                    stats.get('total_debt', None),
                    stats.get('net_cash', None),
                    stats.get('net_cash_per_share', None),
                    stats.get('equity_book_value', None),
                    stats.get('book_value_per_share', None),
                    stats.get('working_capital', None),
                    # Cash Flow
                    stats.get('operating_cash_flow', None),
                    stats.get('capital_expenditures', None),
                    stats.get('free_cash_flow', None),
                    stats.get('fcf_per_share', None),
                    # Margins
                    stats.get('gross_margin', None),
                    stats.get('operating_margin', None),
                    stats.get('pretax_margin', None),
                    stats.get('profit_margin', None),
                    stats.get('ebitda_margin', None),
                    stats.get('ebit_margin', None),
                    stats.get('fcf_margin', None),
                    # Dividends & Yields
                    stats.get('dividend_per_share', None),
                    stats.get('dividend_yield', None),
                    stats.get('dividend_growth_yoy', None),
                    stats.get('years_dividend_growth', None),
                    stats.get('payout_ratio', None),
                    stats.get('buyback_yield', None),
                    stats.get('shareholder_yield', None),
                    stats.get('earnings_yield', None),
                    stats.get('fcf_yield', None),
                    # Stock Splits (strings - use empty string for NULL)
                    stats.get('last_split_date', '') or '',
                    stats.get('split_type', '') or '',
                    stats.get('split_ratio', '') or '',
                    # Scores
                    stats.get('altman_z_score', None),
                    stats.get('piotroski_f_score', None),
                    # Strength Analysis
                    stats.get('strength_score', None),
                    stats.get('analysis_timestamp', None),
                ])
            
            # Column names for insertion (no id field)
            columns = [
                'ticker', 'scraped_at', 'source_url',
                # Total Valuation
                'market_cap', 'enterprise_value',
                # Important Dates
                'earnings_date', 'ex_dividend_date',
                # Stock Price Statistics
                'beta_5y', '52_week_high', '52_week_low', '52_week_change',
                '50_day_ma', '200_day_ma', 'relative_strength_index', 'average_volume_20d',
                # Share Statistics
                'current_share_class', 'shares_outstanding', 'shares_change_yoy', 'shares_change_qoq',
                'percent_insiders', 'percent_institutions', 'shares_float',
                # Short Selling Information
                'short_interest', 'short_previous_month', 'short_percent_shares_out',
                'short_percent_float', 'short_ratio',
                # Valuation Ratios
                'pe_ratio', 'forward_pe', 'ps_ratio', 'forward_ps', 'pb_ratio',
                'p_tbv_ratio', 'p_fcf_ratio', 'p_ocf_ratio', 'peg_ratio',
                # Enterprise Valuation
                'ev_to_earnings', 'ev_to_sales', 'ev_to_ebitda', 'ev_to_ebit', 'ev_to_fcf',
                # Financial Position
                'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_ebitda',
                'debt_to_fcf', 'interest_coverage',
                # Financial Efficiency
                'return_on_equity', 'return_on_assets', 'return_on_invested_capital',
                'return_on_capital_employed', 'revenue_per_employee', 'profits_per_employee',
                'employee_count', 'asset_turnover', 'inventory_turnover',
                # Taxes
                'income_tax', 'effective_tax_rate',
                # Income Statement
                'revenue', 'gross_profit', 'operating_income', 'pretax_income',
                'net_income', 'ebitda', 'ebit', 'earnings_per_share',
                # Balance Sheet
                'cash_and_equivalents', 'total_debt', 'net_cash', 'net_cash_per_share',
                'equity_book_value', 'book_value_per_share', 'working_capital',
                # Cash Flow
                'operating_cash_flow', 'capital_expenditures', 'free_cash_flow', 'fcf_per_share',
                # Margins
                'gross_margin', 'operating_margin', 'pretax_margin', 'profit_margin',
                'ebitda_margin', 'ebit_margin', 'fcf_margin',
                # Dividends & Yields
                'dividend_per_share', 'dividend_yield', 'dividend_growth_yoy',
                'years_dividend_growth', 'payout_ratio', 'buyback_yield',
                'shareholder_yield', 'earnings_yield', 'fcf_yield',
                # Stock Splits
                'last_split_date', 'split_type', 'split_ratio',
                # Scores
                'altman_z_score', 'piotroski_f_score',
                # Strength Analysis
                'strength_score', 'analysis_timestamp'
            ]
            
            result = self.client.insert(
                'News.float_list_detailed_dedup',
                data_rows,
                column_names=columns
            )
            
            logger.info(f"Inserted {len(stats_data)} statistics records into ClickHouse")
            return len(stats_data)
            
        except Exception as e:
            logger.error(f"Error inserting deduplicated statistics: {e}")
            raise

    def get_active_tickers(self) -> List[str]:
        """Get current active ticker list from float_list table"""
        try:
            query = """
            SELECT ticker FROM News.float_list 
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

    def drop_breaking_news_table(self):
        """Drop the breaking_news table to refresh schema"""
        try:
            self.client.command("DROP TABLE IF EXISTS News.breaking_news")
            logger.info("Dropped breaking_news table for schema refresh")
        except Exception as e:
            logger.error(f"Error dropping breaking_news table: {e}")
            raise

    def drop_all_pipeline_tables(self):
        """Drop ALL pipeline tables for complete data flow reset"""
        pipeline_tables = [
            'breaking_news',
            'monitored_tickers', 
            'price_tracking',
            'news_alert'
        ]
        
        try:
            for table in pipeline_tables:
                self.client.command(f"DROP TABLE IF EXISTS News.{table}")
                logger.info(f"Dropped {table} table for pipeline reset")
            
            logger.info("🧹 COMPLETE PIPELINE RESET: All data flow tables cleared")
        except Exception as e:
            logger.error(f"Error dropping pipeline tables: {e}")
            raise

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
            
            # NOTE: breaking_news table is created separately via create_breaking_news_table()
            # Removed duplicate definition to avoid schema conflicts

            # FIXED: Create immediate_notifications table for zero-lag detection
            immediate_notifications_sql = """
            CREATE TABLE IF NOT EXISTS News.immediate_notifications (
                id UUID DEFAULT generateUUIDv4(),
                ticker String,
                timestamp DateTime DEFAULT now(),
                processed UInt8 DEFAULT 0,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (ticker, created_at)
            PARTITION BY toYYYYMM(created_at)
            TTL created_at + INTERVAL 1 HOUR
            """
            self.client.command(immediate_notifications_sql)
            logger.info("Created immediate_notifications table for zero-lag detection")

            # NOTE: float_list table is created via create_float_list_table() method
            # This avoids duplicate schema definitions and ensures recent_split column is included

            # NOTE: price_move table is created via create_price_move_table() method
            # Removed duplicate definition here to avoid schema conflicts

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
                source String DEFAULT 'polygon',
                
                -- Sentiment analysis fields (from associated news)
                sentiment String DEFAULT 'neutral',
                recommendation String DEFAULT 'HOLD',
                confidence String DEFAULT 'low'
            ) ENGINE = MergeTree()
            ORDER BY (ticker, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 7 DAY
            """
            self.client.command(price_tracking_sql)
            logger.info("price_tracking table created/verified with sentiment analysis columns")

            # News alert table
            news_alert_sql = """
            CREATE TABLE IF NOT EXISTS News.news_alert (
                ticker String,
                timestamp DateTime DEFAULT now(),
                alert UInt8 DEFAULT 1,
                price Float64
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

    def force_merge_breaking_news(self):
        """Force merge to deduplicate immediately"""
        try:
            self.client.command("OPTIMIZE TABLE News.breaking_news FINAL")
            logger.info("Forced merge/optimization of breaking_news table")
        except Exception as e:
            logger.error(f"Error forcing merge: {e}")
    
    def check_table_structure(self):
        """Check the current table structure"""
        try:
            query = "DESCRIBE TABLE News.breaking_news"
            result = self.client.query(query)
            logger.info("Current breaking_news table structure:")
            for row in result.result_rows:
                logger.info(f"  {row}")
            
            # Also check the engine
            query = "SHOW CREATE TABLE News.breaking_news"
            result = self.client.query(query)
            logger.info(f"Table creation SQL: {result.result_rows[0][0] if result.result_rows else 'No result'}")
        except Exception as e:
            logger.error(f"Error checking table structure: {e}")

    def create_immediate_trigger(self, ticker: str, timestamp: datetime = None):
        """Create immediate trigger file for zero-lag price checking"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Create triggers directory if it doesn't exist
            trigger_dir = "triggers"
            os.makedirs(trigger_dir, exist_ok=True)
            
            # Create immediate trigger file with timestamp
            trigger_file = os.path.join(trigger_dir, f"immediate_{ticker}_{int(datetime.now().timestamp())}.json")
            trigger_data = {
                "ticker": ticker,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "trigger_type": "immediate_price_check",
                "created_at": datetime.now().isoformat()
            }
            
            with open(trigger_file, 'w') as f:
                json.dump(trigger_data, f)
            
            logger.info(f"🚀 ZERO-LAG TRIGGER: Created {trigger_file} for {ticker} immediately!")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create immediate trigger for {ticker}: {e}")
            return False

def setup_clickhouse_database():
    """Initialize ClickHouse database and tables"""
    ch_manager = ClickHouseManager()
    
    try:
        # Connect to ClickHouse
        ch_manager.connect()
        
        # Create database
        ch_manager.create_database()
        
        # 🧹 COMPLETE PIPELINE RESET: Drop ALL data flow tables for fresh start
        ch_manager.drop_all_pipeline_tables()
        ch_manager.create_breaking_news_table()
        
        # Check table structure to verify schema
        ch_manager.check_table_structure()
        
        # Force immediate merge for deduplication
        ch_manager.force_merge_breaking_news()
        
        # Create other tables
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
        'published_utc': '11:26 ET',
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