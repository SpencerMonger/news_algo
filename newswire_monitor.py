import asyncio
import aiohttp
import feedparser
import hashlib
import logging
import re
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
import pytz
from dataclasses import dataclass
from clickhouse_setup import ClickHouseManager, setup_clickhouse_database
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NewsSource:
    name: str
    rss_url: str
    poll_interval: float
    priority: int
    enabled: bool = True

class NewswireMonitor:
    def __init__(self):
        self.clickhouse_manager = None
        self.session = None
        self.batch_queue = []
        self.last_batch_insert = time.time()
        self.duplicate_hashes = set()
        self.ticker_patterns = {}  # Initialize ticker patterns dict
        
        # Statistics
        self.stats = {
            'articles_processed': 0,
            'articles_inserted': 0,
            'duplicates_filtered': 0,
            'errors': 0,
            'last_reset': time.time()
        }
        
        # Newswire sources optimized for speed
        self.sources = [
            NewsSource("GlobeNewswire", 
                      "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20Releases", 
                      0.5, 1),
            NewsSource("BusinessWire", 
                      "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEF9YXA==", 
                      0.5, 1),
            NewsSource("PRNewswire", 
                      "https://www.prnewswire.com/rss/news-releases-list.rss", 
                      1.0, 2),
            NewsSource("MarketWatch_Bulletins", 
                      "https://feeds.content.dowjones.io/public/rss/mw_bulletins", 
                      0.5, 1)
        ]

    async def initialize(self):
        """Initialize the newswire monitor"""
        # Setup ClickHouse connection
        self.clickhouse_manager = setup_clickhouse_database()
        
        # Drop and recreate breaking_news table for fresh start
        await self.reset_breaking_news_table()
        
        # Initialize HTTP session with optimized settings
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        # Load ticker list
        await self.load_tickers()
        
        # Compile ticker patterns for faster matching
        self.compile_ticker_patterns()
        
        logger.info(f"Initialized NewswireMonitor with {len(self.ticker_list)} tickers")

    async def reset_breaking_news_table(self):
        """Drop and recreate breaking_news table for a fresh start"""
        try:
            logger.info("Resetting breaking_news table...")
            
            # Drop the breaking_news table
            self.clickhouse_manager.client.command("DROP TABLE IF EXISTS News.breaking_news")
            logger.info("Dropped breaking_news table")
            
            # Recreate the breaking_news table with the same structure
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
            self.clickhouse_manager.client.command(breaking_news_sql)
            logger.info("Recreated breaking_news table")
            
        except Exception as e:
            logger.error(f"Error resetting breaking_news table: {e}")
            raise

    async def load_tickers(self):
        """Load ticker list from ClickHouse database"""
        try:
            # Try to get tickers from database first
            db_tickers = self.clickhouse_manager.get_active_tickers()
            
            if db_tickers:
                self.ticker_list = db_tickers
                logger.info(f"Loaded {len(self.ticker_list)} tickers from database")
            else:
                # Fallback to CSV file if database is empty
                logger.warning("No tickers in database, falling back to CSV file")
                csv_path = os.path.join('data_files', 'FV_master_u50float_u10price.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    self.ticker_list = [str(ticker).strip().upper() for ticker in df['Ticker'].tolist() if pd.notna(ticker)]
                    logger.info(f"Loaded {len(self.ticker_list)} tickers from CSV fallback")
                else:
                    logger.error("No CSV fallback file found")
                    self.ticker_list = []
                    
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            self.ticker_list = []

    def compile_ticker_patterns(self):
        """Pre-compile regex patterns for faster ticker matching"""
        for ticker in self.ticker_list:
            # Create multiple patterns for different ticker formats
            patterns = [
                rf'\b{re.escape(ticker)}\b',  # Basic word boundary
                rf'\${re.escape(ticker)}\b',  # $TICKER format
                rf'NASDAQ:{re.escape(ticker)}\b',  # NASDAQ:TICKER
                rf'NYSE:{re.escape(ticker)}\b',   # NYSE:TICKER
            ]
            self.ticker_patterns[ticker] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """Fast ticker extraction using pre-compiled patterns"""
        if not text:
            return []
            
        found_tickers = set()
        text_upper = text.upper()
        
        # Quick check: only process if text contains potential ticker-like patterns
        if not re.search(r'\b[A-Z]{1,5}\b', text_upper):
            return []
        
        for ticker, patterns in self.ticker_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    found_tickers.add(ticker)
                    break  # Found ticker, no need to check other patterns
        
        return list(found_tickers)

    def generate_content_hash(self, title: str, url: str) -> str:
        """Generate hash for duplicate detection based on URL only"""
        return hashlib.md5(url.encode()).hexdigest()

    def is_recent_article(self, published_time: str, max_age_seconds: int = 1800) -> bool:
        """Check if article is recent enough"""
        if not published_time:
            return False
            
        try:
            # Parse the published time
            if isinstance(published_time, str):
                # Try different parsing methods
                try:
                    # Try ISO format first
                    parsed_time = datetime.fromisoformat(published_time.replace('Z', '+00:00'))
                except:
                    try:
                        # Try common RSS date formats
                        from email.utils import parsedate_to_datetime
                        parsed_time = parsedate_to_datetime(published_time)
                    except:
                        try:
                            # Try strptime with common formats
                            for fmt in [
                                '%a, %d %b %Y %H:%M:%S %Z',
                                '%a, %d %b %Y %H:%M:%S %z',
                                '%Y-%m-%d %H:%M:%S %Z',
                                '%Y-%m-%d %H:%M:%S %z',
                                '%a, %d %b %Y %H:%M %Z',
                                '%a, %d %b %Y %H:%M %z'
                            ]:
                                try:
                                    parsed_time = datetime.strptime(published_time, fmt)
                                    break
                                except:
                                    continue
                            else:
                                return False
                        except:
                            return False
            else:
                parsed_time = published_time
                
            # Ensure timezone awareness
            if parsed_time.tzinfo is None:
                parsed_time = pytz.UTC.localize(parsed_time)
                
            current_time = datetime.now(pytz.UTC)
            age = (current_time - parsed_time).total_seconds()
            
            return age <= max_age_seconds
            
        except Exception as e:
            logger.warning(f"Error parsing article time {published_time}: {e}")
            return False

    def parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime string to datetime object"""
        if not date_str:
            return datetime.now()
            
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                # Try email.utils parser for RSS dates
                from email.utils import parsedate_to_datetime
                return parsedate_to_datetime(date_str)
            except:
                try:
                    # Try strptime with common formats
                    for fmt in [
                        '%a, %d %b %Y %H:%M:%S %Z',
                        '%a, %d %b %Y %H:%M:%S %z',
                        '%Y-%m-%d %H:%M:%S %Z',
                        '%Y-%m-%d %H:%M:%S %z',
                        '%a, %d %b %Y %H:%M %Z',
                        '%a, %d %b %Y %H:%M %z'
                    ]:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                except:
                    pass
                
        return datetime.now()

    async def fetch_and_parse_feed(self, source: NewsSource) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed with error handling and retry logic"""
        articles = []
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                start_time = time.time()
                
                # Adjust timeout based on source reliability
                timeout_seconds = 15 if source.name == "GlobeNewswire" else 10
                
                async with self.session.get(source.rss_url, 
                                          timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as response:
                    response_time = time.time() - start_time
                    
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {source.name}")
                        if retry_count < max_retries:
                            retry_count += 1
                            await asyncio.sleep(2)  # Wait before retry
                            continue
                        return articles
                    
                    feed_content = await response.text()
                    
                # Parse feed
                feed = feedparser.parse(feed_content)
                
                if not feed.entries:
                    logger.debug(f"No entries in {source.name} feed")
                    return articles
                    
                processing_start = time.time()
                
                for entry in feed.entries:
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        published = entry.get('published', '')
                        
                        # Skip if not recent
                        if not self.is_recent_article(published, max_age_seconds=1800):
                            continue
                            
                        # Generate hash for duplicate detection
                        content_hash = self.generate_content_hash(title, link)
                        
                        if content_hash in self.duplicate_hashes:
                            self.stats['duplicates_filtered'] += 1
                            continue
                        
                        # Extract tickers from title and summary
                        text_to_search = f"{title} {summary}"
                        found_tickers = self.extract_tickers_from_text(text_to_search)
                        
                        if found_tickers:
                            # Calculate processing latency
                            latency_ms = int((time.time() - processing_start) * 1000)
                            
                            for ticker in found_tickers:
                                article = {
                                    'timestamp': datetime.now(),
                                    'source': source.name,
                                    'ticker': ticker,
                                    'headline': title,
                                    'published_utc': self.parse_datetime(published),
                                    'article_url': link,
                                    'summary': summary,
                                    'full_content': '',  # Can be enhanced later with full text extraction
                                    'detected_at': datetime.now(),
                                    'processing_latency_ms': latency_ms,
                                    'market_relevant': 1,  # Assume relevant if ticker found
                                    'source_check_time': datetime.fromtimestamp(start_time),
                                    'content_hash': content_hash,
                                    'news_type': self.classify_news_type(title, summary),
                                    'urgency_score': self.calculate_urgency_score(title, summary)
                                }
                                articles.append(article)
                                
                            # Add to processed set
                            self.duplicate_hashes.add(content_hash)
                            
                            # Enhanced logging with timing details
                            detection_delay = (datetime.now() - self.parse_datetime(published).replace(tzinfo=None)).total_seconds() if self.parse_datetime(published) else 0
                            logger.info(f"📰 Found news for {found_tickers} from {source.name}: {title[:50]}... "
                                      f"(Published: {published}, Detection delay: {detection_delay:.1f}s)")
                    
                    except Exception as e:
                        logger.warning(f"Error processing entry from {source.name}: {e}")
                        continue
                        
                self.stats['articles_processed'] += len(articles)
                return articles  # Success, exit retry loop
                
            except asyncio.TimeoutError:
                retry_count += 1
                logger.warning(f"Timeout fetching {source.name} (attempt {retry_count}/{max_retries + 1})")
                if retry_count <= max_retries:
                    await asyncio.sleep(3)  # Wait before retry
                else:
                    self.stats['errors'] += 1
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Error fetching {source.name} (attempt {retry_count}/{max_retries + 1}): {e}")
                if retry_count <= max_retries:
                    await asyncio.sleep(3)  # Wait before retry
                else:
                    self.stats['errors'] += 1
                    
        return articles

    def classify_news_type(self, title: str, summary: str) -> str:
        """Classify news type based on content"""
        text = f"{title} {summary}".lower()
        
        if any(word in text for word in ['earnings', 'quarter', 'q1', 'q2', 'q3', 'q4', 'revenue']):
            return 'earnings'
        elif any(word in text for word in ['fda', 'approval', 'clinical', 'trial', 'phase']):
            return 'clinical_trial'
        elif any(word in text for word in ['merger', 'acquisition', 'acquire', 'buyout']):
            return 'merger'
        elif any(word in text for word in ['partnership', 'collaboration', 'agreement']):
            return 'partnership'
        else:
            return 'other'

    def calculate_urgency_score(self, title: str, summary: str) -> int:
        """Calculate urgency score 1-10"""
        text = f"{title} {summary}".lower()
        score = 5  # Default score
        
        # High urgency keywords
        if any(word in text for word in ['breaking', 'urgent', 'halt', 'suspend']):
            score += 3
        elif any(word in text for word in ['announces', 'reports', 'files']):
            score += 1
            
        # FDA/Clinical trial urgency
        if 'fda' in text and any(word in text for word in ['approval', 'breakthrough', 'fast track']):
            score += 2
            
        return min(10, max(1, score))

    async def flush_buffer_to_clickhouse(self):
        """Flush article buffer to ClickHouse"""
        if not self.batch_queue:
            return
            
        try:
            inserted_count = self.clickhouse_manager.insert_articles(self.batch_queue)
            self.stats['articles_inserted'] += inserted_count
            
            logger.info(f"Flushed {inserted_count} articles to ClickHouse")
            self.batch_queue.clear()
            
        except Exception as e:
            logger.error(f"Error flushing buffer to ClickHouse: {e}")
            self.stats['errors'] = self.stats.get('errors', 0) + 1

    async def monitor_source(self, source: NewsSource):
        """Monitor a single news source"""
        logger.info(f"Starting monitor for {source.name} (interval: {source.poll_interval}s)")
        
        while True:
            try:
                articles = await self.fetch_and_parse_feed(source)
                
                if articles:
                    self.batch_queue.extend(articles)
                    logger.info(f"Added {len(articles)} articles from {source.name} to buffer")
                
            except Exception as e:
                logger.error(f"Error monitoring {source.name}: {e}")
                
            await asyncio.sleep(source.poll_interval)

    async def buffer_flusher(self):
        """Periodically flush buffer to ClickHouse"""
        while True:
            try:
                await asyncio.sleep(2)  # Flush every 2 seconds for faster detection
                await self.flush_buffer_to_clickhouse()
                
                # Cleanup processed hashes to prevent memory growth
                if len(self.duplicate_hashes) > 10000:
                    # Keep only the most recent 5000 hashes
                    self.duplicate_hashes = set(list(self.duplicate_hashes)[-5000:])
                    
            except Exception as e:
                logger.error(f"Error in buffer flusher: {e}")

    async def stats_reporter(self):
        """Report performance statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                runtime = time.time() - self.stats['last_reset']
                rate = self.stats['articles_processed'] / runtime if runtime > 0 else 0
                
                logger.info(f"STATS - Runtime: {runtime:.1f}s, "
                          f"Processed: {self.stats['articles_processed']}, "
                          f"Inserted: {self.stats['articles_inserted']}, "
                          f"Duplicates: {self.stats['duplicates_filtered']}, "
                          f"Errors: {self.stats['errors']}, "
                          f"Rate: {rate:.2f} articles/sec")
                          
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")

    async def start_monitoring(self):
        """Start monitoring all sources"""
        logger.info("Starting newswire monitoring...")
        
        # Initialize
        await self.initialize()
        
        # Create tasks for each source
        tasks = []
        
        # Monitor each source
        for source in self.sources:
            if source.enabled:
                task = asyncio.create_task(self.monitor_source(source))
                tasks.append(task)
        
        # Buffer flusher task
        buffer_task = asyncio.create_task(self.buffer_flusher())
        tasks.append(buffer_task)
        
        # Stats reporter task
        stats_task = asyncio.create_task(self.stats_reporter())
        tasks.append(stats_task)
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            # Clean up
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Final flush before shutdown
            await self.flush_buffer_to_clickhouse()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                
            # Close ClickHouse connection
            if self.clickhouse_manager:
                self.clickhouse_manager.close()
                
            logger.info("NewswireMonitor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function to run the newswire monitor"""
    logger.info("Starting Newswire Monitor")
    
    # Create and start monitor
    monitor = NewswireMonitor()
    
    try:
        # Start monitoring (this calls initialize internally)
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Newswire Monitor stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in monitor: {e}")
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 