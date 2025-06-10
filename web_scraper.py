#!/usr/bin/env python3
"""
Direct Website Scraper using Crawl4AI
Scrapes news websites directly for faster news detection than RSS feeds
"""

import asyncio
import time
import re
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
import pytz
from dataclasses import dataclass
from clickhouse_setup import setup_clickhouse_database
import pandas as pd
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import aiohttp
import feedparser
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingTarget:
    name: str
    url: str
    css_selector: str  # CSS selector for news items
    poll_interval: float
    enabled: bool = True

class Crawl4AIScraper:
    def __init__(self):
        self.clickhouse_manager = None
        self.crawler = None
        self.batch_queue = []
        self.ticker_list = []
        
        # Performance tracking
        self.stats = {
            'articles_processed': 0,
            'articles_inserted': 0,
            'errors': 0,
            'total_runtime': 0
        }
        
        # SIMPLE: All three major newswire feeds
        self.sources = {
            'GlobeNewswire': "https://www.globenewswire.com/en/search/date/24HOURS?pageSize=50&page=1",
            'BusinessWire': "https://www.businesswire.com/newsroom?language=en",
            'PRNewswire': "https://www.prnewswire.com/news-releases/news-releases-list/?page=1&pagesize=50"
        }

    async def initialize(self):
        """Initialize the Crawl4AI scraper with better error handling"""
        # Setup ClickHouse connection
        self.clickhouse_manager = setup_clickhouse_database()
        
        # Initialize Crawl4AI AsyncWebCrawler with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.crawler = AsyncWebCrawler(
                    verbose=True,
                    headless=True,
                    browser_type="chromium",
                    # Add these for stability
                    max_idle_time=30000,  # 30 seconds
                    keep_alive=True
                )
                await self.crawler.start()
                logger.info(f"✅ Crawl4AI browser started successfully (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.warning(f"❌ Failed to start Crawl4AI browser (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize Crawl4AI after {max_retries} attempts")
                await asyncio.sleep(2)
        
        # Load ticker list
        await self.load_tickers()
        
        # Compile ticker patterns for faster matching
        self.compile_ticker_patterns()
        
        logger.info(f"Initialized Crawl4AI Scraper with {len(self.ticker_list)} tickers")

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
                if os.path.exists('data_files/FV_master_u50float_u10price.csv'):
                    df = pd.read_csv('data_files/FV_master_u50float_u10price.csv')
                    self.ticker_list = [str(ticker).strip().upper() for ticker in df['Ticker'].tolist() if pd.notna(ticker)]
                    logger.info(f"Loaded {len(self.ticker_list)} tickers from CSV fallback")
                else:
                    logger.error("No CSV fallback file found")
                    self.ticker_list = []
                    
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            self.ticker_list = []

    def compile_ticker_patterns(self):
        """Simple ticker matching - just exact ticker in all caps"""
        pass  # No pre-compilation needed for simple matching

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract tickers from text using multiple patterns including quoted formats"""
        if not text:
            return []
            
        found_tickers = []
        
        for ticker in self.ticker_list:
            ticker_escaped = re.escape(ticker)
            
            # Only match tickers in proper financial contexts - NO broad word matching
            patterns = [
                # Exchange patterns: ":TICKER" (e.g., "Nasdaq: STSS", "NYSE: AAPL")
                rf':\s*{ticker_escaped}\b',
                
                # Quoted pattern: "TICKER" (e.g., "STSS" and "STSSW")
                rf'"{ticker_escaped}"',
                
                # Parenthetical with exchange: (Exchange: TICKER) (e.g., "(NYSE: AAPL)")
                rf'\([^)]*:\s*{ticker_escaped}\)',
                
                # Exchange with quotes: ": "TICKER"" (e.g., ': "STSS"')
                rf':\s*"{ticker_escaped}"',
                
                # Parenthetical ticker only: (TICKER) - but only if 3+ chars to avoid common words
                rf'\({ticker_escaped}\)' if len(ticker) >= 3 else None
            ]
            
            # Remove None patterns and check each remaining pattern
            valid_patterns = [p for p in patterns if p is not None]
            for pattern in valid_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_tickers.append(ticker)
                    logger.debug(f"Found ticker {ticker} using pattern: {pattern}")
                    break  # Found with one pattern, no need to check others for this ticker
        
        return found_tickers

    def generate_content_hash(self, title: str, url: str) -> str:
        """Generate hash for duplicate detection"""
        content = f"{title}{url}"
        return hashlib.md5(content.encode()).hexdigest()

    def parse_relative_time(self, time_text: str) -> datetime:
        """Extract timestamp from article - now handles full datetime formats"""
        if not time_text:
            return datetime.now()
        
        time_text = time_text.strip()
        
        try:
            # Handle full datetime patterns like "June 09, 2025 12:58 ET"
            full_datetime_patterns = [
                # "June 09, 2025 12:58 ET" format
                (r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*ET', '%B %d, %Y %H:%M'),
                (r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*EST', '%B %d, %Y %H:%M'),
                # "June 09, 2025 12:58 PM ET" format
                (r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*([AP]M)\s*ET', '%B %d, %Y %I:%M %p'),
                # ISO format
                (r'(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2}):(\d{2})', '%Y-%m-%d %H:%M:%S'),
                (r'(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})', '%Y-%m-%d %H:%M'),
            ]
            
            for pattern, format_str in full_datetime_patterns:
                if re.search(pattern, time_text):
                    try:
                        # For the full month name patterns, we need to handle the parsing differently
                        if 'A-Za-z' in pattern:
                            # Clean up the text for parsing
                            clean_text = re.sub(r'\s*ET$|\s*EST$', '', time_text)
                            parsed_time = datetime.strptime(clean_text, format_str.replace(' ET', '').replace(' EST', ''))
                        else:
                            parsed_time = datetime.strptime(time_text, format_str)
                        
                        logger.info(f"🕐 PARSED FULL DATETIME: '{time_text}' -> {parsed_time}")
                        return parsed_time
                    except ValueError as e:
                        logger.debug(f"Failed to parse with pattern {pattern}: {e}")
                        continue
            
            # Fallback to time-only parsing (existing logic)
            time_match = re.search(r'(\d{1,2}):(\d{2})', time_text)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                
                # Use today's date with the EXACT time found in the article
                today = datetime.now().date()
                parsed_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
                
                # If this time is more than 12 hours in the future, assume it's from yesterday
                time_diff = (datetime.now() - parsed_time).total_seconds()
                if time_diff < -43200:  # More than 12 hours in future
                    parsed_time = parsed_time - timedelta(days=1)
                    logger.info(f"🕐 FROM YESTERDAY: '{time_text}' -> {parsed_time} (diff: {(datetime.now() - parsed_time).total_seconds():.1f}s)")
                else:
                    logger.info(f"🕐 TIME ONLY: '{time_text}' -> {parsed_time} (diff: {time_diff:.1f}s)")
                
                return parsed_time
                
        except Exception as e:
            logger.debug(f"Error parsing time '{time_text}': {e}")
        
        # Fallback to current time
        logger.warning(f"⚠️ Could not parse time '{time_text}', using current time")
        return datetime.now()

    async def scrape_all_newswires(self) -> List[Dict[str, Any]]:
        """Scrape all newswire sources and find EXACT ticker matches"""
        all_articles = []
        
        for source_name, source_url in self.sources.items():
            try:
                logger.info(f"🔍 Scraping {source_name}...")
                
                result: CrawlResult = await self.crawler.arun(
                    url=source_url,
                    wait_for="css:.news-item, .search-result, .bw-release-story, .newsreleaseheadline",
                    delay_before_return_html=2.0,
                    timeout=30
                )
                
                if not result.success or not result.html:
                    logger.error(f"❌ Failed to scrape {source_name}: {result.error_message}")
                    continue
                
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Find all article links and titles
                article_links = soup.find_all('a', href=True)
                
                logger.info(f"📰 Found {len(article_links)} potential articles from {source_name}")
                
                for link in article_links:
                    try:
                        title = link.get_text(strip=True)
                        url = link.get('href', '')
                        
                        if not title or not url or len(title) < 20:
                            continue
                        
                        # Make URL absolute
                        if url.startswith('/'):
                            if source_name == 'GlobeNewswire':
                                url = "https://www.globenewswire.com" + url
                            elif source_name == 'BusinessWire':
                                url = "https://www.businesswire.com" + url
                            elif source_name == 'PRNewswire':
                                url = "https://www.prnewswire.com" + url
                        
                        # Skip non-news URLs
                        if not any(x in url for x in ['news-release', 'story', 'releases']):
                            continue
                        
                        # Get any available description/summary text
                        description = ""
                        parent = link.parent
                        if parent:
                            # Look for description in nearby elements
                            desc_elem = parent.find('p') or parent.find('div', class_='summary') or parent.find('div', class_='description')
                            if desc_elem:
                                description = desc_elem.get_text(strip=True)
                        
                        # Search for tickers in title AND description (like RSS monitor)
                        text_to_search = f"{title} {description}"
                        found_tickers = self.extract_tickers_from_text(text_to_search)
                        
                        if found_tickers:
                            logger.info(f"✅ TICKER MATCH: {found_tickers} in {source_name} title: {title}")
                            
                            # Extract timestamp from the listing page instead of individual article
                            time_text = self.extract_timestamp_from_listing(link, source_name)
                            
                            # Parse the full datetime
                            parsed_timestamp = self.parse_relative_time(time_text)
                            
                            # SANITY CHECK: Only apply current day filter to GlobeNewswire (24HOURS can include yesterday)
                            # Other sources (BusinessWire, PRNewswire) already filter to current day
                            if source_name == 'GlobeNewswire':
                                current_date = datetime.now().date()
                                article_date = parsed_timestamp.date()
                                
                                if article_date != current_date:
                                    logger.debug(f"⏰ SKIPPING old GlobeNewswire article from {article_date}: {title[:50]}...")
                                    continue
                            
                            # Generate content hash for deduplication
                            content_hash = self.generate_content_hash(title, url)
                            
                            # Create an article for each ticker found
                            for ticker in found_tickers:
                                article = {
                                    'timestamp': datetime.now(),
                                    'source': f'{source_name}_24H',
                                    'ticker': ticker,
                                    'headline': title,
                                    'published_utc': time_text,  # Store raw string as per schema
                                    'article_url': url,
                                    'summary': title,
                                    'full_content': title,
                                    'detected_at': datetime.now(),
                                    'processing_latency_ms': 0,
                                    'market_relevant': 1,
                                    'source_check_time': datetime.now(),
                                    'content_hash': content_hash,
                                    'news_type': 'other',
                                    'urgency_score': 5
                                }
                                
                                # Add article directly to batch (let database handle duplicates)
                                all_articles.append(article)
                                logger.info(f"✅ NEW ARTICLE: {ticker} - {title[:50]}...")
                        else:
                            logger.debug(f"❌ No tickers found in: {title[:50]}...")
                        
                    except Exception as e:
                        logger.debug(f"Error processing article: {e}")
                        continue
                
                logger.info(f"🎯 Found {len([a for a in all_articles if a['source'].startswith(source_name)])} articles with exact ticker matches from {source_name}")
                
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
                continue
        
        return all_articles

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

    async def monitor_all_newswires(self):
        """Monitor ALL newswire sources for exact ticker matches"""
        logger.info("Starting ALL newswire monitoring (GlobeNewswire, BusinessWire, PRNewswire)...")
        
        while True:
            try:
                start_time = time.time()
                
                # Scrape all newswire sources
                articles = await self.scrape_all_newswires()
                
                if articles:
                    self.batch_queue.extend(articles)
                    self.stats['articles_processed'] += len(articles)
                
                total_time = time.time() - start_time
                logger.info(f"🔄 All newswires scan completed: {len(articles)} articles in {total_time:.2f}s")
                
                # Wait before next cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in newswire monitoring: {e}")
                await asyncio.sleep(60)

    async def buffer_flusher(self):
        """Periodically flush buffer to ClickHouse"""
        while True:
            try:
                await asyncio.sleep(3)  # Flush every 3 seconds for fast detection
                await self.flush_buffer_to_clickhouse()
                
            except Exception as e:
                logger.error(f"Error in buffer flusher: {e}")

    async def stats_reporter(self):
        """Report performance statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                runtime = time.time() - self.stats['total_runtime']
                rate = self.stats['articles_processed'] / runtime if runtime > 0 else 0
                
                logger.info(f"BULK SCRAPING STATS - Runtime: {runtime:.1f}s, "
                          f"Processed: {self.stats['articles_processed']}, "
                          f"Inserted: {self.stats['articles_inserted']}, "
                          f"Errors: {self.stats['errors']}, "
                          f"Rate: {rate:.2f} articles/sec")
                          
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")

    async def start_scraping(self):
        """Start scraping using bulk approach - MUCH faster"""
        logger.info("🚀 Starting BULK scraping (get all articles, filter locally - MUCH faster)...")
        
        # Initialize
        await self.initialize()
        
        # Create tasks
        tasks = []
        
        # Bulk scraping monitoring task (only 3 HTTP requests per cycle!)
        bulk_task = asyncio.create_task(self.monitor_all_newswires())
        tasks.append(bulk_task)
        
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
            logger.info("Bulk scraping stopped by user")
        finally:
            # Clean up
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Final flush before shutdown
            await self.flush_buffer_to_clickhouse()
            
            # Close Crawl4AI crawler
            if self.crawler:
                await self.crawler.close()
                
            # Close ClickHouse connection
            if self.clickhouse_manager:
                self.clickhouse_manager.close()
                
            logger.info("Crawl4AI Scraper cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_rss_for_comparison(self, site_config: dict) -> List[str]:
        """Get RSS content to compare what we should be finding"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(site_config['rss_url']) as response:
                    if response.status == 200:
                        rss_content = await response.text()
                        feed = feedparser.parse(rss_content)
                        
                        recent_titles = []
                        for entry in feed.entries[:10]:  # First 10 RSS entries
                            title = entry.get('title', '')
                            published = entry.get('published', '')
                            recent_titles.append(f"{title[:50]}... ({published})")
                        
                        logger.info(f"📊 RSS COMPARISON for {site_config['name']}: Found {len(feed.entries)} RSS entries")
                        for i, title in enumerate(recent_titles[:3]):
                            logger.info(f"   RSS #{i+1}: {title}")
                        
                        return [entry.get('title', '') for entry in feed.entries[:20]]
                        
        except Exception as e:
            logger.warning(f"Could not fetch RSS for comparison: {e}")
            return []

    def extract_timestamp_from_listing(self, link, source_name):
        """Extract timestamp directly from listing page elements - much faster than individual requests"""
        try:
            # Find timestamp in the same container as the link
            container = link.parent
            if container:
                # Try multiple approaches to find timestamp in nearby elements
                
                # Look for time in text content around the link
                container_text = container.get_text()
                
                # Enhanced timestamp patterns that capture full datetime
                time_patterns = [
                    # Full date patterns like "June 09, 2025 12:58 ET"
                    r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*ET\b',     # "June 09, 2025 12:58 ET"
                    r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*EST\b',    # "June 09, 2025 12:58 EST"
                    r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]M\s*ET\b',  # "June 09, 2025 12:58 PM ET"
                    
                    # ISO-like patterns
                    r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b',                     # "2024-01-15 10:30:00"
                    r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\b',                           # "2024-01-15 10:30"
                    
                    # Shorter date patterns  
                    r'\b[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\b',                            # "Jan 15, 2024"
                    
                    # Time-only patterns (fallback)
                    r'\b\d{1,2}:\d{2}\s*ET\b',                                        # "09:30 ET"
                    r'\b\d{1,2}:\d{2}\s*EST\b',                                       # "09:30 EST"
                    r'\b\d{1,2}:\d{2}\s*[AP]M\s*ET\b',                               # "09:30 AM ET"
                    r'\b\d{1,2}:\d{2}\s*[AP]M\s*EST\b',                              # "09:30 AM EST"
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, container_text)
                    if match:
                        found_time = match.group().strip()
                        logger.debug(f"Found timestamp in listing: {found_time}")
                        return found_time
                
                # If no patterns match, look for specific timestamp elements
                for time_elem in container.find_all(['time', 'span', 'div']):
                    # Check for datetime attributes
                    datetime_attr = time_elem.get('datetime')
                    if datetime_attr:
                        return datetime_attr
                    
                    # Check for timestamp-like classes
                    elem_class = time_elem.get('class', [])
                    if any(cls for cls in elem_class if 'time' in cls.lower() or 'date' in cls.lower()):
                        time_text = time_elem.get_text(strip=True)
                        if time_text and len(time_text) > 3:
                            return time_text
                
                # Fallback: look for time pattern in broader parent container
                broader_container = container.parent
                if broader_container:
                    broader_text = broader_container.get_text()
                    for pattern in time_patterns:
                        match = re.search(pattern, broader_text)
                        if match:
                            found_time = match.group().strip()
                            logger.debug(f"Found timestamp in broader container: {found_time}")
                            return found_time
            
        except Exception as e:
            logger.debug(f"Error extracting timestamp from listing for {source_name}: {e}")
        
        # Return current time as fallback - still better than failing completely
        current_time = datetime.now().strftime("%H:%M ET")
        logger.debug(f"Using current time as fallback: {current_time}")
        return current_time

async def main():
    """Main function to run the Crawl4AI web scraper"""
    logger.info("🚀 Starting Crawl4AI Website Scraper - MORE RELIABLE than basic scraping!")
    
    # Create and start scraper
    scraper = Crawl4AIScraper()
    
    try:
        # Start scraping (this calls initialize internally)
        await scraper.start_scraping()
        
    except KeyboardInterrupt:
        logger.info("Crawl4AI Web Scraper stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in scraper: {e}")
    finally:
        await scraper.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 