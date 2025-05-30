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
            'BusinessWire': "https://www.businesswire.com/newsroom",
            'PRNewswire': "https://www.prnewswire.com/news-releases/news-releases-list/"
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
        """Simple ticker extraction - look for exact ticker as whole word in ALL CAPS ONLY"""
        if not text:
            return []
            
        found_tickers = []
        
        for ticker in self.ticker_list:
            # Use word boundaries to match whole words only - NO re.IGNORECASE so only ALL CAPS matches
            pattern = rf'\b{re.escape(ticker)}\b'
            if re.search(pattern, text):  # Removed re.IGNORECASE - only ALL CAPS will match
                found_tickers.append(ticker)
        
        return found_tickers

    def generate_content_hash(self, title: str, url: str) -> str:
        """Generate hash for duplicate detection"""
        content = f"{title}{url}"
        return hashlib.md5(content.encode()).hexdigest()

    def parse_relative_time(self, time_text: str) -> datetime:
        """Extract timestamp directly from article - store exactly what's found"""
        if not time_text:
            return datetime.now()
        
        time_text = time_text.strip()
        
        try:
            # Look for time pattern like "11:26 ET" or "10:59 EST"
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
                    logger.info(f"🕐 EXACT TIME: '{time_text}' -> {parsed_time} (diff: {time_diff:.1f}s)")
                
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
                            
                            # Get timestamp from individual article
                            time_text = await self.get_article_timestamp(url)
                            
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

    async def get_article_timestamp(self, url: str) -> str:
        """Get timestamp from individual article page - enhanced to find more timestamp formats"""
        try:
            result: CrawlResult = await self.crawler.arun(
                url=url,
                timeout=10
            )
            
            if result.success and result.html:
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Comprehensive list of selectors for different news sites
                time_selectors = [
                    ".timestamp", ".date", "time", ".published", ".publish-date",
                    ".article-date", ".news-date", ".release-date", ".story-date",
                    ".byline-timestamp", ".article-timestamp", ".publish-time",
                    "[datetime]", "[data-timestamp]", ".dateline", ".publication-date",
                    ".meta-date", ".entry-date", ".post-date", ".created-date",
                    ".article-meta time", ".byline time", ".header-date",
                    ".news-meta .date", ".article-info .date", ".story-meta .date"
                ]
                
                # Try each selector
                for selector in time_selectors:
                    time_elem = soup.select_one(selector)
                    if time_elem:
                        # Try getting datetime attribute first
                        datetime_attr = time_elem.get('datetime') or time_elem.get('data-timestamp')
                        if datetime_attr:
                            logger.debug(f"Found datetime attribute: {datetime_attr}")
                            return datetime_attr
                        
                        # Get text content
                        time_text = time_elem.get_text(strip=True)
                        if time_text and len(time_text) > 3:  # Minimum reasonable timestamp length
                            logger.debug(f"Found timestamp text: {time_text}")
                            return time_text
                
                # Fallback: search for common date/time patterns in page text
                page_text = soup.get_text()
                
                # Look for patterns like "Jan 15, 2024 10:30 AM EST" or "2024-01-15 10:30:00"
                date_patterns = [
                    r'\b[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]M\s*[A-Z]{2,3}\b',  # Jan 15, 2024 10:30 AM EST
                    r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b',  # 2024-01-15 10:30:00
                    r'\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*[AP]M\b',  # 1/15/2024 10:30 AM
                    r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b',  # January 15, 2024
                    r'\b\d{1,2}:\d{2}\s*[AP]M\s*[A-Z]{2,3}\b',  # 10:30 AM EST
                    r'\b\d{1,2}:\d{2}\s*ET\b'  # 10:30 ET
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, page_text)
                    if match:
                        found_date = match.group()
                        logger.debug(f"Found date pattern: {found_date}")
                        return found_date
                
                # Last resort: look for any time-like text in meta tags
                meta_tags = soup.find_all('meta')
                for meta in meta_tags:
                    name = meta.get('name', '').lower()
                    property_attr = meta.get('property', '').lower()
                    content = meta.get('content', '')
                    
                    if any(keyword in name or keyword in property_attr for keyword in 
                           ['date', 'time', 'published', 'created', 'modified']):
                        if content and len(content) > 5:
                            logger.debug(f"Found meta date: {content}")
                            return content
                
        except Exception as e:
            logger.debug(f"Could not get timestamp from {url}: {e}")
        
        return "NO_TIME"

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
                await asyncio.sleep(30)  # Check every 30 seconds
                
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