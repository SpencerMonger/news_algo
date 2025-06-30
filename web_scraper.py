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
from clickhouse_setup import ClickHouseManager
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
    def __init__(self, enable_old=False):
        self.clickhouse_manager = None
        self.crawler = None
        self.batch_queue = []
        self.ticker_list = []
        self.enable_old = enable_old  # Flag to disable freshness filtering for testing
        
        # Debug logging to confirm enable_old flag
        if self.enable_old:
            logger.info("🔓 FRESHNESS FILTER DISABLED - Will process old news articles")
        else:
            logger.info("⏰ FRESHNESS FILTER ENABLED - Will skip articles older than 2 minutes")
        
        # Performance tracking
        self.stats = {
            'articles_processed': 0,
            'articles_inserted': 0,
            'errors': 0,
            'total_runtime': 0,
            'rss_articles_processed': 0,
            'rss_articles_inserted': 0
        }
        
        # SIMPLE: All major newswire feeds
        self.sources = {
            'GlobeNewswire': "https://www.globenewswire.com/en/search/date/24HOURS?pageSize=50&page=1",
            'BusinessWire': "https://www.businesswire.com/newsroom?language=en",
            'PRNewswire': "https://www.prnewswire.com/news-releases/news-releases-list/?page=1&pagesize=50",
            'AccessNewswire': "https://www.accessnewswire.com/newsroom"
        }
        
        # RSS feeds for comparison testing
        self.rss_feeds = {
            'GlobeNewswire': "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20Releases",
            'BusinessWire': "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEF9YXA==",
            'PRNewswire': "https://www.prnewswire.com/rss/news-releases-list.rss"
        }

    async def initialize(self):
        """Initialize the Crawl4AI scraper with SPEED-OPTIMIZED CPU efficiency"""
        logger.info("🚀 Initializing Crawl4AI scraper with SPEED-OPTIMIZED CPU efficiency...")
        
        # FIXED: Don't call setup_clickhouse_database() as it WIPES all tables!
        # Instead, create direct connection and assume tables already exist
        self.clickhouse_manager = ClickHouseManager()
        self.clickhouse_manager.connect()
        
        # Initialize Crawl4AI AsyncWebCrawler with SPEED-OPTIMIZED CPU efficiency
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.crawler = AsyncWebCrawler(
                    verbose=False,  # Reduced logging to save CPU
                    headless=True,
                    browser_type="chromium",
                    
                    # SPEED-OPTIMIZED: Maintain performance while reducing CPU usage
                    max_idle_time=30000,  # 30s - reasonable timeout
                    keep_alive=True,
                    
                    # EFFICIENT RESOURCE LIMITS - Balance speed vs CPU
                    max_memory_usage=512,  # 512MB - enough for fast processing
                    max_concurrent_sessions=2,  # 2 sessions for parallel processing
                    delay_between_requests=0.5,  # Fast 0.5s delay for speed
                    
                    # CPU-EFFICIENT BROWSER FLAGS - Reduce CPU without slowing down
                    extra_args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        
                        # DISABLE CPU-INTENSIVE FEATURES (keep speed)
                        "--disable-gpu",  # No GPU rendering needed for scraping
                        "--disable-software-rasterizer",
                        "--disable-background-timer-throttling",  # CRITICAL: Don't throttle our timers
                        "--disable-backgrounding-occluded-windows",
                        "--disable-renderer-backgrounding",
                        "--disable-features=TranslateUI",
                        "--disable-ipc-flooding-protection",
                        
                        # EFFICIENT MEMORY MANAGEMENT
                        "--memory-pressure-off",
                        "--max_old_space_size=256",  # Reasonable memory limit
                        "--aggressive-cache-discard",  # Discard unused cache aggressively
                        
                        # DISABLE UNNECESSARY FEATURES (saves CPU)
                        "--disable-extensions",
                        "--disable-plugins",
                        "--disable-images",  # Don't load images - saves bandwidth & CPU
                        "--disable-javascript",  # We only need HTML structure
                        "--disable-web-security",  # Skip security checks for speed
                        "--disable-features=VizDisplayCompositor",
                        
                        # NETWORK OPTIMIZATIONS (faster loading)
                        "--disable-background-networking",
                        "--disable-sync",
                        "--disable-default-apps",
                        "--disable-component-update",
                        
                        # PROCESS OPTIMIZATIONS
                        "--disable-hang-monitor",  # Don't monitor for hangs
                        "--disable-prompt-on-repost",
                        "--disable-client-side-phishing-detection",
                        "--disable-component-extensions-with-background-pages",
                        
                        # PERFORMANCE FLAGS
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-popup-blocking",
                        "--disable-notifications",
                    ]
                )
                await self.crawler.start()
                logger.info(f"✅ Crawl4AI browser started with SPEED-OPTIMIZED efficiency (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.warning(f"❌ Failed to start Crawl4AI browser (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize Crawl4AI after {max_retries} attempts")
                await asyncio.sleep(2)  # Quick retry delay
        
        # Load ticker list efficiently
        await self.load_tickers()
        
        # Compile ticker patterns for faster matching
        self.compile_ticker_patterns()
        
        logger.info(f"🕥 Web scraper initialized with SPEED-OPTIMIZED efficiency - {len(self.ticker_list)} tickers")

    async def load_tickers(self):
        """Load ticker list from ClickHouse database"""
        try:
            # Get tickers from database - NO CSV fallback allowed
            db_tickers = self.clickhouse_manager.get_active_tickers()
            
            if db_tickers:
                self.ticker_list = db_tickers
                logger.info(f"Loaded {len(self.ticker_list)} tickers from database")
            else:
                # NO CSV fallback - system should fail if database is empty
                logger.error("❌ CRITICAL ERROR: No tickers found in float_list table!")
                logger.error("❌ The system requires the Finviz scraper to populate the float_list table first")
                logger.error("❌ Run the system without --skip-list flag to update ticker list")
                raise Exception("No tickers in database - float_list table is empty")
                    
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            raise  # Re-raise the exception to stop the system

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
        """Generate hash for duplicate detection based on URL only"""
        return hashlib.md5(url.encode()).hexdigest()

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
        """Scrape all newswire sources in parallel and find EXACT ticker matches"""
        all_articles = []
        
        # Create scraping tasks for all sources in parallel
        scraping_tasks = []
        for source_name, source_url in self.sources.items():
            task = self.scrape_single_source(source_name, source_url)
            scraping_tasks.append(task)
        
        # Execute all scraping tasks in parallel
        try:
            results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Process results from all sources
            for i, result in enumerate(results):
                source_name = list(self.sources.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error scraping {source_name}: {result}")
                    continue
                elif result:
                    all_articles.extend(result)
                    logger.info(f"🎯 Found {len(result)} articles with exact ticker matches from {source_name}")
        
        except Exception as e:
            logger.error(f"Error in parallel scraping: {e}")
        
        return all_articles

    async def scrape_single_source(self, source_name: str, source_url: str) -> List[Dict[str, Any]]:
        """Scrape a single newswire source with SPEED-OPTIMIZED efficiency"""
        articles = []
        
        try:
            logger.info(f"🔍 Scraping {source_name} with speed optimization...")
            
            result: CrawlResult = await self.crawler.arun(
                url=source_url,
                wait_for="css:.news-item, .search-result, .bw-release-story, .newsreleaseheadline",
                delay_before_return_html=1.0,  # Fast 1s delay for speed
                timeout=15  # Quick 15s timeout for speed
            )
            
            if not result.success or not result.html:
                logger.error(f"❌ Failed to scrape {source_name}: {result.error_message}")
                return articles
            
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Find all article links and titles
            article_links = soup.find_all('a', href=True)
            
            logger.info(f"📰 Found {len(article_links)} potential articles from {source_name}")
            
            # SPEED-OPTIMIZED: Process articles efficiently without unnecessary delays
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
                        elif source_name == 'AccessNewswire':
                            url = "https://www.accessnewswire.com" + url
                    
                    # Skip non-news URLs
                    if not any(x in url for x in ['news-release', 'story', 'releases', 'news/home', 'newsroom', 'press-release']):
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
                        
                        # FRESHNESS CHECK: Compare time portions only (timezone-agnostic) - CONDITIONAL
                        logger.debug(f"🔍 FRESHNESS CHECK: enable_old={self.enable_old}, checking freshness={not self.enable_old}")
                        
                        if not self.enable_old:  # Only check freshness if enable_old is False
                            logger.debug(f"⏰ FRESHNESS FILTERING ENABLED - Checking if article is fresh enough")
                            current_time = datetime.now()
                            
                            # Extract minute:second from current time (detection time)
                            current_min_sec = current_time.strftime("%M:%S")
                            
                            # Extract minute:second from published time
                            published_min_sec = parsed_timestamp.strftime("%M:%S")
                            
                            # Calculate time difference in minutes and seconds only
                            current_total_seconds = current_time.minute * 60 + current_time.second
                            published_total_seconds = parsed_timestamp.minute * 60 + parsed_timestamp.second
                            
                            # Handle minute rollover (e.g., published at 59:30, detected at 01:15)
                            time_diff_seconds = current_total_seconds - published_total_seconds
                            if time_diff_seconds < 0:
                                time_diff_seconds += 3600  # Add 60 minutes worth of seconds
                            
                            if time_diff_seconds > 120:  # More than 2 minutes old
                                logger.info(f"⏰ SKIPPING STALE NEWS: {found_tickers} - Published at :{published_min_sec}, detected at :{current_min_sec} ({time_diff_seconds}s diff > 120s): {title[:50]}...")
                                continue
                            
                            logger.info(f"✅ FRESH NEWS: {found_tickers} - Published at :{published_min_sec}, detected at :{current_min_sec} ({time_diff_seconds}s diff < 120s)")
                        else:
                            logger.info(f"🔓 PROCESSING OLD NEWS (freshness disabled): {found_tickers} - {title[:50]}...")
                        
                        # SANITY CHECK: Only apply current day filter to GlobeNewswire (24HOURS can include yesterday)
                        # Other sources (BusinessWire, PRNewswire) already filter to current day
                        if not self.enable_old and source_name == 'GlobeNewswire':  # Only check date if enable_old is False
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
                            articles.append(article)
                            logger.info(f"✅ NEW ARTICLE: {ticker} - {title[:50]}...")
                            
                            # 🚀 ZERO-LAG: Create immediate trigger file for instant price checking
                            try:
                                self.clickhouse_manager.create_immediate_trigger(ticker, parsed_timestamp)
                            except Exception as e:
                                logger.warning(f"⚠️ Could not create immediate trigger for {ticker}: {e}")
                    else:
                        logger.debug(f"❌ No tickers found in: {title[:50]}...")
                    
                except Exception as e:
                    logger.debug(f"Error processing article: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
        
        return articles

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
        """Monitor ALL newswire sources for exact ticker matches with SPEED-OPTIMIZED efficiency"""
        logger.info("Starting ALL newswire monitoring with SPEED-OPTIMIZED efficiency...")
        
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
                
                # SPEED-OPTIMIZED: Fast cycle times for rapid news detection
                min_wait = 5.0  # Fast 5s cycles for rapid news detection
                if total_time < min_wait:
                    wait_time = min_wait - total_time
                    logger.debug(f"⏱️ SPEED CYCLE: Completed in {total_time:.2f}s, waiting {wait_time:.2f}s for next cycle")
                    await asyncio.sleep(wait_time)
                else:
                    # Cycle took longer than minimum - brief pause to prevent overwhelming
                    buffer_time = 1.0  # Brief 1s buffer for efficiency
                    logger.info(f"⚠️ LONG CYCLE: {total_time:.2f}s (longer than {min_wait}s), adding {buffer_time}s buffer")
                    await asyncio.sleep(buffer_time)
                
            except Exception as e:
                logger.error(f"Error in newswire monitoring: {e}")
                await asyncio.sleep(30)  # Quick error recovery for speed

    async def buffer_flusher(self):
        """Periodically flush buffer to ClickHouse - OPTIMIZED for speed"""
        while True:
            try:
                await asyncio.sleep(0.25)  # Flush every 250ms for ULTRA-fast detection
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
                rss_rate = self.stats['rss_articles_processed'] / runtime if runtime > 0 else 0
                
                logger.info(f"SCRAPING STATS - Runtime: {runtime:.1f}s, "
                          f"Web Processed: {self.stats['articles_processed']}, "
                          f"RSS Processed: {self.stats['rss_articles_processed']}, "
                          f"Total Inserted: {self.stats['articles_inserted']}, "
                          f"Errors: {self.stats['errors']}, "
                          f"Web Rate: {rate:.2f}/sec, RSS Rate: {rss_rate:.2f}/sec")
                          
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")

    async def start_scraping(self):
        """Start scraping using bulk approach - MUCH faster"""
        logger.info("🚀 Starting BULK scraping (get all articles, filter locally - MUCH faster)...")
        
        # Initialize
        await self.initialize()
        
        # Create tasks
        tasks = []
        
        # Bulk scraping monitoring task (only 4 HTTP requests per cycle!)
        bulk_task = asyncio.create_task(self.monitor_all_newswires())
        tasks.append(bulk_task)
        
        # RSS comparison monitoring task
        rss_task = asyncio.create_task(self.rss_comparison_monitor())
        tasks.append(rss_task)
        
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
                        
                        logger.info(f"🔄 RSS COMPARISON for {site_config['name']}: Found {len(feed.entries)} RSS entries")
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

    async def rss_comparison_monitor(self):
        """RSS monitor for comparison with SPEED-OPTIMIZED efficiency"""
        logger.info("Starting RSS comparison monitoring with SPEED-OPTIMIZED efficiency...")
        
        while True:
            try:
                start_time = time.time()
                
                # Scrape all RSS feeds
                articles = await self.scrape_all_rss_feeds()
                
                if articles:
                    # Add to separate RSS batch queue or mark as RSS source
                    for article in articles:
                        article['source'] = f"{article['source']}_RSS"  # Mark as RSS source
                    
                    self.batch_queue.extend(articles)
                    self.stats['rss_articles_processed'] += len(articles)
                
                total_time = time.time() - start_time
                logger.info(f"🔄 RSS feeds scan completed: {len(articles)} articles in {total_time:.2f}s")
                
                # SPEED-OPTIMIZED: Fast RSS cycles for rapid news detection
                min_wait = 8.0  # Fast 8s cycles for RSS (slightly slower than web scraping)
                if total_time < min_wait:
                    wait_time = min_wait - total_time
                    logger.debug(f"⏱️ RSS SPEED CYCLE: Completed in {total_time:.2f}s, waiting {wait_time:.2f}s for next cycle")
                    await asyncio.sleep(wait_time)
                else:
                    # Cycle took longer than minimum - brief pause to prevent overwhelming
                    buffer_time = 2.0  # Brief 2s buffer for RSS efficiency
                    logger.info(f"⚠️ RSS LONG CYCLE: {total_time:.2f}s (longer than {min_wait}s), adding {buffer_time}s buffer")
                    await asyncio.sleep(buffer_time)
                
            except Exception as e:
                logger.error(f"Error in RSS monitoring: {e}")
                await asyncio.sleep(30)  # Quick error recovery for speed

    async def scrape_all_rss_feeds(self) -> List[Dict[str, Any]]:
        """Scrape all RSS feeds in parallel using EXACT same ticker extraction logic"""
        all_articles = []
        
        # Create RSS scraping tasks for all sources in parallel
        rss_tasks = []
        for source_name, rss_url in self.rss_feeds.items():
            task = self.scrape_single_rss_source(source_name, rss_url)
            rss_tasks.append(task)
        
        # Execute all RSS scraping tasks in parallel
        try:
            results = await asyncio.gather(*rss_tasks, return_exceptions=True)
            
            # Process results from all RSS sources
            for i, result in enumerate(results):
                source_name = list(self.rss_feeds.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error scraping RSS {source_name}: {result}")
                    continue
                elif result:
                    all_articles.extend(result)
                    logger.info(f"🎯 Found {len(result)} RSS articles with exact ticker matches from {source_name}")
        
        except Exception as e:
            logger.error(f"Error in parallel RSS scraping: {e}")
        
        return all_articles

    async def scrape_single_rss_source(self, source_name: str, rss_url: str) -> List[Dict[str, Any]]:
        """Scrape a single RSS source"""
        articles = []
        
        try:
            logger.info(f"🔍 Scraping RSS {source_name}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(rss_url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"❌ Failed to fetch RSS {source_name}: HTTP {response.status}")
                        return articles
                    
                    rss_content = await response.text()
                    feed = feedparser.parse(rss_content)
                    
                    if not feed.entries:
                        logger.warning(f"❌ No entries in RSS feed for {source_name}")
                        return articles
                    
                    logger.info(f"📰 Found {len(feed.entries)} RSS entries from {source_name}")
                    
                    for entry in feed.entries:
                        try:
                            title = entry.get('title', '').strip()
                            url = entry.get('link', '').strip()
                            description = entry.get('description', '').strip()
                            published = entry.get('published', '')
                            
                            if not title or not url or len(title) < 20:
                                continue
                            
                            # EXACT same ticker extraction logic as web scraper
                            text_to_search = f"{title} {description}"
                            found_tickers = self.extract_tickers_from_text(text_to_search)
                            
                            if found_tickers:
                                logger.info(f"✅ RSS TICKER MATCH: {found_tickers} in {source_name} title: {title}")
                                
                                # Parse timestamp using EXACT same logic as web scraper
                                parsed_timestamp = self.parse_relative_time(published)
                                
                                # FRESHNESS CHECK: Compare time portions only (timezone-agnostic) - CONDITIONAL
                                logger.debug(f"🔍 FRESHNESS CHECK: enable_old={self.enable_old}, checking freshness={not self.enable_old}")
                                
                                if not self.enable_old:  # Only check freshness if enable_old is False
                                    logger.debug(f"⏰ FRESHNESS FILTERING ENABLED - Checking if article is fresh enough")
                                    current_time = datetime.now()
                                    
                                    # Extract minute:second from current time (detection time)
                                    current_min_sec = current_time.strftime("%M:%S")
                                    
                                    # Extract minute:second from published time
                                    published_min_sec = parsed_timestamp.strftime("%M:%S")
                                    
                                    # Calculate time difference in minutes and seconds only
                                    current_total_seconds = current_time.minute * 60 + current_time.second
                                    published_total_seconds = parsed_timestamp.minute * 60 + parsed_timestamp.second
                                    
                                    # Handle minute rollover (e.g., published at 59:30, detected at 01:15)
                                    time_diff_seconds = current_total_seconds - published_total_seconds
                                    if time_diff_seconds < 0:
                                        time_diff_seconds += 3600  # Add 60 minutes worth of seconds
                                    
                                    if time_diff_seconds > 120:  # More than 2 minutes old
                                        logger.info(f"⏰ SKIPPING STALE RSS NEWS: {found_tickers} - Published at :{published_min_sec}, detected at :{current_min_sec} ({time_diff_seconds}s diff > 120s): {title[:50]}...")
                                        continue
                                    
                                    logger.info(f"✅ FRESH RSS NEWS: {found_tickers} - Published at :{published_min_sec}, detected at :{current_min_sec} ({time_diff_seconds}s diff < 120s)")
                                else:
                                    logger.info(f"🔓 PROCESSING OLD RSS NEWS (freshness disabled): {found_tickers} - {title[:50]}...")
                                
                                # EXACT same current day filtering logic as web scraper
                                if not self.enable_old and source_name == 'GlobeNewswire':  # Only check date if enable_old is False
                                    current_date = datetime.now().date()
                                    article_date = parsed_timestamp.date()
                                    
                                    if article_date != current_date:
                                        logger.debug(f"⏰ SKIPPING old RSS GlobeNewswire article from {article_date}: {title[:50]}...")
                                        continue
                                
                                # EXACT same content hash generation as web scraper
                                content_hash = self.generate_content_hash(title, url)
                                
                                # Create article with EXACT same structure as web scraper
                                for ticker in found_tickers:
                                    article = {
                                        'timestamp': datetime.now(),
                                        'source': f'{source_name}_RSS',  # Mark as RSS source
                                        'ticker': ticker,
                                        'headline': title,
                                        'published_utc': published,  # Store raw RSS timestamp
                                        'article_url': url,
                                        'summary': description if description else title,
                                        'full_content': f"{title} {description}",
                                        'detected_at': datetime.now(),
                                        'processing_latency_ms': 0,
                                        'market_relevant': 1,
                                        'source_check_time': datetime.now(),
                                        'content_hash': content_hash,
                                        'news_type': 'other',
                                        'urgency_score': 5
                                    }
                                    
                                    articles.append(article)
                                    logger.info(f"✅ NEW RSS ARTICLE: {ticker} - {title[:50]}...")
                                    
                                    # 🚀 ZERO-LAG: Create immediate trigger file for instant price checking
                                    try:
                                        self.clickhouse_manager.create_immediate_trigger(ticker, parsed_timestamp)
                                    except Exception as e:
                                        logger.warning(f"⚠️ Could not create immediate trigger for {ticker}: {e}")
                            else:
                                logger.debug(f"❌ No tickers found in RSS: {title[:50]}...")
                                
                        except Exception as e:
                            logger.debug(f"Error processing RSS entry: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"Error scraping RSS {source_name}: {e}")
        
        return articles

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