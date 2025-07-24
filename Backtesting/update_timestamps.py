#!/usr/bin/env python3
"""
Update Article Timestamps for Backtesting
Parses actual publication timestamps from article URLs and updates the database
Uses Crawl4AI like the existing web_scraper.py for reliable content extraction
"""

import asyncio
import logging
import re
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import pytz
from dateutil import parser as date_parser
from crawl4ai import AsyncWebCrawler, CrawlResult

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimestampUpdater:
    """
    Updates article timestamps by parsing actual publication dates from article URLs
    Uses Crawl4AI for reliable content extraction like the existing web_scraper.py
    """
    
    def __init__(self):
        self.ch_manager = None
        self.crawler = None
        
        # Timezone objects
        self.utc_tz = pytz.UTC
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Stats tracking
        self.stats = {
            'articles_processed': 0,
            'timestamps_updated': 0,
            'timestamps_failed': 0,
            'cache_hits': 0,
            'start_time': datetime.now()
        }
        
        # Cache for parsed timestamps
        self.timestamp_cache = {}

    async def initialize(self):
        """Initialize the timestamp updater with Crawl4AI"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Initialize Crawl4AI AsyncWebCrawler (same config as web_scraper.py)
            logger.info("🚀 Initializing Crawl4AI for timestamp parsing...")
            
            self.crawler = AsyncWebCrawler(
                verbose=False,
                headless=True,
                browser_type="chromium",
                max_idle_time=30000,
                keep_alive=True,
                max_memory_usage=512,
                max_concurrent_sessions=2,
                delay_between_requests=0.5,
                extra_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    "--memory-pressure-off",
                    "--max_old_space_size=256",
                    "--aggressive-cache-discard",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",
                    # JavaScript is now ENABLED for timestamp extraction
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-background-networking",
                    "--disable-sync",
                    "--disable-default-apps",
                    "--disable-component-update",
                    "--disable-hang-monitor",
                    "--disable-prompt-on-repost",
                    "--disable-client-side-phishing-detection",
                    "--disable-component-extensions-with-background-pages",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-popup-blocking",
                    "--disable-notifications",
                ]
            )
            await self.crawler.start()
            
            logger.info("✅ Timestamp Updater with Crawl4AI initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing timestamp updater: {e}")
            return False

    async def get_articles_to_update(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Get articles that need timestamp updates"""
        try:
            # Get articles with default timestamps (likely current date) or very recent dates
            query = """
            SELECT ticker, headline, article_url, published_utc, scraped_at, source, newswire_type, article_content, content_hash
            FROM News.historical_news
            WHERE published_utc >= now() - INTERVAL 2 DAY
            OR published_utc > '2025-01-01'  -- Articles with future dates are definitely wrong
            ORDER BY published_utc DESC
            LIMIT %s
            """
            
            result = self.ch_manager.client.query(query, [batch_size])
            
            articles = []
            for row in result.result_rows:
                ticker, headline, article_url, published_utc, scraped_at, source, newswire_type, article_content, content_hash = row
                articles.append({
                    'ticker': ticker,
                    'headline': headline,
                    'article_url': article_url,
                    'published_utc': published_utc,
                    'scraped_at': scraped_at,
                    'source': source,
                    'newswire_type': newswire_type,
                    'article_content': article_content,
                    'content_hash': content_hash
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles to update: {e}")
            return []

    async def parse_article_timestamp_with_crawl4ai(self, article_url: str) -> Optional[datetime]:
        """Parse the actual publication timestamp from an article URL using Crawl4AI"""
        try:
            # Check cache first
            if article_url in self.timestamp_cache:
                self.stats['cache_hits'] += 1
                return self.timestamp_cache[article_url]
            
            logger.debug(f"Parsing timestamp with Crawl4AI from: {article_url}")
            
            # Use Crawl4AI to get the page content (same as web_scraper.py)
            result: CrawlResult = await self.crawler.arun(
                url=article_url,
                wait_for="css:body, .bw-release-main, .release-body",
                delay_before_return_html=3.0,  # Wait longer for JS to load
                timeout=30  # Longer timeout
            )
            
            if not result.success or not result.html:
                logger.warning(f"❌ Crawl4AI failed for {article_url}: {result.error_message}")
                return None
            
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Parse timestamp based on the news source
            timestamp = None
            
            if 'globenewswire.com' in article_url:
                timestamp = await self.parse_globenewswire_timestamp(soup, article_url)
            elif 'prnewswire.com' in article_url:
                timestamp = await self.parse_prnewswire_timestamp(soup, article_url)
            elif 'businesswire.com' in article_url:
                timestamp = await self.parse_businesswire_timestamp(soup, article_url)
            elif 'accesswire.com' in article_url:
                timestamp = await self.parse_accesswire_timestamp(soup, article_url)
            elif 'finviz.com' in article_url:
                timestamp = await self.parse_finviz_timestamp(soup, article_url)
            else:
                # Generic timestamp parsing
                timestamp = await self.parse_generic_timestamp(soup, article_url)
            
            # Cache the result
            self.timestamp_cache[article_url] = timestamp
            
            if timestamp:
                logger.debug(f"Found timestamp: {timestamp} for {article_url}")
            else:
                logger.debug(f"No timestamp found for {article_url}")
            
            return timestamp
                
        except Exception as e:
            logger.error(f"Error parsing timestamp with Crawl4AI from {article_url}: {e}")
            return None

    async def parse_globenewswire_timestamp(self, soup: BeautifulSoup, url: str) -> Optional[datetime]:
        """Parse timestamp from GlobeNewswire articles"""
        try:
            # First, look for specific GlobeNewswire timestamp patterns in text
            full_text = soup.get_text()
            
            # GlobeNewswire patterns - they often show timestamps like:
            # "March 17, 2025 at 8:30 AM ET"
            # "Mar 17, 2025 08:30 ET"
            globenewswire_patterns = [
                # "March 17, 2025 at 8:30 AM ET"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+ET',
                # "March 17, 2025 at 8:30 AM EST"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+EST',
                # "Mar 17, 2025 08:30 ET" (24-hour)
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+ET',
                # "March 17, 2025 8:30 AM Eastern Time"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Time',
                # ISO-like format: "2025-03-17T08:30:00"
                r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})',
                # Simple format: "March 17, 2025 8:30 AM"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)',
            ]
            
            for pattern in globenewswire_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        
                        if pattern.startswith(r'(\d{4})'):  # ISO format
                            year = int(groups[0])
                            month = int(groups[1])
                            day = int(groups[2])
                            hour = int(groups[3])
                            minute = int(groups[4])
                        else:  # Month name format
                            month_name = groups[0]
                            day = int(groups[1])
                            year = int(groups[2])
                            hour = int(groups[3])
                            minute = int(groups[4])
                            
                            # Handle AM/PM if present
                            if len(groups) >= 6 and groups[5]:
                                am_pm = groups[5].upper()
                                if am_pm == 'PM' and hour != 12:
                                    hour += 12
                                elif am_pm == 'AM' and hour == 12:
                                    hour = 0
                            
                            # Parse month name to number
                            month_map = {
                                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                                'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                                'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                            }
                            
                            month = month_map.get(month_name.lower())
                            if not month:
                                continue
                        
                        # Create datetime in EST
                        est_dt = self.est_tz.localize(datetime(year, month, day, hour, minute))
                        utc_dt = est_dt.astimezone(self.utc_tz)
                        
                        logger.info(f"✅ GlobeNewswire full timestamp parsed: {match.group()} → {utc_dt}")
                        return utc_dt
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing GlobeNewswire timestamp match '{match.group()}': {e}")
                        continue
            
            # Fallback to CSS selectors
            selectors = [
                '.article-published-date',
                '.published-date',
                '.timestamp',
                '[data-module="ArticleDateTime"]',
                '.article-meta time',
                'time[datetime]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    # Try datetime attribute first
                    datetime_attr = element.get('datetime')
                    if datetime_attr:
                        timestamp = self.parse_datetime_string(datetime_attr)
                        if timestamp:
                            return timestamp
                    
                    # Try text content
                    text = element.get_text().strip()
                    if text:
                        timestamp = self.parse_datetime_string(text)
                        if timestamp:
                            return timestamp
            
            # Look for date patterns in the URL itself
            url_timestamp = self.extract_timestamp_from_url(url)
            if url_timestamp:
                logger.info(f"⚠️ GlobeNewswire using URL fallback (no time info): {url_timestamp}")
                return url_timestamp
            
            # Look for date patterns in text
            return self.find_timestamp_in_text(soup.get_text())
            
        except Exception as e:
            logger.debug(f"Error parsing GlobeNewswire timestamp: {e}")
            return None

    async def parse_prnewswire_timestamp(self, soup: BeautifulSoup, url: str) -> Optional[datetime]:
        """Parse timestamp from PRNewswire articles"""
        try:
            # First, look for specific PRNewswire timestamp patterns in text
            full_text = soup.get_text()
            
            # PRNewswire patterns - they often show timestamps like:
            # "April 17, 2024, 09:00 AM EDT"
            # "Apr 17, 2024 9:00 AM Eastern Daylight Time"
            prnewswire_patterns = [
                # "April 17, 2024, 09:00 AM EDT"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4}),\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+EDT',
                # "April 17, 2024, 09:00 AM EST"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4}),\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+EST',
                # "April 17, 2024 9:00 AM Eastern Daylight Time"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Daylight\s+Time',
                # "April 17, 2024 9:00 AM Eastern Standard Time"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Standard\s+Time',
                # "Apr 17, 2024 09:00 EDT" (24-hour)
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+EDT',
                # "Apr 17, 2024 09:00 EST" (24-hour)
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+EST',
                # Simple format: "April 17, 2024 9:00 AM"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)',
            ]
            
            for pattern in prnewswire_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        month_name = groups[0]
                        day = int(groups[1])
                        year = int(groups[2])
                        hour = int(groups[3])
                        minute = int(groups[4])
                        
                        # Handle AM/PM if present
                        if len(groups) >= 6 and groups[5]:
                            am_pm = groups[5].upper()
                            if am_pm == 'PM' and hour != 12:
                                hour += 12
                            elif am_pm == 'AM' and hour == 12:
                                hour = 0
                        
                        # Parse month name to number
                        month_map = {
                            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                            'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                        }
                        
                        month = month_map.get(month_name.lower())
                        if not month:
                            continue
                        
                        # Create datetime in EST
                        est_dt = self.est_tz.localize(datetime(year, month, day, hour, minute))
                        utc_dt = est_dt.astimezone(self.utc_tz)
                        
                        logger.info(f"✅ PRNewswire full timestamp parsed: {match.group()} → {utc_dt}")
                        return utc_dt
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing PRNewswire timestamp match '{match.group()}': {e}")
                        continue
            
            # Fallback to CSS selectors
            selectors = [
                '.xn-chron',
                '.release-date',
                '.timestamp',
                'time[datetime]',
                '.date-time',
                '[data-datetime]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    # Try datetime attribute
                    datetime_attr = element.get('datetime') or element.get('data-datetime')
                    if datetime_attr:
                        timestamp = self.parse_datetime_string(datetime_attr)
                        if timestamp:
                            return timestamp
                    
                    # Try text content
                    text = element.get_text().strip()
                    if text:
                        timestamp = self.parse_datetime_string(text)
                        if timestamp:
                            return timestamp
            
            # URL pattern extraction
            url_timestamp = self.extract_timestamp_from_url(url)
            if url_timestamp:
                logger.info(f"⚠️ PRNewswire using URL fallback (no time info): {url_timestamp}")
                return url_timestamp
            
            return self.find_timestamp_in_text(soup.get_text())
            
        except Exception as e:
            logger.debug(f"Error parsing PRNewswire timestamp: {e}")
            return None

    async def parse_businesswire_timestamp(self, soup: BeautifulSoup, url: str) -> Optional[datetime]:
        """Parse timestamp from BusinessWire articles"""
        try:
            # First, look for the specific timestamp format in BusinessWire articles
            # From debug: "Jul 17, 2025 9:00 AM Eastern Daylight Time"
            
            # Look for text containing time patterns
            full_text = soup.get_text()
            
            # Enhanced patterns for BusinessWire timestamps (based on actual format found)
            businesswire_patterns = [
                # "Jul 17, 2025 9:00 AM Eastern Daylight Time" (actual format found)
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Daylight\s+Time',
                # "Jul 17, 2025 9:00 AM Eastern Standard Time"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Standard\s+Time',
                # "Feb 12, 2025 8:00 AM EST"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+EST',
                # "Feb 12, 2025 8:00 AM ET"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+ET',
                # "Feb 12, 2025 8:00 AM EDT"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+EDT',
                # 24-hour format versions
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+Eastern\s+Daylight\s+Time',
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+Eastern\s+Standard\s+Time',
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+EST',
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+ET',
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+EDT',
            ]
            
            for pattern in businesswire_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        month_name = groups[0]
                        day = int(groups[1])
                        year = int(groups[2])
                        hour = int(groups[3])
                        minute = int(groups[4])
                        
                        # Handle AM/PM if present
                        if len(groups) >= 6 and groups[5]:
                            am_pm = groups[5].upper()
                            if am_pm == 'PM' and hour != 12:
                                hour += 12
                            elif am_pm == 'AM' and hour == 12:
                                hour = 0
                        
                        # Parse month name to number
                        month_map = {
                            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                            'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                        }
                        
                        month = month_map.get(month_name.lower())
                        if not month:
                            continue
                        
                        # Create datetime in EST
                        est_dt = self.est_tz.localize(datetime(year, month, day, hour, minute))
                        utc_dt = est_dt.astimezone(self.utc_tz)
                        
                        logger.info(f"✅ BusinessWire full timestamp parsed: {match.group()} → {utc_dt}")
                        return utc_dt
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing BusinessWire timestamp match '{match.group()}': {e}")
                        continue
            
            # Fallback to CSS selectors
            selectors = [
                '.bw-release-timestamp',
                '.release-meta time',
                'time[datetime]',
                '.timestamp',
                '.date-published',
                '.release-date'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    datetime_attr = element.get('datetime')
                    if datetime_attr:
                        timestamp = self.parse_datetime_string(datetime_attr)
                        if timestamp:
                            return timestamp
                    
                    text = element.get_text().strip()
                    if text:
                        timestamp = self.parse_datetime_string(text)
                        if timestamp:
                            return timestamp
            
            # URL pattern extraction as final fallback
            url_timestamp = self.extract_timestamp_from_url(url)
            if url_timestamp:
                logger.info(f"⚠️ BusinessWire using URL fallback (no time info): {url_timestamp}")
                return url_timestamp
            
            # Text search as last resort
            return self.find_timestamp_in_text(soup.get_text())
            
        except Exception as e:
            logger.debug(f"Error parsing BusinessWire timestamp: {e}")
            return None

    async def parse_accesswire_timestamp(self, soup: BeautifulSoup, url: str) -> Optional[datetime]:
        """Parse timestamp from Accesswire articles"""
        try:
            # First, look for specific Accesswire timestamp patterns in text
            full_text = soup.get_text()
            
            # Accesswire patterns - they often show timestamps like:
            # "Thursday, 10 April 2025 08:30 AM"
            # "April 10, 2025 8:30 AM EST"
            accesswire_patterns = [
                # "Thursday, 10 April 2025 08:30 AM"
                r'[A-Za-z]+,\s+(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)',
                # "April 10, 2025 8:30 AM EST"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+EST',
                # "April 10, 2025 8:30 AM ET"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+ET',
                # "10 April 2025 08:30"
                r'(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\s+(\d{1,2}):(\d{2})',
                # "April 10, 2025 8:30 AM"
                r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)',
                # ISO format: "2025-04-10T08:30:00"
                r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})',
            ]
            
            for pattern in accesswire_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        
                        if pattern.startswith(r'(\d{4})'):  # ISO format
                            year = int(groups[0])
                            month = int(groups[1])
                            day = int(groups[2])
                            hour = int(groups[3])
                            minute = int(groups[4])
                        elif pattern.startswith(r'[A-Za-z]+,\s+(\d{1,2})\s+'):  # "Thursday, 10 April 2025" format
                            day = int(groups[0])
                            month_name = groups[1]
                            year = int(groups[2])
                            hour = int(groups[3])
                            minute = int(groups[4])
                            
                            # Handle AM/PM if present
                            if len(groups) >= 6 and groups[5]:
                                am_pm = groups[5].upper()
                                if am_pm == 'PM' and hour != 12:
                                    hour += 12
                                elif am_pm == 'AM' and hour == 12:
                                    hour = 0
                            
                            # Parse month name to number
                            month_map = {
                                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                                'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                                'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                            }
                            
                            month = month_map.get(month_name.lower())
                            if not month:
                                continue
                        elif pattern.startswith(r'(\d{1,2})\s+'):  # "10 April 2025" format
                            day = int(groups[0])
                            month_name = groups[1]
                            year = int(groups[2])
                            hour = int(groups[3])
                            minute = int(groups[4])
                            
                            # Parse month name to number
                            month_map = {
                                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                                'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                                'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                            }
                            
                            month = month_map.get(month_name.lower())
                            if not month:
                                continue
                        else:  # Standard "April 10, 2025" format
                            month_name = groups[0]
                            day = int(groups[1])
                            year = int(groups[2])
                            hour = int(groups[3])
                            minute = int(groups[4])
                            
                            # Handle AM/PM if present
                            if len(groups) >= 6 and groups[5]:
                                am_pm = groups[5].upper()
                                if am_pm == 'PM' and hour != 12:
                                    hour += 12
                                elif am_pm == 'AM' and hour == 12:
                                    hour = 0
                            
                            # Parse month name to number
                            month_map = {
                                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                                'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                                'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                            }
                            
                            month = month_map.get(month_name.lower())
                            if not month:
                                continue
                        
                        # Create datetime in EST
                        est_dt = self.est_tz.localize(datetime(year, month, day, hour, minute))
                        utc_dt = est_dt.astimezone(self.utc_tz)
                        
                        logger.info(f"✅ Accesswire full timestamp parsed: {match.group()} → {utc_dt}")
                        return utc_dt
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing Accesswire timestamp match '{match.group()}': {e}")
                        continue
            
            # Fallback to CSS selectors
            selectors = [
                '.article-date',
                '.published-date',
                'time[datetime]',
                '.timestamp',
                '.release-date'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    datetime_attr = element.get('datetime')
                    if datetime_attr:
                        timestamp = self.parse_datetime_string(datetime_attr)
                        if timestamp:
                            return timestamp
                    
                    text = element.get_text().strip()
                    if text:
                        timestamp = self.parse_datetime_string(text)
                        if timestamp:
                            return timestamp
            
            # URL pattern extraction
            url_timestamp = self.extract_timestamp_from_url(url)
            if url_timestamp:
                logger.info(f"⚠️ Accesswire using URL fallback (no time info): {url_timestamp}")
                return url_timestamp
            
            return self.find_timestamp_in_text(soup.get_text())
            
        except Exception as e:
            logger.debug(f"Error parsing Accesswire timestamp: {e}")
            return None

    async def parse_finviz_timestamp(self, soup: BeautifulSoup, url: str) -> Optional[datetime]:
        """Parse timestamp from Finviz news articles"""
        try:
            # Finviz redirects to original articles, so this might not be needed
            # But we'll handle it just in case
            return self.find_timestamp_in_text(soup.get_text())
            
        except Exception as e:
            logger.debug(f"Error parsing Finviz timestamp: {e}")
            return None

    async def parse_generic_timestamp(self, soup: BeautifulSoup, url: str) -> Optional[datetime]:
        """Generic timestamp parsing for unknown sources"""
        try:
            # Try common HTML5 time elements
            time_elements = soup.find_all('time')
            for time_elem in time_elements:
                datetime_attr = time_elem.get('datetime')
                if datetime_attr:
                    timestamp = self.parse_datetime_string(datetime_attr)
                    if timestamp:
                        return timestamp
            
            # Try URL extraction
            url_timestamp = self.extract_timestamp_from_url(url)
            if url_timestamp:
                return url_timestamp
            
            # Try text parsing
            return self.find_timestamp_in_text(soup.get_text())
            
        except Exception as e:
            logger.debug(f"Error in generic timestamp parsing: {e}")
            return None

    def extract_timestamp_from_url(self, url: str) -> Optional[datetime]:
        """Extract timestamp from URL patterns"""
        try:
            # Common URL timestamp patterns
            patterns = [
                r'/(\d{4})/(\d{2})/(\d{2})/',  # /2024/07/23/
                r'/(\d{4})-(\d{2})-(\d{2})/',  # /2024-07-23/
                r'(\d{4})(\d{2})(\d{2})',      # 20240723
                r'/news-releases/.*?-(\d{6,8})-.*?\.html',  # PRNewswire pattern
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    if len(match.groups()) == 3:
                        year, month, day = match.groups()
                        try:
                            return datetime(int(year), int(month), int(day), tzinfo=self.utc_tz)
                        except ValueError:
                            continue
                    elif len(match.groups()) == 1:
                        # Handle compressed date formats
                        date_str = match.group(1)
                        if len(date_str) == 8:  # YYYYMMDD
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            return datetime(year, month, day, tzinfo=self.utc_tz)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting timestamp from URL: {e}")
            return None

    def find_timestamp_in_text(self, text: str) -> Optional[datetime]:
        """Find timestamp patterns in article text"""
        try:
            # Common timestamp patterns in text
            patterns = [
                r'(\w+\s+\d{1,2},\s+\d{4})',  # "July 23, 2024"
                r'(\d{1,2}/\d{1,2}/\d{4})',   # "7/23/2024"
                r'(\d{4}-\d{2}-\d{2})',       # "2024-07-23"
                r'(\w+\s+\d{1,2}\s+\d{4})',  # "July 23 2024"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    timestamp = self.parse_datetime_string(match)
                    if timestamp and timestamp.year >= 2020:  # Reasonable date range
                        return timestamp
            
            return None
            
        except Exception as e:
            logger.debug(f"Error finding timestamp in text: {e}")
            return None

    def parse_datetime_string(self, date_string: str) -> Optional[datetime]:
        """Parse various datetime string formats"""
        try:
            # Clean the string
            date_string = date_string.strip()
            
            # Try dateutil parser (handles most formats)
            parsed_date = date_parser.parse(date_string, fuzzy=True)
            
            # Ensure timezone awareness
            if parsed_date.tzinfo is None:
                # Assume EST for news articles
                parsed_date = self.est_tz.localize(parsed_date)
            
            # Convert to UTC
            return parsed_date.astimezone(self.utc_tz)
            
        except Exception as e:
            logger.debug(f"Error parsing datetime string '{date_string}': {e}")
            return None

    async def update_article_timestamp(self, article: Dict[str, Any]) -> bool:
        """Update a single article's timestamp by recreating the record"""
        try:
            self.stats['articles_processed'] += 1
            
            article_url = article['article_url']
            content_hash = article['content_hash']
            
            # Parse the actual timestamp using Crawl4AI
            actual_timestamp = await self.parse_article_timestamp_with_crawl4ai(article_url)
            
            if not actual_timestamp:
                self.stats['timestamps_failed'] += 1
                logger.debug(f"❌ Could not parse timestamp for {article['ticker']}: {article_url}")
                return False
            
            # Since we can't UPDATE a key column, we need to:
            # 1. Delete the old record
            # 2. Insert a new record with the correct timestamp
            
            try:
                # Delete the old record
                delete_query = """
                DELETE FROM News.historical_news
                WHERE content_hash = %s
                """
                
                self.ch_manager.client.command(delete_query, [content_hash])
                
                # Insert the new record with correct timestamp
                insert_data = [(
                    article['ticker'],
                    article['headline'],
                    article['article_url'],
                    actual_timestamp,  # Use the parsed timestamp
                    article['scraped_at'],
                    article['source'],
                    article['newswire_type'],
                    article['article_content'],
                    article['content_hash']
                )]
                
                self.ch_manager.client.insert(
                    'News.historical_news',
                    insert_data,
                    column_names=['ticker', 'headline', 'article_url', 'published_utc', 'scraped_at', 'source', 'newswire_type', 'article_content', 'content_hash']
                )
                
                self.stats['timestamps_updated'] += 1
                logger.info(f"✅ Updated {article['ticker']}: {article['published_utc']} → {actual_timestamp}")
                
                return True
                
            except Exception as db_error:
                logger.error(f"Database error updating {article['ticker']}: {db_error}")
                # Try to restore the original record if something went wrong
                try:
                    restore_data = [(
                        article['ticker'],
                        article['headline'],
                        article['article_url'],
                        article['published_utc'],  # Original timestamp
                        article['scraped_at'],
                        article['source'],
                        article['newswire_type'],
                        article['article_content'],
                        article['content_hash']
                    )]
                    
                    self.ch_manager.client.insert(
                        'News.historical_news',
                        restore_data,
                        column_names=['ticker', 'headline', 'article_url', 'published_utc', 'scraped_at', 'source', 'newswire_type', 'article_content', 'content_hash']
                    )
                    logger.info(f"Restored original record for {article['ticker']}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore record for {article['ticker']}: {restore_error}")
                
                self.stats['timestamps_failed'] += 1
                return False
            
        except Exception as e:
            self.stats['timestamps_failed'] += 1
            logger.error(f"Error updating timestamp for {article.get('ticker', 'UNKNOWN')}: {e}")
            return False

    async def run_timestamp_updates(self, batch_size: int = 50):
        """Run the complete timestamp update process"""
        try:
            logger.info("🕒 Starting Article Timestamp Updates with Crawl4AI...")
            
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize timestamp updater")
                return False
            
            total_processed = 0
            batch_count = 0
            
            while True:
                batch_count += 1
                
                # Get next batch of articles to update
                logger.info(f"📊 BATCH {batch_count}: Getting articles to update...")
                articles = await self.get_articles_to_update(batch_size)
                
                if not articles:
                    logger.info("✅ No more articles to update")
                    break
                
                logger.info(f"🕒 BATCH {batch_count}: Updating timestamps for {len(articles)} articles...")
                
                # Process articles with controlled parallelism
                semaphore = asyncio.Semaphore(5)  # Conservative limit for Crawl4AI
                
                async def update_with_semaphore(article):
                    async with semaphore:
                        return await self.update_article_timestamp(article)
                
                # Execute updates
                update_tasks = [update_with_semaphore(article) for article in articles]
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Count successful updates
                successful_updates = sum(1 for result in results if result is True)
                
                total_processed += len(articles)
                
                # Progress logging
                logger.info(f"📈 BATCH {batch_count} COMPLETE: {successful_updates}/{len(articles)} timestamps updated")
                logger.info(f"🔄 TOTAL PROGRESS: {total_processed} articles processed")
                
                # Rate limiting between batches
                await asyncio.sleep(2)
            
            # Final stats
            elapsed = datetime.now() - self.stats['start_time']
            logger.info("🎉 TIMESTAMP UPDATES COMPLETE!")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Articles processed: {self.stats['articles_processed']}")
            logger.info(f"  • Timestamps updated: {self.stats['timestamps_updated']}")
            logger.info(f"  • Timestamps failed: {self.stats['timestamps_failed']}")
            logger.info(f"  • Cache hits: {self.stats['cache_hits']}")
            logger.info(f"  • Time elapsed: {elapsed}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in timestamp updates: {e}")
            return False
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        if self.crawler:
            await self.crawler.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("✅ Timestamp updater cleanup completed")

async def main():
    """Main function"""
    updater = TimestampUpdater()
    success = await updater.run_timestamp_updates()
    
    if success:
        print("\n✅ Timestamp updates completed successfully!")
    else:
        print("\n❌ Timestamp updates failed!")

if __name__ == "__main__":
    asyncio.run(main()) 