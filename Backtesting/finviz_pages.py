#!/usr/bin/env python3
"""
Finviz Historical News Scraper for Backtesting
Scrapes ticker lists and 6 months of newswire articles from Finviz
Only scrapes newswires articles published between 5am-9am EST
"""

import asyncio
import logging
import re
import os
import sys
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from bs4 import BeautifulSoup
import pytz
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler, CrawlResult

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed back to INFO for normal operation
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinvizHistoricalScraper:
    def __init__(self):
        self.ch_manager = None
        self.crawler = None
        
        # Finviz screener URLs for ticker discovery
        self.screener_urls = [
            "https://elite.finviz.com/screener.ashx?v=111&f=geo_usa|asia|latinamerica|argentina|china|denmark|france|greece|hungary|india|ireland|italy|jordan|philippines|russia|spain|switzerland|thailand|unitedarabemirates|europe|bric|australia|belgium|bermuda|canada|chile|chinahongkong|cyprus|brazil|benelux|colombia|luxembourg|malta|monaco|newzealand|panama|southafrica|uruguay|vietnam|unitedkingdom|turkey|taiwan|sweden|southkorea|singapore|portugal|peru|norway|netherlands|mexico|malaysia|kazakhstan|japan|indonesia|iceland|hongkong|germany|finland,sec_healthcare|technology|industrials|consumerdefensive|communicationservices|energy|consumercyclical|basicmaterials|utilities,sh_float_u100,sh_price_u3&ft=4",
            "https://elite.finviz.com/screener.ashx?v=111&f=geo_usa|asia|latinamerica|argentina|china|denmark|france|greece|hungary|india|ireland|italy|jordan|philippines|russia|spain|switzerland|thailand|unitedarabemirates|europe|bric|australia|belgium|bermuda|canada|chile|chinahongkong|cyprus|brazil|benelux|colombia|luxembourg|malta|monaco|newzealand|panama|southafrica|uruguay|vietnam|unitedkingdom|turkey|taiwan|sweden|southkorea|singapore|portugal|peru|norway|netherlands|mexico|malaysia|kazakhstan|japan|indonesia|iceland|hongkong|germany|finland,sec_healthcare|technology|industrials|consumerdefensive|communicationservices|energy|consumercyclical|basicmaterials|utilities,sh_float_u100,sh_price_3to10&ft=4"
        ]
        
        # Target newswire sources
        self.target_newswires = {
            'PRNewswire', 'BusinessWire', 'GlobeNewswire', 'Accesswire',
            'PR Newswire', 'Business Wire', 'Globe Newswire', 'AccessWire'
        }
        
        # EST timezone for filtering articles
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Stats tracking
        self.stats = {
            'tickers_found': 0,
            'tickers_processed': 0,
            'articles_found': 0,
            'articles_filtered': 0,
            'articles_stored': 0,
            'start_time': datetime.now()
        }

    async def initialize(self):
        """Initialize the scraper with Crawl4AI"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Initialize Crawl4AI AsyncWebCrawler with optimized settings for backtesting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.crawler = AsyncWebCrawler(
                        verbose=False,  # Reduced logging for cleaner output
                        headless=True,
                        browser_type="chromium",
                        
                        # Optimized settings for backtesting (slower pace than real-time)
                        max_idle_time=30000,  # 30s timeout
                        keep_alive=True,
                        
                        # Resource limits suitable for backtesting
                        max_memory_usage=512,  # 512MB
                        max_concurrent_sessions=2,  # 2 sessions for parallel processing
                        delay_between_requests=1.0,  # 1s delay for respectful scraping
                        
                        # Browser flags optimized for scraping
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
                            "--disable-images",  # Don't load images for faster scraping
                            "--disable-javascript",  # We only need HTML structure
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
                    logger.info(f"✅ Crawl4AI browser started successfully (attempt {attempt + 1})")
                    break
                except Exception as e:
                    logger.warning(f"❌ Failed to start Crawl4AI browser (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to initialize Crawl4AI after {max_retries} attempts")
                    await asyncio.sleep(2)
            
            logger.info("✅ Finviz Historical Scraper with Crawl4AI initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing scraper: {e}")
            return False

    def parse_table_value(self, value_str: str) -> float:
        """Parse table values (handle B, M, K suffixes)"""
        if not value_str or value_str == '-' or value_str == 'N/A':
            return 0.0
            
        value_str = value_str.replace('%', '')
        
        multiplier = 1
        if value_str.endswith('B'):
            multiplier = 1e9
            value_str = value_str[:-1]
        elif value_str.endswith('M'):
            multiplier = 1e6
            value_str = value_str[:-1]
        elif value_str.endswith('K'):
            multiplier = 1e3
            value_str = value_str[:-1]
        
        try:
            return float(value_str) * multiplier
        except:
            return 0.0

    async def get_ticker_list_from_screeners(self) -> List[Dict[str, Any]]:
        """Get complete ticker list from Finviz screener URLs using Crawl4AI"""
        all_tickers = []
        
        logger.info(f"📊 Scraping ticker lists from {len(self.screener_urls)} screener URLs with Crawl4AI...")
        
        for i, screener_url in enumerate(self.screener_urls):
            url_desc = f"price under $3" if i == 0 else f"price $3 to $10"
            logger.info(f"🔍 Scraping {url_desc} stocks from screener...")
            
            page_num = 1
            url_tickers = 0
            
            while True:
                # Construct URL for current page
                if page_num == 1:
                    page_url = screener_url
                else:
                    page_url = f"{screener_url}&r={(page_num-1)*20+1}"
                
                logger.info(f"📄 Scraping {url_desc} page {page_num}")
                
                try:
                    # Use Crawl4AI instead of aiohttp
                    result: CrawlResult = await self.crawler.arun(
                        url=page_url,
                        wait_for="css:table.screener_table, table[width='100%']",
                        delay_before_return_html=2.0,  # Wait for page to load
                        timeout=30  # 30s timeout for screener pages
                    )
                    
                    if not result.success or not result.html:
                        logger.warning(f"Failed to scrape page {page_num}: {result.error_message}")
                        break
                    
                    soup = BeautifulSoup(result.html, 'html.parser')
                    
                    # Find screener table
                    table = soup.find('table', {'class': 'screener_table'})
                    if not table:
                        table = soup.find('table', {'width': '100%'})
                        if not table:
                            logger.warning(f"No table found on page {page_num}, stopping")
                            break
                    
                    # Get table rows
                    rows = table.find_all('tr')
                    if len(rows) < 2:
                        logger.info(f"No data rows on page {page_num}, stopping")
                        break
                    
                    # Parse header row
                    header_row = rows[0]
                    headers = [th.get_text().strip().lower() for th in header_row.find_all(['td', 'th'])]
                    
                    # Create column mapping
                    col_map = {}
                    for idx, header in enumerate(headers):
                        col_map[header] = idx
                    
                    # Process data rows
                    page_tickers = 0
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) < len(headers):
                            continue
                        
                        try:
                            # Extract ticker (usually first or second column)
                            ticker_text = cells[col_map.get('ticker', 1)].get_text().strip()
                            if not ticker_text or len(ticker_text) < 3 or len(ticker_text) > 4:
                                continue
                            
                            # Build ticker data
                            ticker_data = {
                                'ticker': ticker_text,
                                'company_name': cells[col_map.get('company', 2)].get_text().strip() if col_map.get('company', 2) < len(cells) else '',
                                'sector': cells[col_map.get('sector', 3)].get_text().strip() if col_map.get('sector', 3) < len(cells) else '',
                                'industry': cells[col_map.get('industry', 4)].get_text().strip() if col_map.get('industry', 4) < len(cells) else '',
                                'country': cells[col_map.get('country', 5)].get_text().strip() if col_map.get('country', 5) < len(cells) else '',
                                'market_cap': self.parse_table_value(cells[col_map.get('market cap', 6)].get_text().strip() if col_map.get('market cap', 6) < len(cells) else '0'),
                                'price': self.parse_table_value(cells[col_map.get('price', 7)].get_text().strip() if col_map.get('price', 7) < len(cells) else '0'),
                                'volume': int(self.parse_table_value(cells[col_map.get('volume', 8)].get_text().strip() if col_map.get('volume', 8) < len(cells) else '0')),
                                'float_shares': self.parse_table_value(cells[col_map.get('float', 9)].get_text().strip() if col_map.get('float', 9) < len(cells) else '0')
                            }
                            
                            if ticker_data['ticker']:
                                all_tickers.append(ticker_data)
                                page_tickers += 1
                                
                        except Exception as e:
                            logger.debug(f"Error parsing ticker row: {e}")
                            continue
                    
                    if page_tickers == 0:
                        logger.info(f"No valid tickers on page {page_num}, stopping")
                        break
                    
                    url_tickers += page_tickers
                    page_num += 1
                    
                    # Safety limit
                    if page_num > 50:
                        logger.warning(f"Reached page limit for {url_desc}")
                        break
                    
                    # Delay between pages for respectful scraping
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error scraping page {page_num}: {e}")
                    break
            
            logger.info(f"✅ Completed {url_desc}: {url_tickers} tickers found")
        
        # Remove duplicates by ticker
        unique_tickers = {}
        for ticker_data in all_tickers:
            ticker = ticker_data['ticker']
            if ticker not in unique_tickers:
                unique_tickers[ticker] = ticker_data
        
        final_tickers = list(unique_tickers.values())
        self.stats['tickers_found'] = len(final_tickers)
        
        logger.info(f"📊 Total unique tickers found: {len(final_tickers)}")
        return final_tickers

    async def get_ticker_list_from_database(self, single_ticker: str = None) -> List[Dict[str, Any]]:
        """Get ticker list from existing float_list table instead of scraping screeners"""
        try:
            if single_ticker:
                logger.info(f"📊 Getting data for single ticker: {single_ticker}")
                
                # Query for specific ticker
                query = """
                SELECT 
                    ticker,
                    company_name,
                    sector,
                    industry,
                    country,
                    market_cap,
                    price,
                    volume,
                    float_shares
                FROM News.float_list 
                WHERE ticker = %s
                """
                
                result = self.ch_manager.client.query(query, parameters=[single_ticker.upper()])
                
                if not result.result_rows:
                    logger.warning(f"Ticker {single_ticker} not found in float_list table, creating default entry")
                    # Create a default entry for the ticker
                    tickers = [{
                        'ticker': single_ticker.upper(),
                        'company_name': '',
                        'sector': '',
                        'industry': '',
                        'country': '',
                        'market_cap': 0.0,
                        'price': 0.0,
                        'volume': 0,
                        'float_shares': 0.0
                    }]
                else:
                    # Process the found ticker
                    tickers = []
                    for row in result.result_rows:
                        ticker_data = {
                            'ticker': row[0],
                            'company_name': row[1] or '',
                            'sector': row[2] or '',
                            'industry': row[3] or '',
                            'country': row[4] or '',
                            'market_cap': float(row[5]) if row[5] else 0.0,
                            'price': float(row[6]) if row[6] else 0.0,
                            'volume': int(row[7]) if row[7] else 0,
                            'float_shares': float(row[8]) if row[8] else 0.0
                        }
                        tickers.append(ticker_data)
            else:
                logger.info("📊 Getting ticker list from existing float_list table...")
                
                # Query the existing float_list table
                query = """
                SELECT 
                    ticker,
                    company_name,
                    sector,
                    industry,
                    country,
                    market_cap,
                    price,
                    volume,
                    float_shares
                FROM News.float_list 
                WHERE ticker IS NOT NULL 
                AND ticker != ''
                ORDER BY ticker
                """
                
                result = self.ch_manager.client.query(query)
                
                tickers = []
                for row in result.result_rows:
                    ticker_data = {
                        'ticker': row[0],
                        'company_name': row[1] or '',
                        'sector': row[2] or '',
                        'industry': row[3] or '',
                        'country': row[4] or '',
                        'market_cap': float(row[5]) if row[5] else 0.0,
                        'price': float(row[6]) if row[6] else 0.0,
                        'volume': int(row[7]) if row[7] else 0,
                        'float_shares': float(row[8]) if row[8] else 0.0
                    }
                    tickers.append(ticker_data)
            
            self.stats['tickers_found'] = len(tickers)
            logger.info(f"📊 Found {len(tickers)} tickers from float_list table")
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting tickers from database: {e}")
            if single_ticker:
                # For single ticker, don't fall back to screener scraping
                logger.error(f"Cannot proceed with single ticker {single_ticker}")
                return []
            else:
                logger.info("Falling back to screener scraping...")
                return await self.get_ticker_list_from_screeners()

    def is_newswire_article(self, article_text: str, article_url: str) -> str:
        """Check if article is from target newswires and return the type"""
        article_text_lower = article_text.lower()
        article_url_lower = article_url.lower()
        
        # Check URL patterns
        if 'globenewswire.com' in article_url_lower:
            return 'GlobeNewswire'
        elif 'prnewswire.com' in article_url_lower:
            return 'PRNewswire'
        elif 'businesswire.com' in article_url_lower:
            return 'BusinessWire'
        elif 'accesswire.com' in article_url_lower:
            return 'Accesswire'
        
        # Check text patterns
        for newswire in self.target_newswires:
            if newswire.lower() in article_text_lower:
                return newswire
        
        return None

    def filter_by_time(self, published_est: datetime) -> bool:
        """Filter articles to only 5am-9am EST"""
        try:
            # Convert UTC to EST
            est_time = published_est.replace(tzinfo=pytz.UTC).astimezone(self.est_tz)
            
            # Check if time is between 5am and 9am EST
            return 5 <= est_time.hour < 9
            
        except Exception as e:
            logger.debug(f"Error filtering time: {e}")
            return False

    def generate_content_hash(self, headline: str, article_url: str) -> str:
        """Generate hash for duplicate detection"""
        content = f"{headline}|{article_url}".strip()
        return hashlib.md5(content.encode()).hexdigest()

    def extract_timestamp_from_finviz_news(self, link, soup) -> str:
        """Extract timestamp from Finviz news listing - improved approach"""
        try:
            # Enhanced timestamp patterns for Finviz pages
            time_patterns = [
                # Full date patterns like "Jun 09, 2025 12:58 ET"
                r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*ET\b',     # "June 09, 2025 12:58 ET"
                r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*EST\b',    # "June 09, 2025 12:58 EST"
                r'\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]M\s*ET\b',  # "June 09, 2025 12:58 PM ET"
                
                # Finviz specific patterns
                r'Today\s+\d{1,2}:\d{2}[AP]M',                                     # "Today 08:30AM"
                r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',                        # "Jul-21-25 08:20AM"
                r'\w{3}-\d{2}-\d{2}',                                              # "Jul-23-25"
                
                # ISO-like patterns
                r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b',                     # "2024-01-15 10:30:00"
                r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\b',                           # "2024-01-15 10:30"
                
                # Time-only patterns (fallback)
                r'\b\d{1,2}:\d{2}\s*[AP]M\s*ET\b',                               # "09:30 AM ET"
                r'\b\d{1,2}:\d{2}\s*[AP]M\s*EST\b',                              # "09:30 AM EST"
                r'\b\d{1,2}:\d{2}\s*ET\b',                                        # "09:30 ET"
                r'\b\d{1,2}:\d{2}\s*EST\b',                                       # "09:30 EST"
            ]
            
            # Method 1: Look for timestamp in the same table row or container as the link
            container = link.parent
            if container:
                # Look for timestamp patterns in the container text
                container_text = container.get_text()
                logger.debug(f"🔍 Container text: '{container_text[:100]}...'")
                
                for pattern in time_patterns:
                    match = re.search(pattern, container_text)
                    if match:
                        found_time = match.group().strip()
                        logger.debug(f"✅ Found timestamp in container: {found_time}")
                        return found_time
                
                # Method 2: Look for timestamp in sibling elements (table cells, etc.)
                parent = container.parent
                if parent:
                    # Check all siblings for timestamp information
                    for sibling in parent.find_all(['td', 'span', 'div']):
                        sibling_text = sibling.get_text().strip()
                        if sibling_text and len(sibling_text) < 50:  # Timestamps are usually short
                            for pattern in time_patterns:
                                match = re.search(pattern, sibling_text)
                                if match:
                                    found_time = match.group().strip()
                                    logger.debug(f"✅ Found timestamp in sibling: {found_time}")
                                    return found_time
                
                # Method 3: Look for specific timestamp elements with datetime attributes
                for time_elem in container.find_all(['time', 'span', 'div']):
                    # Check for datetime attributes
                    datetime_attr = time_elem.get('datetime')
                    if datetime_attr:
                        logger.debug(f"✅ Found datetime attribute: {datetime_attr}")
                        return datetime_attr
                    
                    # Check for timestamp-like classes
                    elem_class = time_elem.get('class', [])
                    if any(cls for cls in elem_class if 'time' in cls.lower() or 'date' in cls.lower()):
                        time_text = time_elem.get_text(strip=True)
                        if time_text and len(time_text) > 3:
                            logger.debug(f"✅ Found timestamp by class: {time_text}")
                            return time_text
                
                # Method 4: Look in broader container (table row, etc.)
                broader_container = parent.parent if parent else None
                if broader_container:
                    broader_text = broader_container.get_text()
                    logger.debug(f"🔍 Broader container text: '{broader_text[:100]}...'")
                    for pattern in time_patterns:
                        match = re.search(pattern, broader_text)
                        if match:
                            found_time = match.group().strip()
                            logger.debug(f"✅ Found timestamp in broader container: {found_time}")
                            return found_time
            
            # Method 5: Look for news table structure - Finviz often uses tables for news
            news_tables = soup.find_all('table')
            for table in news_tables:
                # Check if this table contains our link
                if link in table.find_all('a'):
                    # Look for timestamp in the same row
                    row = link.find_parent('tr')
                    if row:
                        row_text = row.get_text()
                        logger.debug(f"🔍 Table row text: '{row_text[:100]}...'")
                        for pattern in time_patterns:
                            match = re.search(pattern, row_text)
                            if match:
                                found_time = match.group().strip()
                                logger.debug(f"✅ Found timestamp in table row: {found_time}")
                                return found_time
            
            # Method 6: Look for timestamp anywhere in the vicinity of the link
            # Sometimes timestamps might be in nearby text nodes
            link_text = link.get_text()
            logger.debug(f"🔍 Link text: '{link_text}'")
            
            # Check if the link text itself contains a timestamp
            for pattern in time_patterns:
                match = re.search(pattern, link_text)
                if match:
                    found_time = match.group().strip()
                    logger.debug(f"✅ Found timestamp in link text: {found_time}")
                    return found_time
            
            # Method 7: Look in all text around the link (more aggressive search)
            if container:
                # Get all text from parent containers up to 3 levels up
                search_containers = [container]
                current = container
                for level in range(3):
                    if current.parent:
                        current = current.parent
                        search_containers.append(current)
                
                for search_container in search_containers:
                    search_text = search_container.get_text()
                    logger.debug(f"🔍 Level {search_containers.index(search_container)} container: '{search_text[:100]}...'")
                    
                    for pattern in time_patterns:
                        match = re.search(pattern, search_text)
                        if match:
                            found_time = match.group().strip()
                            logger.debug(f"✅ Found timestamp at level {search_containers.index(search_container)}: {found_time}")
                            return found_time
            
        except Exception as e:
            logger.debug(f"❌ Error extracting timestamp from Finviz news: {e}")
        
        # Return current time as fallback
        current_time = datetime.now().strftime("%H:%M ET")
        logger.warning(f"⚠️ No timestamp found, using current time as fallback: {current_time}")
        return current_time

    def parse_finviz_timestamp(self, time_text: str) -> datetime:
        """Parse timestamp from Finviz - enhanced version with better format support"""
        if not time_text:
            logger.warning("⚠️ Empty time_text provided to parse_finviz_timestamp")
            return datetime.now()
        
        time_text = time_text.strip()
        logger.debug(f"🕐 Attempting to parse timestamp: '{time_text}'")
        
        try:
            # Handle full datetime patterns
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
                        # Clean up the text for parsing
                        clean_text = re.sub(r'\s*ET$|\s*EST$', '', time_text)
                        parsed_time = datetime.strptime(clean_text, format_str.replace(' ET', '').replace(' EST', ''))
                        logger.debug(f"✅ PARSED FULL DATETIME: '{time_text}' -> {parsed_time}")
                        return parsed_time
                    except ValueError as e:
                        logger.debug(f"❌ Failed to parse with pattern {pattern}: {e}")
                        continue
            
            # Handle Finviz-specific patterns
            if 'Today' in time_text:
                # "Today 08:30AM" format
                time_match = re.search(r'Today\s+(\d{1,2}):(\d{2})([AP]M)', time_text)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    ampm = time_match.group(3)
                    
                    # Convert to 24-hour format
                    if ampm == 'PM' and hour != 12:
                        hour += 12
                    elif ampm == 'AM' and hour == 12:
                        hour = 0
                    
                    today = datetime.now().date()
                    parsed_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
                    logger.debug(f"✅ PARSED TODAY: '{time_text}' -> {parsed_time}")
                    return parsed_time
                else:
                    logger.debug(f"❌ 'Today' found but regex didn't match: '{time_text}'")
            
            # Handle "Jul-21-25 08:20AM" format
            finviz_match = re.search(r'(\w{3})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})([AP]M)', time_text)
            if finviz_match:
                month_str = finviz_match.group(1)
                day = int(finviz_match.group(2))
                year = int(f"20{finviz_match.group(3)}")  # Convert 25 to 2025
                hour = int(finviz_match.group(4))
                minute = int(finviz_match.group(5))
                ampm = finviz_match.group(6)
                
                # Convert to 24-hour format
                if ampm == 'PM' and hour != 12:
                    hour += 12
                elif ampm == 'AM' and hour == 12:
                    hour = 0
                
                # Parse month
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month = month_map.get(month_str, 1)
                
                parsed_time = datetime(year, month, day, hour, minute)
                logger.debug(f"✅ PARSED FINVIZ FORMAT: '{time_text}' -> {parsed_time}")
                return parsed_time
            
            # Handle "Jul-23-25" date-only format
            date_only_match = re.search(r'(\w{3})-(\d{2})-(\d{2})$', time_text)
            if date_only_match:
                month_str = date_only_match.group(1)
                day = int(date_only_match.group(2))
                year = int(f"20{date_only_match.group(3)}")
                
                # Parse month
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month = month_map.get(month_str, 1)
                
                parsed_time = datetime(year, month, day, 9, 0)  # Default to 9 AM
                logger.debug(f"✅ PARSED DATE ONLY: '{time_text}' -> {parsed_time}")
                return parsed_time
            
            # Fallback to time-only parsing (existing logic)
            time_match = re.search(r'(\d{1,2}):(\d{2})([AP]M)?', time_text)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                ampm = time_match.group(3)
                
                # Convert to 24-hour format if AM/PM is specified
                if ampm:
                    if ampm == 'PM' and hour != 12:
                        hour += 12
                    elif ampm == 'AM' and hour == 12:
                        hour = 0
                
                # Use today's date with the EXACT time found in the article
                today = datetime.now().date()
                parsed_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
                
                # If this time is more than 12 hours in the future, assume it's from yesterday
                time_diff = (datetime.now() - parsed_time).total_seconds()
                if time_diff < -43200:  # More than 12 hours in future
                    parsed_time = parsed_time - timedelta(days=1)
                    logger.debug(f"✅ FROM YESTERDAY: '{time_text}' -> {parsed_time}")
                else:
                    logger.debug(f"✅ TIME ONLY: '{time_text}' -> {parsed_time}")
                
                return parsed_time
            else:
                logger.debug(f"❌ No time patterns matched: '{time_text}'")
                
        except Exception as e:
            logger.debug(f"❌ Error parsing time '{time_text}': {e}")
        
        # Fallback to current time
        logger.warning(f"⚠️ Could not parse time '{time_text}', using current time")
        return datetime.now()

    def debug_finviz_structure(self, link, soup, ticker):
        """Debug function to understand Finviz page structure"""
        try:
            logger.debug(f"🔍 DEBUGGING STRUCTURE for {ticker}")
            
            # Get link details
            href = link.get('href', '')
            text = link.get_text().strip()
            logger.debug(f"📎 Link: '{text}' -> {href}")
            
            # Check parent structure
            container = link.parent
            if container:
                logger.debug(f"📦 Parent tag: {container.name}")
                logger.debug(f"📦 Parent classes: {container.get('class', [])}")
                logger.debug(f"📦 Parent text: '{container.get_text()[:200]}...'")
                
                # Check if parent is a table cell
                if container.name == 'td':
                    row = container.parent
                    if row and row.name == 'tr':
                        cells = row.find_all('td')
                        logger.debug(f"📊 Table row has {len(cells)} cells:")
                        for i, cell in enumerate(cells):
                            cell_text = cell.get_text().strip()
                            logger.debug(f"  Cell {i}: '{cell_text[:50]}...'")
                            
                            # Check for timestamp patterns in each cell
                            time_patterns = [
                                r'Today\s+\d{1,2}:\d{2}[AP]M',
                                r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                                r'\w{3}-\d{2}-\d{2}',
                                r'\b\d{1,2}:\d{2}\s*[AP]M\s*ET\b',
                                r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\b'
                            ]
                            
                            for pattern in time_patterns:
                                if re.search(pattern, cell_text):
                                    logger.debug(f"  ✅ TIMESTAMP PATTERN FOUND in cell {i}: '{cell_text}'")
                
                # Check grandparent
                grandparent = container.parent
                if grandparent:
                    logger.debug(f"👴 Grandparent tag: {grandparent.name}")
                    logger.debug(f"👴 Grandparent text: '{grandparent.get_text()[:200]}...'")
            
            # Look for nearby timestamp elements
            logger.debug("🔍 Looking for nearby timestamp elements...")
            
            # Check for time elements
            time_elements = soup.find_all('time')
            if time_elements:
                logger.debug(f"⏰ Found {len(time_elements)} <time> elements on page")
                for time_elem in time_elements:
                    logger.debug(f"  Time element: {time_elem}")
            
            # Check for elements with timestamp-like classes
            timestamp_classes = ['time', 'date', 'timestamp', 'published', 'datetime']
            for class_name in timestamp_classes:
                elements = soup.find_all(class_=lambda x: x and class_name in str(x).lower())
                if elements:
                    logger.debug(f"📅 Found {len(elements)} elements with '{class_name}' in class")
                    for elem in elements[:3]:  # Show first 3
                        logger.debug(f"  {elem.name}: {elem.get_text()[:50]}")
            
        except Exception as e:
            logger.debug(f"❌ Error in debug structure: {e}")

    async def scrape_ticker_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Scrape news for a specific ticker using proper cell-pairing logic"""
        articles = []
        
        # Construct ticker page URL
        ticker_url = f"https://elite.finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=i1"
        
        logger.info(f"📰 Scraping news for {ticker} with cell-pairing logic...")
        
        try:
            # Use Crawl4AI to get the page
            result: CrawlResult = await self.crawler.arun(
                url=ticker_url,
                wait_for="css:table",
                delay_before_return_html=3.0,
                timeout=30
            )
            
            if not result.success or not result.html:
                logger.warning(f"Failed to scrape {ticker}: {result.error_message}")
                return articles
            
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Find the news table - it's the table with both timestamps and newswire links
            news_table = None
            tables = soup.find_all('table')
            
            target_newswires = [
                'GlobeNewswire', 'Globe Newswire', 'GLOBENEWSWIRE', 'GLOBE NEWSWIRE',
                'PRNewswire', 'PR Newswire', 'PRNEWSWIRE', 'PR NEWSWIRE', 
                'BusinessWire', 'Business Wire', 'BUSINESSWIRE', 'BUSINESS WIRE',
                'Accesswire', 'AccessWire', 'ACCESSWIRE', 'ACCESS WIRE'
            ]
            
            timestamp_pattern = r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M'
            time_only_pattern = r'^\d{1,2}:\d{2}[AP]M$'
            
            for table in tables:
                table_text = table.get_text()
                has_timestamps = bool(re.search(timestamp_pattern, table_text))
                has_newswires = any(wire in table_text for wire in target_newswires)
                
                if has_timestamps and has_newswires:
                    news_table = table
                    logger.debug(f"Found news table with {len(table.find_all('tr'))} rows")
                    break
            
            if not news_table:
                logger.warning(f"No news table found for {ticker}")
                return articles
            
            # Process table rows: each row has timestamp cell and article cell
            processed_articles = set()
            last_full_date = None  # Keep track of last full date for time-only timestamps
            
            news_rows = news_table.find_all('tr')
            logger.debug(f"Found {len(news_rows)} news rows to process")
            
            for row in news_rows:
                try:
                    cells = row.find_all(['td', 'th'])
                    
                    # Skip rows that don't have exactly 2 cells (timestamp + article)
                    if len(cells) != 2:
                        continue
                    
                    timestamp_cell = cells[0]
                    article_cell = cells[1]
                    
                    timestamp_text = timestamp_cell.get_text().strip()
                    article_links = article_cell.find_all('a', href=True)
                    
                    # Skip rows without article links
                    if not article_links:
                        continue
                    
                    # Check if timestamp cell contains a timestamp
                    is_full_timestamp = bool(re.search(timestamp_pattern, timestamp_text))
                    is_time_only = bool(re.match(time_only_pattern, timestamp_text))
                    
                    current_timestamp = None
                    
                    if is_full_timestamp:
                        # Extract the full timestamp and remember the date part
                        timestamp_match = re.search(timestamp_pattern, timestamp_text)
                        if timestamp_match:
                            current_timestamp = timestamp_match.group()
                            last_full_date = current_timestamp.split()[0]  # Extract date part (e.g., "Jul-08-25")
                            logger.debug(f"Row with full timestamp: {current_timestamp}")
                            
                    elif is_time_only and last_full_date:
                        # Combine time-only with last known date
                        current_timestamp = f"{last_full_date} {timestamp_text}"
                        logger.debug(f"Row with time-only '{timestamp_text}' combined with date '{last_full_date}' -> '{current_timestamp}'")
                    
                    else:
                        # Not a timestamp row, skip
                        logger.debug(f"Skipping row - no valid timestamp: '{timestamp_text}'")
                        continue
                    
                    # Process all links in this article cell
                    for link in article_links:
                        try:
                            href = link.get('href', '')
                            headline_text = link.get_text().strip()
                            
                            # Skip if link is too short or invalid
                            if not href or not headline_text or len(headline_text) < 15:
                                continue
                            
                            # Check if this specific link is associated with a newswire
                            newswire_type = self.find_newswire_for_link(link, article_cell, target_newswires)
                            
                            if not newswire_type:
                                logger.debug(f"    ❌ No newswire found for link: {headline_text[:30]}...")
                                continue
                            
                            # Convert relative URLs to absolute
                            if href.startswith('/'):
                                article_url = f"https://elite.finviz.com{href}"
                            elif href.startswith('http'):
                                article_url = href
                            else:
                                logger.debug(f"    ❌ Invalid URL format: {href}")
                                continue
                            
                            # Parse timestamp
                            published_est = self.parse_finviz_timestamp(current_timestamp)
                            
                            # Create unique key to avoid duplicates
                            article_key = (headline_text, article_url)
                            if article_key in processed_articles:
                                logger.debug(f"    ❌ Duplicate article")
                                continue
                            processed_articles.add(article_key)
                            
                            # Create article record
                            article = {
                                'ticker': ticker,
                                'headline': headline_text,
                                'article_url': article_url,
                                'published_est': published_est,
                                'newswire_type': newswire_type,
                                'content_hash': self.generate_content_hash(headline_text, article_url)
                            }
                            
                            articles.append(article)
                            self.stats['articles_found'] += 1
                            
                            logger.debug(f"    ✅ ROW PAIRED: {current_timestamp} → {headline_text[:40]}... ({newswire_type})")
                            
                        except Exception as e:
                            logger.debug(f"    ❌ Error processing link: {e}")
                            continue
                
                except Exception as e:
                    logger.debug(f"Error processing row: {e}")
                    continue
            
            logger.info(f"✅ {ticker}: Found {len(articles)} properly paired newswire articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping news for {ticker}: {e}")
            return articles

    def find_newswire_for_link(self, link, container, target_newswires) -> str:
        """Find the specific newswire type for a given link"""
        try:
            # Method 1: Look for newswire indicators in the immediate vicinity of the link
            
            # Get the parent elements to search for newswire indicators
            search_elements = [link]
            current = link
            
            # Go up the DOM tree to find newswire indicators
            for level in range(3):  # Search up to 3 levels up
                if current.parent:
                    current = current.parent
                    search_elements.append(current)
            
            # Also search siblings
            if link.parent:
                for sibling in link.parent.find_all(['span', 'div', 'td', 'a']):
                    search_elements.append(sibling)
            
            # Look for newswire indicators in these elements
            for element in search_elements:
                element_text = element.get_text()
                
                # Check for parenthetical newswire indicators first (more reliable)
                for newswire in target_newswires:
                    if f'({newswire})' in element_text:
                        return self.normalize_newswire_name(newswire)
                
                # Check for non-parenthetical newswire mentions
                for newswire in target_newswires:
                    if newswire in element_text:
                        # Make sure it's not just part of another word
                        if re.search(rf'\b{re.escape(newswire)}\b', element_text, re.IGNORECASE):
                            return self.normalize_newswire_name(newswire)
            
            # Method 2: Check if the link URL itself indicates a newswire
            href = link.get('href', '')
            if 'globenewswire.com' in href.lower():
                return 'GlobeNewswire'
            elif 'prnewswire.com' in href.lower():
                return 'PRNewswire'
            elif 'businesswire.com' in href.lower():
                return 'BusinessWire'
            elif 'accesswire.com' in href.lower():
                return 'Accesswire'
            
            return None
            
        except Exception as e:
            logger.debug(f"Error finding newswire for link: {e}")
            return None

    def normalize_newswire_name(self, newswire: str) -> str:
        """Normalize newswire names to consistent format"""
        mapping = {
            'ACCESSWIRE': 'Accesswire',
            'PRNewswire': 'PRNewswire', 
            'BusinessWire': 'BusinessWire',
            'GlobeNewswire': 'GlobeNewswire',
            'TipRanks': 'TipRanks'
        }
        return mapping.get(newswire, newswire)

    async def store_ticker_list(self, tickers: List[Dict[str, Any]]):
        """Store ticker list in ClickHouse"""
        if not tickers:
            return
        
        try:
            # Prepare data for insertion
            ticker_data = []
            for ticker in tickers:
                ticker_data.append((
                    ticker['ticker'],
                    ticker['company_name'],
                    ticker['sector'],
                    ticker['industry'],
                    ticker['country'],
                    ticker['market_cap'],
                    ticker['price'],
                    ticker['volume'],
                    ticker['float_shares'],
                    datetime.now()
                ))
            
            # Clear existing data
            self.ch_manager.client.command("TRUNCATE TABLE News.ticker_master_backtest")
            
            # Insert new data
            self.ch_manager.client.insert(
                'News.ticker_master_backtest',
                ticker_data,
                column_names=['ticker', 'company_name', 'sector', 'industry', 'country', 'market_cap', 'price', 'volume', 'float_shares', 'scraped_at']
            )
            
            logger.info(f"✅ Stored {len(ticker_data)} tickers in database")
            
        except Exception as e:
            logger.error(f"Error storing ticker list: {e}")

    async def store_articles(self, articles: List[Dict[str, Any]]):
        """Store articles in ClickHouse"""
        if not articles:
            return
        
        try:
            # Prepare data for insertion
            article_data = []
            for article in articles:
                article_data.append((
                    article['ticker'],
                    article['headline'],
                    article['article_url'],
                    article['published_est'],
                    datetime.now(),
                    'finviz',
                    article['newswire_type'],
                    '',  # article_content (will be filled by sentiment analysis)
                    article['content_hash']
                ))
            
            # Insert articles
            self.ch_manager.client.insert(
                'News.historical_news',
                article_data,
                column_names=['ticker', 'headline', 'article_url', 'published_est', 'scraped_at', 'source', 'newswire_type', 'article_content', 'content_hash']
            )
            
            self.stats['articles_stored'] += len(article_data)
            logger.info(f"✅ Stored {len(article_data)} articles in database")
            
        except Exception as e:
            logger.error(f"Error storing articles: {e}")

    async def run_historical_scrape(self, ticker_limit: int = None, single_ticker: str = None):
        """Run the complete historical scraping process"""
        try:
            if single_ticker:
                limit_desc = f" (single ticker: {single_ticker})"
            elif ticker_limit:
                limit_desc = f" (limited to {ticker_limit} tickers)"
            else:
                limit_desc = ""
                
            logger.info(f"🚀 Starting Finviz Historical News Scraping with Crawl4AI{limit_desc}...")
            
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize scraper")
                return False
            
            # Step 1: Get ticker list from database (much faster than scraping)
            logger.info("📊 STEP 1: Getting ticker list from database...")
            tickers = await self.get_ticker_list_from_database(single_ticker=single_ticker)
            
            if not tickers:
                logger.error("No tickers found from database")
                return False
            
            # Apply ticker limit if specified (but not if single ticker is specified)
            if not single_ticker and ticker_limit and ticker_limit < len(tickers):
                logger.info(f"🔢 Limiting tickers from {len(tickers)} to {ticker_limit} for testing")
                tickers = tickers[:ticker_limit]
            
            # Store ticker list
            await self.store_ticker_list(tickers)
            
            # Step 2: Scrape news for each ticker
            logger.info(f"📰 STEP 2: Scraping 6 months of news for {len(tickers)} tickers...")
            
            all_articles = []
            processed = 0
            
            for ticker_data in tickers:
                ticker = ticker_data['ticker']
                
                try:
                    # Scrape news for this ticker
                    ticker_articles = await self.scrape_ticker_news(ticker)
                    all_articles.extend(ticker_articles)
                    
                    processed += 1
                    self.stats['tickers_processed'] = processed
                    
                    # Store articles in batches
                    if len(all_articles) >= 100:
                        await self.store_articles(all_articles)
                        all_articles = []
                    
                    # Progress logging
                    if processed % 10 == 0 or single_ticker:  # Always log for single ticker
                        logger.info(f"📈 PROGRESS: {processed}/{len(tickers)} tickers processed, {self.stats['articles_stored']} articles stored")
                    
                    # Rate limiting for respectful scraping
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    continue
            
            # Store remaining articles
            if all_articles:
                await self.store_articles(all_articles)
            
            # Final stats
            elapsed = datetime.now() - self.stats['start_time']
            logger.info("🎉 HISTORICAL SCRAPING COMPLETE!")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Tickers found: {self.stats['tickers_found']}")
            logger.info(f"  • Tickers processed: {self.stats['tickers_processed']}")
            logger.info(f"  • Articles found: {self.stats['articles_found']}")
            logger.info(f"  • Articles stored: {self.stats['articles_stored']}")
            logger.info(f"  • Time elapsed: {elapsed}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in historical scrape: {e}")
            return False
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Close Crawl4AI crawler
            if self.crawler:
                await self.crawler.close()
                
            # Close ClickHouse connection
            if self.ch_manager:
                self.ch_manager.close()
                
            logger.info("✅ Finviz scraper cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function"""
    scraper = FinvizHistoricalScraper()
    success = await scraper.run_historical_scrape()
    
    if success:
        print("\n✅ Historical news scraping completed successfully!")
    else:
        print("\n❌ Historical news scraping failed!")

if __name__ == "__main__":
    asyncio.run(main()) 