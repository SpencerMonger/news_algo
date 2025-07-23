#!/usr/bin/env python3
"""
Finviz Historical News Scraper for Backtesting
Scrapes ticker lists and 6 months of newswire articles from Finviz
Only scrapes newswires articles published between 5am-9am EST
"""

import asyncio
import aiohttp
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed back from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinvizHistoricalScraper:
    def __init__(self):
        self.ch_manager = None
        self.session = None
        
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
        
        # Browser headers to avoid detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Cache-Control': 'max-age=0'
        }
        
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
        """Initialize the scraper"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create HTTP session with longer timeout for news scraping
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout, 
                connector=connector,
                headers=self.headers
            )
            
            logger.info("✅ Finviz Historical Scraper initialized")
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
        """Get complete ticker list from Finviz screener URLs"""
        all_tickers = []
        
        logger.info(f"📊 Scraping ticker lists from {len(self.screener_urls)} screener URLs...")
        
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
                    async with self.session.get(page_url) as response:
                        if response.status != 200:
                            logger.warning(f"HTTP {response.status} for page {page_num}, stopping")
                            break
                        
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
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
                        
                        # Delay between pages
                        await asyncio.sleep(1)
                        
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

    def filter_by_time(self, published_utc: datetime) -> bool:
        """Filter articles to only 5am-9am EST"""
        try:
            # Convert UTC to EST
            est_time = published_utc.replace(tzinfo=pytz.UTC).astimezone(self.est_tz)
            
            # Check if time is between 5am and 9am EST
            return 5 <= est_time.hour < 9
            
        except Exception as e:
            logger.debug(f"Error filtering time: {e}")
            return False

    def generate_content_hash(self, headline: str, article_url: str) -> str:
        """Generate hash for duplicate detection"""
        content = f"{headline}|{article_url}".strip()
        return hashlib.md5(content.encode()).hexdigest()

    async def scrape_ticker_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Scrape 6 months of news for a specific ticker"""
        articles = []
        
        # Construct ticker page URL - use the same format as shown in screenshot
        ticker_url = f"https://elite.finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=i1"
        
        logger.info(f"📰 Scraping news for {ticker}...")
        
        try:
            async with self.session.get(ticker_url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {ticker}")
                    return articles
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Debug: Log the page structure to understand the layout
                logger.debug(f"Page title: {soup.title.string if soup.title else 'No title'}")
                
                # Find all links on the page that could be news articles
                all_links = soup.find_all('a', href=True)
                
                logger.debug(f"Found {len(all_links)} total links on {ticker} page")
                
                # Look for news links - they should contain news URLs or newswire domains
                news_links_found = 0
                
                for link in all_links:
                    try:
                        href = link.get('href', '')
                        text = link.get_text().strip()
                        
                        if not href or not text:
                            continue
                        
                        # Check if this is a news link by URL patterns
                        is_news_link = False
                        
                        # Direct newswire URLs
                        if any(domain in href.lower() for domain in [
                            'businesswire.com', 'globenewswire.com', 
                            'prnewswire.com', 'accesswire.com'
                        ]):
                            is_news_link = True
                        
                        # Finviz news URLs
                        elif 'finviz.com/news' in href.lower():
                            is_news_link = True
                        
                        # Yahoo Finance news URLs
                        elif 'finance.yahoo.com/news' in href.lower():
                            is_news_link = True
                            
                        # Other news patterns
                        elif any(pattern in href.lower() for pattern in [
                            '/news/', 'news.', 'press-release', 'article'
                        ]):
                            # Additional check - make sure it's not a navigation link
                            if not any(nav in href.lower() for nav in [
                                'screener', 'portfolio', 'insider', 'futures', 
                                'forex', 'crypto', 'backtests', 'pricing'
                            ]):
                                is_news_link = True
                        
                        if not is_news_link:
                            continue
                        
                        news_links_found += 1
                        
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            article_url = f"https://elite.finviz.com{href}"
                        elif href.startswith('http'):
                            article_url = href
                        else:
                            continue
                        
                        # Try to extract date from the link text or surrounding elements
                        published_utc = None
                        
                        # Look for date patterns in the link text first
                        date_patterns = [
                            r'Today (\d{2}:\d{2}AM)',                # Today 07:56AM
                            r'(\w{3}-\d{2}-\d{2}) (\d{2}:\d{2}AM)',  # Jul-21-25 08:20AM
                            r'(\w{3}-\d{2}-\d{2})',                  # Jul-23-25
                            r'(\d{2}-\d{2}-\d{2})',                  # MM-DD-YY
                            r'(\d{4}-\d{2}-\d{2})',                  # YYYY-MM-DD
                            r'(\d{1,2}/\d{1,2}/\d{4})',              # M/D/YYYY
                            r'(\w{3} \d{1,2})',                      # Jul 23
                        ]
                        
                        # Check the link text and parent elements for dates
                        search_text = text
                        parent = link.parent
                        if parent:
                            search_text += " " + parent.get_text()
                        
                        for pattern in date_patterns:
                            match = re.search(pattern, search_text)
                            if match:
                                try:
                                    if pattern.startswith(r'Today'):
                                        # Today format - use current date
                                        time_str = match.group(1)
                                        today = datetime.now().date()
                                        time_obj = datetime.strptime(time_str, '%H:%M%p').time()
                                        published_utc = datetime.combine(today, time_obj)
                                        
                                    elif r'AM\)' in pattern:
                                        # Date with time format
                                        date_str = match.group(1)
                                        time_str = match.group(2)
                                        
                                        # Parse date part
                                        if re.match(r'\w{3}-\d{2}-\d{2}', date_str):
                                            date_obj = datetime.strptime(date_str, '%b-%d-%y').date()
                                            time_obj = datetime.strptime(time_str, '%H:%M%p').time()
                                            published_utc = datetime.combine(date_obj, time_obj)
                                            
                                    else:
                                        # Date only formats
                                        date_str = match.group(1)
                                        
                                        if re.match(r'\w{3}-\d{2}-\d{2}', date_str):
                                            # Jul-23-25 format
                                            published_utc = datetime.strptime(date_str, '%b-%d-%y')
                                        elif re.match(r'\d{2}-\d{2}-\d{2}', date_str):
                                            # MM-DD-YY format
                                            published_utc = datetime.strptime(date_str, '%m-%d-%y')
                                        elif re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                                            # YYYY-MM-DD format
                                            published_utc = datetime.strptime(date_str, '%Y-%m-%d')
                                        elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                                            # M/D/YYYY format
                                            published_utc = datetime.strptime(date_str, '%m/%d/%Y')
                                        elif re.match(r'\w{3} \d{1,2}', date_str):
                                            # Jul 23 format - assume current year
                                            published_utc = datetime.strptime(f"{date_str} {datetime.now().year}", '%b %d %Y')
                                    
                                    if published_utc:
                                        break
                                        
                                except Exception as e:
                                    logger.debug(f"Error parsing date '{match.group()}': {e}")
                                    continue
                        
                        # If no date found, try to extract from URL or skip
                        if not published_utc:
                            logger.debug(f"No date found for article: {text[:50]}...")
                            # For now, assign current date and let filtering handle it
                            published_utc = datetime.now()
                        
                        # Filter by 6 months
                        cutoff_date = datetime.now() - timedelta(days=180)
                        if published_utc < cutoff_date:
                            logger.debug(f"Article too old: {published_utc}")
                            continue
                        
                        # Check if it's a newswire article
                        newswire_type = self.is_newswire_article(text, article_url)
                        if not newswire_type:
                            logger.debug(f"Not a newswire article: {text[:50]}...")
                            continue
                        
                        # Filter by time (5am-9am EST) - temporarily disabled for more data collection
                        # We can analyze timing patterns later and re-enable if needed
                        # if not self.filter_by_time(published_utc):
                        #     logger.debug(f"Article outside 5am-9am EST window: {published_utc}")
                        #     continue
                        
                        # Create article record
                        article = {
                            'ticker': ticker,
                            'headline': text,
                            'article_url': article_url,
                            'published_utc': published_utc,
                            'newswire_type': newswire_type,
                            'content_hash': self.generate_content_hash(text, article_url)
                        }
                        
                        articles.append(article)
                        self.stats['articles_found'] += 1
                        
                        logger.debug(f"Found article: {text[:50]}... ({newswire_type})")
                        
                        # Log progress
                        if len(articles) % 10 == 0:
                            logger.info(f"  📰 {ticker}: Found {len(articles)} qualifying articles...")
                        
                    except Exception as e:
                        logger.debug(f"Error processing news link for {ticker}: {e}")
                        continue
                
                logger.debug(f"Total news links found on {ticker} page: {news_links_found}")
                
        except Exception as e:
            logger.error(f"Error scraping news for {ticker}: {e}")
        
        logger.info(f"✅ {ticker}: Found {len(articles)} newswire articles")
        return articles

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
                    article['published_utc'],
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
                column_names=['ticker', 'headline', 'article_url', 'published_utc', 'scraped_at', 'source', 'newswire_type', 'article_content', 'content_hash']
            )
            
            self.stats['articles_stored'] += len(article_data)
            logger.info(f"✅ Stored {len(article_data)} articles in database")
            
        except Exception as e:
            logger.error(f"Error storing articles: {e}")

    async def run_historical_scrape(self):
        """Run the complete historical scraping process"""
        try:
            logger.info("🚀 Starting Finviz Historical News Scraping...")
            
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize scraper")
                return False
            
            # Step 1: Get ticker list from screeners
            logger.info("📊 STEP 1: Getting ticker list from Finviz screeners...")
            tickers = await self.get_ticker_list_from_screeners()
            
            if not tickers:
                logger.error("No tickers found from screeners")
                return False
            
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
                    if processed % 10 == 0:
                        logger.info(f"📈 PROGRESS: {processed}/{len(tickers)} tickers processed, {self.stats['articles_stored']} articles stored")
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
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
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("✅ Finviz scraper cleanup completed")

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