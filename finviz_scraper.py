import asyncio
import aiohttp
import logging
import re
import os
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import pandas as pd
from clickhouse_setup import ClickHouseManager, setup_clickhouse_database
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinvizScraper:
    def __init__(self, clickhouse_manager: ClickHouseManager):
        self.ch_manager = clickhouse_manager
        self.session = None
        
        # Finviz Elite credentials
        self.email = os.getenv('FINVIZ_EMAIL', '')
        self.password = os.getenv('FINVIZ_PASSWORD', '')
        
        # Screener URLs for low float stocks - split into two URLs to capture all tickers
        # First URL: price under $3
        self.screener_url_1 = "https://elite.finviz.com/screener.ashx?v=111&f=geo_usa|asia|latinamerica|argentina|china|denmark|france|greece|hungary|india|ireland|italy|jordan|philippines|russia|spain|switzerland|thailand|unitedarabemirates|europe|bric|australia|belgium|bermuda|canada|chile|chinahongkong|cyprus|brazil|benelux|colombia|luxembourg|malta|monaco|newzealand|panama|southafrica|uruguay|vietnam|unitedkingdom|turkey|taiwan|sweden|southkorea|singapore|portugal|peru|norway|netherlands|mexico|malaysia|kazakhstan|japan|indonesia|iceland|hongkong|germany|finland,sec_healthcare|technology|industrials|consumerdefensive|communicationservices|energy|consumercyclical|basicmaterials|utilities,sh_float_u100,sh_price_u3&ft=4"
        # Second URL: price $3 to $10  
        self.screener_url_2 = "https://elite.finviz.com/screener.ashx?v=111&f=geo_usa|asia|latinamerica|argentina|china|denmark|france|greece|hungary|india|ireland|italy|jordan|philippines|russia|spain|switzerland|thailand|unitedarabemirates|europe|bric|australia|belgium|bermuda|canada|chile|chinahongkong|cyprus|brazil|benelux|colombia|luxembourg|malta|monaco|newzealand|panama|southafrica|uruguay|vietnam|unitedkingdom|turkey|taiwan|sweden|southkorea|singapore|portugal|peru|norway|netherlands|mexico|malaysia|kazakhstan|japan|indonesia|iceland|hongkong|germany|finland,sec_healthcare|technology|industrials|consumerdefensive|communicationservices|energy|consumercyclical|basicmaterials|utilities,sh_float_u100,sh_price_3to10&ft=4"
        
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
        
        if not self.email or not self.password:
            logger.warning("FINVIZ_EMAIL and FINVIZ_PASSWORD environment variables not set")

    async def login(self):
        """Login to Finviz Elite account with better session handling"""
        try:
            # First, visit the login page to get any necessary tokens/cookies
            login_page_url = "https://elite.finviz.com/login.ashx"
            
            async with self.session.get(login_page_url, headers=self.headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to access login page: HTTP {response.status}")
                    return False
                
                login_page_html = await response.text()
                logger.info("Successfully accessed login page")
            
            # Parse the login page for any hidden form fields
            soup = BeautifulSoup(login_page_html, 'html.parser')
            login_form = soup.find('form')
            
            # Prepare login data
            login_data = {
                'email': self.email,
                'password': self.password,
                'remember': 'on'
            }
            
            # Add any hidden form fields
            if login_form:
                for hidden_input in login_form.find_all('input', type='hidden'):
                    name = hidden_input.get('name')
                    value = hidden_input.get('value', '')
                    if name:
                        login_data[name] = value
            
            # Submit login form
            login_submit_url = "https://elite.finviz.com/login_submit.ashx"
            
            # Add referer header for login
            login_headers = self.headers.copy()
            login_headers['Referer'] = login_page_url
            login_headers['Content-Type'] = 'application/x-www-form-urlencoded'
            
            async with self.session.post(login_submit_url, data=login_data, headers=login_headers, allow_redirects=True) as response:
                response_text = await response.text()
                
                # Check for successful login by looking for elite features or lack of login form
                if response.status in [200, 302]:
                    # Check if we're redirected to a page that indicates successful login
                    if 'elite.finviz.com' in str(response.url) and 'login' not in str(response.url).lower():
                        logger.info("Successfully logged into Finviz Elite")
                        return True
                    elif 'screener' in response_text.lower() or 'portfolio' in response_text.lower():
                        logger.info("Successfully logged into Finviz Elite (detected via page content)")
                        return True
                    else:
                        logger.error("Login may have failed - unexpected page content")
                        # Log first 500 chars of response for debugging
                        logger.debug(f"Response content preview: {response_text[:500]}")
                        return False
                else:
                    logger.error(f"Login request failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False

    async def test_screener_access(self):
        """Test if we can access the screener page"""
        try:
            # Try accessing a simpler screener URL first
            test_url = "https://elite.finviz.com/screener.ashx"
            
            test_headers = self.headers.copy()
            test_headers['Referer'] = "https://elite.finviz.com/"
            
            async with self.session.get(test_url, headers=test_headers) as response:
                if response.status == 200:
                    logger.info("Successfully accessed basic screener page")
                    return True
                else:
                    logger.warning(f"Test screener access failed: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error testing screener access: {e}")
            return False

    def parse_table_value(self, value_str: str) -> float:
        """Parse table values (handle B, M, K suffixes and percentages)"""
        if not value_str or value_str == '-' or value_str == 'N/A':
            return 0.0
            
        # Remove percentage sign
        value_str = value_str.replace('%', '')
        
        # Handle multipliers
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

    async def scrape_screener_page(self, page_url: str) -> List[Dict[str, Any]]:
        """Scrape a single page of the screener results with better error handling"""
        tickers = []
        
        try:
            # Add appropriate headers
            scraper_headers = self.headers.copy()
            scraper_headers['Referer'] = "https://elite.finviz.com/"
            
            async with self.session.get(page_url, headers=scraper_headers) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Skip login detection since we're not logging in
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for the screener table - try different possible selectors
                    table = soup.find('table', {'class': 'screener_table'})
                    if not table:
                        # Try alternative selectors
                        table = soup.find('table', {'width': '100%'})
                        if not table:
                            logger.warning("Could not find screener table on page")
                            logger.debug(f"Page content preview: {html[:1000]}")
                            return tickers
                    
                    # Get all rows
                    rows = table.find_all('tr')
                    if len(rows) < 2:
                        logger.warning("Not enough rows in table")
                        return tickers
                    
                    # Get header row to map column indices
                    header_row = rows[0]
                    headers = [th.get_text().strip() for th in header_row.find_all(['td', 'th'])]
                    
                    # Create column mapping (case insensitive)
                    col_map = {}
                    for i, header in enumerate(headers):
                        col_map[header.lower()] = i
                    
                    logger.info(f"Found table with columns: {headers}")
                    
                    # Process data rows
                    data_rows = rows[1:]  # Skip header
                    
                    for row in data_rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) < len(headers):
                            continue
                            
                        try:
                            # Extract ticker (usually first or second column)
                            ticker_text = cells[col_map.get('ticker', 1)].get_text().strip()
                            if not ticker_text or len(ticker_text) > 10:  # Skip invalid tickers
                                continue
                            
                            # Apply 2-4 letter restriction - only store tickers with 2, 3, or 4 letters
                            if len(ticker_text) < 2 or len(ticker_text) > 4:
                                continue
                            
                            ticker_data = {
                                'ticker': ticker_text,
                                'company_name': cells[col_map.get('company', 2)].get_text().strip() if col_map.get('company', 2) < len(cells) else '',
                                'sector': cells[col_map.get('sector', 3)].get_text().strip() if col_map.get('sector', 3) < len(cells) else '',
                                'industry': cells[col_map.get('industry', 4)].get_text().strip() if col_map.get('industry', 4) < len(cells) else '',
                                'country': cells[col_map.get('country', 5)].get_text().strip() if col_map.get('country', 5) < len(cells) else '',
                                'market_cap': self.parse_table_value(cells[col_map.get('market cap', 6)].get_text().strip() if col_map.get('market cap', 6) < len(cells) else '0'),
                                'price': self.parse_table_value(cells[col_map.get('price', 7)].get_text().strip() if col_map.get('price', 7) < len(cells) else '0'),
                                'volume': int(self.parse_table_value(cells[col_map.get('volume', 8)].get_text().strip() if col_map.get('volume', 8) < len(cells) else '0')),
                                'float_shares': self.parse_table_value(cells[col_map.get('float', 9)].get_text().strip() if col_map.get('float', 9) < len(cells) else '0'),
                                'pe_ratio': self.parse_table_value(cells[col_map.get('p/e', 10)].get_text().strip() if col_map.get('p/e', 10) < len(cells) else '0'),
                                'eps': self.parse_table_value(cells[col_map.get('eps (ttm)', 11)].get_text().strip() if col_map.get('eps (ttm)', 11) < len(cells) else '0'),
                                'analyst_rating': cells[col_map.get('analyst recom', 12)].get_text().strip() if col_map.get('analyst recom', 12) and col_map.get('analyst recom', 12) < len(cells) else '',
                                'insider_ownership': self.parse_table_value(cells[col_map.get('insider own', 13)].get_text().strip() if col_map.get('insider own', 13) and col_map.get('insider own', 13) < len(cells) else '0'),
                                'institutional_ownership': self.parse_table_value(cells[col_map.get('inst own', 14)].get_text().strip() if col_map.get('inst own', 14) and col_map.get('inst own', 14) < len(cells) else '0')
                            }
                            
                            # Filter out Malaysian stocks and only add valid tickers
                            if (ticker_data['ticker'] and 
                                ticker_data['ticker'] != '-' and 
                                ticker_data['country'].lower() != 'malaysia'):
                                tickers.append(ticker_data)
                                
                        except Exception as e:
                            logger.warning(f"Error parsing row: {e}")
                            continue
                    
                    logger.info(f"Scraped {len(tickers)} tickers from page")
                    
                elif response.status == 403:
                    logger.error("403 Forbidden - Finviz may be blocking automated access")
                    return tickers
                else:
                    logger.error(f"Failed to fetch screener page: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error scraping screener page: {e}")
            
        return tickers

    async def scrape_all_pages(self) -> List[Dict[str, Any]]:
        """Scrape all pages of screener results from both URLs with better error handling"""
        all_tickers = []
        
        # Skip the test access since we're not logging in
        logger.info("Starting direct scraping without authentication")
        
        # URLs to scrape
        screener_urls = [
            ("price under $3", self.screener_url_1),
            ("price $3 to $10", self.screener_url_2)
        ]
        
        # Scrape both URLs sequentially
        for url_desc, base_url in screener_urls:
            logger.info(f"Starting to scrape {url_desc} stocks...")
            page_num = 1
            url_tickers = 0
            
            while True:
                # Construct URL for current page
                if page_num == 1:
                    page_url = base_url
                else:
                    page_url = f"{base_url}&r={(page_num-1)*20+1}"
                
                logger.info(f"Scraping {url_desc} page {page_num}: {page_url}")
                
                page_tickers = await self.scrape_screener_page(page_url)
                
                if not page_tickers:
                    logger.info(f"No more data found on {url_desc} page {page_num}, stopping")
                    break
                    
                all_tickers.extend(page_tickers)
                url_tickers += len(page_tickers)
                page_num += 1
                
                # Safety limit per URL
                if page_num > 50:  # Reduced from 100 to be more conservative
                    logger.warning(f"Reached page limit of 50 for {url_desc}, stopping")
                    break
                    
                # Longer delay between requests to avoid rate limiting
                await asyncio.sleep(2)
            
            logger.info(f"Completed scraping {url_desc}: {url_tickers} tickers found")
        
        logger.info(f"Total tickers scraped from both URLs: {len(all_tickers)}")
        return all_tickers

    async def update_ticker_database(self):
        """Main function to update ticker database with retry logic"""
        logger.info("Starting Finviz ticker database update")
        
        max_attempts = 3
        attempt = 1
        
        while attempt <= max_attempts:
            logger.info(f"Attempt {attempt} of {max_attempts} to update ticker database")
            
            # Create HTTP session with longer timeout
            timeout = aiohttp.ClientTimeout(total=60)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
            try:
                # Skip login - the screener URL works without authentication
                logger.info("Skipping login - accessing public screener directly")
                
                # Small delay before starting
                await asyncio.sleep(1)
                
                # Scrape all tickers
                tickers = await self.scrape_all_pages()
                
                if not tickers:
                    logger.error(f"No tickers scraped on attempt {attempt}")
                    raise Exception("No tickers scraped from Finviz")
                
                # Drop the existing float_list table to refresh data
                logger.info("Dropping existing float_list table for complete refresh")
                self.ch_manager.drop_float_list_table()
                
                # Recreate the float_list table
                logger.info("Recreating float_list table")
                self.ch_manager.create_float_list_table()
                
                # Insert into ClickHouse
                inserted_count = self.ch_manager.insert_tickers(tickers)
                
                if inserted_count == 0:
                    logger.error(f"No tickers inserted into database on attempt {attempt}")
                    raise Exception("No tickers inserted into database")
                
                logger.info(f"Successfully updated {inserted_count} tickers in database on attempt {attempt}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating ticker database on attempt {attempt}: {e}")
                
                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} attempts failed. Ticker database update failed permanently.")
                    return False
                else:
                    logger.info(f"Retrying in 5 seconds... (attempt {attempt + 1} of {max_attempts})")
                    await asyncio.sleep(5)
                    
            finally:
                if self.session:
                    await self.session.close()
                    
            attempt += 1
        
        return False

async def main():
    """Main function to run the Finviz scraper"""
    # Setup ClickHouse
    ch_manager = setup_clickhouse_database()
    
    # Create and run scraper
    scraper = FinvizScraper(ch_manager)
    
    try:
        success = await scraper.update_ticker_database()
        if success:
            logger.info("Ticker database update completed successfully")
        else:
            logger.error("Ticker database update failed")
    except KeyboardInterrupt:
        logger.info("Scraper stopped by user")
    finally:
        ch_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 