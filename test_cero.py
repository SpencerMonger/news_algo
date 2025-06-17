#!/usr/bin/env python3
"""
Test script to find the CERO article across ALL pages
and test direct article access
"""

import asyncio
import re
from crawl4ai import AsyncWebCrawler, CrawlResult
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_cero_multipage():
    """Test ALL pages and direct article access"""
    
    # The specific CERO article URL
    direct_cero_url = "https://www.globenewswire.com/news-release/2025/06/17/3100623/0/en/CERo-Therapeutics-Holdings-Inc-Announces-FDA-Orphan-Drug-Designation-Granted-to-CER-1236-for-the-Treatment-of-Acute-Myeloid-Leukemia-AML.html"
    
    crawler = None
    try:
        # Initialize crawler
        crawler = AsyncWebCrawler(
            verbose=True,
            headless=True,
            browser_type="chromium"
        )
        await crawler.start()
        
        # First, test direct access to the CERO article
        logger.info("🎯 TESTING DIRECT ACCESS TO CERO ARTICLE")
        logger.info(f"URL: {direct_cero_url}")
        
        result = await crawler.arun(
            url=direct_cero_url,
            delay_before_return_html=2.0,
            timeout=30
        )
        
        if result.success and result.html:
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Get the article title and content
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "No title found"
            
            # Look for the ticker information in the article body
            article_text = soup.get_text()
            
            logger.info(f"✅ DIRECT ARTICLE ACCESS SUCCESS")
            logger.info(f"Title: {title}")
            logger.info(f"Article contains '(Nasdaq: CERO)': {'(Nasdaq: CERO)' in article_text}")
            logger.info(f"Article contains 'CERO': {'CERO' in article_text}")
            
            # Test ticker patterns on the full article
            test_ticker_patterns_on_full_article(article_text)
            
        else:
            logger.error(f"❌ Failed to access direct article: {result.error_message}")
        
        logger.info("\n" + "="*80)
        logger.info("🔍 TESTING ALL LISTING PAGES")
        
        # Test ALL pages of the 24-hour listing until we find CERO or hit max pages
        page_num = 1
        max_pages = 20  # Safety limit to prevent infinite loop
        found_cero = False
        
        while page_num <= max_pages and not found_cero:
            url = f"https://www.globenewswire.com/en/search/date/24HOURS?pageSize=50&page={page_num}"
            logger.info(f"\n📄 Checking page {page_num}: {url}")
            
            result = await crawler.arun(
                url=url,
                wait_for="css:.news-item, .search-result",
                delay_before_return_html=2.0,
                timeout=30
            )
            
            if not result.success or not result.html:
                logger.error(f"❌ Failed to scrape page {page_num}: {result.error_message}")
                page_num += 1
                continue
            
            soup = BeautifulSoup(result.html, 'html.parser')
            article_links = soup.find_all('a', href=True)
            
            logger.info(f"📰 Found {len(article_links)} total links on page {page_num}")
            
            # Look for CERO article
            for link in article_links:
                url_href = link.get('href', '')
                title = link.get_text(strip=True)
                
                # Check for CERO article by ID or title content
                if ("3100623" in url_href or 
                    ("cero" in title.lower() and len(title) > 20) or
                    ("cer-1236" in title.lower()) or
                    ("orphan drug" in title.lower() and "cero" in title.lower())):
                    
                    logger.info(f"🎯 FOUND CERO ARTICLE ON PAGE {page_num}!")
                    logger.info(f"   Title: {title}")
                    logger.info(f"   URL: {url_href}")
                    
                    # Get description
                    description = ""
                    parent = link.parent
                    if parent:
                        desc_elem = parent.find('p') or parent.find('div', class_='summary') or parent.find('div', class_='description')
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                    
                    logger.info(f"   Description: '{description}'")
                    combined_text = f"{title} {description}"
                    logger.info(f"   Combined: '{combined_text}'")
                    
                    # Test patterns
                    test_ticker_patterns(combined_text)
                    found_cero = True
                    break
            
            if not found_cero:
                logger.info(f"❌ CERO article not found on page {page_num}")
                
                # Show a few sample articles from this page for debugging
                sample_count = 0
                logger.info(f"📋 Sample articles from page {page_num}:")
                for link in article_links:
                    title = link.get_text(strip=True)
                    url_href = link.get('href', '')
                    if (title and len(title) > 20 and 
                        any(x in url_href for x in ['news-release', 'story', 'releases'])):
                        logger.info(f"   {sample_count+1}. {title[:60]}...")
                        sample_count += 1
                        if sample_count >= 3:  # Show first 3 articles
                            break
            
            page_num += 1
        
        if not found_cero:
            logger.warning(f"❌ CERO article not found in any of the {page_num-1} pages checked")
            logger.info("This confirms the article is NOT in the GlobeNewswire 24-hour search results")
            logger.info("Your scraper will miss this article because it only checks these listing pages")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        
    finally:
        if crawler:
            await crawler.close()

def test_ticker_patterns(text):
    """Test ticker patterns on listing text"""
    ticker = "CERO"
    ticker_escaped = re.escape(ticker)
    
    patterns = [
        (rf':\s*{ticker_escaped}\b', "Exchange colon pattern"),
        (rf'"{ticker_escaped}"', "Quoted pattern"),
        (rf'\([^)]*:\s*{ticker_escaped}\)', "Parenthetical with exchange"),
        (rf':\s*"{ticker_escaped}"', "Exchange with quotes"),
        (rf'\({ticker_escaped}\)', "Parenthetical ticker only"),
    ]
    
    logger.info(f"🔍 Testing ticker patterns on listing text: '{text[:100]}...'")
    
    found_any = False
    for pattern, description in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.info(f"   ✅ MATCH: {description}")
            found_any = True
        else:
            logger.info(f"   ❌ NO MATCH: {description}")
    
    if not found_any:
        logger.warning("   ⚠️  NO PATTERNS MATCHED in listing text")

def test_ticker_patterns_on_full_article(article_text):
    """Test ticker patterns on full article text"""
    ticker = "CERO"
    ticker_escaped = re.escape(ticker)
    
    patterns = [
        (rf':\s*{ticker_escaped}\b', "Exchange colon pattern"),
        (rf'"{ticker_escaped}"', "Quoted pattern"),
        (rf'\([^)]*:\s*{ticker_escaped}\)', "Parenthetical with exchange"),
        (rf':\s*"{ticker_escaped}"', "Exchange with quotes"),
        (rf'\({ticker_escaped}\)', "Parenthetical ticker only"),
    ]
    
    logger.info(f"🔍 Testing ticker patterns on FULL ARTICLE text:")
    
    found_any = False
    for pattern, description in patterns:
        matches = re.findall(pattern, article_text, re.IGNORECASE)
        if matches:
            logger.info(f"   ✅ MATCH: {description} - Found: {matches}")
            found_any = True
        else:
            logger.info(f"   ❌ NO MATCH: {description}")
    
    if found_any:
        logger.info("   ✅ TICKER PATTERNS WORK ON FULL ARTICLE!")
    else:
        logger.warning("   ⚠️  NO PATTERNS MATCHED even in full article")

if __name__ == "__main__":
    asyncio.run(test_cero_multipage())