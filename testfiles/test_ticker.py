#!/usr/bin/env python3
"""
Test script to see what title and description text the scraper gets
for the SPECIFIC CERO article from GlobeNewswire listing page
"""

import asyncio
import re
from crawl4ai import AsyncWebCrawler, CrawlResult
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_specific_cero_article():
    """Test what text we can extract from GlobeNewswire listing for the SPECIFIC CERO article"""
    
    # The specific CERO article URL we're looking for
    target_cero_url = "/news-release/2025/06/17/3100623/0/en/CERo-Therapeutics-Holdings-Inc-Announces-FDA-Orphan-Drug-Designation-Granted-to-CER-1236-for-the-Treatment-of-Acute-Myeloid-Leukemia-AML.html"
    
    crawler = None
    try:
        # Initialize crawler
        crawler = AsyncWebCrawler(
            verbose=True,
            headless=True,
            browser_type="chromium"
        )
        await crawler.start()
        
        # Scrape GlobeNewswire 24-hour page
        url = "https://www.globenewswire.com/en/search/date/24HOURS?pageSize=50&page=1"
        logger.info(f"🔍 Scraping: {url}")
        logger.info(f"🎯 Looking for CERO article: {target_cero_url}")
        
        result: CrawlResult = await crawler.arun(
            url=url,
            wait_for="css:.news-item, .search-result",
            delay_before_return_html=2.0,
            timeout=30
        )
        
        if not result.success or not result.html:
            logger.error(f"❌ Failed to scrape: {result.error_message}")
            return
        
        soup = BeautifulSoup(result.html, 'html.parser')
        article_links = soup.find_all('a', href=True)
        
        logger.info(f"📰 Found {len(article_links)} total links")
        
        # Look for the SPECIFIC CERO article
        found_cero = False
        
        for link in article_links:
            title = link.get_text(strip=True)
            url_href = link.get('href', '')
            
            if not title or len(title) < 20:
                continue
            
            # Check if this is the SPECIFIC CERO article we want
            if target_cero_url in url_href or "3100623" in url_href:
                
                logger.info(f"🎯 FOUND THE SPECIFIC CERO ARTICLE!")
                logger.info(f"   Title: {title}")
                logger.info(f"   URL: {url_href}")
                
                # Get description from parent elements (same logic as scraper)
                description = ""
                parent = link.parent
                if parent:
                    desc_elem = parent.find('p') or parent.find('div', class_='summary') or parent.find('div', class_='description')
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)
                
                logger.info(f"   Description: '{description}'")
                logger.info(f"   Combined Text: '{title} {description}'")
                
                # Test ticker extraction patterns on this EXACT text
                combined_text = f"{title} {description}"
                logger.info("=" * 80)
                test_ticker_patterns(combined_text)
                logger.info("=" * 80)
                
                found_cero = True
                break
            
            # Also check if title contains CERO-related keywords
            elif any(keyword in title.lower() for keyword in ['cero', 'cer-1236', 'orphan drug']):
                logger.info(f"📋 CERO-related article found:")
                logger.info(f"   Title: {title}")
                logger.info(f"   URL: {url_href}")
        
        if not found_cero:
            logger.warning("❌ SPECIFIC CERO ARTICLE NOT FOUND in listing!")
            logger.info("This could mean:")
            logger.info("1. The article is not in the current 24-hour listing")
            logger.info("2. It's on a different page")
            logger.info("3. The URL structure changed")
            
            # Show some articles for debugging
            logger.info("\n📋 First 10 articles in listing:")
            count = 0
            for link in article_links:
                title = link.get_text(strip=True)
                url_href = link.get('href', '')
                
                if title and len(title) > 20 and any(x in url_href for x in ['news-release', 'story', 'releases']):
                    logger.info(f"   {count+1}. {title[:80]}...")
                    logger.info(f"      URL: {url_href}")
                    count += 1
                    if count >= 10:
                        break
            
    except Exception as e:
        logger.error(f"Error during test: {e}")
        
    finally:
        if crawler:
            await crawler.close()

def test_ticker_patterns(text):
    """Test the ticker extraction patterns on the given text"""
    ticker = "CERO"
    ticker_escaped = re.escape(ticker)
    
    patterns = [
        # Exchange patterns: ":TICKER" (e.g., "Nasdaq: STSS", "NYSE: AAPL")
        (rf':\s*{ticker_escaped}\b', "Exchange colon pattern"),
        
        # Quoted pattern: "TICKER" (e.g., "STSS" and "STSSW")
        (rf'"{ticker_escaped}"', "Quoted pattern"),
        
        # Parenthetical with exchange: (Exchange: TICKER) (e.g., "(NYSE: AAPL)")
        (rf'\([^)]*:\s*{ticker_escaped}\)', "Parenthetical with exchange"),
        
        # Exchange with quotes: ": "TICKER"" (e.g., ': "STSS"')
        (rf':\s*"{ticker_escaped}"', "Exchange with quotes"),
        
        # Parenthetical ticker only: (TICKER) - but only if 3+ chars to avoid common words
        (rf'\({ticker_escaped}\)', "Parenthetical ticker only"),
    ]
    
    logger.info(f"🔍 TESTING TICKER PATTERNS on EXACT CERO article text:")
    logger.info(f"Text: '{text}'")
    logger.info(f"Text length: {len(text)} characters")
    
    found_any = False
    for pattern, description in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.info(f"   ✅ MATCH: {description} - Pattern: {pattern}")
            found_any = True
        else:
            logger.info(f"   ❌ NO MATCH: {description} - Pattern: {pattern}")
    
    if not found_any:
        logger.warning(f"   ⚠️  NO TICKER PATTERNS MATCHED for CERO in the listing text")
        logger.warning(f"   This confirms the ticker info is NOT in the listing page text!")
        logger.warning(f"   The '(Nasdaq: CERO)' must be in the FULL ARTICLE BODY only.")

if __name__ == "__main__":
    asyncio.run(test_specific_cero_article())