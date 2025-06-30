#!/usr/bin/env python3
"""
Test script to debug GlobeNewswire RSS feed and web scraping for ARTL article
Checks both RSS feed and web scraping to see why ARTL wasn't detected
"""

import asyncio
import aiohttp
import feedparser
import re
from datetime import datetime
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlResult
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARTLTester:
    def __init__(self):
        self.target_article_url = "https://www.globenewswire.com/news-release/2025/06/30/3107392/0/en/Artelo-Biosciences-Announces-Positive-First-in-Human-Data-for-ART26-12-a-Novel-Non-Opioid-Treatment-Candidate-for-Persistent-Pain.html"
        self.rss_url = "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20Releases"
        self.web_scrape_url = "https://www.globenewswire.com/newsroom"
        self.ticker_list = ["ARTL"]  # Just test ARTL
        
    def extract_tickers_from_text(self, text: str) -> list:
        """Extract tickers using same logic as web_scraper.py"""
        if not text:
            return []
            
        found_tickers = []
        
        for ticker in self.ticker_list:
            ticker_escaped = re.escape(ticker)
            
            patterns = [
                rf':\s*{ticker_escaped}\b',  # ":TICKER"
                rf'"{ticker_escaped}"',      # "TICKER"
                rf'\([^)]*:\s*{ticker_escaped}\)',  # (Exchange: TICKER)
                rf':\s*"{ticker_escaped}"',  # ': "TICKER"'
                rf'\({ticker_escaped}\)' if len(ticker) >= 3 else None  # (TICKER)
            ]
            
            valid_patterns = [p for p in patterns if p is not None]
            for pattern in valid_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_tickers.append(ticker)
                    logger.info(f"✅ Found ticker {ticker} using pattern: {pattern}")
                    break
        
        return found_tickers
    
    async def test_rss_feed(self):
        """Test GlobeNewswire RSS feed for ARTL article"""
        logger.info("🔍 Testing GlobeNewswire RSS feed...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.rss_url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"❌ RSS feed failed: HTTP {response.status}")
                        return
                    
                    rss_content = await response.text()
                    feed = feedparser.parse(rss_content)
                    
                    logger.info(f"📰 RSS feed has {len(feed.entries)} entries")
                    
                    artl_found = False
                    for i, entry in enumerate(feed.entries):
                        title = entry.get('title', '').strip()
                        url = entry.get('link', '').strip()
                        description = entry.get('description', '').strip()
                        published = entry.get('published', '')
                        
                        # Check if this is the ARTL article
                        if self.target_article_url in url or "Artelo" in title or "ARTL" in title:
                            artl_found = True
                            logger.info(f"🎯 FOUND ARTL in RSS entry #{i+1}:")
                            logger.info(f"   Title: {title}")
                            logger.info(f"   URL: {url}")
                            logger.info(f"   Published: {published}")
                            logger.info(f"   Description: {description[:200]}...")
                            
                            # Test ticker extraction
                            text_to_search = f"{title} {description}"
                            found_tickers = self.extract_tickers_from_text(text_to_search)
                            logger.info(f"   Tickers found: {found_tickers}")
                            
                        # Also check for any ARTL mentions
                        text_to_search = f"{title} {description}"
                        if "ARTL" in text_to_search or "Artelo" in text_to_search:
                            logger.info(f"📋 RSS Entry #{i+1} has ARTL/Artelo mention:")
                            logger.info(f"   Title: {title}")
                            logger.info(f"   Published: {published}")
                    
                    if not artl_found:
                        logger.warning("❌ ARTL article NOT found in RSS feed")
                        logger.info("📋 First 5 RSS entries:")
                        for i, entry in enumerate(feed.entries[:5]):
                            logger.info(f"   #{i+1}: {entry.get('title', '')} - {entry.get('published', '')}")
                        
        except Exception as e:
            logger.error(f"Error testing RSS feed: {e}")
    
    async def test_web_scraping(self):
        """Test GlobeNewswire web scraping for ARTL article"""
        logger.info("🔍 Testing GlobeNewswire web scraping...")
        
        crawler = None
        try:
            crawler = AsyncWebCrawler(
                verbose=False,
                headless=True,
                browser_type="chromium"
            )
            await crawler.start()
            
            result: CrawlResult = await crawler.arun(
                url=self.web_scrape_url,
                wait_for="css:.news-item, .search-result, .bw-release-story, .newsreleaseheadline",
                delay_before_return_html=2.0,
                timeout=30
            )
            
            if not result.success or not result.html:
                logger.error(f"❌ Web scraping failed: {result.error_message}")
                return
            
            soup = BeautifulSoup(result.html, 'html.parser')
            article_links = soup.find_all('a', href=True)
            
            logger.info(f"📰 Found {len(article_links)} potential articles from web scraping")
            
            artl_found = False
            valid_articles = 0
            
            for link in article_links:
                title = link.get_text(strip=True)
                url = link.get('href', '')
                
                if not title or not url or len(title) < 20:
                    continue
                
                # Make URL absolute
                if url.startswith('/'):
                    url = "https://www.globenewswire.com" + url
                
                # Skip non-news URLs
                if not any(x in url for x in ['news-release', 'story', 'releases']):
                    continue
                
                valid_articles += 1
                
                # Check if this is the ARTL article
                if self.target_article_url in url or "Artelo" in title or "ARTL" in title:
                    artl_found = True
                    logger.info(f"🎯 FOUND ARTL in web scraping:")
                    logger.info(f"   Title: {title}")
                    logger.info(f"   URL: {url}")
                    
                    # Test ticker extraction
                    found_tickers = self.extract_tickers_from_text(title)
                    logger.info(f"   Tickers found: {found_tickers}")
                
                # Also check for any ARTL mentions
                if "ARTL" in title or "Artelo" in title:
                    logger.info(f"📋 Web scraping found ARTL/Artelo mention:")
                    logger.info(f"   Title: {title}")
                    logger.info(f"   URL: {url}")
            
            logger.info(f"📊 Web scraping found {valid_articles} valid news articles")
            
            if not artl_found:
                logger.warning("❌ ARTL article NOT found in web scraping")
                logger.info("📋 First 5 valid articles:")
                count = 0
                for link in article_links:
                    if count >= 5:
                        break
                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    if title and url and len(title) >= 20 and any(x in url for x in ['news-release', 'story', 'releases']):
                        if url.startswith('/'):
                            url = "https://www.globenewswire.com" + url
                        logger.info(f"   #{count+1}: {title}")
                        count += 1
                        
        except Exception as e:
            logger.error(f"Error testing web scraping: {e}")
        finally:
            if crawler:
                await crawler.close()
    
    async def test_direct_article_access(self):
        """Test direct access to the ARTL article to verify it exists"""
        logger.info("🔍 Testing direct access to ARTL article...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.target_article_url, timeout=30) as response:
                    if response.status == 200:
                        logger.info("✅ ARTL article is accessible directly")
                        content = await response.text()
                        
                        # Check for ARTL ticker mentions
                        if "ARTL" in content:
                            logger.info("✅ Article contains 'ARTL' text")
                        if "(Nasdaq: ARTL)" in content:
                            logger.info("✅ Article contains '(Nasdaq: ARTL)' format")
                        if "Artelo" in content:
                            logger.info("✅ Article contains 'Artelo' company name")
                            
                        # Test ticker extraction on full content
                        found_tickers = self.extract_tickers_from_text(content)
                        logger.info(f"✅ Ticker extraction from full article: {found_tickers}")
                        
                    else:
                        logger.error(f"❌ Cannot access ARTL article: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error testing direct article access: {e}")
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting ARTL debugging tests...")
        logger.info(f"🎯 Target article: {self.target_article_url}")
        logger.info("=" * 80)
        
        await self.test_direct_article_access()
        logger.info("=" * 80)
        
        await self.test_rss_feed()
        logger.info("=" * 80)
        
        await self.test_web_scraping()
        logger.info("=" * 80)
        
        logger.info("✅ All tests completed!")

async def main():
    tester = ARTLTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())