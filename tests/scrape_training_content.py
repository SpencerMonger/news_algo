#!/usr/bin/env python3
"""
Scrape Full Article Content for RAG Training Dataset

This script fetches the full article content from URLs in the rag_training_dataset table
and updates the full_content column with the scraped text. This is needed because the
training dataset currently only has headlines, not the full article body text required
for meaningful vector generation.

Uses Crawl4AI like the live system for successful scraping.

Usage:
    python3 tests/scrape_training_content.py --batch-size 50 --max-chars 6000
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
from bs4 import BeautifulSoup

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from crawl4ai import AsyncWebCrawler, CrawlResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingContentScraper:
    """Scrape full article content for training dataset articles using Crawl4AI"""
    
    def __init__(self, batch_size: int = 50, max_chars: int = 4500):  # 6K tokens ≈ 4500 chars
        self.ch_manager = ClickHouseManager()
        self.batch_size = batch_size
        self.max_chars = max_chars  # 6K token limit ≈ 4500 characters
        self.crawler: Optional[AsyncWebCrawler] = None
        
        # Stats tracking
        self.stats = {
            'total_articles': 0,
            'articles_with_urls': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'updated_articles': 0
        }
        
    async def initialize(self):
        """Initialize the scraper with Crawl4AI"""
        logger.info("🧪 Initializing Training Content Scraper with Crawl4AI...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize Crawl4AI AsyncWebCrawler - SAME CONFIG AS WEB_SCRAPER.PY
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
                logger.info(f"✅ Crawl4AI browser started successfully (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.warning(f"❌ Failed to start Crawl4AI browser (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize Crawl4AI after {max_retries} attempts")
                await asyncio.sleep(2)  # Quick retry delay
        
        logger.info("✅ Training content scraper initialized successfully")
    
    async def scrape_article_content(self, url: str) -> str:
        """
        Scrape article content from URL using Crawl4AI + AGGRESSIVE CLEAN EXTRACTION
        Focus ONLY on main article body paragraphs, filter out all navigation/page elements
        """
        try:
            result: CrawlResult = await self.crawler.arun(
                url=url,
                delay_before_return_html=1.0,
                timeout=15
            )
            
            if not result.success or not result.html:
                logger.error(f"❌ Failed to scrape {url}: {result.error_message}")
                return ""
            
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # AGGRESSIVE CLEANUP - Remove ALL non-content elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", 
                               "iframe", "form", "button", "input", "select", "textarea",
                               "noscript", "meta", "link", "title"]):
                element.decompose()
            
            # Remove common navigation and UI elements by class/id
            nav_selectors = [
                '[class*="nav"]', '[class*="menu"]', '[class*="header"]', '[class*="footer"]',
                '[class*="sidebar"]', '[class*="widget"]', '[class*="ad"]', '[class*="banner"]',
                '[class*="social"]', '[class*="share"]', '[class*="comment"]', '[class*="related"]',
                '[class*="breadcrumb"]', '[class*="pagination"]', '[class*="tag"]',
                '[id*="nav"]', '[id*="menu"]', '[id*="header"]', '[id*="footer"]',
                '[id*="sidebar"]', '[id*="ad"]', '[id*="banner"]'
            ]
            
            for selector in nav_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # FOCUSED ARTICLE EXTRACTION - Try specific article content selectors first
            article_content = ""
            
            # Site-specific extraction logic
            if 'benzinga.com' in url:
                # Benzinga: Target specific article content classes
                article_paragraphs = soup.find_all('p')
                content_paragraphs = []
                
                for p in article_paragraphs:
                    if p.parent and p.parent.get('class'):
                        parent_classes = p.parent.get('class', [])
                        if any('cAazyy' in str(cls) or 'dIYChw' in str(cls) for cls in parent_classes):
                            text = p.get_text().strip()
                            if len(text) > 30:  # Only substantial paragraphs
                                content_paragraphs.append(text)
                
                article_content = ' '.join(content_paragraphs)
            
            elif 'finviz.com' in url:
                # Finviz: Look for main content area
                main_content = soup.find('div', class_='t-text-gray-700') or soup.find('div', class_='news-content')
                if main_content:
                    paragraphs = main_content.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
            
            elif 'globenewswire.com' in url:
                # GlobeNewswire: Target press release body
                release_body = soup.find('div', class_='bw-release-story') or soup.find('div', class_='article-wrap')
                if release_body:
                    paragraphs = release_body.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
            
            elif 'businesswire.com' in url:
                # BusinessWire: Target release body
                release_body = soup.find('div', class_='bw-release-story') or soup.find('[data-module="ArticleBody"]')
                if release_body:
                    paragraphs = release_body.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
            
            elif 'prnewswire.com' in url:
                # PRNewswire: Target story content
                story_content = soup.find('div', class_='prnews-story') or soup.find('section', class_='release-body')
                if story_content:
                    paragraphs = story_content.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
            
            elif 'accesswire.com' in url:
                # AccessWire: Target article content
                content_div = soup.find('div', class_='news-content') or soup.find('div', class_='article-content')
                if content_div:
                    paragraphs = content_div.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
            
            elif 'yahoo.com' in url:
                # Yahoo Finance: Target story body
                story_body = soup.find('div', class_='caas-body') or soup.find('div', class_='canvas-body')
                if story_body:
                    paragraphs = story_body.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
            
            # FALLBACK: Generic article extraction if site-specific fails
            if not article_content or len(article_content) < 200:
                # Try common article selectors in priority order
                selectors_to_try = [
                    'article p',  # Paragraphs within article tags
                    '.article-content p', '.story-body p', '.post-content p',
                    '.content p', '.article-body p', '.entry-content p',
                    '.news-content p', '.release-body p',
                    'article',  # Full article tag
                    '.article-content', '.story-body', '.post-content',
                    '.content', '.article-body', '.entry-content'
                ]
                
                for selector in selectors_to_try:
                    elements = soup.select(selector)
                    if elements:
                        if 'p' in selector:
                            # Extract text from paragraphs
                            texts = [elem.get_text().strip() for elem in elements if len(elem.get_text().strip()) > 30]
                            article_content = ' '.join(texts)
                        else:
                            # Extract text from container
                            article_content = elements[0].get_text()
                        
                        if len(article_content) > 200:
                            break
                
                # LAST RESORT: Extract all meaningful paragraphs from page
                if not article_content or len(article_content) < 200:
                    all_paragraphs = soup.find_all('p')
                    meaningful_paragraphs = []
                    
                    for p in all_paragraphs:
                        text = p.get_text().strip()
                        # Filter out short paragraphs and common navigation text
                        if (len(text) > 50 and 
                            not any(skip_word in text.lower() for skip_word in 
                                   ['subscribe', 'follow us', 'share', 'tweet', 'facebook', 
                                    'linkedin', 'instagram', 'newsletter', 'cookie', 'privacy',
                                    'terms of service', 'all rights reserved', 'copyright',
                                    'click here', 'read more', 'view all', 'see all'])):
                            meaningful_paragraphs.append(text)
                    
                    article_content = ' '.join(meaningful_paragraphs)
            
            # FINAL CLEANUP - Remove extra whitespace and limit length
            if article_content:
                # Clean up whitespace
                lines = (line.strip() for line in article_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit to max_chars (6K token limit ≈ 4500 characters)
                if len(clean_content) > self.max_chars:
                    clean_content = clean_content[:self.max_chars]
                
                logger.debug(f"Scraped clean content: {len(clean_content)} characters from {url}")
                return clean_content
            else:
                logger.warning(f"No meaningful content extracted from {url}")
                return ""
            
        except Exception as e:
            logger.error(f"Error scraping content from {url}: {e}")
            return ""
    
    async def get_articles_needing_content(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get articles that need content scraping - ALL OF THEM since none have body content"""
        try:
            query = """
            SELECT 
                ticker,
                headline,
                full_content,
                article_url,
                outcome_type,
                original_content_hash
            FROM News.rag_training_dataset
            WHERE article_url IS NOT NULL 
              AND article_url != ''
            ORDER BY outcome_type, ticker
            LIMIT %s OFFSET %s
            """
            
            result = self.ch_manager.client.query(query, parameters=[limit, offset])
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'full_content': row[2],
                    'article_url': row[3],
                    'outcome_type': row[4],
                    'original_content_hash': row[5]
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching articles needing content: {e}")
            return []
    
    async def update_article_content(self, content_hash: str, scraped_content: str) -> bool:
        """Update article with scraped content"""
        try:
            update_query = """
            ALTER TABLE News.rag_training_dataset
            UPDATE full_content = %s
            WHERE original_content_hash = %s
            """
            
            self.ch_manager.client.command(update_query, parameters=[scraped_content, content_hash])
            return True
            
        except Exception as e:
            logger.error(f"Error updating article content for hash {content_hash}: {e}")
            return False
    
    async def get_content_stats(self):
        """Get statistics about content availability"""
        try:
            stats_query = """
            SELECT 
                outcome_type,
                COUNT(*) as total,
                COUNT(CASE WHEN article_url IS NOT NULL AND article_url != '' THEN 1 END) as has_url
            FROM News.rag_training_dataset 
            GROUP BY outcome_type
            ORDER BY outcome_type
            """
            
            result = self.ch_manager.client.query(stats_query)
            
            logger.info("📊 Articles to Scrape by Outcome Type:")
            total_to_scrape = 0
            
            for row in result.result_rows:
                outcome, total, has_url = row
                logger.info(f"  • {outcome}:")
                logger.info(f"    - Total articles: {total}")
                logger.info(f"    - Has URL: {has_url} ({has_url/total*100:.1f}%)")
                logger.info(f"    - Will scrape: {has_url}")
                total_to_scrape += has_url
            
            logger.info(f"📋 Total articles to scrape: {total_to_scrape}")
            return total_to_scrape
            
        except Exception as e:
            logger.error(f"Error getting content stats: {e}")
            return 0
    
    async def scrape_all_missing_content(self):
        """Scrape content for ALL articles since none have body content"""
        logger.info("🚀 Starting content scraping for ALL training dataset articles...")
        
        # Get statistics
        total_needing_scraping = await self.get_content_stats()
        
        if total_needing_scraping == 0:
            logger.info("❌ No articles with URLs found!")
            return
        
        self.stats['total_articles'] = total_needing_scraping
        
        processed = 0
        start_time = datetime.now()
        
        # Process in batches
        while processed < total_needing_scraping:
            batch_start = datetime.now()
            logger.info(f"📦 Processing batch {processed//self.batch_size + 1}: articles {processed+1}-{min(processed+self.batch_size, total_needing_scraping)}")
            
            # Get batch of articles needing content
            articles = await self.get_articles_needing_content(offset=processed, limit=self.batch_size)
            if not articles:
                break
            
            # Process articles in this batch
            for i, article in enumerate(articles, 1):
                ticker = article['ticker']
                url = article['article_url']
                content_hash = article['original_content_hash']
                
                logger.info(f"  🔍 Scraping content for {ticker} ({i}/{len(articles)})")
                
                if not url:
                    logger.warning(f"    ⚠️ No URL for {ticker}")
                    continue
                
                self.stats['articles_with_urls'] += 1
                
                # Scrape content - NO CHECKING, just scrape everything
                try:
                    scraped_content = await self.scrape_article_content(url)
                    
                    if scraped_content and len(scraped_content) > 50:
                        # Update database
                        success = await self.update_article_content(content_hash, scraped_content)
                        
                        if success:
                            logger.info(f"    ✅ {ticker} updated with {len(scraped_content)} characters")
                            self.stats['successful_scrapes'] += 1
                            self.stats['updated_articles'] += 1
                        else:
                            logger.error(f"    ❌ {ticker} failed to update database")
                            self.stats['failed_scrapes'] += 1
                    else:
                        logger.warning(f"    ⚠️ {ticker} no meaningful content scraped")
                        self.stats['failed_scrapes'] += 1
                        
                except Exception as e:
                    logger.error(f"    ❌ {ticker} scraping failed: {e}")
                    self.stats['failed_scrapes'] += 1
                
                # Small delay to be respectful to servers
                await asyncio.sleep(0.5)
            
            processed += len(articles)
            batch_time = (datetime.now() - batch_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Batch completed in {batch_time:.1f}s | Progress: {processed}/{total_needing_scraping} ({processed/total_needing_scraping*100:.1f}%)")
            logger.info(f"⏱️ Total time: {total_time:.1f}s")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Content scraping completed in {total_time:.1f}s!")
        
        # Print final statistics
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  • Total articles processed: {self.stats['total_articles']}")
        logger.info(f"  • Articles with URLs: {self.stats['articles_with_urls']}")
        logger.info(f"  • Successful scrapes: {self.stats['successful_scrapes']}")
        logger.info(f"  • Failed scrapes: {self.stats['failed_scrapes']}")
        logger.info(f"  • Articles updated: {self.stats['updated_articles']}")
        
        success_rate = (self.stats['successful_scrapes'] / max(1, self.stats['articles_with_urls']) * 100)
        logger.info(f"  • Success rate: {success_rate:.1f}%")
    
    async def verify_content_after_scraping(self):
        """Verify content was successfully scraped and updated"""
        logger.info("🔍 Verifying scraped content...")
        
        await self.get_content_stats()
        
        # Sample some updated articles
        sample_query = """
        SELECT 
            ticker, 
            outcome_type, 
            LENGTH(full_content) as content_length,
            SUBSTRING(full_content, 1, 200) as content_preview
        FROM News.rag_training_dataset 
        WHERE full_content IS NOT NULL 
          AND full_content != '' 
          AND LENGTH(full_content) > 100
        ORDER BY outcome_type, ticker
        LIMIT 10
        """
        
        result = self.ch_manager.client.query(sample_query)
        
        logger.info("📋 Sample of articles with scraped content:")
        for row in result.result_rows:
            ticker, outcome, length, preview = row
            logger.info(f"  • {ticker} ({outcome}): {length} chars")
            logger.info(f"    Preview: {preview[:150]}...")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.crawler:
            await self.crawler.close()
        
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Scrape Full Article Content for RAG Training Dataset')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size for processing')
    parser.add_argument('--max-chars', type=int, default=4500, help='Maximum characters per article (6K token limit ≈ 4500 chars)')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing content')
    parser.add_argument('--stats-only', action='store_true', help='Only show content statistics')
    
    args = parser.parse_args()
    
    scraper = TrainingContentScraper(batch_size=args.batch_size, max_chars=args.max_chars)
    
    try:
        await scraper.initialize()
        
        if args.stats_only:
            await scraper.get_content_stats()
        elif args.verify_only:
            await scraper.verify_content_after_scraping()
        else:
            await scraper.scrape_all_missing_content()
            await scraper.verify_content_after_scraping()
        
    except Exception as e:
        logger.error(f"Content scraping failed: {e}")
        raise
    finally:
        await scraper.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 