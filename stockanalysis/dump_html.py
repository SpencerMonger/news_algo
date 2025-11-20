#!/usr/bin/env python3
"""
Simple HTML dumper for StockAnalysis.com statistics pages
Use this to verify what content is available for extraction
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawl4ai import AsyncWebCrawler, CrawlResult
from bs4 import BeautifulSoup


async def dump_html(ticker: str):
    """Dump the raw HTML and parsed text from a ticker's statistics page"""
    
    url = f"https://stockanalysis.com/stocks/{ticker.lower()}/statistics/"
    
    print(f"Fetching: {url}")
    print("=" * 80)
    
    # Initialize crawler with same settings as the scraper
    crawler = AsyncWebCrawler(
        verbose=False,
        headless=True,
        browser_type="chromium"
    )
    
    try:
        await crawler.start()
        print("✅ Browser started")
        
        # Fetch the page
        result: CrawlResult = await crawler.arun(
            url=url,
            wait_for="css:.mb-4, .grid, .space-y-5",
            delay_before_return_html=2.0,
            timeout=30
        )
        
        if not result.success:
            print(f"❌ Failed to fetch page: {result.error_message}")
            return
        
        print(f"✅ Page fetched successfully")
        print("=" * 80)
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(result.html, 'html.parser')
        all_text = soup.get_text()
        
        # Save raw HTML to file
        html_file = f"{ticker}_raw.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(result.html)
        print(f"📄 Saved raw HTML to: {html_file}")
        
        # Save parsed text to file
        text_file = f"{ticker}_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(all_text)
        print(f"📄 Saved parsed text to: {text_file}")
        
        # Display first 2000 characters of text
        print("=" * 80)
        print("PARSED TEXT (first 2000 chars):")
        print("=" * 80)
        print(all_text[:2000])
        print("...")
        print("=" * 80)
        print(f"\nTotal text length: {len(all_text)} characters")
        print(f"Total HTML length: {len(result.html)} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await crawler.close()
        print("✅ Browser closed")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 dump_html.py <TICKER>")
        print("Example: python3 dump_html.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    print(f"Dumping HTML for: {ticker}")
    print()
    
    asyncio.run(dump_html(ticker))


if __name__ == "__main__":
    main()

