#!/usr/bin/env python3
"""
Test Finviz Page Structure
Debug what we're actually getting from Finviz ticker pages
"""

import asyncio
import sys
import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlResult

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_finviz_page_structure(ticker="ABEO"):
    """Test what we actually get from a Finviz ticker page"""
    
    print(f"🔍 Testing Finviz page structure for {ticker}")
    print("=" * 60)
    
    # Initialize Crawl4AI
    crawler = AsyncWebCrawler(
        verbose=False,
        headless=True,
        browser_type="chromium"
    )
    
    try:
        await crawler.start()
        
        # Construct ticker URL
        ticker_url = f"https://elite.finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=i1"
        print(f"📍 URL: {ticker_url}")
        print()
        
        # Scrape the page
        result: CrawlResult = await crawler.arun(
            url=ticker_url,
            wait_for="css:body",
            delay_before_return_html=3.0,
            timeout=30
        )
        
        if not result.success:
            print(f"❌ Failed to scrape: {result.error_message}")
            return
        
        soup = BeautifulSoup(result.html, 'html.parser')
        
        print(f"✅ Page loaded successfully")
        print(f"📄 Title: {soup.title.string if soup.title else 'No title'}")
        print()
        
        # 1. Find all links that could be news
        print("🔗 ANALYZING ALL LINKS")
        print("-" * 40)
        
        all_links = soup.find_all('a', href=True)
        news_links = []
        
        for i, link in enumerate(all_links):
            href = link.get('href', '')
            text = link.get_text().strip()
            
            # Check if it's a news link
            is_news = any(domain in href.lower() for domain in [
                'businesswire.com', 'globenewswire.com', 
                'prnewswire.com', 'accesswire.com',
                'finviz.com/news', 'finance.yahoo.com/news'
            ]) or any(pattern in href.lower() for pattern in [
                '/news/', 'news.', 'press-release', 'article'
            ])
            
            if is_news and text and len(text) > 10:
                news_links.append((link, href, text))
                print(f"📰 News Link #{len(news_links)}:")
                print(f"   Text: {text[:80]}...")
                print(f"   URL:  {href}")
                
                # Analyze the structure around this link
                analyze_link_structure(link, len(news_links))
                print()
        
        print(f"📊 Total links found: {len(all_links)}")
        print(f"📰 News links found: {len(news_links)}")
        print()
        
        # 2. Look for tables that might contain news
        print("📋 ANALYZING TABLES")
        print("-" * 40)
        
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables on page")
        
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) > 1:  # Skip single-row tables
                print(f"\n📋 Table {i+1}: {len(rows)} rows")
                
                # Check if this table contains news links
                table_links = table.find_all('a', href=True)
                news_in_table = 0
                
                for link in table_links:
                    href = link.get('href', '')
                    if any(domain in href.lower() for domain in [
                        'businesswire.com', 'globenewswire.com', 
                        'prnewswire.com', 'accesswire.com'
                    ]):
                        news_in_table += 1
                
                if news_in_table > 0:
                    print(f"   ✅ Contains {news_in_table} news links")
                    
                    # Analyze first few rows in detail
                    for row_idx, row in enumerate(rows[:5]):
                        cells = row.find_all(['td', 'th'])
                        if len(cells) > 1:
                            print(f"   Row {row_idx}: {len(cells)} cells")
                            for cell_idx, cell in enumerate(cells):
                                cell_text = cell.get_text().strip()
                                if cell_text:
                                    print(f"     Cell {cell_idx}: {cell_text[:60]}...")
                                    
                                    # Check for timestamp patterns
                                    timestamp_patterns = [
                                        r'Jul-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                                        r'Today\s+\d{1,2}:\d{2}[AP]M',
                                        r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                                        r'\w{3}-\d{2}-\d{2}',
                                        r'\d{1,2}:\d{2}[AP]M'
                                    ]
                                    
                                    for pattern in timestamp_patterns:
                                        if re.search(pattern, cell_text):
                                            print(f"       🎯 TIMESTAMP PATTERN FOUND: '{cell_text}'")
                else:
                    print(f"   ❌ No news links")
        
        # 3. Look for specific timestamp patterns anywhere on the page
        print("\n⏰ SEARCHING FOR TIMESTAMP PATTERNS")
        print("-" * 40)
        
        page_text = soup.get_text()
        timestamp_patterns = [
            (r'Jul-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', 'Finviz date-time format'),
            (r'Today\s+\d{1,2}:\d{2}[AP]M', 'Today format'),
            (r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', 'Month-day-year time'),
            (r'\w{3}-\d{2}-\d{2}', 'Date only'),
            (r'\d{1,2}:\d{2}[AP]M', 'Time only')
        ]
        
        for pattern, description in timestamp_patterns:
            matches = re.findall(pattern, page_text)
            if matches:
                print(f"✅ {description}: Found {len(matches)} matches")
                for match in matches[:5]:  # Show first 5
                    print(f"   {match}")
            else:
                print(f"❌ {description}: No matches")
        
        # 4. Save raw HTML for manual inspection
        html_filename = f"testfiles/finviz_{ticker}_page.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(result.html)
        print(f"\n💾 Raw HTML saved to: {html_filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await crawler.close()

def analyze_link_structure(link, link_number):
    """Analyze the structure around a news link"""
    print(f"   🔍 Structure Analysis:")
    
    # Parent analysis
    parent = link.parent
    if parent:
        print(f"     Parent: <{parent.name}> class={parent.get('class', [])}")
        parent_text = parent.get_text().strip()
        if parent_text != link.get_text().strip():
            print(f"     Parent text: {parent_text[:100]}...")
        
        # Check if parent is a table cell
        if parent.name == 'td':
            row = parent.parent
            if row and row.name == 'tr':
                cells = row.find_all('td')
                print(f"     Table row with {len(cells)} cells:")
                for i, cell in enumerate(cells):
                    cell_text = cell.get_text().strip()
                    if cell_text:
                        print(f"       Cell {i}: {cell_text[:40]}...")
                        
                        # Look for timestamps in each cell
                        timestamp_patterns = [
                            r'Jul-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                            r'Today\s+\d{1,2}:\d{2}[AP]M',
                            r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                            r'\w{3}-\d{2}-\d{2}',
                        ]
                        
                        for pattern in timestamp_patterns:
                            if re.search(pattern, cell_text):
                                print(f"         🎯 TIMESTAMP: {cell_text}")
        
        # Grandparent analysis
        grandparent = parent.parent
        if grandparent and grandparent.name != 'tr':  # Skip if already analyzed as table row
            print(f"     Grandparent: <{grandparent.name}> class={grandparent.get('class', [])}")

async def main():
    """Main test function"""
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "ABEO"  # Default to ABEO as shown in screenshot
    
    await test_finviz_page_structure(ticker)

if __name__ == "__main__":
    asyncio.run(main()) 