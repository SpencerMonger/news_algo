#!/usr/bin/env python3
"""
Test ABLV specifically to debug missing articles issue
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

async def test_ablv_page():
    """Test ABLV page specifically to debug the missing articles issue"""
    
    ticker = "ABLV"
    print(f"🔍 Testing {ticker} page structure")
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
            wait_for="css:table",
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
        
        # 1. Check all tables and their news content
        tables = soup.find_all('table')
        print(f"📋 Found {len(tables)} tables on page")
        
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            print(f"\n📋 Table {i+1}: {len(rows)} rows")
            
            # Check for news links in this table
            table_links = table.find_all('a', href=True)
            news_links = []
            
            for link in table_links:
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
                    news_links.append((href, text))
            
            print(f"   📰 News links in table: {len(news_links)}")
            
            if news_links:
                print(f"   ✅ Table {i+1} contains news!")
                
                # Show first few news links
                for j, (href, text) in enumerate(news_links[:3]):
                    print(f"     {j+1}. {text[:60]}...")
                    print(f"        URL: {href}")
                
                # Analyze table structure for timestamps
                print(f"   🔍 Analyzing table structure for timestamps...")
                
                for row_idx, row in enumerate(rows[:5]):  # Check first 5 rows
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        print(f"     Row {row_idx}: {len(cells)} cells")
                        
                        for cell_idx, cell in enumerate(cells):
                            cell_text = cell.get_text().strip()
                            if cell_text:
                                print(f"       Cell {cell_idx}: '{cell_text[:50]}...'")
                                
                                # Check for timestamp patterns
                                timestamp_patterns = [
                                    r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                                    r'Today\s+\d{1,2}:\d{2}[AP]M',
                                    r'\d{1,2}:\d{2}[AP]M',
                                    r'\w{3}-\d{2}-\d{2}'
                                ]
                                
                                for pattern in timestamp_patterns:
                                    if re.search(pattern, cell_text):
                                        print(f"         🎯 TIMESTAMP FOUND: '{cell_text}'")
                        print()
            else:
                print(f"   ❌ No news links in table {i+1}")
        
        # 2. Look for ALL links that could be news (not just in tables)
        print("\n🔗 SEARCHING ALL LINKS ON PAGE")
        print("-" * 40)
        
        all_links = soup.find_all('a', href=True)
        all_news_links = []
        
        for link in all_links:
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
                all_news_links.append((href, text, link))
        
        print(f"📊 Total news links found on entire page: {len(all_news_links)}")
        
        if all_news_links:
            print("\n📰 Sample news links:")
            for i, (href, text, link) in enumerate(all_news_links[:5]):
                print(f"  {i+1}. {text[:60]}...")
                print(f"     URL: {href}")
                
                # Try to find timestamp for this link
                print(f"     🕐 Looking for timestamp...")
                
                # Check parent structure
                parent = link.parent
                if parent:
                    parent_text = parent.get_text().strip()
                    print(f"     Parent text: '{parent_text[:100]}...'")
                    
                    # Look for timestamps in parent
                    timestamp_patterns = [
                        r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M',
                        r'Today\s+\d{1,2}:\d{2}[AP]M',
                        r'\d{1,2}:\d{2}[AP]M'
                    ]
                    
                    found_timestamp = False
                    for pattern in timestamp_patterns:
                        matches = re.findall(pattern, parent_text)
                        if matches:
                            print(f"     ✅ Timestamp found: {matches[0]}")
                            found_timestamp = True
                            break
                    
                    if not found_timestamp:
                        # Check if parent is in a table row
                        row = parent.find_parent('tr')
                        if row:
                            row_text = row.get_text().strip()
                            print(f"     Table row text: '{row_text[:100]}...'")
                            
                            for pattern in timestamp_patterns:
                                matches = re.findall(pattern, row_text)
                                if matches:
                                    print(f"     ✅ Timestamp in row: {matches[0]}")
                                    found_timestamp = True
                                    break
                    
                    if not found_timestamp:
                        print(f"     ❌ No timestamp found")
                
                print()
        else:
            print("❌ No news links found on entire page!")
        
        # 3. Save HTML for manual inspection
        html_filename = f"testfiles/finviz_{ticker}_debug.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(result.html)
        print(f"\n💾 Raw HTML saved to: {html_filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await crawler.close()

if __name__ == "__main__":
    asyncio.run(test_ablv_page()) 