#!/usr/bin/env python3
"""
Analyze Finviz HTML structure to understand timestamp-article associations
"""

import asyncio
import sys
import os
import re
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlResult

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def analyze_finviz_structure():
    """Analyze the HTML structure of a Finviz ticker page"""
    
    print("🔍 Analyzing Finviz HTML structure for CLDI...")
    
    # Initialize Crawl4AI
    crawler = AsyncWebCrawler(
        verbose=False,
        headless=True,
        browser_type="chromium",
        max_idle_time=30000,
        keep_alive=True,
        extra_args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-images",
            "--disable-javascript",
        ]
    )
    
    try:
        await crawler.start()
        
        # Scrape CLDI page
        ticker_url = "https://elite.finviz.com/quote.ashx?t=CLDI&ty=c&ta=1&p=i1"
        print(f"📰 Scraping: {ticker_url}")
        
        result = await crawler.arun(
            url=ticker_url,
            wait_for="css:table",
            delay_before_return_html=3.0,
            timeout=30
        )
        
        if not result.success:
            print(f"❌ Failed to scrape: {result.error_message}")
            return
        
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Find all tables and analyze their structure
        tables = soup.find_all('table')
        print(f"\n📊 Found {len(tables)} tables on the page")
        
        # Look for news-related tables
        for i, table in enumerate(tables):
            table_text = table.get_text()
            
            # Check if this table contains news content
            has_timestamps = bool(re.search(r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', table_text))
            has_newswires = any(wire in table_text for wire in ['GlobeNewswire', 'PRNewswire', 'BusinessWire', 'Accesswire'])
            
            if has_timestamps or has_newswires:
                print(f"\n🎯 NEWS TABLE {i} (has timestamps: {has_timestamps}, has newswires: {has_newswires}):")
                print(f"   Table size: {len(table_text)} chars")
                print(f"   Rows: {len(table.find_all('tr'))}")
                
                # Analyze table structure
                rows = table.find_all('tr')
                for j, row in enumerate(rows[:5]):  # Show first 5 rows
                    cells = row.find_all(['td', 'th'])
                    row_text = row.get_text().strip()
                    
                    if len(row_text) > 20:  # Skip empty/header rows
                        print(f"   Row {j}: {len(cells)} cells, {len(row_text)} chars")
                        print(f"     Text: '{row_text[:150]}...'")
                        
                        # Check for timestamps in this row
                        timestamps = re.findall(r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', row_text)
                        if timestamps:
                            print(f"     📅 Timestamps: {timestamps}")
                        
                        # Check for links in this row
                        links = row.find_all('a', href=True)
                        if links:
                            print(f"     🔗 Links: {len(links)}")
                            for link in links[:2]:  # Show first 2 links
                                href = link.get('href', '')
                                text = link.get_text().strip()
                                print(f"       '{text[:50]}...' -> {href}")
                
                print(f"   Full table preview:")
                print(f"   '{table_text[:500]}...'")
                print("-" * 80)
        
        # Look for specific news containers based on web search findings
        print(f"\n🔍 Looking for specific news container patterns...")
        
        # Pattern 1: Look for table rows with timestamps and links
        news_rows = []
        for table in tables:
            for row in table.find_all('tr'):
                row_text = row.get_text()
                if re.search(r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', row_text):
                    links = row.find_all('a', href=True)
                    if links:
                        news_rows.append(row)
        
        print(f"📰 Found {len(news_rows)} rows with timestamps and links")
        
        # Analyze the structure of these news rows
        for i, row in enumerate(news_rows[:3]):  # Show first 3
            cells = row.find_all(['td', 'th'])
            print(f"\nNews Row {i+1}:")
            print(f"  Cells: {len(cells)}")
            
            for j, cell in enumerate(cells):
                cell_text = cell.get_text().strip()
                cell_links = cell.find_all('a', href=True)
                timestamps = re.findall(r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', cell_text)
                
                print(f"    Cell {j}: {len(cell_text)} chars, {len(cell_links)} links, {len(timestamps)} timestamps")
                if timestamps:
                    print(f"      Timestamps: {timestamps}")
                if cell_links:
                    for link in cell_links:
                        print(f"      Link: '{link.get_text().strip()[:40]}...' -> {link.get('href', '')}")
                if len(cell_text) > 0:
                    print(f"      Text: '{cell_text[:100]}...'")
        
        # Save the HTML for manual inspection
        with open('cldi_finviz_structure.html', 'w', encoding='utf-8') as f:
            f.write(result.html)
        print(f"\n💾 Saved full HTML to 'cldi_finviz_structure.html' for manual inspection")
        
    finally:
        await crawler.close()

if __name__ == "__main__":
    asyncio.run(analyze_finviz_structure()) 