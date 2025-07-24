#!/usr/bin/env python3
"""
Debug Timestamp Extraction
Test what content we're getting from BusinessWire articles
"""

import asyncio
import sys
import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlResult
import pytz

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def debug_businesswire_content():
    """Debug what content we get from a BusinessWire article"""
    
    # Test URL from your logs
    test_url = "https://www.businesswire.com/news/home/20250716321422/en/"
    
    print(f"🔍 Testing BusinessWire content extraction from: {test_url}")
    
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
            # Removed --disable-javascript to allow JS execution
        ]
    )
    
    try:
        await crawler.start()
        
        # Crawl the page
        result: CrawlResult = await crawler.arun(
            url=test_url,
            wait_for="css:time, .timestamp, .date, .published",
            delay_before_return_html=2.0,
            timeout=20
        )
        
        if not result.success:
            print(f"❌ Crawl failed: {result.error_message}")
            return
        
        soup = BeautifulSoup(result.html, 'html.parser')
        full_text = soup.get_text()
        
        print("\n" + "="*80)
        print("📄 SCRAPED CONTENT (first 1000 chars):")
        print("="*80)
        print(full_text[:1000])
        print("...")
        
        print("\n" + "="*80)
        print("🔍 FINDING EXACT TIMESTAMP TEXT:")
        print("="*80)
        
        # Look for the timestamp more carefully
        if 'Jul' in full_text and 'AM' in full_text:
            jul_idx = full_text.find('Jul')
            print(f"Found 'Jul' at index {jul_idx}")
            
            # Get a larger context around the timestamp
            start_idx = max(0, jul_idx - 50)
            end_idx = min(len(full_text), jul_idx + 100)
            timestamp_context = full_text[start_idx:end_idx]
            
            print(f"Context around timestamp:")
            print(f"Raw: {repr(timestamp_context)}")
            print(f"Display: {timestamp_context}")
            
            # Look for the exact timestamp pattern in this context
            pattern = r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Daylight\s+Time'
            
            match = re.search(pattern, timestamp_context, re.IGNORECASE)
            if match:
                print(f"✅ PATTERN MATCHED in context!")
                print(f"   Match: {match.group()}")
                print(f"   Groups: {match.groups()}")
            else:
                print(f"❌ Pattern still doesn't match in context")
                
                # Try to find any timestamp-like patterns
                simple_patterns = [
                    r'Jul \d+, \d{4} \d+:\d+ [AP]M',
                    r'[A-Za-z]+ \d+, \d{4} \d+:\d+ [AP]M',
                    r'\d+:\d+ [AP]M'
                ]
                
                for i, simple_pattern in enumerate(simple_patterns):
                    simple_match = re.search(simple_pattern, timestamp_context)
                    if simple_match:
                        print(f"✅ Simple pattern {i+1} matched: {simple_match.group()}")
                    else:
                        print(f"❌ Simple pattern {i+1} no match")
        
        print("\n" + "="*80)
        print("🔍 LOOKING FOR ALL TIMESTAMP-LIKE TEXT:")
        print("="*80)
        
        # Find all lines that contain timestamp-like patterns
        lines = full_text.split('\n')
        timestamp_lines = []
        
        for line in lines:
            line = line.strip()
            if re.search(r'\d{1,2}:\d{2}\s+[AP]M', line):
                timestamp_lines.append(line)
        
        if timestamp_lines:
            print(f"Found {len(timestamp_lines)} lines with time patterns:")
            for i, line in enumerate(timestamp_lines):
                print(f"  {i+1}. {repr(line)}")
                print(f"     Display: {line}")
                
                # Test our pattern against each line
                pattern = r'([A-Za-z]{3,9})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s+([AP]M)\s+Eastern\s+Daylight\s+Time'
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    print(f"     ✅ MATCHES our pattern!")
                else:
                    print(f"     ❌ Doesn't match our pattern")
        else:
            print("No lines with time patterns found")
        
    finally:
        await crawler.close()

if __name__ == "__main__":
    asyncio.run(debug_businesswire_content()) 