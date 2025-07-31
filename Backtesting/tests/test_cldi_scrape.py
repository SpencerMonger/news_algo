#!/usr/bin/env python3
"""
Test CLDI scraping to see raw timestamp data and associations
"""

import asyncio
import logging
import sys
import os
import re
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlResult

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_cldi_scrape():
    """Test CLDI scraping to see raw data"""
    
    print("🔍 Testing CLDI scraping to see raw timestamp data...")
    
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
        
        # Find all timestamps on the page
        timestamp_pattern = r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M'
        page_text = soup.get_text()
        all_timestamps = re.findall(timestamp_pattern, page_text)
        
        print(f"\n📅 ALL TIMESTAMPS FOUND ON PAGE ({len(all_timestamps)}):")
        for i, ts in enumerate(set(all_timestamps)):  # Remove duplicates
            print(f"  {i+1}. {ts}")
        
        # Find potential news containers
        potential_containers = []
        
        # Method 1: Table rows
        for tr in soup.find_all('tr'):
            tr_text = tr.get_text().strip()
            if 20 < len(tr_text) < 2000:
                potential_containers.append(('tr', tr))
        
        # Method 2: Divs with timestamps
        for div in soup.find_all('div'):
            div_text = div.get_text().strip()
            if 50 < len(div_text) < 3000:
                if re.search(timestamp_pattern, div_text):
                    potential_containers.append(('div', div))
        
        # Method 3: Table cells
        for td in soup.find_all('td'):
            td_text = td.get_text().strip()
            if 30 < len(td_text) < 1500:
                if td.find('a', href=True) and len(td.find_all('a', href=True)) <= 10:
                    potential_containers.append(('td', td))
        
        print(f"\n📦 POTENTIAL NEWS CONTAINERS FOUND: {len(potential_containers)}")
        
        # Analyze first 5 containers
        target_newswires = [
            'GlobeNewswire', 'Globe Newswire', 'GLOBENEWSWIRE', 'GLOBE NEWSWIRE',
            'PRNewswire', 'PR Newswire', 'PRNEWSWIRE', 'PR NEWSWIRE', 
            'BusinessWire', 'Business Wire', 'BUSINESSWIRE', 'BUSINESS WIRE',
            'Accesswire', 'AccessWire', 'ACCESSWIRE', 'ACCESS WIRE'
        ]
        
        articles_found = []
        
        # First, find containers that actually have both timestamps AND newswire content
        containers_with_news = []
        
        for i, (container_type, container) in enumerate(potential_containers):
            container_text = container.get_text()
            timestamps_in_container = re.findall(timestamp_pattern, container_text)
            links = container.find_all('a', href=True)
            
            # Check if this container has newswire content
            has_newswire = False
            for newswire in target_newswires:
                if f'({newswire})' in container_text or re.search(rf'\b{re.escape(newswire)}\b', container_text, re.IGNORECASE):
                    has_newswire = True
                    break
            
            if has_newswire and (timestamps_in_container or len(links) > 0):
                containers_with_news.append((i, container_type, container, has_newswire))
        
        print(f"\n🎯 CONTAINERS WITH NEWSWIRE CONTENT: {len(containers_with_news)}")
        
        for container_idx, container_type, container, has_newswire in containers_with_news[:5]:  # Show first 5 news containers
            container_text = container.get_text()
            timestamps_in_container = re.findall(timestamp_pattern, container_text)
            links = container.find_all('a', href=True)
            
            print(f"\n📦 NEWS CONTAINER {container_idx+1} ({container_type}):")
            print(f"  Size: {len(container_text)} chars")
            print(f"  Timestamps: {len(timestamps_in_container)} -> {timestamps_in_container}")
            print(f"  Links: {len(links)}")
            print(f"  Text preview: '{container_text[:300]}...'")
            
            # If no timestamps in immediate container, look in parent containers
            if not timestamps_in_container:
                print(f"    🔍 No timestamps in immediate container, searching parent containers...")
                current = container
                for level in range(3):  # Search up to 3 parent levels
                    if current.parent:
                        current = current.parent
                        parent_text = current.get_text()
                        parent_timestamps = re.findall(timestamp_pattern, parent_text)
                        print(f"      Level {level+1} parent ({len(parent_text)} chars): {len(parent_timestamps)} timestamps")
                        if parent_timestamps:
                            timestamps_in_container = parent_timestamps
                            print(f"    ✅ Found {len(parent_timestamps)} timestamps in parent container (level {level+1}): {parent_timestamps}")
                            break
                    else:
                        print(f"      Level {level+1}: No parent container")
                
                if not timestamps_in_container:
                    print(f"    ❌ No timestamps found in any parent containers")
            
            # Check for newswire articles in this container
            for link in links:
                href = link.get('href', '')
                headline = link.get_text().strip()
                
                if len(headline) < 15:
                    continue
                
                # Check for newswire indicators
                newswire_found = None
                search_text = container_text
                
                for newswire in target_newswires:
                    if f'({newswire})' in search_text or re.search(rf'\b{re.escape(newswire)}\b', search_text, re.IGNORECASE):
                        newswire_found = newswire
                        break
                
                if newswire_found:
                    print(f"    🎯 NEWSWIRE ARTICLE FOUND:")
                    print(f"      Headline: '{headline[:60]}...'")
                    print(f"      Newswire: {newswire_found}")
                    print(f"      URL: {href}")
                    print(f"      Available timestamps: {timestamps_in_container}")
                    
                    # Show which timestamp would be selected
                    if len(timestamps_in_container) == 1:
                        selected_ts = timestamps_in_container[0]
                        print(f"      ✅ Selected timestamp (only one): {selected_ts}")
                    elif len(timestamps_in_container) > 1:
                        print(f"      ⚠️ Multiple timestamps - would need proximity logic")
                        
                        # Try to extract date from URL
                        url_date_match = re.search(r'/(\d{4})[/-](\d{1,2})[/-](\d{1,2})/', href)
                        if url_date_match:
                            year, month, day = int(url_date_match.group(1)), int(url_date_match.group(2)), int(url_date_match.group(3))
                            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            if 1 <= month <= 12:
                                expected_date_prefix = f"{month_names[month]}-{day:02d}-{str(year)[-2:]}"
                                matching_timestamps = [ts for ts in timestamps_in_container if ts.startswith(expected_date_prefix)]
                                print(f"      📅 URL date: {year}-{month:02d}-{day:02d} -> {expected_date_prefix}")
                                print(f"      📅 Matching timestamps: {matching_timestamps}")
                                if matching_timestamps:
                                    print(f"      ✅ Would select: {matching_timestamps[0]}")
                    else:
                        print(f"      ❌ No timestamps available")
                    
                    articles_found.append({
                        'headline': headline,
                        'newswire': newswire_found,
                        'timestamps': timestamps_in_container,
                        'url': href
                    })
                    print()
        
        print(f"\n🎉 SUMMARY:")
        print(f"📊 Total timestamps on page: {len(set(all_timestamps))}")
        print(f"📦 Containers analyzed: {min(10, len(potential_containers))}")
        print(f"📰 Newswire articles found: {len(articles_found)}")
        
        print(f"\n📋 TOP ARTICLES:")
        for i, article in enumerate(articles_found[:5]):
            print(f"  {i+1}. '{article['headline'][:60]}...' ({article['newswire']})")
            print(f"      Timestamps: {article['timestamps']}")
        
    finally:
        await crawler.close()

if __name__ == "__main__":
    asyncio.run(test_cldi_scrape()) 