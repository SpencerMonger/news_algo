import asyncio
import aiohttp
import feedparser
import pandas as pd
import os
from datetime import datetime
import pytz

async def debug_feeds():
    """Debug RSS feeds to see what articles are available"""
    
    # Load ticker list
    csv_path = os.path.join('data_files', 'FV_master_u50float_u10price.csv')
    df = pd.read_csv(csv_path)
    ticker_list = [str(ticker).strip().upper() for ticker in df['Ticker'].tolist()[:10] if pd.notna(ticker)]  # Just first 10 for testing
    print(f"Testing with {len(ticker_list)} tickers: {ticker_list}")
    
    sources = [
        ("GlobeNewswire", "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20Releases"),
        ("BusinessWire", "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEF9YXA==")
    ]
    
    async with aiohttp.ClientSession() as session:
        for name, url in sources:
            print(f"\n=== {name} ===")
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        print(f"Found {len(feed.entries)} total articles")
                        
                        # Check first 5 articles
                        for i, entry in enumerate(feed.entries[:5]):
                            title = entry.get('title', '')
                            published = entry.get('published', '')
                            
                            # Check for tickers in title
                            found_tickers = []
                            for ticker in ticker_list:
                                if ticker in title.upper():
                                    found_tickers.append(ticker)
                            
                            # Parse publish time
                            try:
                                from email.utils import parsedate_to_datetime
                                parsed_time = parsedate_to_datetime(published)
                                if parsed_time.tzinfo is None:
                                    parsed_time = pytz.UTC.localize(parsed_time)
                                current_time = datetime.now(pytz.UTC)
                                age_minutes = (current_time - parsed_time).total_seconds() / 60
                            except:
                                age_minutes = "unknown"
                            
                            print(f"  {i+1}. {title[:80]}...")
                            print(f"     Published: {published} (Age: {age_minutes} min)")
                            print(f"     Tickers found: {found_tickers}")
                            print()
                            
                    else:
                        print(f"HTTP Error: {response.status}")
                        
            except Exception as e:
                print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    asyncio.run(debug_feeds()) 