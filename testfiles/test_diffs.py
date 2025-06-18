#!/usr/bin/env python3

from clickhouse_setup import ClickHouseManager

def find_actual_duplicates():
    """Find the actual duplicates in your database"""
    
    print("🔍 Finding actual duplicates in your database...")
    print("=" * 60)
    
    m = ClickHouseManager()
    m.connect()
    
    try:
        # Look for duplicate URLs across ALL data (not just recent)
        query = """
        SELECT 
            article_url,
            groupArray(source) as sources,
            groupArray(ticker) as tickers,
            groupArray(headline) as headlines,
            groupArray(timestamp) as timestamps,
            count() as count
        FROM News.breaking_news 
        GROUP BY article_url
        HAVING count > 1
        ORDER BY count DESC
        LIMIT 20
        """
        
        result = m.client.query(query)
        
        if result.result_rows:
            print(f"Found {len(result.result_rows)} URLs with multiple entries:")
            print()
            
            for i, (url, sources, tickers, headlines, timestamps, count) in enumerate(result.result_rows, 1):
                print(f"{i}. DUPLICATE URL ({count} entries):")
                print(f"   URL: {url}")
                print(f"   Sources: {sources}")
                print(f"   Tickers: {tickers}")
                print(f"   Timestamps: {timestamps}")
                print(f"   Headlines: {[h[:50]+'...' if len(h) > 50 else h for h in headlines]}")
                
                # Check if this is the RSS vs Web scraper issue
                has_rss = any('RSS' in str(s) for s in sources)
                has_24h = any('24H' in str(s) for s in sources)
                
                if has_rss and has_24h:
                    print(f"   🎯 RSS vs Web Scraper duplicate!")
                
                print()
        else:
            print("No duplicate URLs found in the database.")
        
        print("\n" + "=" * 60)
        
        # Look specifically for KLTO duplicates
        query2 = """
        SELECT 
            source,
            ticker,
            headline,
            article_url,
            timestamp,
            content_hash
        FROM News.breaking_news 
        WHERE ticker = 'KLTO'
        ORDER BY timestamp DESC
        """
        
        result2 = m.client.query(query2)
        
        if result2.result_rows:
            print(f"KLTO articles ({len(result2.result_rows)} found):")
            print()
            
            for i, (source, ticker, headline, url, timestamp, content_hash) in enumerate(result2.result_rows, 1):
                print(f"{i}. Source: {source}")
                print(f"   Ticker: {ticker}")
                print(f"   Headline: {headline[:60]}...")
                print(f"   URL: {url}")
                print(f"   Timestamp: {timestamp}")
                print(f"   Content Hash: {content_hash}")
                print()
        
        print("\n" + "=" * 60)
        
        # Check the ReplacingMergeTree ORDER BY effectiveness
        query3 = """
        SELECT 
            content_hash,
            article_url,
            groupArray(source) as sources,
            count() as count
        FROM News.breaking_news 
        GROUP BY content_hash, article_url
        HAVING count > 1
        ORDER BY count DESC
        LIMIT 10
        """
        
        result3 = m.client.query(query3)
        
        if result3.result_rows:
            print(f"Duplicates by ORDER BY key (content_hash, article_url):")
            print()
            
            for i, (content_hash, url, sources, count) in enumerate(result3.result_rows, 1):
                print(f"{i}. Hash: {content_hash}")
                print(f"   URL: {url}")
                print(f"   Sources: {sources}")
                print(f"   Count: {count}")
                print()
        else:
            print("No duplicates found by ORDER BY key - ReplacingMergeTree should work!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        m.close()

if __name__ == "__main__":
    find_actual_duplicates()