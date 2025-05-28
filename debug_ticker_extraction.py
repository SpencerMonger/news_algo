import asyncio
from clickhouse_setup import setup_clickhouse_database

async def debug_ticker_extraction():
    """Debug recent ticker extractions to find false positives"""
    
    print("Debugging recent ticker extractions...")
    
    # Setup ClickHouse
    ch = setup_clickhouse_database()
    
    try:
        # Get recent articles with their tickers and headlines
        query = """
        SELECT ticker, headline, source 
        FROM News.breaking_news 
        WHERE detected_at >= now() - INTERVAL 60 MINUTE 
        ORDER BY detected_at DESC 
        LIMIT 20
        """
        
        result = ch.client.query(query)
        
        print(f"\n=== Recent Articles ===")
        for row in result.result_rows:
            ticker = row[0]
            headline = row[1]
            source = row[2]
            print(f"{ticker:6} | {source:15} | {headline}")
        
        # Check for suspicious tickers
        suspicious = ['LE', 'LAW', 'HIT', 'MOVE', 'GAME']
        
        print(f"\n=== Suspicious Ticker Details ===")
        for ticker in suspicious:
            query = f"""
            SELECT headline, source 
            FROM News.breaking_news 
            WHERE ticker = '{ticker}' AND detected_at >= now() - INTERVAL 60 MINUTE
            """
            result = ch.client.query(query)
            
            if result.result_rows:
                print(f"\n{ticker} found in:")
                for row in result.result_rows:
                    print(f"  {row[1]}: {row[0]}")
                    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ch.close()

if __name__ == "__main__":
    asyncio.run(debug_ticker_extraction()) 