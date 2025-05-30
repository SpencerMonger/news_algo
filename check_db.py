#!/usr/bin/env python3

from clickhouse_setup import ClickHouseManager
import os

def check_database():
    m = ClickHouseManager()
    
    # Show connection details
    print(f"Connecting to: {m.host}:{m.port}")
    print(f"Username: {m.username}")
    print(f"Database: {m.database}")
    print(f"ENV CLICKHOUSE_HOST: {os.getenv('CLICKHOUSE_HOST', 'NOT SET')}")
    print()
    
    m.connect()
    
    try:
        # Show what database we're actually in
        result = m.client.query('SELECT currentDatabase()')
        current_db = result.result_rows[0][0]
        print(f"Current database: {current_db}")
        
        # Show all databases
        result = m.client.query('SHOW DATABASES')
        databases = [row[0] for row in result.result_rows]
        print(f"Available databases: {databases}")
        
        # Show tables in News database
        result = m.client.query('SHOW TABLES FROM News')
        tables = [row[0] for row in result.result_rows]
        print(f"Tables in News database: {tables}")
        print()
        
        # Check count
        result = m.client.query('SELECT COUNT(*) FROM News.breaking_news')
        count = result.result_rows[0][0]
        print(f"Total rows in News.breaking_news: {count}")
        
        if count > 0:
            # Get latest records
            result = m.client.query('SELECT ticker, headline, timestamp FROM News.breaking_news ORDER BY timestamp DESC LIMIT 3')
            
            print(f"\nLatest articles:")
            for i, row in enumerate(result.result_rows, 1):
                print(f"{i}. Ticker: {row[0]}")
                print(f"   Headline: {row[1][:60]}...")
                print(f"   Timestamp: {row[2]}")
                print()
        else:
            print("No articles found in News.breaking_news!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    m.close()

if __name__ == "__main__":
    check_database() 