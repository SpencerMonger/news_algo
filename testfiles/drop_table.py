#!/usr/bin/env python3

from clickhouse_setup import ClickHouseManager

def drop_breaking_news_table():
    m = ClickHouseManager()
    m.connect()
    
    print("Dropping News.breaking_news table...")
    m.client.command("DROP TABLE IF EXISTS News.breaking_news")
    print("✅ Table dropped successfully!")
    print("Now run the web scraper to recreate with fixed schema.")
    
    m.close()

if __name__ == "__main__":
    drop_breaking_news_table() 