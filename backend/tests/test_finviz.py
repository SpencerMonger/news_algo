import pandas as pd
from datetime import datetime
import os
import requests

# Add required headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

URL = "https://elite.finviz.com/news_export.ashx?v=111&f=geo_usa,sh_float_u20&ft=4&auth=e1fdfc26-3d70-4153-9f54-9f44f5ddd633"

response = requests.get(URL, headers=headers)

try:
    # Read CSV and filter for 'stock' category
    df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
    stock_news_df = df[df['Category'] == 'stock']
    
    print("\nDEBUG INFO:")
    print("All available columns:", df.columns.tolist())
    print("\nUnique categories:", df['Category'].unique())
    print(f"\nFound {len(stock_news_df)} stock-related news items out of {len(df)} total")
    print("\nFirst 5 stock news items:")
    print(stock_news_df.head())
    
    # Save filtered results
    stock_news_df.to_csv("stock_news_export.csv", index=False)
    print("\nSaved filtered results to stock_news_export.csv")
    
except Exception as e:
    print(f"Error processing data: {str(e)}")
    print("Raw response content:")
    print(response.content[:500])