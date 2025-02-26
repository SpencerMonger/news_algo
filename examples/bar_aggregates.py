import logging
import aiohttp
import os
from dotenv import load_dotenv
import asyncio
import json
import pandas as pd
from datetime import datetime
import pytz
from config.query_config import default_config  # Changed from configs.query_config
import urllib.parse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_aggregates(symbol="AAPL", date="2025-02-14", start_time="09:30", end_time="10:30"):
    """
    Get 1-minute bars for a specific symbol and date using proxy setup from app.py
    """
    try:
        # Convert times to milliseconds since epoch for the API
        est = pytz.timezone('US/Eastern')
        start_dt = est.localize(datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M"))
        end_dt = est.localize(datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M"))
        
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        url = f"http://3.128.134.41/v2/aggs/ticker/{symbol}/range/1/minute/{start_ms}/{end_ms}"
        api_key = os.getenv('POLYGON_API_KEY')
        
        params = {
            'adjusted': "true",
            'sort': 'asc',
            'limit': 50000,
            'apiKey': api_key
        }
        
        # Configure proxy
        proxy = "http://3.128.134.41:80"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results'):
                        # Convert to DataFrame
                        df = pd.DataFrame(data['results'])
                        
                        # Convert timestamp to EST
                        est = pytz.timezone('US/Eastern')
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert(est)
                        
                        # Rename columns for clarity
                        df = df.rename(columns={
                            'v': 'volume',
                            'vw': 'vwap',
                            'o': 'open',
                            'c': 'close',
                            'h': 'high',
                            'l': 'low',
                            'n': 'trades'
                        })
                        
                        # Reorder columns
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']
                        df = df[columns]
                        
                        # Remove CSV saving and return the processed DataFrame
                        return df.to_dict('records')  # Convert DataFrame to list of dictionaries
                    
                logger.error("No results in response")
                return None
                
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

async def main():
    data = await get_aggregates()
    if data and data.get('results'):
        # Save raw response for debugging
        with open('polygon_response.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved raw response to polygon_response.json")
        
        # Load the CSV and display sample
        csv_filename = f"AAPL_2025-02-14_bars.csv"
        df = pd.read_csv(csv_filename)
        print(f"\nFirst few rows from {csv_filename}:")
        print(df.head())
        print(f"\nTotal rows: {len(df)}")

if __name__ == "__main__":
    asyncio.run(main())
