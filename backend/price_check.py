import requests
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Union
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PriceMonitor:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.base_url = os.getenv('PROXY_URL', 'http://3.128.134.41')  # Default if not in .env
        self.proxy = {
            'http': f"{self.base_url}:80",
            'https': f"{self.base_url}:80"
        }
        self.session = requests.Session()
        self.session.proxies.update(self.proxy)
        
    def check_price_vs_high(self, ticker: str) -> Optional[Dict[str, Union[float, bool]]]:
        """
        Checks if current price is above trailing 15-minute high
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Contains current price, 15min high, and boolean result
        """
        try:
            # Get current price
            current_price = self._get_last_trade(ticker)
            if current_price is None:
                return None
                
            # Get 15-minute bars
            fifteen_min_high = self._get_fifteen_min_high(ticker)
            if fifteen_min_high is None:
                return None
                
            return {
                "current_price": current_price,
                "fifteen_min_high": fifteen_min_high,
                "is_above_high": current_price > fifteen_min_high
            }
            
        except Exception as e:
            print(f"Error checking price vs high: {str(e)}")
            return None
            
    def _get_last_trade(self, ticker: str) -> Optional[float]:
        """Gets the last trade price for the ticker"""
        try:
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {'apiKey': self.api_key}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return float(data['results']['p'])
            
        except Exception as e:
            print(f"Error fetching last trade: {str(e)}")
            return None
            
    def _get_fifteen_min_high(self, ticker: str) -> Optional[float]:
        """Gets the highest price in the last 15 minutes"""
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=15)
            
            # Format timestamps
            from_ts = start_time.strftime('%Y-%m-%d')
            to_ts = end_time.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute/{from_ts}/{to_ts}"
            params = {'apiKey': self.api_key}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('results'):
                return None
                
            return max(bar['h'] for bar in data['results'])
            
        except Exception as e:
            print(f"Error fetching 15-min high: {str(e)}")
            return None

def main():
    monitor = PriceMonitor()
    ticker = os.getenv('SYMBOL', 'AMD')  # Default to AMD if not specified
    
    print(f"Starting price monitor for {ticker}")
    print(f"Using proxy: {monitor.proxy['http']}")
    
    while True:
        result = monitor.check_price_vs_high(ticker)
        if result:
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"15-min High: ${result['fifteen_min_high']:.2f}")
            print(f"Above High: {'Yes' if result['is_above_high'] else 'No'}")
        
        time.sleep(60)  # Wait 1 minute before next check

if __name__ == "__main__":
    main()
