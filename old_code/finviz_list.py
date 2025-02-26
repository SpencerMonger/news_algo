import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class FinvizScreener:
    def __init__(self):
        self._cached_data = None
        self._last_update = None
        self._cache_duration = timedelta(minutes=15)  # Cache data for 15 minutes
        
    def get_stock_list(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetches stock list from Finviz and caches it locally.
        
        Args:
            force_refresh (bool): If True, forces a refresh of cached data
            
        Returns:
            pd.DataFrame: DataFrame containing stock data
        """
        # Return cached data if it's still valid
        if not force_refresh and self._is_cache_valid():
            return self._cached_data
            
        try:
            # Add required headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Replace with your actual Finviz URL and auth token
            url = "https://elite.finviz.com/screener.ashx?v=111&f=geo_usa,sh_float_u20&ft=4&auth=e1fdfc26-3d70-4153-9f54-9f44f5ddd633"
            
            # Use existing URL but add headers
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Add error handling for CSV parsing
            try:
                df = pd.read_csv(
                    pd.io.common.StringIO(response.content.decode('utf-8')),
                    sep=',',          # Explicitly specify separator
                    on_bad_lines='skip'  # Skip problematic lines
                )
                print(f"Successfully loaded {len(df)} rows of data")
                
                # Update cache
                self._cached_data = df
                self._last_update = datetime.now()
                
                return df
            except Exception as csv_error:
                print(f"CSV parsing error: {str(csv_error)}")
                print("Raw response content:")
                print(response.content[:500])  # Print first 500 chars to debug
                return None
            
        except Exception as e:
            print(f"Error fetching Finviz data: {str(e)}")
            return None
            
    def _is_cache_valid(self) -> bool:
        """Checks if cached data is still valid"""
        if self._cached_data is None or self._last_update is None:
            return False
        return datetime.now() - self._last_update < self._cache_duration
