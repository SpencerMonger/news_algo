import pandas as pd
from finviz_list import FinvizScreener
from typing import Optional, Set
import requests
from datetime import datetime, timedelta
from price_check import PriceMonitor
import time

class NewsMonitor:
    def __init__(self):
        self.screener = FinvizScreener()
        self._cached_news = None
        self._last_news_update = None
        self._cache_duration = timedelta(minutes=5)
        self._seen_articles = set()  # Track articles we've already processed
        self.price_monitor = PriceMonitor()
        
    def monitor_news_and_price(self, check_interval: int = 10):
        """
        Continuously monitors news and triggers price checks for new articles
        
        Args:
            check_interval (int): Seconds between news checks
        """
        print("Starting news and price monitor...")
        
        while True:
            try:
                new_articles = self._check_for_new_articles()
                if new_articles:
                    self._process_new_articles(new_articles)
                    
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in news monitoring loop: {str(e)}")
                time.sleep(check_interval)
    
    def _check_for_new_articles(self) -> Optional[pd.DataFrame]:
        """Checks for new articles and returns only unseen ones"""
        news_df = self.get_filtered_news(force_refresh=True)
        if news_df is None or news_df.empty:
            return None
            
        # Create unique identifier for each article (combine date and title)
        news_df['article_id'] = news_df['Date'] + news_df['Title']
        
        # Filter to only new articles
        new_articles = news_df[~news_df['article_id'].isin(self._seen_articles)]
        
        if not new_articles.empty:
            # Add new articles to seen set
            self._seen_articles.update(new_articles['article_id'].tolist())
            return new_articles
            
        return None
    
    def _process_new_articles(self, new_articles: pd.DataFrame):
        """Process new articles and trigger price checks"""
        for _, article in new_articles.iterrows():
            ticker = article['Ticker']
            print(f"\nNew article detected for {ticker}:")
            print(f"Title: {article['Title']}")
            print(f"Date: {article['Date']}")
            
            # Check price vs 15-min high
            price_result = self.price_monitor.check_price_vs_high(ticker)
            if price_result:
                print(f"Price Check Results for {ticker}:")
                print(f"Current Price: ${price_result['current_price']:.2f}")
                print(f"15-min High: ${price_result['fifteen_min_high']:.2f}")
                print(f"Above High: {'Yes' if price_result['is_above_high'] else 'No'}")
            print("-" * 50)
    
    def get_filtered_news(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetches news for stocks in our screener list.
        
        Args:
            force_refresh (bool): If True, forces a refresh of cached data
            
        Returns:
            pd.DataFrame: DataFrame containing filtered news
        """
        if not force_refresh and self._is_cache_valid():
            return self._cached_news
            
        try:
            # Add required headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Get our stock list
            stocks_df = self.screener.get_stock_list()
            if stocks_df is None:
                return None
                
            # Get comma-separated list of tickers
            tickers = ','.join(stocks_df['Ticker'].tolist())
            
            # Construct URL with screener filters
            url = (
                "https://elite.finviz.com/news_export.ashx?"
                "f=geo_usa,sh_float_u20&"  # Use screener filters directly
                "v=1&"                      # News view
                "auth=e1fdfc26-3d70-4153-9f54-9f44f5ddd633"
            )
            
            # Use existing URL but add headers
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Add error handling for CSV parsing
            try:
                news_df = pd.read_csv(
                    pd.io.common.StringIO(response.content.decode('utf-8')),
                    sep=',',
                    on_bad_lines='skip'
                )
                print("\nDEBUG INFO:")
                print("Available columns:", news_df.columns.tolist())  # Debug column names
                print("\nFirst 5 rows of data:")
                print(news_df.head())  # Show first 5 rows
                print("\nData types of columns:")
                print(news_df.dtypes)
                
                # Rename columns if needed (Finviz might use different names)
                column_mapping = {
                    'Symbol': 'Ticker',  # Common alternative name
                    'No.': 'Ticker'      # Another possibility
                }
                news_df = news_df.rename(columns=column_mapping)
                
                # Update cache
                self._cached_news = news_df
                self._last_news_update = datetime.now()
                
                return news_df
                
            except Exception as e:
                print(f"\nError details:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if news_df is not None:
                    print("\nDataFrame info:")
                    print(news_df.info())
                return None
            
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
            return None
            
    def _is_cache_valid(self) -> bool:
        """Checks if cached news is still valid"""
        if self._cached_news is None or self._last_news_update is None:
            return False
        return datetime.now() - self._last_news_update < self._cache_duration

def main():
    monitor = NewsMonitor()
    monitor.monitor_news_and_price(check_interval=10)

if __name__ == "__main__":
    main()
