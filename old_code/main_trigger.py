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
            
            # Modified URL to include category=1 for Stock News
            URL = "https://elite.finviz.com/news_export.ashx?v=111&f=geo_usa,sh_float_u20&ft=4&auth=e1fdfc26-3d70-4153-9f54-9f44f5ddd633"
            
            response = requests.get(URL, headers=headers)
            
            # Read CSV without category filter
            news_df = pd.read_csv(pd.io.common.StringIO(response.content.decode('utf-8')))
            
            print("\nDEBUG INFO:")
            print("All available columns:", news_df.columns.tolist())
            print("\nUnique categories:", news_df['Category'].unique())
            print(f"\nFound {len(news_df)} total news items")
            print("\nFirst 5 news items:")
            print(news_df.head())
            
            # Instead of filtering for 'Stock News', we'll look for stock tickers in the titles
            # This is more reliable since the API doesn't seem to have a 'Stock News' category
            
            # Extract potential stock tickers from titles (words in all caps, 1-5 letters)
            def extract_tickers(title):
                import re
                # Find all uppercase words 1-5 letters long that could be tickers
                potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', title)
                return potential_tickers
            
            # Add potential tickers column
            news_df['potential_tickers'] = news_df['Title'].apply(extract_tickers)
            
            # Filter to only include news with potential stock tickers
            news_with_tickers = news_df[news_df['potential_tickers'].apply(len) > 0]
            print(f"\nAfter filtering for news with potential tickers: {len(news_with_tickers)} news items")
            
            # Get our screener tickers
            screener_tickers = self.screener.get_tickers()
            
            # Filter to only include news with tickers from our screener
            if screener_tickers:
                def has_screener_ticker(ticker_list):
                    return any(ticker in screener_tickers for ticker in ticker_list)
                
                news_with_screener_tickers = news_with_tickers[news_with_tickers['potential_tickers'].apply(has_screener_ticker)]
                print(f"\nAfter filtering for news with screener tickers: {len(news_with_screener_tickers)} news items")
                
                # Use this filtered dataframe
                news_df = news_with_screener_tickers
            else:
                # If no screener tickers available, use all news with potential tickers
                news_df = news_with_tickers
            
            # Add ticker column for easier processing
            def get_first_matching_ticker(ticker_list):
                if not screener_tickers:
                    return ticker_list[0] if ticker_list else None
                
                for ticker in ticker_list:
                    if ticker in screener_tickers:
                        return ticker
                return ticker_list[0] if ticker_list else None
            
            news_df['Ticker'] = news_df['potential_tickers'].apply(get_first_matching_ticker)
            
            # Update cache
            self._cached_news = news_df
            self._last_news_update = datetime.now()
            
            return news_df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            print("Raw response content:")
            print(response.content[:500])
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
