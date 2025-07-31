#!/usr/bin/env python3
"""
Price Movement Analyzer for Historical News Articles
Analyzes price movements to identify articles that preceded significant price increases
Creates a new table with successful matches instead of modifying existing tables
Enhanced to detect "false pumps" - cases where price increases 30%+ but falls back to within 10% of initial price
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pytz

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceMovementAnalyzer:
    """
    Analyzes historical news articles to identify those that preceded significant price movements
    Creates a new table with results instead of modifying existing tables
    Enhanced to detect "false pumps" where price increases but then falls back
    """
    
    def __init__(self):
        self.ch_manager = None
        self.est_tz = pytz.timezone('US/Eastern')
        self.stats = {
            'articles_analyzed': 0,
            'articles_with_data': 0,
            'articles_with_30pct_increase': 0,
            'articles_with_false_pump': 0,
            'articles_no_price_data': 0,
            'start_time': datetime.now()
        }
    
    async def initialize(self):
        """Initialize the analyzer"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create the results table
            await self.create_results_table()
            
            logger.info("✅ Price Movement Analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            return False
    
    async def create_results_table(self):
        """Create a new table to store price movement analysis results with false pump detection"""
        try:
            # Drop existing table if it exists
            logger.info("🗑️ Dropping existing price_movement_analysis table (if exists)...")
            self.ch_manager.client.command("DROP TABLE IF EXISTS News.price_movement_analysis")
            
            # Create new table with enhanced schema
            create_table_sql = """
            CREATE TABLE News.price_movement_analysis (
                ticker String,
                headline String,
                article_url String,
                published_est DateTime,
                newswire_type String,
                content_hash String,
                entry_time DateTime,
                exit_time DateTime,
                entry_price Float64,
                exit_price Float64,
                max_price Float64,
                price_increase_ratio Float64,
                max_price_ratio Float64,
                has_30pct_increase UInt8,
                is_false_pump UInt8,
                analysis_date DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (ticker, published_est)
            PARTITION BY toYYYYMM(published_est)
            """
            
            self.ch_manager.client.command(create_table_sql)
            logger.info("✅ Created fresh price_movement_analysis table with false pump detection")
            
        except Exception as e:
            logger.error(f"Error creating results table: {e}")
            raise
    
    async def get_articles_for_analysis(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Get articles that need price movement analysis"""
        try:
            # Calculate 6 months ago from today
            six_months_ago = datetime.now() - timedelta(days=180)
            cutoff_date = six_months_ago.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get all articles in the time window - removed artificial limit to capture all months
            query = """
            SELECT 
                ticker, 
                headline, 
                article_url, 
                published_est,
                newswire_type,
                content_hash
            FROM News.historical_news 
            WHERE published_est >= %s
            AND EXTRACT(HOUR FROM published_est) BETWEEN 6 AND 9
            ORDER BY published_est ASC
            """
            
            result = self.ch_manager.client.query(query, parameters=[cutoff_date])
            
            articles = []
            for row in result.result_rows:
                articles.append({
                    'ticker': row[0],
                    'headline': row[1],
                    'article_url': row[2],
                    'published_est': row[3],
                    'newswire_type': row[4],
                    'content_hash': row[5]
                })
            
            logger.info(f"📄 Found {len(articles)} articles to analyze (last 6 months, 6:00-8:59 AM)")
            
            # Log distribution by month for better visibility
            from collections import defaultdict
            monthly_counts = defaultdict(int)
            for article in articles:
                month_key = article['published_est'].strftime('%Y-%m')
                monthly_counts[month_key] += 1
            
            logger.info("📊 Articles by month:")
            for month in sorted(monthly_counts.keys()):
                logger.info(f"  • {month}: {monthly_counts[month]} articles")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles for analysis: {e}")
            return []
    
    async def get_price_data_for_analysis(self, ticker: str, published_est: datetime) -> Optional[Dict[str, Any]]:
        """Get price data needed for movement analysis including all bars for max price tracking"""
        try:
            # Calculate key timestamps
            published_est_tz = published_est.replace(tzinfo=pytz.UTC).astimezone(self.est_tz)
            
            # Price at published_est + 30 seconds
            entry_time = published_est + timedelta(seconds=30)
            
            # Price at 9:28 AM EST on the same date
            exit_time_est = published_est_tz.replace(hour=9, minute=28, second=0, microsecond=0)
            exit_time_utc = exit_time_est.astimezone(pytz.UTC)
            
            # Query price data from published time to 9:30 AM EST
            start_time = published_est
            end_time = published_est_tz.replace(hour=9, minute=30, second=0, microsecond=0).astimezone(pytz.UTC)
            
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM News.historical_price
            WHERE ticker = '{ticker}'
            AND timestamp >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND timestamp <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
            ORDER BY timestamp
            """
            
            result = self.ch_manager.client.query(query)
            
            if len(result.result_rows) == 0:
                logger.debug(f"No price data found for {ticker} at {published_est}")
                return None
            
            bars = []
            for row in result.result_rows:
                bars.append({
                    'timestamp': row[0],
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5]
                })
            
            # Find closest bars to our target times
            entry_price = self.find_closest_price(bars, entry_time)
            exit_price = self.find_closest_price(bars, exit_time_utc)
            
            if entry_price is None or exit_price is None:
                logger.debug(f"Could not find entry/exit prices for {ticker}")
                return None
            
            # Find maximum price during the entire period
            max_price = self.find_max_price_in_period(bars, entry_time, exit_time_utc)
            
            if max_price is None:
                max_price = max(entry_price, exit_price)  # Fallback to higher of entry/exit
            
            return {
                'entry_time': entry_time,
                'exit_time': exit_time_utc,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'max_price': max_price,
                'bars_count': len(bars)
            }
            
        except Exception as e:
            logger.debug(f"Error getting price data for {ticker}: {e}")
            return None
    
    def find_closest_price(self, bars: List[Dict[str, Any]], target_time: datetime) -> Optional[float]:
        """Find the closest price to the target time"""
        if not bars:
            return None
        
        closest_bar = None
        min_diff = float('inf')
        
        for bar in bars:
            bar_time = bar['timestamp']
            if isinstance(bar_time, str):
                bar_time = datetime.fromisoformat(bar_time.replace('Z', '+00:00'))
            elif bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=pytz.UTC)
            
            # Ensure target_time has timezone info
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=pytz.UTC)
            
            time_diff = abs((bar_time - target_time).total_seconds())
            
            if time_diff < min_diff:
                min_diff = time_diff
                closest_bar = bar
        
        return closest_bar['close'] if closest_bar else None
    
    def find_max_price_in_period(self, bars: List[Dict[str, Any]], start_time: datetime, end_time: datetime) -> Optional[float]:
        """Find the maximum price (using high values) during the specified period"""
        if not bars:
            return None
        
        max_price = None
        
        for bar in bars:
            bar_time = bar['timestamp']
            if isinstance(bar_time, str):
                bar_time = datetime.fromisoformat(bar_time.replace('Z', '+00:00'))
            elif bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=pytz.UTC)
            
            # Ensure times have timezone info
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=pytz.UTC)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=pytz.UTC)
            
            # Check if bar is within our time period
            if start_time <= bar_time <= end_time:
                bar_high = bar['high']
                if max_price is None or bar_high > max_price:
                    max_price = bar_high
        
        return max_price
    
    def calculate_price_movement(self, entry_price: float, exit_price: float, max_price: float) -> tuple[float, float, bool, bool]:
        """
        Calculate price movement ratios and detect both 30% increases and false pumps
        
        Args:
            entry_price: Price at entry time (published_est + 30 seconds)
            exit_price: Price at exit time (9:28 AM EST)
            max_price: Maximum price reached during the period
        
        Returns:
            (price_increase_ratio, max_price_ratio, is_30_percent_increase, is_false_pump)
        """
        if entry_price <= 0:  # Avoid division by zero
            return 0.0, 0.0, False, False
        
        # Calculate ratios
        price_increase_ratio = exit_price / entry_price
        max_price_ratio = max_price / entry_price
        
        # Check for 30% increase from entry to exit
        is_30_percent_increase = price_increase_ratio >= 1.30
        
        # Check for false pump:
        # 1. Max price during period was 20%+ above entry price (reduced from 30%)
        # 2. Exit price fell back to within 10% of entry price (between 0.90x and 1.10x)
        had_20pct_pump = max_price_ratio >= 1.20
        fell_back_to_initial = 0.90 <= price_increase_ratio <= 1.10
        is_false_pump = had_20pct_pump and fell_back_to_initial
        
        return price_increase_ratio, max_price_ratio, is_30_percent_increase, is_false_pump
    
    async def analyze_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single article for price movement including false pump detection"""
        try:
            ticker = article['ticker']
            published_est = article['published_est']
            
            # Get price data
            price_data = await self.get_price_data_for_analysis(ticker, published_est)
            
            if price_data is None:
                self.stats['articles_no_price_data'] += 1
                return None  # No price data available
            
            # Calculate price movement and false pump detection
            price_ratio, max_price_ratio, has_30pct_increase, is_false_pump = self.calculate_price_movement(
                price_data['entry_price'], 
                price_data['exit_price'],
                price_data['max_price']
            )
            
            self.stats['articles_with_data'] += 1
            if has_30pct_increase:
                self.stats['articles_with_30pct_increase'] += 1
                logger.info(f"📈 30%+ increase found: {ticker} - {article['headline'][:50]}... ({price_ratio:.2f}x)")
            
            if is_false_pump:
                self.stats['articles_with_false_pump'] += 1
                logger.info(f"🎢 False pump detected: {ticker} - {article['headline'][:50]}... (max: {max_price_ratio:.2f}x, final: {price_ratio:.2f}x)")
            
            # Return complete analysis result
            return {
                'ticker': ticker,
                'headline': article['headline'],
                'article_url': article['article_url'],
                'published_est': published_est,
                'newswire_type': article['newswire_type'],
                'content_hash': article['content_hash'],
                'entry_time': price_data['entry_time'],
                'exit_time': price_data['exit_time'],
                'entry_price': price_data['entry_price'],
                'exit_price': price_data['exit_price'],
                'max_price': price_data['max_price'],
                'price_increase_ratio': price_ratio,
                'max_price_ratio': max_price_ratio,
                'has_30pct_increase': 1 if has_30pct_increase else 0,
                'is_false_pump': 1 if is_false_pump else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing article {article.get('content_hash', 'unknown')}: {e}")
            return None
    
    async def store_results(self, results: List[Dict[str, Any]]):
        """Store analysis results in the new table with false pump data"""
        if not results:
            return
        
        try:
            # Prepare data for insertion
            result_data = []
            for result in results:
                result_data.append((
                    result['ticker'],
                    result['headline'],
                    result['article_url'],
                    result['published_est'],
                    result['newswire_type'],
                    result['content_hash'],
                    result['entry_time'],
                    result['exit_time'],
                    result['entry_price'],
                    result['exit_price'],
                    result['max_price'],
                    result['price_increase_ratio'],
                    result['max_price_ratio'],
                    result['has_30pct_increase'],
                    result['is_false_pump'],
                    datetime.now()
                ))
            
            # Insert results
            self.ch_manager.client.insert(
                'News.price_movement_analysis',
                result_data,
                column_names=[
                    'ticker', 'headline', 'article_url', 'published_est', 'newswire_type', 
                    'content_hash', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 
                    'max_price', 'price_increase_ratio', 'max_price_ratio', 'has_30pct_increase', 
                    'is_false_pump', 'analysis_date'
                ]
            )
            
            logger.info(f"✅ Stored {len(result_data)} analysis results")
            
        except Exception as e:
            logger.error(f"Error storing results: {e}")
    
    async def run_analysis(self, batch_size: int = 50):
        """Run the complete price movement analysis with false pump detection"""
        try:
            logger.info("🚀 Starting Price Movement Analysis with False Pump Detection...")
            
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize analyzer")
                return False
            
            # Get all articles to analyze (removed hardcoded limit)
            articles = await self.get_articles_for_analysis()
            
            if not articles:
                logger.info("No articles found to analyze")
                return True
            
            logger.info(f"📊 Analyzing {len(articles)} articles...")
            
            results = []
            
            for i, article in enumerate(articles):
                try:
                    # Analyze the article
                    result = await self.analyze_article(article)
                    
                    if result:
                        results.append(result)
                    
                    self.stats['articles_analyzed'] += 1
                    
                    # Progress logging every 100 articles and show current month being processed
                    if (i + 1) % 100 == 0:
                        current_month = article['published_est'].strftime('%Y-%m')
                        logger.info(f"📈 PROGRESS: {i + 1}/{len(articles)} articles analyzed (currently processing {current_month})")
                        logger.info(f"  • With price data: {self.stats['articles_with_data']}")
                        logger.info(f"  • With 30%+ increase: {self.stats['articles_with_30pct_increase']}")
                        logger.info(f"  • With false pumps: {self.stats['articles_with_false_pump']}")
                        logger.info(f"  • No price data: {self.stats['articles_no_price_data']}")
                    
                    # Store results in batches
                    if len(results) >= 20:
                        await self.store_results(results)
                        results = []
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.get('content_hash', 'unknown')}: {e}")
                    continue
            
            # Store remaining results
            if results:
                await self.store_results(results)
            
            # Final statistics
            elapsed = datetime.now() - self.stats['start_time']
            logger.info("🎉 PRICE MOVEMENT ANALYSIS COMPLETE!")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Total articles analyzed: {self.stats['articles_analyzed']}")
            logger.info(f"  • Articles with price data: {self.stats['articles_with_data']}")
            logger.info(f"  • Articles with 30%+ increase: {self.stats['articles_with_30pct_increase']}")
            logger.info(f"  • Articles with false pumps: {self.stats['articles_with_false_pump']}")
            logger.info(f"  • Articles without price data: {self.stats['articles_no_price_data']}")
            logger.info(f"  • Time elapsed: {elapsed}")
            
            if self.stats['articles_with_data'] > 0:
                success_rate = (self.stats['articles_with_30pct_increase'] / self.stats['articles_with_data']) * 100
                false_pump_rate = (self.stats['articles_with_false_pump'] / self.stats['articles_with_data']) * 100
                logger.info(f"  • 30%+ increase rate: {success_rate:.2f}%")
                logger.info(f"  • False pump rate: {false_pump_rate:.2f}%")
            
            logger.info(f"📋 Results stored in News.price_movement_analysis table")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in price movement analysis: {e}")
            return False
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.ch_manager:
                self.ch_manager.close()
            logger.info("✅ Price Movement Analyzer cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function"""
    analyzer = PriceMovementAnalyzer()
    success = await analyzer.run_analysis()
    
    if success:
        print("\n✅ Price movement analysis completed successfully!")
        print("📋 Results are stored in the News.price_movement_analysis table")
        print("\nQuery examples:")
        print("-- Get articles with 30%+ increases:")
        print("SELECT * FROM News.price_movement_analysis WHERE has_30pct_increase = 1;")
        print("\n-- Get articles with false pumps:")
        print("SELECT * FROM News.price_movement_analysis WHERE is_false_pump = 1;")
        print("\n-- Get summary statistics:")
        print("SELECT has_30pct_increase, is_false_pump, COUNT(*) FROM News.price_movement_analysis GROUP BY has_30pct_increase, is_false_pump;")
        print("\n-- Get false pump examples with details:")
        print("SELECT ticker, headline, entry_price, max_price, exit_price, max_price_ratio, price_increase_ratio")
        print("FROM News.price_movement_analysis WHERE is_false_pump = 1 ORDER BY max_price_ratio DESC;")
    else:
        print("\n❌ Price movement analysis failed!")

if __name__ == "__main__":
    asyncio.run(main()) 