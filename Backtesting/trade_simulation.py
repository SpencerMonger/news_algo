#!/usr/bin/env python3
"""
Trade Simulation for Backtesting
Uses Polygon API 10-second aggregate bars for historical price data
Entry: BUY 30 seconds after initial timestamp if conditions met
Exit: SELL at exactly 9:28 AM EST
Trade conditions:
1. Price increases 5%+ within first 40 seconds of initial timestamp
2. Sentiment analysis shows 'BUY' with 'high' confidence
3. Only trades articles published between 7am-9:30am EST
"""

import asyncio
import aiohttp
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pytz
import uuid

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

class TradeSimulator:
    """
    Trade simulation system for backtesting historical news-based trades
    Uses Polygon API 10-second aggregate bars for price data
    """
    
    def __init__(self, skip_sentiment_check: bool = False):
        self.ch_manager = None
        self.session = None
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        
        # NEW: Skip sentiment requirement for testing
        self.skip_sentiment_check = skip_sentiment_check
        
        if not self.polygon_api_key:
            logger.error("POLYGON_API_KEY environment variable not set")
            raise ValueError("Polygon API key is required")
        
        # Use PROXY_URL if available
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.base_url = proxy_url.rstrip('/')
            logger.info(f"Using proxy URL: {self.base_url}")
        else:
            self.base_url = "https://api.polygon.io"
            logger.info("Using official Polygon API")
        
        # EST timezone for trade timing
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Trade parameters
        self.default_quantity = 100  # Default shares per trade
        self.exit_time_est = "09:28:00"  # Exit time in EST
        self.trading_start_time_est = "07:00:00"  # Start of trading window
        self.trading_end_time_est = "09:30:00"  # End of trading window
        self.price_increase_threshold = 5.0  # 5% price increase required
        self.price_check_window_seconds = 40  # Check for price increase within 40 seconds
        
        # Stats tracking
        self.stats = {
            'articles_processed': 0,
            'articles_filtered_time': 0,
            'articles_filtered_no_price_data': 0,
            'articles_filtered_no_5pct_move': 0,
            'trades_simulated': 0,
            'trades_profitable': 0,
            'trades_unprofitable': 0,
            'total_pnl': 0.0,
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time()
        }

    async def initialize(self):
        """Initialize the trade simulator"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            # Create HTTP session for Polygon API
            timeout = aiohttp.ClientTimeout(total=60, connect=15)  # Increased timeout for historical data
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Test Polygon API connectivity
            if not await self.test_polygon_api():
                logger.error("❌ Failed to connect to Polygon API")
                return False
            
            logger.info("✅ Trade Simulator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trade simulator: {e}")
            return False

    async def test_polygon_api(self) -> bool:
        """Test Polygon API connectivity"""
        try:
            logger.info("🔍 Testing Polygon API connection...")
            
            # Use aggregates endpoint for testing
            test_url = f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02"
            params = {'apikey': self.polygon_api_key}
            
            async with self.session.get(test_url, params=params) as response:
                if response.status == 200:
                    logger.info("✅ Polygon API connection successful")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(f"❌ Polygon API test failed: {response.status} - {response_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Polygon API test failed: {e}")
            return False

    def is_trading_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within trading hours (7am-9:30am EST)"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        
        est_time = timestamp.astimezone(self.est_tz)
        time_str = est_time.strftime("%H:%M:%S")
        
        return self.trading_start_time_est <= time_str <= self.trading_end_time_est

    async def get_articles_for_trading(self, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Get articles with sentiment analysis that meet trading criteria, filtered by time"""
        try:
            if self.skip_sentiment_check:
                # Skip sentiment requirements - get ANY articles for testing
                logger.info("🔄 TESTING MODE: Skipping sentiment requirements - getting all articles in trading hours")
                query = """
                SELECT 
                    hn.ticker,
                    hn.headline,
                    hn.article_url,
                    hn.published_utc,
                    'neutral' as sentiment,
                    'HOLD' as recommendation,
                    'medium' as confidence,
                    'Testing mode - sentiment bypassed' as explanation,
                    hn.content_hash
                FROM News.historical_news hn
                LEFT JOIN News.backtest_trades bt
                    ON hn.content_hash = bt.article_url  -- Using article_url as unique identifier
                WHERE bt.trade_id IS NULL  -- Not already traded
                AND hn.ticker != ''
                ORDER BY hn.published_utc ASC
                LIMIT %s
                """
            else:
                # Normal mode - require BUY/high sentiment
                query = """
                SELECT 
                    hs.ticker,
                    hs.headline,
                    hs.article_url,
                    hs.published_utc,
                    hs.sentiment,
                    hs.recommendation,
                    hs.confidence,
                    hs.explanation,
                    hs.content_hash
                FROM News.historical_sentiment hs
                LEFT JOIN News.backtest_trades bt
                    ON hs.content_hash = bt.article_url  -- Using article_url as unique identifier
                WHERE bt.trade_id IS NULL  -- Not already traded
                AND hs.recommendation = 'BUY'
                AND hs.confidence = 'high'
                AND hs.ticker != ''
                ORDER BY hs.published_utc ASC
                LIMIT %s
                """
            
            result = self.ch_manager.client.query(query, [batch_size])
            
            articles = []
            for row in result.result_rows:
                ticker, headline, article_url, published_utc, sentiment, recommendation, confidence, explanation, content_hash = row
                
                # Filter by trading hours (7am-9:30am EST)
                if not self.is_trading_hours(published_utc):
                    self.stats['articles_filtered_time'] += 1
                    logger.debug(f"⏰ Filtered out {ticker} - published outside trading hours: {published_utc}")
                    continue
                
                articles.append({
                    'ticker': ticker,
                    'headline': headline,
                    'article_url': article_url,
                    'published_utc': published_utc,
                    'sentiment': sentiment,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'explanation': explanation,
                    'content_hash': content_hash
                })
            
            logger.info(f"📊 Found {len(articles)} articles in trading hours (filtered {self.stats['articles_filtered_time']} outside 7am-9:30am EST)")
            if self.skip_sentiment_check:
                logger.info("🔄 TESTING MODE: Sentiment requirements bypassed - all articles will be tested for price movement only")
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles for trading: {e}")
            return []

    async def get_polygon_bars(self, ticker: str, from_timestamp: datetime, to_timestamp: datetime) -> List[Dict[str, Any]]:
        """Get 10-second aggregate bars from Polygon API"""
        self.stats['api_calls'] += 1
        
        try:
            # Convert timestamps to date strings for Polygon API
            from_date = from_timestamp.strftime('%Y-%m-%d')
            to_date = to_timestamp.strftime('%Y-%m-%d')
            
            # Polygon aggregates endpoint - 10 second bars
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/10/second/{from_date}/{to_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,  # Get all bars for the day
                'apikey': self.polygon_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        bars = []
                        for bar in data['results']:
                            bar_timestamp = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.UTC)
                            
                            # Filter bars to only include the time range we need
                            if from_timestamp <= bar_timestamp <= to_timestamp:
                                bars.append({
                                    'timestamp': bar_timestamp,
                                    'open': bar['o'],
                                    'high': bar['h'],
                                    'low': bar['l'],
                                    'close': bar['c'],
                                    'volume': bar['v']
                                })
                        
                        logger.debug(f"📊 Retrieved {len(bars)} 10-second bars for {ticker} from {from_timestamp} to {to_timestamp}")
                        return bars
                    else:
                        logger.debug(f"No bar data available for {ticker} on {from_date}")
                        return []
                        
                elif response.status == 429:
                    logger.warning(f"Rate limited for {ticker} bars - waiting before retry")
                    await asyncio.sleep(1)
                    return []
                else:
                    response_text = await response.text()
                    logger.warning(f"Polygon bars API error for {ticker}: {response.status} - {response_text}")
                    self.stats['api_errors'] += 1
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting bars for {ticker}: {e}")
            self.stats['api_errors'] += 1
            return []

    def check_price_increase_condition(self, bars: List[Dict[str, Any]], initial_timestamp: datetime) -> Dict[str, Any]:
        """Check if price increased 5% within first 40 seconds of initial timestamp"""
        if not bars:
            return {'meets_condition': False, 'reason': 'No price data available'}
        
        # Find the initial price (closest bar to initial timestamp)
        initial_bar = None
        min_time_diff = float('inf')
        
        for bar in bars:
            time_diff = abs((bar['timestamp'] - initial_timestamp).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                initial_bar = bar
        
        if not initial_bar:
            return {'meets_condition': False, 'reason': 'No initial price bar found'}
        
        initial_price = initial_bar['close']
        cutoff_time = initial_timestamp + timedelta(seconds=self.price_check_window_seconds)
        
        # Check for 5% increase within 40 seconds
        max_price = initial_price
        max_price_time = initial_timestamp
        
        for bar in bars:
            if initial_timestamp <= bar['timestamp'] <= cutoff_time:
                if bar['high'] > max_price:
                    max_price = bar['high']
                    max_price_time = bar['timestamp']
        
        price_increase_pct = ((max_price - initial_price) / initial_price) * 100
        
        meets_condition = price_increase_pct >= self.price_increase_threshold
        
        return {
            'meets_condition': meets_condition,
            'initial_price': initial_price,
            'max_price': max_price,
            'price_increase_pct': price_increase_pct,
            'max_price_time': max_price_time,
            'time_to_max': (max_price_time - initial_timestamp).total_seconds(),
            'reason': f'Price moved {price_increase_pct:.2f}% (need {self.price_increase_threshold}%)'
        }

    def get_entry_exit_prices(self, bars: List[Dict[str, Any]], entry_time: datetime, exit_time: datetime) -> tuple:
        """Get entry and exit prices from bars data"""
        entry_price = None
        exit_price = None
        
        # Find entry price (closest bar to entry time)
        min_entry_diff = float('inf')
        for bar in bars:
            time_diff = abs((bar['timestamp'] - entry_time).total_seconds())
            if time_diff < min_entry_diff:
                min_entry_diff = time_diff
                entry_price = bar['close']  # Use close price for entry
        
        # Find exit price (closest bar to exit time)
        min_exit_diff = float('inf')
        for bar in bars:
            time_diff = abs((bar['timestamp'] - exit_time).total_seconds())
            if time_diff < min_exit_diff:
                min_exit_diff = time_diff
                exit_price = bar['close']  # Use close price for exit
        
        return entry_price, exit_price

    async def simulate_trade(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate a single trade based on new conditions"""
        try:
            ticker = article['ticker']
            published_utc = article['published_utc']
            
            logger.info(f"💰 Analyzing potential trade for {ticker} published at {published_utc}")
            self.stats['articles_processed'] += 1
            
            # Convert published time to EST for calculations
            if published_utc.tzinfo is None:
                published_utc = published_utc.replace(tzinfo=pytz.UTC)
            
            published_est = published_utc.astimezone(self.est_tz)
            
            # Calculate time range: from initial timestamp to 9:30 AM EST
            end_time_est = published_est.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Convert back to UTC for API calls
            from_timestamp_utc = published_utc
            to_timestamp_utc = end_time_est.astimezone(pytz.UTC)
            
            # Get 10-second bars from Polygon
            logger.debug(f"📊 Getting 10-second bars for {ticker} from {from_timestamp_utc} to {to_timestamp_utc}")
            bars = await self.get_polygon_bars(ticker, from_timestamp_utc, to_timestamp_utc)
            
            if not bars:
                logger.warning(f"❌ No price data available for {ticker} - skipping")
                self.stats['articles_filtered_no_price_data'] += 1
                return None
            
            # Check if price increased 5% within first 40 seconds
            price_condition = self.check_price_increase_condition(bars, published_utc)
            
            if not price_condition['meets_condition']:
                logger.info(f"❌ {ticker} - {price_condition['reason']} - skipping trade")
                self.stats['articles_filtered_no_5pct_move'] += 1
                return None
            
            logger.info(f"✅ {ticker} - Price condition met: {price_condition['price_increase_pct']:.2f}% increase in {price_condition['time_to_max']:.0f}s")
            
            # Both conditions met - proceed with trade simulation
            # Entry: 30 seconds after initial timestamp
            entry_time_utc = published_utc + timedelta(seconds=30)
            
            # Exit: 9:28 AM EST
            exit_time_est = published_est.replace(hour=9, minute=28, second=0, microsecond=0)
            exit_time_utc = exit_time_est.astimezone(pytz.UTC)
            
            # Get entry and exit prices from bars
            entry_price, exit_price = self.get_entry_exit_prices(bars, entry_time_utc, exit_time_utc)
            
            if not entry_price or not exit_price:
                logger.warning(f"❌ Cannot find entry/exit prices for {ticker} - skipping")
                return None
            
            # Calculate trade results
            quantity = self.default_quantity
            pnl = (exit_price - entry_price) * quantity
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            trade_duration = int((exit_time_utc - entry_time_utc).total_seconds())
            
            # Generate unique trade ID
            trade_id = f"{ticker}_{int(published_utc.timestamp())}_{uuid.uuid4().hex[:8]}"
            
            # Create trade record
            trade_record = {
                'trade_id': trade_id,
                'ticker': ticker,
                'article_url': article['article_url'],
                'published_utc': published_utc,
                'entry_time': entry_time_utc,
                'exit_time': exit_time_utc,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'entry_type': 'BUY',
                'exit_type': 'SELL',
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'trade_duration_seconds': trade_duration,
                'sentiment': article['sentiment'],
                'recommendation': article['recommendation'],
                'confidence': article['confidence'],
                'explanation': article['explanation'],
                'initial_price': price_condition['initial_price'],
                'max_price_in_40s': price_condition['max_price'],
                'price_increase_pct': price_condition['price_increase_pct'],
                'bars_count': len(bars)
            }
            
            # Update stats
            self.stats['trades_simulated'] += 1
            if pnl > 0:
                self.stats['trades_profitable'] += 1
            else:
                self.stats['trades_unprofitable'] += 1
            self.stats['total_pnl'] += pnl
            
            # Log trade result
            result_emoji = "✅" if pnl > 0 else "❌"
            logger.info(f"{result_emoji} TRADE EXECUTED: {ticker}")
            logger.info(f"   📊 Entry: ${entry_price:.4f} at {entry_time_utc.astimezone(self.est_tz).strftime('%H:%M:%S')} EST")
            logger.info(f"   📊 Exit:  ${exit_price:.4f} at {exit_time_utc.astimezone(self.est_tz).strftime('%H:%M:%S')} EST")
            logger.info(f"   💰 P&L:   ${pnl:.2f} ({pnl_percent:+.2f}%)")
            logger.info(f"   📈 Price trigger: {price_condition['price_increase_pct']:.2f}% in {price_condition['time_to_max']:.0f}s")
            logger.info(f"   ⏱️ Duration: {trade_duration}s ({len(bars)} price bars)")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error simulating trade for {article.get('ticker', 'UNKNOWN')}: {e}")
            return None

    async def store_trade_results(self, trade_records: List[Dict[str, Any]]):
        """Store trade results in ClickHouse"""
        if not trade_records:
            return
        
        try:
            # Prepare data for insertion
            trade_data = []
            for trade in trade_records:
                trade_data.append((
                    trade['trade_id'],
                    trade['ticker'],
                    trade['article_url'],  # Used as unique identifier
                    trade['published_utc'],
                    trade['entry_time'],
                    trade['exit_time'],
                    trade['entry_price'],
                    trade['exit_price'],
                    trade['quantity'],
                    trade['entry_type'],
                    trade['exit_type'],
                    trade['pnl'],
                    trade['pnl_percent'],
                    trade['trade_duration_seconds'],
                    trade['sentiment'],
                    trade['recommendation'],
                    trade['confidence'],
                    trade['explanation'],
                    datetime.now()
                ))
            
            # Insert trade data
            self.ch_manager.client.insert(
                'News.backtest_trades',
                trade_data,
                column_names=['trade_id', 'ticker', 'article_url', 'published_utc', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'quantity', 'entry_type', 'exit_type', 'pnl', 'pnl_percent', 'trade_duration_seconds', 'sentiment', 'recommendation', 'confidence', 'explanation', 'created_at']
            )
            
            logger.info(f"✅ Stored {len(trade_data)} trade results in database")
            
        except Exception as e:
            logger.error(f"Error storing trade results: {e}")

    async def run_trade_simulation(self, batch_size: int = 20):
        """Run the complete trade simulation process"""
        try:
            if self.skip_sentiment_check:
                logger.info("💰 Starting TESTING MODE Trade Simulation - Price movement detection ONLY (sentiment bypassed)...")
            else:
                logger.info("💰 Starting Enhanced Trade Simulation with 10-second bars and 5% price movement detection...")
            
            # Initialize
            if not await self.initialize():
                logger.error("Failed to initialize trade simulator")
                return False
            
            total_processed = 0
            batch_count = 0
            
            while True:
                batch_count += 1
                
                # Get next batch of articles to trade
                logger.info(f"📊 BATCH {batch_count}: Getting articles for trading...")
                articles = await self.get_articles_for_trading(batch_size)
                
                if not articles:
                    logger.info("✅ No more articles to trade")
                    break
                
                logger.info(f"💰 BATCH {batch_count}: Analyzing {len(articles)} articles for trade conditions...")
                
                # Process trades with controlled parallelism to avoid rate limits
                trade_records = []
                semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls for historical data
                
                async def simulate_with_semaphore(article):
                    async with semaphore:
                        return await self.simulate_trade(article)
                
                # Execute trade simulations
                trade_tasks = [simulate_with_semaphore(article) for article in articles]
                trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)
                
                # Filter out exceptions and None results
                for i, result in enumerate(trade_results):
                    if isinstance(result, Exception):
                        logger.error(f"Exception in trade simulation for article {i}: {result}")
                        continue
                    if result:
                        trade_records.append(result)
                
                # Store trade results in database
                if trade_records:
                    await self.store_trade_results(trade_records)
                
                total_processed += len(articles)
                
                # Progress logging
                logger.info(f"📈 BATCH {batch_count} COMPLETE: {len(trade_records)}/{len(articles)} trades executed")
                logger.info(f"📊 FILTERING STATS: {self.stats['articles_filtered_time']} time filtered, {self.stats['articles_filtered_no_price_data']} no data, {self.stats['articles_filtered_no_5pct_move']} no 5% move")
                logger.info(f"🔄 TOTAL PROGRESS: {total_processed} articles processed, {self.stats['trades_simulated']} trades simulated")
                
                # Rate limiting between batches
                await asyncio.sleep(5)
            
            # Final stats
            elapsed = time.time() - self.stats['start_time']
            win_rate = (self.stats['trades_profitable'] / self.stats['trades_simulated'] * 100) if self.stats['trades_simulated'] > 0 else 0
            avg_pnl = self.stats['total_pnl'] / self.stats['trades_simulated'] if self.stats['trades_simulated'] > 0 else 0
            
            logger.info("🎉 ENHANCED TRADE SIMULATION COMPLETE!")
            if self.skip_sentiment_check:
                logger.info("🔄 TESTING MODE: Sentiment requirements were bypassed - trades based on price movement only")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Articles processed: {self.stats['articles_processed']}")
            logger.info(f"  • Time filtered (outside 7am-9:30am): {self.stats['articles_filtered_time']}")
            logger.info(f"  • No price data available: {self.stats['articles_filtered_no_price_data']}")
            logger.info(f"  • No 5% price movement: {self.stats['articles_filtered_no_5pct_move']}")
            logger.info(f"  • Trades simulated: {self.stats['trades_simulated']}")
            logger.info(f"  • Profitable trades: {self.stats['trades_profitable']}")
            logger.info(f"  • Unprofitable trades: {self.stats['trades_unprofitable']}")
            logger.info(f"  • Win rate: {win_rate:.1f}%")
            logger.info(f"  • Total P&L: ${self.stats['total_pnl']:.2f}")
            logger.info(f"  • Average P&L per trade: ${avg_pnl:.2f}")
            logger.info(f"  • API calls made: {self.stats['api_calls']}")
            logger.info(f"  • API errors: {self.stats['api_errors']}")
            logger.info(f"  • Time elapsed: {elapsed/60:.1f} minutes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trade simulation: {e}")
            return False
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.ch_manager:
            self.ch_manager.close()
        logger.info("✅ Trade simulator cleanup completed")

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trade Simulation for Backtesting')
    parser.add_argument('--skip-sentiment', action='store_true', 
                       help='Skip sentiment requirements for testing (price movement only)')
    
    args = parser.parse_args()
    
    simulator = TradeSimulator(skip_sentiment_check=args.skip_sentiment)
    success = await simulator.run_trade_simulation()
    
    if success:
        print("\n✅ Enhanced trade simulation completed successfully!")
    else:
        print("\n❌ Enhanced trade simulation failed!")

if __name__ == "__main__":
    asyncio.run(main()) 