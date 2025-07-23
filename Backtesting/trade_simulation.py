#!/usr/bin/env python3
"""
Trade Simulation for Backtesting
Simulates trades using Polygon API for historical bid/ask prices
Entry: BUY on ask 30 seconds after article publication
Exit: SELL on bid at exactly 9:28 AM EST
Only trades articles with sentiment='BUY' and confidence='high'
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
    Uses Polygon API for bid/ask price data
    """
    
    def __init__(self):
        self.ch_manager = None
        self.session = None
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        
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
        
        # Stats tracking
        self.stats = {
            'articles_processed': 0,
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
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
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
            
            # Use a simple status endpoint
            test_url = f"{self.base_url}/v1/marketstatus/now"
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

    async def get_articles_for_trading(self, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Get articles with sentiment analysis that meet trading criteria"""
        try:
            # Get articles with BUY recommendation and high confidence that haven't been traded yet
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
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles for trading: {e}")
            return []

    def calculate_entry_exit_times(self, published_utc: datetime) -> tuple:
        """Calculate entry and exit times for the trade"""
        try:
            # Convert published time to EST
            if published_utc.tzinfo is None:
                published_utc = published_utc.replace(tzinfo=pytz.UTC)
            
            published_est = published_utc.astimezone(self.est_tz)
            
            # Entry time: 30 seconds after publication
            entry_time_est = published_est + timedelta(seconds=30)
            
            # Exit time: 9:28 AM EST on the same day
            exit_time_est = published_est.replace(hour=9, minute=28, second=0, microsecond=0)
            
            # If publication is after 9:28 AM, skip this trade
            if published_est.time() >= datetime.strptime(self.exit_time_est, "%H:%M:%S").time():
                return None, None
            
            # Convert back to UTC for API calls
            entry_time_utc = entry_time_est.astimezone(pytz.UTC)
            exit_time_utc = exit_time_est.astimezone(pytz.UTC)
            
            return entry_time_utc, exit_time_utc
            
        except Exception as e:
            logger.error(f"Error calculating trade times: {e}")
            return None, None

    async def get_nbbo_quote(self, ticker: str, timestamp: datetime, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Get NBBO (National Best Bid and Offer) quote from Polygon API"""
        self.stats['api_calls'] += 1
        
        for attempt in range(max_retries):
            try:
                # Format timestamp for Polygon API (nanoseconds)
                timestamp_ns = int(timestamp.timestamp() * 1_000_000_000)
                
                # Polygon NBBO endpoint
                url = f"{self.base_url}/v3/quotes/{ticker}"
                params = {
                    'timestamp': timestamp_ns,
                    'order': 'desc',
                    'limit': 1,
                    'apikey': self.polygon_api_key
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            quote = data['results'][0]
                            
                            bid_price = quote.get('bid', 0.0)
                            ask_price = quote.get('ask', 0.0)
                            bid_size = quote.get('bid_size', 0)
                            ask_size = quote.get('ask_size', 0)
                            quote_timestamp = quote.get('participant_timestamp', timestamp_ns)
                            
                            if bid_price > 0 and ask_price > 0:
                                return {
                                    'bid_price': bid_price,
                                    'ask_price': ask_price,
                                    'bid_size': bid_size,
                                    'ask_size': ask_size,
                                    'timestamp': datetime.fromtimestamp(quote_timestamp / 1_000_000_000, tz=pytz.UTC),
                                    'spread': ask_price - bid_price,
                                    'mid_price': (bid_price + ask_price) / 2.0
                                }
                        
                        logger.debug(f"No valid quotes found for {ticker} at {timestamp}")
                        return None
                        
                    elif response.status == 429:
                        # Rate limit - wait and retry
                        wait_time = (2 ** attempt) * 1
                        logger.warning(f"Rate limited for {ticker}, waiting {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        response_text = await response.text()
                        logger.warning(f"Polygon API error for {ticker}: {response.status} - {response_text}")
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        else:
                            self.stats['api_errors'] += 1
                            return None
                            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting quote for {ticker} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.stats['api_errors'] += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting quote for {ticker}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    self.stats['api_errors'] += 1
                    return None
        
        return None

    async def simulate_trade(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate a single trade based on article sentiment"""
        try:
            ticker = article['ticker']
            published_utc = article['published_utc']
            
            logger.info(f"💰 Simulating trade for {ticker} published at {published_utc}")
            
            # Calculate entry and exit times
            entry_time, exit_time = self.calculate_entry_exit_times(published_utc)
            
            if not entry_time or not exit_time:
                logger.warning(f"❌ Cannot calculate trade times for {ticker} - skipping")
                return None
            
            # Get entry quote (buy on ask)
            logger.debug(f"📈 Getting entry quote for {ticker} at {entry_time}")
            entry_quote = await self.get_nbbo_quote(ticker, entry_time)
            
            if not entry_quote:
                logger.warning(f"❌ No entry quote available for {ticker} - skipping trade")
                return None
            
            # Get exit quote (sell on bid)
            logger.debug(f"📉 Getting exit quote for {ticker} at {exit_time}")
            exit_quote = await self.get_nbbo_quote(ticker, exit_time)
            
            if not exit_quote:
                logger.warning(f"❌ No exit quote available for {ticker} - skipping trade")
                return None
            
            # Calculate trade results
            entry_price = entry_quote['ask_price']  # BUY on ask
            exit_price = exit_quote['bid_price']    # SELL on bid
            quantity = self.default_quantity
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * quantity
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            
            # Calculate trade duration
            trade_duration = int((exit_time - entry_time).total_seconds())
            
            # Generate unique trade ID
            trade_id = f"{ticker}_{int(published_utc.timestamp())}_{uuid.uuid4().hex[:8]}"
            
            # Create trade record
            trade_record = {
                'trade_id': trade_id,
                'ticker': ticker,
                'article_url': article['article_url'],
                'published_utc': published_utc,
                'entry_time': entry_time,
                'exit_time': exit_time,
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
                'entry_spread': entry_quote['spread'],
                'exit_spread': exit_quote['spread'],
                'entry_mid_price': entry_quote['mid_price'],
                'exit_mid_price': exit_quote['mid_price']
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
            logger.info(f"{result_emoji} TRADE COMPLETE: {ticker}")
            logger.info(f"   📊 Entry: ${entry_price:.4f} at {entry_time.strftime('%H:%M:%S')} EST")
            logger.info(f"   📊 Exit:  ${exit_price:.4f} at {exit_time.strftime('%H:%M:%S')} EST")
            logger.info(f"   💰 P&L:   ${pnl:.2f} ({pnl_percent:+.2f}%)")
            logger.info(f"   ⏱️ Duration: {trade_duration}s")
            
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
            logger.info("💰 Starting Trade Simulation...")
            
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
                
                logger.info(f"💰 BATCH {batch_count}: Simulating trades for {len(articles)} articles...")
                self.stats['articles_processed'] += len(articles)
                
                # Process trades with controlled parallelism to avoid rate limits
                trade_records = []
                semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
                
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
                logger.info(f"📈 BATCH {batch_count} COMPLETE: {len(trade_records)}/{len(articles)} successful trades")
                logger.info(f"🔄 TOTAL PROGRESS: {total_processed} articles processed, {self.stats['trades_simulated']} trades simulated")
                
                # Rate limiting between batches
                await asyncio.sleep(3)
            
            # Final stats
            elapsed = time.time() - self.stats['start_time']
            win_rate = (self.stats['trades_profitable'] / self.stats['trades_simulated'] * 100) if self.stats['trades_simulated'] > 0 else 0
            avg_pnl = self.stats['total_pnl'] / self.stats['trades_simulated'] if self.stats['trades_simulated'] > 0 else 0
            
            logger.info("🎉 TRADE SIMULATION COMPLETE!")
            logger.info(f"📊 FINAL STATS:")
            logger.info(f"  • Articles processed: {self.stats['articles_processed']}")
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
    simulator = TradeSimulator()
    success = await simulator.run_trade_simulation()
    
    if success:
        print("\n✅ Trade simulation completed successfully!")
    else:
        print("\n❌ Trade simulation failed!")

if __name__ == "__main__":
    asyncio.run(main()) 