#!/usr/bin/env python3
"""
Standalone Price Monitor
Reads news from ClickHouse and monitors for price moves independently of news collection
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
import pytz
from price_checker import PriceChecker
from clickhouse_setup import setup_clickhouse_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandalonePriceMonitor:
    def __init__(self):
        self.clickhouse_manager = None
        self.price_checker = None
        self.processed_articles = set()
        
        # Statistics
        self.stats = {
            'articles_checked': 0,
            'price_moves_detected': 0,
            'errors': 0,
            'start_time': time.time()
        }

    async def initialize(self):
        """Initialize the price monitor"""
        logger.info("Initializing standalone price monitor...")
        
        # Setup ClickHouse connection
        self.clickhouse_manager = setup_clickhouse_database()
        
        # Setup price checker
        self.price_checker = PriceChecker()
        
        logger.info("Standalone price monitor initialized")

    async def get_recent_news(self, minutes_back=30):
        """Get recent news articles from ClickHouse"""
        try:
            # Calculate time threshold
            cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
            
            # Query for recent articles with ticker info
            query = f"""
            SELECT DISTINCT 
                ticker,
                headline,
                published_utc,
                article_url,
                processing_latency_ms,
                content_hash
            FROM breaking_news 
            WHERE published_utc >= '{cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND ticker != ''
            ORDER BY published_utc DESC
            """
            
            result = self.clickhouse_manager.client.execute(query)
            
            articles = []
            for row in result:
                article = {
                    'ticker': row[0],
                    'headline': row[1], 
                    'published_utc': row[2],
                    'article_url': row[3],
                    'processing_latency_ms': row[4],
                    'content_hash': row[5]
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []

    async def process_article(self, article):
        """Process a single article for price checking"""
        try:
            ticker = article['ticker']
            content_hash = article['content_hash']
            
            # Skip if already processed
            if content_hash in self.processed_articles:
                return
            
            logger.info(f"Checking price for {ticker}: {article['headline'][:50]}...")
            
            # Format news data for price checker
            news_data = {
                'headline': article['headline'],
                'published_utc': article['published_utc'],
                'article_url': article['article_url'],
                'detected_at': datetime.now(pytz.UTC)
            }
            
            # Check for price move
            price_move_data = await self.price_checker.check_price_move(ticker, news_data)
            
            if price_move_data:
                logger.info(f"🚨 PRICE MOVE DETECTED: {ticker}")
                logger.info(f"   Current: ${price_move_data['current_price']:.2f}")
                logger.info(f"   Prev Close: ${price_move_data['previous_close']:.2f}")
                logger.info(f"   Price Change: {price_move_data['price_change_percentage']:.1f}%")
                
                # Insert into price_move table
                await self.log_price_move(ticker, article, price_move_data)
                
                self.stats['price_moves_detected'] += 1
            
            # Mark as processed
            self.processed_articles.add(content_hash)
            self.stats['articles_checked'] += 1
            
        except Exception as e:
            logger.error(f"Error processing article {article.get('content_hash', 'unknown')}: {e}")
            self.stats['errors'] += 1

    async def log_price_move(self, ticker, article, price_move_data):
        """Log price move to ClickHouse"""
        try:
            # Insert into price_move table
            query = """
            INSERT INTO price_move VALUES
            """
            
            values = [(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ticker,
                article['headline'],
                article['published_utc'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(article['published_utc'], datetime) else str(article['published_utc']),
                article['article_url'],
                float(price_move_data['current_price']),
                float(price_move_data['previous_close']),
                float(price_move_data['price_change_percentage']),
                int(price_move_data.get('volume_change_percentage', 0)),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )]
            
            self.clickhouse_manager.client.execute(query, values)
            logger.info(f"Logged price move for {ticker} to database")
            
        except Exception as e:
            logger.error(f"Error logging price move: {e}")

    async def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting price monitoring loop...")
        
        while True:
            try:
                # Get recent news articles
                articles = await self.get_recent_news(minutes_back=30)
                
                if articles:
                    logger.info(f"Found {len(articles)} recent articles to check")
                    
                    # Process each article
                    for article in articles:
                        await self.process_article(article)
                        
                        # Small delay between checks to avoid rate limiting
                        await asyncio.sleep(0.5)
                
                # Report stats every 10 cycles
                if self.stats['articles_checked'] % 50 == 0 and self.stats['articles_checked'] > 0:
                    await self.report_stats()
                
                # Wait before next check
                logger.info("Waiting 60 seconds before next price check cycle...")
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(10)

    async def report_stats(self):
        """Report monitoring statistics"""
        runtime = time.time() - self.stats['start_time']
        rate = self.stats['articles_checked'] / runtime if runtime > 0 else 0
        
        logger.info(f"📊 PRICE MONITOR STATS:")
        logger.info(f"   Runtime: {runtime/60:.1f} minutes")
        logger.info(f"   Articles Checked: {self.stats['articles_checked']}")
        logger.info(f"   Price Moves Found: {self.stats['price_moves_detected']}")
        logger.info(f"   Errors: {self.stats['errors']}")
        logger.info(f"   Check Rate: {rate:.2f} articles/sec")
        
        # Clean up processed set to prevent memory growth
        if len(self.processed_articles) > 1000:
            # Keep only the most recent 500
            self.processed_articles = set(list(self.processed_articles)[-500:])

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.price_checker:
                await self.price_checker.close()
            
            if self.clickhouse_manager:
                self.clickhouse_manager.close()
                
            logger.info("Price monitor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def start(self):
        """Start the price monitor"""
        try:
            await self.initialize()
            await self.monitoring_loop()
        except KeyboardInterrupt:
            logger.info("Price monitor stopped by user")
        except Exception as e:
            logger.error(f"Error in price monitor: {e}")
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    logger.info("🚀 Starting Standalone Price Monitor")
    logger.info("This monitor reads news from ClickHouse and checks for price moves")
    
    monitor = StandalonePriceMonitor()
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main()) 