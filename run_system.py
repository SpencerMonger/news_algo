import asyncio
import logging
import sys
import argparse
from finviz_scraper import FinvizScraper
from web_scraper import Crawl4AIScraper
from price_checker import ContinuousPriceMonitor
from clickhouse_setup import setup_clickhouse_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_news_monitor(enable_old=False):
    """Run the news monitoring system"""
    try:
        logger.info("Starting news monitoring...")
        monitor = Crawl4AIScraper(enable_old=enable_old)
        await monitor.start_scraping()
    except Exception as e:
        logger.error(f"News monitor error: {e}")
        raise

async def run_price_monitor():
    """Run the price monitoring system"""
    try:
        logger.info("Starting price monitoring...")
        price_monitor = ContinuousPriceMonitor()
        await price_monitor.start()
    except Exception as e:
        logger.error(f"Price monitor error: {e}")
        raise

async def main():
    """Main function to update tickers and start both monitoring systems"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='News & Price Monitoring System')
    parser.add_argument('--skip-list', action='store_true', 
                       help='Skip the Finviz ticker list update step')
    parser.add_argument('--enable-old', action='store_true',
                       help='Disable freshness filter - allow processing of old news (for testing)')
    args = parser.parse_args()
    
    logger.info("=== Starting News & Price Monitoring System ===")
    
    # Setup ClickHouse
    ch_manager = setup_clickhouse_database()
    
    try:
        # Step 1: Update ticker list from Finviz (conditional)
        if not args.skip_list:
            logger.info("Step 1: Updating ticker list from Finviz...")
            scraper = FinvizScraper(ch_manager)
            success = await scraper.update_ticker_database()
            
            if not success:
                logger.warning("Ticker update failed, proceeding with existing data")
        else:
            logger.info("Step 1: Skipping Finviz ticker list update (--skip-list flag used)")
        
        # Step 2: Start monitoring systems with proper sequencing
        logger.info("Step 2: Starting monitoring systems with optimized sequencing...")
        
        if args.enable_old:
            logger.info("🔓 FRESHNESS FILTER DISABLED - Processing old news for testing")
        
        # FIXED: Initialize price checker first (needs to be ready for incoming tickers)
        logger.info("🚀 Phase 1: Initializing price checker...")
        price_monitor = ContinuousPriceMonitor()
        await price_monitor.initialize()
        logger.info("✅ Price checker initialized and ready for new tickers")
        
        # Start price monitoring loop in background
        price_task = asyncio.create_task(price_monitor.ultra_fast_monitoring_loop())
        
        # Small delay to ensure price checker is fully running
        await asyncio.sleep(0.5)
        
        # FIXED: Start news monitor (will start inserting articles immediately)
        logger.info("🚀 Phase 2: Starting news monitor...")
        news_monitor = Crawl4AIScraper(enable_old=args.enable_old)
        news_task = asyncio.create_task(news_monitor.start_scraping())
        
        logger.info("✅ Both systems running concurrently with optimized sequencing")
        
        # Wait for both tasks to complete
        await asyncio.gather(news_task, price_task, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        ch_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 