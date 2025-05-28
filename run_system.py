import asyncio
import logging
import sys
import argparse
from finviz_scraper import FinvizScraper
from newswire_monitor import NewswireMonitor
from price_checker import ContinuousPriceMonitor
from clickhouse_setup import setup_clickhouse_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_news_monitor():
    """Run the news monitoring system"""
    try:
        logger.info("Starting news monitoring...")
        monitor = NewswireMonitor()
        await monitor.start_monitoring()
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
        
        # Step 2: Start both monitoring systems concurrently
        logger.info("Step 2: Starting news and price monitoring systems...")
        
        # Run both monitors concurrently
        await asyncio.gather(
            run_news_monitor(),
            run_price_monitor(),
            return_exceptions=True
        )
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        ch_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 