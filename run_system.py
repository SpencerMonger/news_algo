import asyncio
import logging
import sys
from finviz_scraper import FinvizScraper
from newswire_monitor import NewswireMonitor
from clickhouse_setup import setup_clickhouse_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to update tickers and start monitoring"""
    logger.info("=== Starting News Monitoring System ===")
    
    # Setup ClickHouse
    ch_manager = setup_clickhouse_database()
    
    try:
        # Step 1: Update ticker list from Finviz
        logger.info("Step 1: Updating ticker list from Finviz...")
        scraper = FinvizScraper(ch_manager)
        success = await scraper.update_ticker_database()
        
        if not success:
            logger.warning("Ticker update failed, proceeding with existing data")
        
        # Step 2: Start news monitoring
        logger.info("Step 2: Starting news monitoring...")
        monitor = NewswireMonitor()
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        ch_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 