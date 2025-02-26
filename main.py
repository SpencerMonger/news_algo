import logging
import asyncio
import signal
import sys
from news_checker import monitor_news
from price_checker import check_price_on_news
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_price_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info("Shutdown signal received, finishing current cycle...")
    shutdown_requested = True

async def main():
    """
    Main function to run the news and price monitoring system
    """
    global shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting News and Price Monitoring System")
    
    try:
        # Configuration
        check_interval = 20  # seconds between news check cycles
        batch_size = 100     # number of tickers to process in each batch
        
        logger.info(f"Configuration: check_interval={check_interval}s, batch_size={batch_size} tickers")
        
        # Start the news monitoring process, passing the price checker as callback
        monitoring_task = asyncio.create_task(
            monitor_news(
                price_checker_callback=check_price_on_news,
                interval=check_interval,
                batch_size=batch_size
            )
        )
        
        # Check for shutdown request
        while not shutdown_requested:
            await asyncio.sleep(1)
        
        # Cancel the monitoring task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in main monitoring loop: {e}")
    finally:
        logger.info("Shutting down monitoring system")

if __name__ == "__main__":
    asyncio.run(main()) 