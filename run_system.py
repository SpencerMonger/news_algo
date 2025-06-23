#!/usr/bin/env python3
"""
Main orchestrator for the news scraping and price monitoring system
"""

import asyncio
import logging
import argparse
import subprocess
import sys
import time
import os
import signal
from clickhouse_setup import ClickHouseManager
from web_scraper import Crawl4AIScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def setup_database():
    """Initialize ClickHouse database and tables"""
    logger.info("Setting up ClickHouse database...")
    
    # Initialize ClickHouse
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    
    # Clear tables for fresh start
    ch_manager.drop_all_pipeline_tables()
    
    # Setup all required tables
    ch_manager.create_tables()
    ch_manager.create_breaking_news_table()
    ch_manager.create_float_list_table()
    ch_manager.create_price_move_table()
    
    ch_manager.close()
    logger.info("ClickHouse setup completed successfully")

def start_price_checker_process():
    """Start price checker in a separate process for complete isolation"""
    logger.info("🚀 Starting ZERO-LAG price checker in separate process...")
    
    # Start price checker as completely separate process with real-time output
    price_process = subprocess.Popen([
        sys.executable, 'price_checker.py'
    ], 
    stdout=None,  # Let it print to main console
    stderr=None,  # Let it print to main console
    preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group on Unix
    )
    
    logger.info("✅ ZERO-LAG price checker started as separate process (PID: {})".format(price_process.pid))
    return price_process

async def start_news_monitor(enable_old: bool = False):
    """Start news monitoring with Chromium browser"""
    logger.info("🚀 Starting news monitor with Chromium browser...")
    
    scraper = Crawl4AIScraper(enable_old=enable_old)
    await scraper.initialize()
    await scraper.start_scraping()

def terminate_process_group(process):
    """Terminate process and all its children"""
    try:
        if hasattr(os, 'killpg'):
            # Unix: kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            # Windows: just terminate the process
            process.terminate()
        
        # Wait for process to terminate
        try:
            process.wait(timeout=5)
            logger.info("✅ Price checker process terminated gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            logger.info("⚡ Price checker process force-killed")
    except Exception as e:
        logger.error(f"Error terminating price checker process: {e}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='News & Price Monitoring System')
    parser.add_argument('--skip-list', action='store_true', help='Skip Finviz ticker list update')
    parser.add_argument('--enable-old', action='store_true', help='Process old news articles (disable freshness filter)')
    
    args = parser.parse_args()
    
    logger.info("=== Starting News & Price Monitoring System ===")
    
    price_process = None
    
    try:
        # Step 1: Setup database
        await setup_database()
        
        # Step 2: Skip ticker list update if requested
        if args.skip_list:
            logger.info("Step 1: Skipping Finviz ticker list update (--skip-list flag used)")
        else:
            logger.info("Step 1: Updating Finviz ticker list...")
            # Add ticker list update logic here if needed
        
        logger.info("Step 2: Starting monitoring systems with PROCESS ISOLATION...")
        
        if args.enable_old:
            logger.info("🔓 FRESHNESS FILTER DISABLED - Processing old news for testing")
        
        # Step 3: Start price checker in separate process for COMPLETE isolation
        logger.info("🚀 Phase 1: Starting ZERO-LAG price checker in separate process...")
        price_process = start_price_checker_process()
        
        # Give price checker time to initialize
        logger.info("⏳ Waiting 10 seconds for price checker process to fully initialize...")
        await asyncio.sleep(10)
        
        # Step 4: Start news monitor in current process
        logger.info("🚀 Phase 2: Starting news monitor with Chromium browser...")
        logger.info("✅ PROCESS ISOLATION: Price checker runs separately → Zero resource contention")
        
        # Start news monitoring
        await start_news_monitor(enable_old=args.enable_old)
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error in main system: {e}")
        raise
    finally:
        # Always clean up the price checker process
        if price_process and price_process.poll() is None:
            logger.info("🛑 Terminating price checker process...")
            terminate_process_group(price_process)

if __name__ == "__main__":
    asyncio.run(main()) 