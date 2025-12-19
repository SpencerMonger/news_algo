#!/usr/bin/env python3
"""
Main orchestrator for the news scraping and price monitoring system
"""

import asyncio
import logging
import argparse
import subprocess
import sys
import os
import signal
from clickhouse_setup import ClickHouseManager
from log_manager import setup_comprehensive_logging

# SENTIMENT ANALYSIS SERVICE INITIALIZATION
from sentiment_service import get_sentiment_service

# Global log manager for cleanup
log_manager = None

async def setup_database():
    """Initialize ClickHouse database and tables"""
    logging.info("Setting up ClickHouse database...")
    
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
    logging.info("ClickHouse setup completed successfully")

def start_price_checker_process():
    """Start price checker in a separate process for complete isolation"""
    logging.info("🚀 Starting ZERO-LAG price checker in separate process...")
    
    # Start price checker as completely separate process with real-time output
    price_process = subprocess.Popen([
        sys.executable, 'price_checker.py'
    ], 
    stdout=subprocess.PIPE,  # Capture output for logging
    stderr=subprocess.STDOUT,  # Merge stderr with stdout
    universal_newlines=True,
    bufsize=1,  # Line buffered
    preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group on Unix
    )
    
    logging.info("✅ ZERO-LAG price checker started as separate process (PID: {})".format(price_process.pid))
    
    # Start thread to capture and log subprocess output
    import threading
    def log_subprocess_output():
        try:
            for line in iter(price_process.stdout.readline, ''):
                if line.strip():
                    logging.info(f"[PRICE_CHECKER] {line.strip()}")
        except Exception as e:
            logging.error(f"Error reading price checker output: {e}")
    
    output_thread = threading.Thread(target=log_subprocess_output, daemon=True)
    output_thread.start()
    
    return price_process

async def start_news_monitor(enable_old: bool = False, use_websocket: bool = False, process_any_ticker: bool = False):
    """Start news monitoring with either web scraper (default) or WebSocket scraper"""
    
    if use_websocket:
        logging.info("🚀 Starting Benzinga WebSocket news monitor...")
        from benzinga_websocket import Crawl4AIScraper
        scraper_type = "Benzinga WebSocket"
        # Pass both enable_old and process_any_ticker to WebSocket scraper
        scraper = Crawl4AIScraper(enable_old=enable_old, process_any_ticker=process_any_ticker)
    else:
        logging.info("🚀 Starting web scraper news monitor with Chromium browser...")
        from web_scraper import Crawl4AIScraper
        scraper_type = "Web Scraper"
        # Web scraper only uses enable_old parameter
        scraper = Crawl4AIScraper(enable_old=enable_old)
    
    logging.info(f"📡 Selected scraper: {scraper_type}")
    
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
            logging.info("✅ Price checker process terminated gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            logging.info("⚡ Price checker process force-killed")
    except Exception as e:
        logging.error(f"Error terminating price checker process: {e}")

async def initialize_sentiment_service():
    """Initialize the sentiment analysis service"""
    logging.info("🧠 Initializing sentiment analysis service...")
    
    try:
        # Get sentiment service instance
        sentiment_service = await get_sentiment_service()
        
        # Initialize the service
        is_initialized = await sentiment_service.initialize()
        
        if is_initialized:
            logging.info("✅ Sentiment analysis service initialized successfully")
            logging.info("🤖 Claude API connection established - AI analysis ready")
            return True
        else:
            logging.warning("⚠️ Sentiment analysis service failed to initialize")
            logging.warning("🤖 Claude API connection failed - sentiment analysis will be disabled")
            return False
    
    except Exception as e:
        logging.error(f"❌ Error initializing sentiment service: {e}")
        logging.warning("🤖 Sentiment analysis will be disabled - system will continue with price-only alerts")
        return False

async def main():
    """Main function"""
    global log_manager
    
    # Setup comprehensive logging first
    log_manager = setup_comprehensive_logging(log_dir="logs", retention_days=5)
    
    parser = argparse.ArgumentParser(description='News & Price Monitoring System')
    parser.add_argument('--enable-old', action='store_true', help='Process old news articles (disable freshness filter)')
    parser.add_argument('--socket', action='store_true', help='Use Benzinga WebSocket instead of web scraper (default: web scraper)')
    parser.add_argument('--any', action='store_true', help='Process any ticker symbols found (only works with --socket, bypasses ticker list filtering)')
    parser.add_argument('--no-sentiment', action='store_true', help='Skip sentiment analysis initialization (for testing)')
    
    args = parser.parse_args()
    
    logging.info("=== Starting News & Price Monitoring System ===")
    logging.info(f"Command line arguments: {vars(args)}")
    
    # Log scraper selection
    if args.socket:
        logging.info("🔌 WEBSOCKET MODE: Using Benzinga WebSocket for real-time news")
        if args.any:
            logging.info("🎯 ANY TICKER MODE: Will process any ticker symbols found (bypassing database list)")
        else:
            logging.info("📋 DATABASE TICKER MODE: Will only process tickers from database list")
    else:
        logging.info("🌐 WEB SCRAPER MODE: Using traditional web scraping (default)")
        if args.any:
            logging.warning("⚠️ --any flag ignored: Only works with --socket mode")
    
    # Log sentiment analysis mode
    if args.no_sentiment:
        logging.info("🔇 SENTIMENT ANALYSIS DISABLED: Running in price-only mode")
    else:
        logging.info("🧠 SENTIMENT ANALYSIS ENABLED: AI-enhanced price alerts")
    
    price_process = None
    
    try:
        # Step 1: Setup database
        await setup_database()
        
        # Step 2: Initialize sentiment service (unless explicitly disabled)
        if not args.no_sentiment:
            sentiment_initialized = await initialize_sentiment_service()
            if sentiment_initialized:
                logging.info("✅ Sentiment analysis ready - alerts will use AI analysis")
            else:
                logging.warning("⚠️ Sentiment analysis failed - falling back to price-only alerts")
        else:
            logging.info("🔇 Sentiment analysis skipped - price-only alerts enabled")
        
        logging.info("Step 3: Starting monitoring systems with PROCESS ISOLATION...")
        
        if args.enable_old:
            logging.info("🔓 FRESHNESS FILTER DISABLED - Processing old news for testing")
        
        # Step 4: Start price checker in separate process for COMPLETE isolation
        logging.info("Step 4: Starting ZERO-LAG price checker in separate process...")
        price_process = start_price_checker_process()
        
        # Give price checker time to initialize
        logging.info("⏳ Waiting 10 seconds for price checker process to fully initialize...")
        await asyncio.sleep(10)
        
        # Step 5: Start news monitor in current process
        logging.info("Step 5: Starting news monitor...")
        if args.socket:
            logging.info("🚀 Phase 2: Starting Benzinga WebSocket news monitor...")
            logging.info("⚡ REAL-TIME MODE: WebSocket provides sub-second news detection")
        else:
            logging.info("🚀 Phase 2: Starting web scraper news monitor with Chromium browser...")
            logging.info("🌐 TRADITIONAL MODE: Web scraping with browser automation")
            
        logging.info("✅ PROCESS ISOLATION: Price checker runs separately → Zero resource contention")
        logging.info("🧠 SENTIMENT INTEGRATION: AI analysis enhances price alerts")
        
        # Start news monitoring
        await start_news_monitor(enable_old=args.enable_old, use_websocket=args.socket, process_any_ticker=args.any)
        
    except KeyboardInterrupt:
        logging.info("🛑 Received interrupt signal")
    except Exception as e:
        logging.error(f"Fatal error in main system: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Always clean up the price checker process
        if price_process and price_process.poll() is None:
            logging.info("🛑 Terminating price checker process...")
            terminate_process_group(price_process)
        
        # Log shutdown and cleanup
        if log_manager:
            log_manager.log_shutdown_banner()
            log_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 