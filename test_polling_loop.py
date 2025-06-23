#!/usr/bin/env python3
"""
Test script to verify the continuous polling loop is working
"""
import asyncio
import logging
import time
from dotenv import load_dotenv
from price_checker import ContinuousPriceMonitor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_polling_only():
    """Test ONLY the continuous polling loop without file triggers"""
    logger.info("🧪 TESTING: Continuous polling loop in isolation")
    
    # Create price monitor
    monitor = ContinuousPriceMonitor()
    await monitor.initialize()
    
    # Manually add some test tickers
    monitor.active_tickers.add("AAPL")
    monitor.active_tickers.add("TSLA")
    logger.info(f"🎯 TEST: Added test tickers: {monitor.active_tickers}")
    
    # Run ONLY the continuous polling loop for 60 seconds
    logger.info("🔄 TEST: Starting 60-second continuous polling test...")
    
    try:
        await asyncio.wait_for(monitor.continuous_polling_loop(), timeout=60.0)
    except asyncio.TimeoutError:
        logger.info("✅ TEST: 60-second polling test completed")
    except Exception as e:
        logger.error(f"❌ TEST: Polling loop failed: {e}")
    finally:
        await monitor.cleanup()

async def test_dual_system():
    """Test both file triggers AND continuous polling together"""
    logger.info("🧪 TESTING: Dual system (file triggers + continuous polling)")
    
    # Create price monitor
    monitor = ContinuousPriceMonitor()
    await monitor.initialize()
    
    # Manually add some test tickers
    monitor.active_tickers.add("AAPL")
    monitor.active_tickers.add("MSFT")
    logger.info(f"🎯 TEST: Added test tickers: {monitor.active_tickers}")
    
    # Run both systems for 30 seconds
    logger.info("🔄 TEST: Starting 30-second dual system test...")
    
    try:
        await asyncio.wait_for(
            asyncio.gather(
                monitor.file_trigger_monitor_async(),
                monitor.continuous_polling_loop()
            ),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        logger.info("✅ TEST: 30-second dual system test completed")
    except Exception as e:
        logger.error(f"❌ TEST: Dual system failed: {e}")
    finally:
        await monitor.cleanup()

async def main():
    """Main test function"""
    logger.info("🚀 STARTING POLLING LOOP DIAGNOSTIC TESTS")
    
    print("\n" + "="*60)
    print("TEST 1: Continuous Polling Loop Only (60 seconds)")
    print("="*60)
    await test_polling_only()
    
    print("\n" + "="*60)
    print("TEST 2: Dual System - File Triggers + Polling (30 seconds)")
    print("="*60)
    await test_dual_system()
    
    logger.info("🏁 ALL TESTS COMPLETED")

if __name__ == "__main__":
    asyncio.run(main()) 