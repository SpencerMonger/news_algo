#!/usr/bin/env python3
"""
Test script to verify the double API call logic for new tickers
"""
import asyncio
import logging
from datetime import datetime
from price_checker import ContinuousPriceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_double_call_logic():
    """Test the double call logic for new vs existing tickers"""
    logger.info("🧪 Testing double API call logic...")
    
    monitor = ContinuousPriceMonitor()
    await monitor.initialize()
    
    # Test ticker
    test_ticker = "AAPL"
    
    # Test 1: New ticker (should use double call)
    logger.info(f"🔬 TEST 1: Adding {test_ticker} as NEW ticker...")
    monitor.ticker_timestamps[test_ticker] = datetime.now()  # Mark as newly added
    monitor.active_tickers.add(test_ticker)
    
    result1 = await monitor.get_current_price(test_ticker)
    if result1:
        logger.info(f"✅ NEW TICKER RESULT: {test_ticker} = ${result1['price']:.4f} (source: {result1['source']})")
        if result1['source'] == 'trade_verified':
            logger.info("🎯 SUCCESS: Double call was used for new ticker!")
        else:
            logger.warning("⚠️ UNEXPECTED: Single call was used for new ticker")
    else:
        logger.error("❌ FAILED: No price returned for new ticker")
    
    # Wait to make ticker "old"
    await asyncio.sleep(11)  # Wait 11 seconds to make ticker "old"
    
    # Test 2: Existing ticker (should use single call)
    logger.info(f"🔬 TEST 2: Testing {test_ticker} as EXISTING ticker...")
    result2 = await monitor.get_current_price(test_ticker)
    if result2:
        logger.info(f"✅ EXISTING TICKER RESULT: {test_ticker} = ${result2['price']:.4f} (source: {result2['source']})")
        if result2['source'] == 'trade':
            logger.info("🎯 SUCCESS: Single call was used for existing ticker!")
        else:
            logger.warning("⚠️ UNEXPECTED: Double call was used for existing ticker")
    else:
        logger.error("❌ FAILED: No price returned for existing ticker")
    
    await monitor.cleanup()
    logger.info("🧪 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_double_call_logic()) 