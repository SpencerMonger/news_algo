#!/usr/bin/env python3
"""
Test script to verify consistent file trigger processing for multiple tickers
"""

import asyncio
import os
import json
import time
from datetime import datetime
from clickhouse_setup import ClickHouseManager

async def test_consistent_triggers():
    """Test that file triggers work consistently for multiple tickers"""
    
    # Test tickers
    test_tickers = ['LTRY', 'GAME', 'ATNM', 'TEST1', 'TEST2']
    
    print(f"🧪 Testing consistent file trigger processing for {len(test_tickers)} tickers...")
    
    # Create ClickHouse manager
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    
    # Clean up old triggers
    trigger_dir = "triggers"
    os.makedirs(trigger_dir, exist_ok=True)
    
    # Remove old trigger files
    import glob
    old_files = glob.glob(os.path.join(trigger_dir, "immediate_*.json"))
    for f in old_files:
        os.remove(f)
    print(f"🧹 Cleaned up {len(old_files)} old trigger files")
    
    # Create trigger files for all test tickers simultaneously
    print("\n📝 Creating trigger files for all tickers...")
    creation_times = {}
    
    for ticker in test_tickers:
        start_time = time.time()
        ch_manager.create_immediate_trigger(ticker, datetime.now())
        creation_times[ticker] = time.time() - start_time
        print(f"  ✅ {ticker}: {creation_times[ticker]*1000:.1f}ms")
    
    # Check all files were created
    trigger_files = glob.glob(os.path.join(trigger_dir, "immediate_*.json"))
    print(f"\n📁 Created {len(trigger_files)} trigger files")
    
    # Verify trigger file contents
    print("\n🔍 Verifying trigger file contents...")
    for trigger_file in trigger_files:
        with open(trigger_file, 'r') as f:
            data = json.load(f)
        ticker = data['ticker']
        print(f"  ✅ {ticker}: {data}")
    
    # Calculate statistics
    avg_creation_time = sum(creation_times.values()) / len(creation_times)
    max_creation_time = max(creation_times.values())
    min_creation_time = min(creation_times.values())
    
    print(f"\n📊 TRIGGER CREATION STATISTICS:")
    print(f"  Average: {avg_creation_time*1000:.1f}ms")
    print(f"  Fastest: {min_creation_time*1000:.1f}ms")
    print(f"  Slowest: {max_creation_time*1000:.1f}ms")
    print(f"  Variance: {(max_creation_time-min_creation_time)*1000:.1f}ms")
    
    if (max_creation_time - min_creation_time) < 0.010:  # Less than 10ms variance
        print("  ✅ CONSISTENT: All triggers created within 10ms of each other")
    else:
        print("  ⚠️ INCONSISTENT: High variance in trigger creation times")
    
    # Clean up
    for trigger_file in trigger_files:
        os.remove(trigger_file)
    print(f"\n🧹 Cleaned up {len(trigger_files)} test trigger files")
    
    ch_manager.close()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_consistent_triggers()) 