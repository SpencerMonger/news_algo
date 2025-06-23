#!/usr/bin/env python3
"""
Test script to verify file trigger monitor works correctly
"""

import asyncio
import os
import json
import glob
import time
from datetime import datetime

class SimpleTriggerMonitor:
    def __init__(self):
        self.active_tickers = set()
        self.processed_count = 0
        
    async def file_trigger_monitor_async(self):
        """Async file trigger monitor test"""
        trigger_dir = "triggers"
        print("🚀 Starting ASYNC FILE TRIGGER MONITOR TEST!")
        
        start_time = time.time()
        
        while True:
            try:
                # Check for immediate trigger files
                trigger_pattern = os.path.join(trigger_dir, "immediate_*.json")
                trigger_files = glob.glob(trigger_pattern)
                
                if trigger_files:
                    print(f"🔥 FOUND {len(trigger_files)} TRIGGER FILES!")
                    
                    for trigger_file in trigger_files:
                        try:
                            # Read trigger data
                            with open(trigger_file, 'r') as f:
                                trigger_data = json.load(f)
                            
                            ticker = trigger_data['ticker']
                            timestamp = trigger_data['timestamp']
                            
                            processing_time = time.time()
                            lag = processing_time - start_time
                            
                            print(f"⚡ PROCESSING: {ticker} at {timestamp}")
                            print(f"📊 PROCESSING LAG: {lag:.3f}s from monitor start")
                            
                            # Add to active tickers
                            self.active_tickers.add(ticker)
                            self.processed_count += 1
                            
                            # Simulate price check
                            await asyncio.sleep(0.1)  # Simulate API call
                            print(f"💰 SIMULATED PRICE CHECK: {ticker} = $1.23")
                            
                            # Remove trigger file
                            os.remove(trigger_file)
                            print(f"✅ PROCESSED: {ticker} (removed trigger file)")
                            
                        except Exception as e:
                            print(f"❌ ERROR processing {trigger_file}: {e}")
                            try:
                                os.remove(trigger_file)
                            except:
                                pass
                
                # Check every 5ms for maximum speed
                await asyncio.sleep(0.005)
                
                # Stop after 30 seconds or when we've processed all files
                if time.time() - start_time > 30:
                    print(f"⏰ Test completed after 30 seconds")
                    print(f"📊 RESULTS: Processed {self.processed_count} triggers")
                    print(f"📊 ACTIVE TICKERS: {self.active_tickers}")
                    break
                    
            except Exception as e:
                print(f"❌ ERROR in file trigger monitor: {e}")
                await asyncio.sleep(0.005)

async def main():
    print("🧪 TESTING FILE TRIGGER MONITOR IN ISOLATION")
    print("📋 This test will process any existing trigger files in triggers/")
    
    # Check if trigger files exist
    trigger_files = glob.glob("triggers/immediate_*.json")
    if not trigger_files:
        print("⚠️  No trigger files found in triggers/ directory")
        print("💡 Run the main system first to generate trigger files, then run this test")
        return
    
    print(f"📁 Found {len(trigger_files)} trigger files to process")
    
    monitor = SimpleTriggerMonitor()
    await monitor.file_trigger_monitor_async()

if __name__ == "__main__":
    asyncio.run(main()) 