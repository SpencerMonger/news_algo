#!/usr/bin/env python3
"""
Comprehensive end-to-end lag test - Tests the ENTIRE flow:
1. Insert article into breaking_news
2. Create trigger file
3. Run price checker
4. Time every single step including API calls
"""

import asyncio
import time
import json
import os
from datetime import datetime
from clickhouse_setup import ClickHouseManager
from price_checker import ContinuousPriceMonitor

class EndToEndLagTester:
    def __init__(self):
        self.ch_manager = None
        self.price_monitor = None
        self.test_results = {}
        
    async def setup(self):
        """Initialize all components"""
        print("🔧 Setting up test environment...")
        
        # Initialize ClickHouse
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        print("✅ ClickHouse connected")
        
        # Initialize price monitor
        self.price_monitor = ContinuousPriceMonitor()
        await self.price_monitor.initialize()
        print("✅ Price monitor initialized")
        
        # Clean up old triggers
        trigger_dir = "triggers"
        os.makedirs(trigger_dir, exist_ok=True)
        import glob
        old_files = glob.glob(os.path.join(trigger_dir, "immediate_*.json"))
        for f in old_files:
            os.remove(f)
        print(f"🧹 Cleaned up {len(old_files)} old trigger files")
        
    async def test_single_ticker_complete_flow(self, ticker="TEST"):
        """Test complete flow for a single ticker with detailed timing"""
        print(f"\n🧪 TESTING COMPLETE FLOW FOR {ticker}")
        print("=" * 60)
        
        results = {
            'ticker': ticker,
            'steps': {},
            'total_time': 0,
            'success': False
        }
        
        overall_start = time.time()
        
        try:
            # STEP 1: Insert article into breaking_news
            print(f"📝 STEP 1: Inserting article for {ticker}...")
            step1_start = time.time()
            
            article_data = [(
                datetime.now(),  # timestamp
                "TestSource",    # source
                ticker,          # ticker
                f"Test news for {ticker}",  # headline
                "2025-06-23T15:40:00Z",     # published_utc
                f"https://test.com/{ticker.lower()}", # article_url
                f"Test summary for {ticker}",        # summary
                f"Full test content for {ticker}",   # full_content
                datetime.now(),  # detected_at
                0,               # processing_latency_ms
                1,               # market_relevant
                datetime.now(),  # source_check_time
                f"test_hash_{ticker}_{int(time.time())}", # content_hash
                "earnings",      # news_type
                5                # urgency_score
            )]
            
            self.ch_manager.client.insert(
                'News.breaking_news',
                article_data,
                column_names=[
                    'timestamp', 'source', 'ticker', 'headline', 'published_utc',
                    'article_url', 'summary', 'full_content', 'detected_at',
                    'processing_latency_ms', 'market_relevant', 'source_check_time',
                    'content_hash', 'news_type', 'urgency_score'
                ]
            )
            
            step1_time = time.time() - step1_start
            results['steps']['article_insertion'] = step1_time
            print(f"   ✅ Article inserted in {step1_time*1000:.1f}ms")
            
            # STEP 2: Create trigger file
            print(f"🚀 STEP 2: Creating trigger file for {ticker}...")
            step2_start = time.time()
            
            self.ch_manager.create_immediate_trigger(ticker, datetime.now())
            
            step2_time = time.time() - step2_start
            results['steps']['trigger_creation'] = step2_time
            print(f"   ✅ Trigger file created in {step2_time*1000:.1f}ms")
            
            # STEP 3: Add ticker to active tracking
            print(f"🎯 STEP 3: Adding {ticker} to active tracking...")
            step3_start = time.time()
            
            self.price_monitor.active_tickers.add(ticker)
            
            step3_time = time.time() - step3_start
            results['steps']['add_to_tracking'] = step3_time
            print(f"   ✅ Added to tracking in {step3_time*1000:.1f}ms")
            
            # STEP 4: Test API call timing
            print(f"🌐 STEP 4: Testing API call for {ticker}...")
            step4_start = time.time()
            
            price_result = await self.price_monitor.get_current_price(ticker)
            
            step4_time = time.time() - step4_start
            results['steps']['api_call'] = step4_time
            
            if price_result:
                print(f"   ✅ API call successful in {step4_time*1000:.1f}ms - Price: ${price_result['price']:.4f}")
                results['price'] = price_result['price']
                results['api_source'] = price_result.get('source', 'unknown')
            else:
                print(f"   ❌ API call failed in {step4_time*1000:.1f}ms")
                results['price'] = None
                results['api_source'] = 'failed'
            
            # STEP 5: Database insertion of price
            if price_result:
                print(f"💾 STEP 5: Inserting price data for {ticker}...")
                step5_start = time.time()
                
                price_data = [(
                    datetime.now(),
                    ticker,
                    price_result['price'],
                    0,  # volume
                    price_result.get('source', 'polygon')
                )]
                
                self.ch_manager.client.insert(
                    'News.price_tracking',
                    price_data,
                    column_names=['timestamp', 'ticker', 'price', 'volume', 'source']
                )
                
                step5_time = time.time() - step5_start
                results['steps']['price_insertion'] = step5_time
                print(f"   ✅ Price data inserted in {step5_time*1000:.1f}ms")
                results['success'] = True
            else:
                results['steps']['price_insertion'] = 0
                print(f"   ⚠️ Skipping price insertion (API call failed)")
            
            # Calculate total time
            results['total_time'] = time.time() - overall_start
            
            print(f"\n📊 COMPLETE FLOW RESULTS FOR {ticker}:")
            print(f"   Article Insertion: {results['steps']['article_insertion']*1000:.1f}ms")
            print(f"   Trigger Creation:  {results['steps']['trigger_creation']*1000:.1f}ms")
            print(f"   Add to Tracking:   {results['steps']['add_to_tracking']*1000:.1f}ms")
            print(f"   API Call:          {results['steps']['api_call']*1000:.1f}ms")
            if results['success']:
                print(f"   Price Insertion:   {results['steps']['price_insertion']*1000:.1f}ms")
                print(f"   Final Price:       ${results['price']:.4f} ({results['api_source']})")
            print(f"   TOTAL END-TO-END:  {results['total_time']*1000:.1f}ms")
            
            if results['total_time'] < 1.0:
                print(f"   🎉 EXCELLENT: Under 1 second!")
            elif results['total_time'] < 5.0:
                print(f"   ✅ GOOD: Under 5 seconds")
            else:
                print(f"   ⚠️ SLOW: Over 5 seconds - investigate!")
                
        except Exception as e:
            results['error'] = str(e)
            results['total_time'] = time.time() - overall_start
            print(f"   ❌ ERROR: {e}")
            
        return results
    
    async def test_multiple_tickers(self, tickers=["TEST1", "TEST2", "TEST3"]):
        """Test multiple tickers to see consistency"""
        print(f"\n🔄 TESTING MULTIPLE TICKERS: {tickers}")
        print("=" * 80)
        
        all_results = []
        
        for ticker in tickers:
            result = await self.test_single_ticker_complete_flow(ticker)
            all_results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(0.1)
        
        # Analyze results
        print(f"\n📈 MULTI-TICKER ANALYSIS:")
        print("-" * 40)
        
        successful_tests = [r for r in all_results if r['success']]
        failed_tests = [r for r in all_results if not r['success']]
        
        print(f"Success Rate: {len(successful_tests)}/{len(all_results)} ({len(successful_tests)/len(all_results)*100:.1f}%)")
        
        if successful_tests:
            total_times = [r['total_time'] for r in successful_tests]
            api_times = [r['steps']['api_call'] for r in successful_tests]
            
            print(f"Total Time - Avg: {sum(total_times)/len(total_times)*1000:.1f}ms, Min: {min(total_times)*1000:.1f}ms, Max: {max(total_times)*1000:.1f}ms")
            print(f"API Time   - Avg: {sum(api_times)/len(api_times)*1000:.1f}ms, Min: {min(api_times)*1000:.1f}ms, Max: {max(api_times)*1000:.1f}ms")
            
            # Check consistency
            time_variance = max(total_times) - min(total_times)
            if time_variance < 1.0:
                print(f"✅ CONSISTENT: All tests within 1 second of each other")
            else:
                print(f"⚠️ INCONSISTENT: {time_variance:.1f}s variance between fastest and slowest")
        
        if failed_tests:
            print(f"❌ Failed tickers: {[r['ticker'] for r in failed_tests]}")
            
        return all_results
    
    async def test_file_trigger_monitor(self, ticker="TRIGGER_TEST"):
        """Test the file trigger monitor specifically"""
        print(f"\n🔍 TESTING FILE TRIGGER MONITOR FOR {ticker}")
        print("=" * 60)
        
        # Create a trigger file manually
        trigger_file = f"triggers/immediate_{ticker}_{int(time.time())}.json"
        trigger_data = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "trigger_type": "immediate_price_check",
            "created_at": datetime.now().isoformat()
        }
        
        with open(trigger_file, 'w') as f:
            json.dump(trigger_data, f)
        
        print(f"📁 Created trigger file: {trigger_file}")
        
        # Monitor for processing (simulate the file trigger monitor)
        start_time = time.time()
        processed = False
        timeout = 10.0  # 10 second timeout
        
        print("👀 Monitoring for trigger file processing...")
        
        while time.time() - start_time < timeout:
            if not os.path.exists(trigger_file):
                processed = True
                process_time = time.time() - start_time
                print(f"✅ Trigger file processed in {process_time*1000:.1f}ms")
                break
            await asyncio.sleep(0.001)  # Check every 1ms
        
        if not processed:
            print(f"❌ Trigger file NOT processed within {timeout}s timeout")
            # Clean up
            if os.path.exists(trigger_file):
                os.remove(trigger_file)
        
        return processed
    
    async def cleanup(self):
        """Clean up test environment"""
        if self.price_monitor:
            await self.price_monitor.cleanup()
        if self.ch_manager:
            self.ch_manager.close()
        print("🧹 Test environment cleaned up")

async def main():
    """Run comprehensive end-to-end lag testing"""
    print("🚀 COMPREHENSIVE END-TO-END LAG TESTING")
    print("=" * 80)
    
    tester = EndToEndLagTester()
    
    try:
        # Setup
        await tester.setup()
        
        # Test 1: Single ticker complete flow
        result1 = await tester.test_single_ticker_complete_flow("LTRY")
        
        # Test 2: Multiple tickers for consistency
        result2 = await tester.test_multiple_tickers(["GAME", "ATNM", "TEST"])
        
        # Test 3: File trigger monitor (if we want to test that specifically)
        # result3 = await tester.test_file_trigger_monitor("MONITOR_TEST")
        
        print(f"\n🎯 FINAL SUMMARY:")
        print("=" * 40)
        print("This test shows exactly where lag occurs in the system.")
        print("Look at the API call times - if they're >1000ms, that's your bottleneck!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 