#!/usr/bin/env python3
"""
Test script to verify price checker endpoint changes
Tests both last trade and NBBO endpoints to confirm the switch
"""

import asyncio
import aiohttp
import os
import time
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PriceCheckerTest:
    def __init__(self):
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
        
        # Use PROXY_URL if available
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.base_url = proxy_url.rstrip('/')
            print(f"Using proxy URL: {self.base_url}")
        else:
            self.base_url = "https://api.polygon.io"
            print("Using official Polygon API")

    async def get_last_trade_price(self, ticker: str):
        """Test the last trade endpoint (new primary)"""
        url = f"{self.base_url}/v2/last/trade/{ticker}"
        params = {'apikey': self.polygon_api_key}
        
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                start_time = time.time()
                async with session.get(url, params=params) as response:
                    api_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'results' in data and data['results']:
                            result = data['results']
                            price = result.get('p', 0.0)  # trade price
                            size = result.get('s', 0)     # trade size
                            timestamp = result.get('t', 0) # trade timestamp
                            
                            return {
                                'price': price,
                                'size': size,
                                'timestamp': timestamp,
                                'api_time': api_time,
                                'source': 'last_trade',
                                'status': 'success'
                            }
                    else:
                        return {
                            'status': 'error',
                            'code': response.status,
                            'api_time': api_time,
                            'source': 'last_trade'
                        }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'source': 'last_trade'
                }

    async def get_nbbo_price(self, ticker: str):
        """Test the NBBO endpoint (old primary, now fallback)"""
        url = f"{self.base_url}/v2/last/nbbo/{ticker}"
        params = {'apikey': self.polygon_api_key}
        
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                start_time = time.time()
                async with session.get(url, params=params) as response:
                    api_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'results' in data and data['results']:
                            result = data['results']
                            bid = result.get('P', 0.0)    # bid price
                            ask = result.get('p', 0.0)    # ask price
                            bid_size = result.get('S', 0) # bid size
                            ask_size = result.get('s', 0) # ask size
                            
                            midpoint = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                            
                            return {
                                'bid': bid,
                                'ask': ask,
                                'midpoint': midpoint,
                                'bid_size': bid_size,
                                'ask_size': ask_size,
                                'spread': ask - bid if bid > 0 and ask > 0 else 0,
                                'api_time': api_time,
                                'source': 'nbbo',
                                'status': 'success'
                            }
                    else:
                        return {
                            'status': 'error',
                            'code': response.status,
                            'api_time': api_time,
                            'source': 'nbbo'
                        }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'source': 'nbbo'
                }

    async def test_price_checker_implementation(self, ticker: str):
        """Test the actual price checker implementation from price_checker.py"""
        from price_checker import ContinuousPriceMonitor
        
        monitor = ContinuousPriceMonitor()
        
        # Initialize session
        timeout = aiohttp.ClientTimeout(total=5.0)
        monitor.session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            start_time = time.time()
            result = await monitor.get_current_price(ticker)
            api_time = time.time() - start_time
            
            if result:
                result['api_time'] = api_time
                result['test_source'] = 'price_checker_implementation'
            
            return result
        finally:
            await monitor.session.close()

    async def run_comparison_test(self, ticker: str):
        """Run comparison test between endpoints"""
        print(f"\n🧪 TESTING PRICE ENDPOINTS FOR {ticker}")
        print("=" * 60)
        
        # Test all endpoints in parallel
        tasks = [
            self.get_last_trade_price(ticker),
            self.get_nbbo_price(ticker),
            self.test_price_checker_implementation(ticker)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        trade_result, nbbo_result, implementation_result = results
        
        # Display results
        print(f"\n📊 RESULTS FOR {ticker}:")
        print("-" * 40)
        
        # Last Trade Results
        print("\n🎯 LAST TRADE ENDPOINT (NEW PRIMARY):")
        if isinstance(trade_result, dict) and trade_result.get('status') == 'success':
            print(f"   ✅ Price: ${trade_result['price']:.4f}")
            print(f"   📦 Size: {trade_result['size']:,} shares")
            print(f"   ⏱️  API Time: {trade_result['api_time']:.3f}s")
            print(f"   🕒 Trade Time: {datetime.fromtimestamp(trade_result['timestamp']/1000)}")
        else:
            print(f"   ❌ Error: {trade_result}")
        
        # NBBO Results
        print("\n📈 NBBO ENDPOINT (FALLBACK):")
        if isinstance(nbbo_result, dict) and nbbo_result.get('status') == 'success':
            print(f"   💰 Bid: ${nbbo_result['bid']:.4f} (size: {nbbo_result['bid_size']:,})")
            print(f"   💰 Ask: ${nbbo_result['ask']:.4f} (size: {nbbo_result['ask_size']:,})")
            print(f"   📊 Midpoint: ${nbbo_result['midpoint']:.4f}")
            print(f"   📏 Spread: ${nbbo_result['spread']:.4f}")
            print(f"   ⏱️  API Time: {nbbo_result['api_time']:.3f}s")
        else:
            print(f"   ❌ Error: {nbbo_result}")
        
        # Implementation Results
        print("\n⚙️  PRICE CHECKER IMPLEMENTATION:")
        if isinstance(implementation_result, dict):
            print(f"   ✅ Price: ${implementation_result['price']:.4f}")
            print(f"   📡 Source: {implementation_result['source']}")
            print(f"   ⏱️  API Time: {implementation_result['api_time']:.3f}s")
        else:
            print(f"   ❌ Error: {implementation_result}")
        
        # Analysis
        print("\n🔍 ANALYSIS:")
        if (isinstance(trade_result, dict) and trade_result.get('status') == 'success' and
            isinstance(implementation_result, dict) and implementation_result.get('price')):
            
            trade_price = trade_result['price']
            impl_price = implementation_result['price']
            impl_source = implementation_result.get('source', 'unknown')
            
            if abs(trade_price - impl_price) < 0.01:  # Within 1 cent
                print(f"   ✅ CORRECT: Implementation using {impl_source} matches last trade (${impl_price:.4f})")
            else:
                print(f"   ⚠️  MISMATCH: Trade=${trade_price:.4f}, Implementation=${impl_price:.4f} (source: {impl_source})")
        
        if (isinstance(nbbo_result, dict) and nbbo_result.get('status') == 'success' and
            isinstance(trade_result, dict) and trade_result.get('status') == 'success'):
            
            trade_price = trade_result['price']
            nbbo_midpoint = nbbo_result['midpoint']
            spread = nbbo_result['spread']
            
            price_diff = abs(trade_price - nbbo_midpoint)
            print(f"   📊 Trade vs NBBO: ${trade_price:.4f} vs ${nbbo_midpoint:.4f} (diff: ${price_diff:.4f})")
            print(f"   📏 Spread: ${spread:.4f} ({(spread/nbbo_midpoint)*100:.2f}%)")
            
            if price_diff > spread/2:
                print(f"   ⚠️  SIGNIFICANT DIFFERENCE: Trade price outside bid/ask spread")
            else:
                print(f"   ✅ REASONABLE: Trade price within expected range")

async def main():
    """Run the price checker test"""
    print("🚀 PRICE CHECKER ENDPOINT TEST")
    print("Testing the switch from NBBO to Last Trade endpoint")
    print("=" * 60)
    
    # Test tickers
    test_tickers = ['FEAM', 'AAPL', 'TSLA']  # Include FEAM to test the problematic ticker
    
    tester = PriceCheckerTest()
    
    for ticker in test_tickers:
        try:
            await tester.run_comparison_test(ticker)
            print("\n" + "="*60)
        except Exception as e:
            print(f"❌ Error testing {ticker}: {e}")
    
    print("\n🎯 TEST SUMMARY:")
    print("- Last Trade endpoint should be used as PRIMARY")
    print("- NBBO endpoint should be used as FALLBACK")
    print("- Implementation should prefer actual trade prices over bid/ask midpoints")
    print("- This should eliminate phantom prices like the $4.05 FEAM issue")

if __name__ == "__main__":
    asyncio.run(main()) 