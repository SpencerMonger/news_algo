#!/usr/bin/env python3
"""
Diagnostic script to investigate incorrect initial price values
UPDATED: Now includes testing of the new double-call logic vs old single-call logic
"""

import asyncio
import aiohttp
import os
import time
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from clickhouse_setup import ClickHouseManager

load_dotenv()

class PriceDiagnostic:
    def __init__(self):
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.base_url = os.getenv('PROXY_URL', 'https://api.polygon.io').rstrip('/')
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        
    async def test_double_call_vs_single_call(self, ticker: str):
        """Test the new double-call logic vs old single-call logic"""
        print(f"\n🧪 TESTING DOUBLE-CALL FIX FOR {ticker}")
        print("=" * 60)
        
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Test 1: OLD LOGIC (Single Call) - What we used to get
            print("📊 OLD LOGIC (Single Call):")
            print("-" * 30)
            
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {'apikey': self.polygon_api_key}
            
            try:
                start_time = time.time()
                async with session.get(url, params=params) as response:
                    api_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'results' in data and data['results']:
                            old_price = data['results'].get('p', 0.0)
                            print(f"   Single API Call: ${old_price:.4f} in {api_time:.3f}s")
                            print(f"   Source: trade (old logic)")
                        else:
                            print("   ❌ No results in API response")
                            return
                    else:
                        print(f"   ❌ API returned status {response.status}")
                        return
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return
            
            # Test 2: NEW LOGIC (Double Call) - What we get now
            print("\n📊 NEW LOGIC (Double Call):")
            print("-" * 30)
            
            try:
                # First call - expect garbage data, discard it
                start_time = time.time()
                async with session.get(url, params=params) as response1:
                    if response1.status == 200:
                        garbage_data = await response1.json()
                        if 'results' in garbage_data and garbage_data['results']:
                            garbage_price = garbage_data['results'].get('p', 0.0)
                            print(f"   First Call (DISCARDED): ${garbage_price:.4f}")
                
                # Small delay between calls to avoid rate limiting
                await asyncio.sleep(0.1)
                
                # Second call - expect correct data, use this one
                async with session.get(url, params=params) as response2:
                    api_time = time.time() - start_time
                    
                    if response2.status == 200:
                        data = await response2.json()
                        
                        if 'results' in data and data['results']:
                            new_price = data['results'].get('p', 0.0)
                            print(f"   Second Call (USED): ${new_price:.4f}")
                            print(f"   Total Time: {api_time:.3f}s")
                            print(f"   Source: trade_verified (new logic)")
                        else:
                            print("   ❌ No results in second API response")
                            return
                    else:
                        print(f"   ❌ Second API returned status {response2.status}")
                        return
                        
            except Exception as e:
                print(f"   ❌ Error in double call: {e}")
                return
            
            # Compare the results
            print(f"\n🔍 COMPARISON ANALYSIS:")
            print("-" * 25)
            
            if old_price > 0 and new_price > 0:
                price_diff = abs(old_price - new_price)
                price_diff_pct = (price_diff / old_price) * 100
                
                print(f"   Old Logic Price: ${old_price:.4f}")
                print(f"   New Logic Price: ${new_price:.4f}")
                print(f"   Difference: ${price_diff:.4f} ({price_diff_pct:.2f}%)")
                
                if price_diff_pct > 1.0:  # More than 1% difference
                    print(f"   🚨 SIGNIFICANT DIFFERENCE: {price_diff_pct:.2f}%")
                    print(f"   💡 This confirms the first call returned garbage data")
                    
                    # Simulate what would happen with price alerts
                    print(f"\n📈 PRICE ALERT SIMULATION:")
                    print("-" * 25)
                    
                    # Simulate a 2% price increase from the baseline
                    simulated_current_price = new_price * 1.02
                    
                    old_alert_pct = ((simulated_current_price - old_price) / old_price) * 100
                    new_alert_pct = ((simulated_current_price - new_price) / new_price) * 100
                    
                    print(f"   Simulated Current Price: ${simulated_current_price:.4f}")
                    print(f"   OLD Logic Alert: {old_alert_pct:+.1f}% (from ${old_price:.4f})")
                    print(f"   NEW Logic Alert: {new_alert_pct:+.1f}% (from ${new_price:.4f})")
                    
                    if old_alert_pct >= 5.0 and new_alert_pct < 5.0:
                        print(f"   ✅ FIX SUCCESS: Old logic would trigger FALSE alert, new logic prevents it!")
                    elif old_alert_pct < 5.0 and new_alert_pct < 5.0:
                        print(f"   ✅ BOTH CORRECT: Neither logic would trigger alert (good)")
                    else:
                        print(f"   ⚠️ BOTH TRIGGER: Both logics would trigger alert")
                        
                else:
                    print(f"   ✅ MINIMAL DIFFERENCE: Both calls returned similar prices")
                    print(f"   💡 This ticker doesn't suffer from the garbage data issue")
            
            # Also test with some delay to see if prices stabilize
            print(f"\n⏱️ STABILITY TEST (after 2 seconds):")
            print("-" * 35)
            
            await asyncio.sleep(2)
            
            try:
                async with session.get(url, params=params) as response3:
                    if response3.status == 200:
                        data = await response3.json()
                        
                        if 'results' in data and data['results']:
                            stable_price = data['results'].get('p', 0.0)
                            print(f"   Price after 2s delay: ${stable_price:.4f}")
                            
                            stable_diff = abs(stable_price - new_price)
                            stable_diff_pct = (stable_diff / new_price) * 100 if new_price > 0 else 0
                            
                            print(f"   Difference from new logic: ${stable_diff:.4f} ({stable_diff_pct:.2f}%)")
                            
                            if stable_diff_pct < 0.5:
                                print(f"   ✅ STABLE: New logic price is consistent over time")
                            else:
                                print(f"   ⚠️ VOLATILE: Price is still changing significantly")
                        
            except Exception as e:
                print(f"   ❌ Error in stability test: {e}")

    async def diagnose_ticker_prices(self, ticker: str, hours_back: int = 24):
        """Diagnose price issues for a specific ticker"""
        print(f"\n🔍 DIAGNOSING PRICE ISSUES FOR {ticker}")
        print("=" * 60)
        
        # Get recent price tracking data
        query = f"""
        SELECT 
            timestamp,
            price,
            source,
            formatDateTime(timestamp, '%Y-%m-%d %H:%M:%S') as formatted_time
        FROM News.price_tracking
        WHERE ticker = '{ticker}'
        AND timestamp >= now() - INTERVAL {hours_back} HOUR
        ORDER BY timestamp ASC
        LIMIT 50
        """
        
        result = self.ch_manager.client.query(query)
        
        if not result.result_rows:
            print(f"❌ No price data found for {ticker} in last {hours_back} hours")
            return
        
        print(f"📊 Found {len(result.result_rows)} price records:")
        print("Time                 | Price    | Source | Notes")
        print("-" * 55)
        
        prices = []
        for i, row in enumerate(result.result_rows):
            timestamp, price, source, formatted_time = row
            prices.append(price)
            
            # Detect anomalies
            notes = []
            if i > 0:
                prev_price = prices[i-1]
                change_pct = ((price - prev_price) / prev_price) * 100
                if abs(change_pct) > 10:  # More than 10% change
                    notes.append(f"🚨 {change_pct:+.1f}% change")
            
            if i == 0:
                notes.append("🎯 FIRST PRICE")
            
            # Highlight double-call verified prices
            if source == 'trade_verified':
                notes.append("✅ DOUBLE-CALL VERIFIED")
            
            notes_str = " | ".join(notes) if notes else ""
            print(f"{formatted_time} | ${price:8.4f} | {source:6} | {notes_str}")
        
        # Calculate statistics
        if len(prices) > 1:
            first_price = prices[0]
            last_price = prices[-1]
            total_change = ((last_price - first_price) / first_price) * 100
            
            print(f"\n📈 PRICE ANALYSIS:")
            print(f"   First Price: ${first_price:.4f}")
            print(f"   Last Price:  ${last_price:.4f}")
            print(f"   Total Change: {total_change:+.2f}%")
            print(f"   Price Range: ${min(prices):.4f} - ${max(prices):.4f}")
            
            # Check for suspicious first price
            if len(prices) >= 3:
                avg_price = sum(prices[1:]) / len(prices[1:])  # Average excluding first
                first_deviation = abs(first_price - avg_price) / avg_price * 100
                
                if first_deviation > 20:  # First price is >20% different from average
                    print(f"🚨 SUSPICIOUS FIRST PRICE: {first_deviation:.1f}% deviation from average")
                    print(f"   This suggests the first API call returned stale/incorrect data")
                    
                    # Check current live price
                    await self.check_live_price_comparison(ticker, first_price)
        
        # Check for alerts that might be affected
        await self.check_affected_alerts(ticker)
    
    async def check_live_price_comparison(self, ticker: str, suspicious_price: float):
        """Compare suspicious price with current live data"""
        print(f"\n🔴 LIVE PRICE COMPARISON FOR {ticker}")
        print("-" * 40)
        
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Test multiple endpoints
            endpoints = [
                ('/v2/last/trade/', 'Last Trade'),
                ('/v2/last/nbbo/', 'NBBO Quote'),
                ('/v2/snapshot/locale/us/markets/stocks/tickers/', 'Snapshot')
            ]
            
            for endpoint, name in endpoints:
                try:
                    if endpoint == '/v2/snapshot/locale/us/markets/stocks/tickers/':
                        url = f"{self.base_url}{endpoint}{ticker}"
                    else:
                        url = f"{self.base_url}{endpoint}{ticker}"
                    
                    params = {'apikey': self.polygon_api_key}
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if endpoint == '/v2/last/trade/':
                                if 'results' in data and data['results']:
                                    price = data['results'].get('p', 0)
                                    trade_time = data['results'].get('t', 0)
                                    if trade_time:
                                        trade_dt = datetime.fromtimestamp(trade_time/1000, tz=pytz.UTC)
                                        age = datetime.now(pytz.UTC) - trade_dt
                                        print(f"   {name}: ${price:.4f} (age: {age})")
                                    else:
                                        print(f"   {name}: ${price:.4f}")
                                        
                            elif endpoint == '/v2/last/nbbo/':
                                if 'results' in data and data['results']:
                                    bid = data['results'].get('P', 0)
                                    ask = data['results'].get('p', 0)
                                    if bid > 0 and ask > 0:
                                        midpoint = (bid + ask) / 2
                                        print(f"   {name}: ${midpoint:.4f} (bid: ${bid:.4f}, ask: ${ask:.4f})")
                                        
                            elif endpoint == '/v2/snapshot/locale/us/markets/stocks/tickers/':
                                if 'results' in data and data['results']:
                                    result = data['results'][0]
                                    if 'value' in result:
                                        price = result['value']
                                        print(f"   {name}: ${price:.4f}")
                                        
                except Exception as e:
                    print(f"   {name}: Error - {e}")
        
        print(f"\n   🎯 Original Suspicious Price: ${suspicious_price:.4f}")
        print(f"   💡 If live prices are very different, this confirms stale API data")
    
    async def check_affected_alerts(self, ticker: str):
        """Check if this ticker generated any alerts with suspicious percentages"""
        query = f"""
        SELECT 
            ticker,
            timestamp,
            alert,
            price,
            formatDateTime(timestamp, '%Y-%m-%d %H:%M:%S') as formatted_time
        FROM News.news_alert
        WHERE ticker = '{ticker}'
        AND timestamp >= now() - INTERVAL 24 HOUR
        ORDER BY timestamp DESC
        """
        
        result = self.ch_manager.client.query(query)
        
        if result.result_rows:
            print(f"\n⚠️  ALERTS GENERATED FOR {ticker}:")
            print("Time                 | Alert | Price")
            print("-" * 40)
            
            for row in result.result_rows:
                ticker, timestamp, alert, price, formatted_time = row
                print(f"{formatted_time} | {alert:5} | ${price:.4f}")
        else:
            print(f"\n✅ No alerts generated for {ticker} in last 24 hours")

async def main():
    """Main diagnostic function"""
    diagnostic = PriceDiagnostic()
    
    # Test with the specific problematic tickers from previous diagnostic
    problematic_tickers = ['TTEC', 'BCTX', 'CYRX', 'BTCT', 'RLYB', 'CRVO', 'POAI']
    
    print("🧪 TESTING DOUBLE-CALL FIX ON PREVIOUSLY PROBLEMATIC TICKERS")
    print("=" * 70)
    print("This will test the new double-call logic against the old single-call logic")
    print("for tickers that previously showed garbage first price data.")
    print()
    
    # Test the double-call fix on problematic tickers
    for ticker in problematic_tickers[:3]:  # Test first 3 tickers
        await diagnostic.test_double_call_vs_single_call(ticker)
        await asyncio.sleep(1)  # Rate limiting between tests
    
    # If no specific tickers provided, find recent ones with large price swings
    print("\n" + "=" * 70)
    print("🔍 FINDING CURRENT TICKERS WITH SUSPICIOUS PRICE MOVEMENTS...")
    
    query = """
    SELECT 
        ticker,
        ((argMax(price, timestamp) - argMin(price, timestamp)) / argMin(price, timestamp)) * 100 as change_pct,
        argMin(price, timestamp) as first_price,
        argMax(price, timestamp) as last_price,
        count() as price_count
    FROM News.price_tracking
    WHERE timestamp >= now() - INTERVAL 24 HOUR
    GROUP BY ticker
    HAVING price_count >= 5 AND (change_pct > 50 OR change_pct < -30)
    ORDER BY change_pct DESC
    LIMIT 10
    """
    
    result = diagnostic.ch_manager.client.query(query)
    
    if result.result_rows:
        print("\n🚨 Found tickers with suspicious price movements:")
        print("Ticker | Change% | First Price | Last Price | Count")
        print("-" * 50)
        
        test_tickers = []
        for row in result.result_rows:
            ticker, change_pct, first_price, last_price, price_count = row
            print(f"{ticker:6} | {change_pct:+7.1f}% | ${first_price:8.4f} | ${last_price:8.4f} | {price_count:5}")
            test_tickers.append(ticker)
        
        # Diagnose each ticker
        for ticker in test_tickers[:3]:  # Limit to first 3 for detailed analysis
            await diagnostic.diagnose_ticker_prices(ticker)
            await asyncio.sleep(1)  # Rate limiting
    else:
        print("✅ No tickers with suspicious movements found")

if __name__ == "__main__":
    asyncio.run(main())