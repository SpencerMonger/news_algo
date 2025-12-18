#!/usr/bin/env python3
"""
Standalone IBKR TWS API Connection Test

This script tests the IBKR connection independently before full integration.
It connects to TWS/Gateway, subscribes to a ticker, and displays price updates.

Can use either:
1. The standalone test client (for basic testing)
2. The production ibkr_client module (for integration testing)

Requirements:
    - TWS or IB Gateway must be running
    - API must be enabled in TWS settings (File > Global Configuration > API > Settings)
    - Enable "Enable ActiveX and Socket Clients"

Usage:
    python testfiles/test_ibkr_connection.py
    python testfiles/test_ibkr_connection.py --port 7496   # Live trading
    python testfiles/test_ibkr_connection.py --ticker MSFT  # Different ticker
    python testfiles/test_ibkr_connection.py --use-module   # Use production ibkr_client module
"""

import sys
import os
import time
import threading
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
except ImportError:
    print("❌ ERROR: ibapi not installed!")
    print("   Run: pip install 'ibapi>=10.30.1'")
    sys.exit(1)


class TestIBKRClient(EWrapper, EClient):
    """
    Test IBKR client for verifying connection and market data.
    Combines EWrapper (callbacks) and EClient (requests) into one class.
    """
    
    def __init__(self, host: str, port: int, client_id: int):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Connection state
        self.connected = False
        self.next_order_id = None
        
        # Subscription tracking
        self.next_req_id = 1000
        self.reqid_to_ticker = {}
        self.ticker_to_reqid = {}
        
        # Price tracking
        self.prices = {}  # ticker -> list of (timestamp, price, volume)
        self.last_volume = {}  # ticker -> last trade volume
        
        # Statistics
        self.tick_count = 0
        self.start_time = None
    
    # === CONNECTION CALLBACKS ===
    
    def nextValidId(self, orderId: int):
        """Called when connection is established - signals ready state"""
        self.next_order_id = orderId
        self.connected = True
        self.start_time = time.time()
        print(f"✅ CONNECTED to IBKR!")
        print(f"   Host: {self.host}:{self.port}")
        print(f"   Client ID: {self.client_id}")
        print(f"   Next Order ID: {orderId}")
        print(f"   Mode: {'PAPER TRADING' if self.port == 7497 else 'LIVE TRADING'}")
        print("-" * 50)
    
    def connectionClosed(self):
        """Called when connection is lost"""
        self.connected = False
        print("❌ Connection closed")
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Handle errors from IBKR"""
        # Informational messages (not errors)
        if errorCode in [2104, 2106, 2158]:
            # 2104 = Market data farm connection OK
            # 2106 = HMDS data farm connection OK
            # 2158 = Sec-def data farm connection OK
            print(f"   ℹ️  [{errorCode}] {errorString}")
            return
        
        # Market data not subscribed (common issue)
        if errorCode == 354:
            print(f"   ⚠️  [{errorCode}] Market data not subscribed - you may need IBKR market data subscription")
            return
        
        # No security definition found
        if errorCode == 200:
            ticker = self.reqid_to_ticker.get(reqId, "UNKNOWN")
            print(f"   ❌ [{errorCode}] No security definition for {ticker}: {errorString}")
            return
        
        # 10167 = Delayed market data (not real-time) - this is informational
        if errorCode == 10167:
            print(f"   ℹ️  [{errorCode}] Delayed market data in use (real-time requires subscription)")
            return
        
        # Other errors
        print(f"   ⚠️  Error [{errorCode}] reqId={reqId}: {errorString}")
    
    # === MARKET DATA CALLBACKS ===
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        """
        Called when price updates arrive
        
        tickType values we care about:
        - 1 = BID
        - 2 = ASK  
        - 4 = LAST (most important - actual trade price)
        - 6 = HIGH
        - 7 = LOW
        - 9 = CLOSE
        """
        ticker = self.reqid_to_ticker.get(reqId)
        if not ticker:
            return
        
        if price <= 0:
            return
        
        # Only print LAST price (tickType 4) to reduce noise
        tick_names = {1: "BID", 2: "ASK", 4: "LAST", 6: "HIGH", 7: "LOW", 9: "CLOSE"}
        tick_name = tick_names.get(tickType, f"TYPE_{tickType}")
        
        # Track all prices for LAST trades
        if tickType == 4:  # LAST - actual executed trade
            self.tick_count += 1
            timestamp = datetime.now()
            volume = self.last_volume.get(ticker, 0)
            
            if ticker not in self.prices:
                self.prices[ticker] = []
            
            self.prices[ticker].append({
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            })
            
            print(f"   📈 {ticker} LAST: ${price:.4f} (vol: {volume}) @ {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        else:
            # Print other tick types more quietly
            print(f"   📊 {ticker} {tick_name}: ${price:.4f}")
    
    def tickSize(self, reqId: int, tickType: int, size: int):
        """
        Called when size/volume updates arrive
        
        tickType values:
        - 0 = BID_SIZE
        - 3 = ASK_SIZE
        - 5 = LAST_SIZE (volume of last trade)
        - 8 = VOLUME (cumulative daily volume)
        """
        if tickType == 5:  # LAST_SIZE - volume of most recent trade
            ticker = self.reqid_to_ticker.get(reqId)
            if ticker:
                self.last_volume[ticker] = size
    
    def tickGeneric(self, reqId: int, tickType: int, value: float):
        """Handle generic tick data"""
        pass  # Ignore for basic testing
    
    def tickString(self, reqId: int, tickType: int, value: str):
        """Handle string tick data (contains timestamp info)"""
        pass  # Ignore for basic testing
    
    # === HELPER METHODS ===
    
    def _get_next_req_id(self) -> int:
        """Get next unique request ID"""
        req_id = self.next_req_id
        self.next_req_id += 1
        return req_id
    
    def _create_stock_contract(self, ticker: str) -> Contract:
        """Create IBKR Contract object for a stock ticker"""
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"  # SMART routing for best execution
        contract.currency = "USD"
        return contract
    
    def subscribe_ticker(self, ticker: str) -> bool:
        """Subscribe to real-time market data for a ticker"""
        if not self.connected:
            print(f"❌ Cannot subscribe to {ticker} - not connected")
            return False
        
        try:
            contract = self._create_stock_contract(ticker)
            req_id = self._get_next_req_id()
            
            # Track the mapping
            self.reqid_to_ticker[req_id] = ticker
            self.ticker_to_reqid[ticker] = req_id
            
            # Initialize price tracking
            self.prices[ticker] = []
            self.last_volume[ticker] = 0
            
            # Request streaming market data
            # genericTickList="" for basic price data
            # snapshot=False for streaming (not one-time)
            # regulatorySnapshot=False for real-time data
            self.reqMktData(req_id, contract, "", False, False, [])
            
            print(f"📡 Subscribed to {ticker} (reqId={req_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error subscribing to {ticker}: {e}")
            return False
    
    def unsubscribe_ticker(self, ticker: str):
        """Cancel market data subscription for a ticker"""
        req_id = self.ticker_to_reqid.get(ticker)
        if req_id:
            try:
                self.cancelMktData(req_id)
                print(f"📡 Unsubscribed from {ticker}")
            except Exception as e:
                print(f"Error unsubscribing from {ticker}: {e}")
            
            # Clean up tracking
            del self.ticker_to_reqid[ticker]
            del self.reqid_to_ticker[req_id]
    
    def print_summary(self):
        """Print summary statistics"""
        if self.start_time:
            runtime = time.time() - self.start_time
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)
            print(f"   Runtime: {runtime:.1f} seconds")
            print(f"   Total ticks received: {self.tick_count}")
            
            for ticker, prices in self.prices.items():
                if prices:
                    print(f"\n   {ticker}:")
                    print(f"      Price updates: {len(prices)}")
                    print(f"      First price: ${prices[0]['price']:.4f}")
                    print(f"      Last price: ${prices[-1]['price']:.4f}")
                    
                    # Calculate price range
                    all_prices = [p['price'] for p in prices]
                    print(f"      Low: ${min(all_prices):.4f}")
                    print(f"      High: ${max(all_prices):.4f}")


def run_test(host: str, port: int, client_id: int, ticker: str, duration: int):
    """Run the IBKR connection test"""
    
    print("=" * 50)
    print("🔌 IBKR TWS API CONNECTION TEST")
    print("=" * 50)
    print(f"   Target: {host}:{port}")
    print(f"   Client ID: {client_id}")
    print(f"   Test Ticker: {ticker}")
    print(f"   Duration: {duration} seconds")
    print("=" * 50)
    print()
    
    # Create client
    client = TestIBKRClient(host=host, port=port, client_id=client_id)
    
    try:
        # Connect to TWS/Gateway
        print("🔌 Connecting to IBKR...")
        client.connect(host, port, client_id)
        
        # Start message processing in separate thread
        api_thread = threading.Thread(target=client.run, daemon=True)
        api_thread.start()
        
        # Wait for connection confirmation (nextValidId callback)
        timeout = 10
        start = time.time()
        while not client.connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not client.connected:
            print("❌ Connection timeout - is TWS/Gateway running?")
            print()
            print("📋 TROUBLESHOOTING:")
            print("   1. Make sure TWS or IB Gateway is running")
            print("   2. Check API settings: File > Global Configuration > API > Settings")
            print("   3. Ensure 'Enable ActiveX and Socket Clients' is checked")
            print(f"   4. Verify the correct port ({port}) is being used")
            print("   5. Check if another app is using the same client_id")
            return False
        
        # Subscribe to test ticker
        print(f"\n📡 Subscribing to {ticker}...")
        client.subscribe_ticker(ticker)
        
        # Wait for market data
        print(f"\n⏳ Listening for {duration} seconds...\n")
        print("-" * 50)
        
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        # Unsubscribe and print summary
        client.unsubscribe_ticker(ticker)
        client.print_summary()
        
        # Disconnect
        print("\n🔌 Disconnecting...")
        client.disconnect()
        
        return client.tick_count > 0
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_module_test(host: str, port: int, client_id: int, ticker: str, duration: int):
    """Run test using the production ibkr_client module"""
    try:
        from ibkr_client import IBKRClient
    except ImportError:
        print("❌ ERROR: ibkr_client module not found!")
        print("   Make sure ibkr_client.py exists in the project root")
        return False
    
    print("=" * 50)
    print("🔌 IBKR MODULE TEST (using ibkr_client.py)")
    print("=" * 50)
    print(f"   Target: {host}:{port}")
    print(f"   Client ID: {client_id}")
    print(f"   Test Ticker: {ticker}")
    print(f"   Duration: {duration} seconds")
    print("=" * 50)
    print()
    
    # Create client using the production module
    client = IBKRClient(host=host, port=port, client_id=client_id)
    
    try:
        # Connect
        print("🔌 Connecting to IBKR using production module...")
        success = client.connect_and_run()
        
        if not success:
            print("❌ Connection failed!")
            return False
        
        # Subscribe to test ticker
        print(f"\n📡 Subscribing to {ticker}...")
        client.subscribe_ticker(ticker)
        
        # Wait for market data
        print(f"\n⏳ Listening for {duration} seconds...\n")
        print("-" * 50)
        
        # Poll for updates
        start = time.time()
        while (time.time() - start) < duration:
            time.sleep(1)
            # Check buffer
            last_price = client.get_last_price(ticker)
            if last_price:
                print(f"   📈 {ticker} LAST: ${last_price:.4f}")
        
        # Get final buffer stats
        buffer = client.get_and_clear_buffer()
        stats = client.get_stats()
        
        print("\n" + "=" * 50)
        print("📊 MODULE TEST SUMMARY")
        print("=" * 50)
        print(f"   Runtime: {duration} seconds")
        print(f"   Ticks Received: {stats['ticks_received']}")
        print(f"   Connected: {stats['connected']}")
        print(f"   Mode: {stats['mode']}")
        
        if ticker in buffer:
            prices = buffer[ticker]
            print(f"\n   {ticker}:")
            print(f"      Price updates: {len(prices)}")
            if prices:
                all_prices = [p['price'] for p in prices]
                print(f"      First price: ${prices[0]['price']:.4f}")
                print(f"      Last price: ${prices[-1]['price']:.4f}")
                print(f"      Low: ${min(all_prices):.4f}")
                print(f"      High: ${max(all_prices):.4f}")
        
        # Disconnect
        print("\n🔌 Disconnecting...")
        client.disconnect_safely()
        
        return stats['ticks_received'] > 0
        
    except Exception as e:
        print(f"❌ Error during module test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test IBKR TWS API connection")
    parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway port (7497=paper, 7496=live)")
    parser.add_argument("--client-id", type=int, default=99, help="Client ID (default: 99 for testing)")
    parser.add_argument("--ticker", default="AAPL", help="Test ticker symbol (default: AAPL)")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds (default: 15)")
    parser.add_argument("--use-module", action="store_true", help="Use production ibkr_client module instead of standalone test")
    
    args = parser.parse_args()
    
    if args.use_module:
        success = run_module_test(
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            ticker=args.ticker,
            duration=args.duration
        )
    else:
        success = run_test(
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            ticker=args.ticker,
            duration=args.duration
        )
    
    print()
    if success:
        print("✅ TEST PASSED - IBKR connection working!")
        if args.use_module:
            print("   Production ibkr_client module is ready for use.")
        else:
            print("   You can proceed with full integration.")
    else:
        print("❌ TEST FAILED - No price data received")
        print("   Check the troubleshooting steps above.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

