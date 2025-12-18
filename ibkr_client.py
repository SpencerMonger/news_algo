#!/usr/bin/env python3
"""
IBKR TWS API Client for NewsHead Price Monitoring

Handles connection, market data subscriptions, and price callbacks.
Replaces Polygon WebSocket as the primary data source.

Requirements:
    - TWS or IB Gateway must be running
    - API must be enabled in TWS settings
    - Use client_id=10 to avoid conflict with tradehead (client_id=1)

Environment Variables:
    - IBKR_HOST: TWS/Gateway host (default: 127.0.0.1)
    - IBKR_PORT: TWS/Gateway port (7497=paper, 7496=live)
    - IBKR_CLIENT_ID: Unique client ID (default: 10)
"""

import threading
import logging
import time
import os
from datetime import datetime
from typing import Dict, Set, Optional, Callable, List

import pytz
from dotenv import load_dotenv

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

load_dotenv()

logger = logging.getLogger(__name__)

# Reduce ibapi logging verbosity (very spammy at DEBUG level)
logging.getLogger('ibapi').setLevel(logging.WARNING)
logging.getLogger('ibapi.client').setLevel(logging.WARNING)
logging.getLogger('ibapi.wrapper').setLevel(logging.WARNING)
logging.getLogger('ibapi.decoder').setLevel(logging.WARNING)
logging.getLogger('ibapi.connection').setLevel(logging.WARNING)
logging.getLogger('ibapi.reader').setLevel(logging.WARNING)


class IBKRClient(EWrapper, EClient):
    """
    IBKR TWS API client that handles:
    - Connection to TWS/Gateway
    - Market data subscriptions
    - Price tick callbacks
    - Thread-safe price buffer updates
    """
    
    def __init__(self, 
                 host: str = "127.0.0.1",
                 port: int = 7497,
                 client_id: int = 10,
                 price_callback: Optional[Callable] = None):
        """
        Initialize IBKR client
        
        Args:
            host: TWS/Gateway host (default localhost)
            port: TWS/Gateway port (7497=paper, 7496=live)
            client_id: Unique client ID (must differ from other connections)
            price_callback: Optional callback function for price updates
        """
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        self.price_callback = price_callback
        
        # Connection state
        self.connected = False
        self.next_order_id = None
        self._connection_error: Optional[str] = None
        
        # Subscription tracking
        self.next_req_id = 1000
        self.reqid_to_ticker: Dict[int, str] = {}
        self.ticker_to_reqid: Dict[str, int] = {}
        self.active_subscriptions: Set[str] = set()
        
        # Price buffer (thread-safe access required)
        self.price_buffer: Dict[str, List[Dict]] = {}
        self.price_buffer_lock = threading.Lock()
        
        # Volume tracking (tickSize arrives separately from tickPrice)
        self.last_volume: Dict[str, int] = {}
        
        # Failed tickers - don't retry these (e.g., "No security definition")
        self.failed_tickers: Dict[str, str] = {}  # ticker -> error reason
        
        # Last known prices (for quick access without buffer)
        self.last_prices: Dict[str, float] = {}
        
        # Message processing thread
        self.api_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'ticks_received': 0,
            'subscriptions_active': 0,
            'errors': 0,
            'reconnections': 0
        }

    # =========================================================================
    # CONNECTION CALLBACKS
    # =========================================================================
    
    def nextValidId(self, orderId: int):
        """Called when connection is established - signals ready state"""
        self.next_order_id = orderId
        self.connected = True
        self._connection_error = None
        logger.info(f"✅ IBKR Connected. Next Order ID: {orderId}")
        logger.info(f"   Mode: {'PAPER TRADING' if self.port == 7497 else 'LIVE TRADING'}")
    
    def connectionClosed(self):
        """Called when connection is lost"""
        self.connected = False
        logger.warning("❌ IBKR Connection closed")
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Handle errors from IBKR"""
        # Informational messages (not errors)
        if errorCode in [2104, 2106, 2158]:
            # 2104 = Market data farm connection OK
            # 2106 = HMDS data farm connection OK
            # 2158 = Sec-def data farm connection OK
            logger.debug(f"IBKR Info [{errorCode}]: {errorString}")
            return
        
        # Connection-related errors
        if errorCode == 502:
            # Couldn't connect to TWS
            self._connection_error = f"Cannot connect to TWS/Gateway: {errorString}"
            logger.error(f"❌ [{errorCode}] {self._connection_error}")
            self.stats['errors'] += 1
            return
        
        if errorCode == 504:
            # Not connected
            self._connection_error = f"Not connected to TWS/Gateway: {errorString}"
            logger.error(f"❌ [{errorCode}] {self._connection_error}")
            self.stats['errors'] += 1
            return
        
        # Security definition errors - PERMANENT failure, don't retry
        if errorCode == 200:
            ticker = self.reqid_to_ticker.get(reqId, "UNKNOWN")
            logger.error(f"❌ No security definition for {ticker}: {errorString}")
            # Mark as permanently failed - won't retry this ticker
            self.failed_tickers[ticker] = f"No security definition: {errorString}"
            # Remove from active subscriptions since it failed
            self.active_subscriptions.discard(ticker)
            # Clean up tracking
            if reqId in self.reqid_to_ticker:
                del self.reqid_to_ticker[reqId]
            if ticker in self.ticker_to_reqid:
                del self.ticker_to_reqid[ticker]
            self.stats['errors'] += 1
            logger.warning(f"⛔ {ticker} marked as failed - will not retry subscription")
            return
        
        # Market data subscription errors
        if errorCode == 354:
            logger.error(f"❌ Market data not subscribed: {errorString}")
            logger.error("   You may need an IBKR market data subscription")
            self.stats['errors'] += 1
            return
        
        # Delayed market data notification (informational)
        if errorCode == 10167:
            logger.info(f"ℹ️  [{errorCode}] Delayed market data in use (real-time requires subscription)")
            return
        
        # Duplicate ticker ID
        if errorCode == 321:
            ticker = self.reqid_to_ticker.get(reqId, "UNKNOWN")
            logger.warning(f"⚠️  [{errorCode}] Duplicate ticker ID for {ticker}: {errorString}")
            return
        
        # Other errors
        logger.error(f"IBKR Error [{errorCode}] reqId={reqId}: {errorString}")
        self.stats['errors'] += 1

    # =========================================================================
    # MARKET DATA CALLBACKS
    # =========================================================================
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        """
        Called when price updates arrive
        
        tickType values:
        - 1 = BID
        - 2 = ASK  
        - 4 = LAST (most important - actual trade price)
        - 6 = HIGH
        - 7 = LOW
        - 9 = CLOSE
        """
        # Only process LAST price (tickType 4) - actual executed trades
        if tickType != 4:
            return
        
        ticker = self.reqid_to_ticker.get(reqId)
        if not ticker:
            return
        
        if price <= 0:
            return
        
        current_time = datetime.now(pytz.UTC)
        volume = self.last_volume.get(ticker, 0)
        
        # Update last known price
        self.last_prices[ticker] = price
        
        # Thread-safe buffer update
        with self.price_buffer_lock:
            if ticker not in self.price_buffer:
                self.price_buffer[ticker] = []
            
            self.price_buffer[ticker].append({
                'price': price,
                'volume': volume,
                'timestamp': current_time,
                'source': 'ibkr_last_trade'
            })
        
        self.stats['ticks_received'] += 1
        
        logger.debug(f"📈 IBKR TICK {ticker}: ${price:.4f} (vol: {volume})")
        
        # Optional callback for immediate processing
        if self.price_callback:
            try:
                self.price_callback(ticker, price, volume, current_time)
            except Exception as e:
                logger.error(f"Error in price callback for {ticker}: {e}")
    
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
        """Handle generic tick data (optional - can ignore for basic price tracking)"""
        pass
    
    def tickString(self, reqId: int, tickType: int, value: str):
        """Handle string tick data (optional - contains timestamp info)"""
        pass

    # =========================================================================
    # CONNECTION METHODS
    # =========================================================================
    
    def connect_and_run(self) -> bool:
        """
        Connect to TWS/Gateway and start message processing thread
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"🔌 Connecting to IBKR at {self.host}:{self.port} (client_id={self.client_id})")
        
        try:
            self.connect(self.host, self.port, self.client_id)
            
            # Start message processing in separate thread
            self.api_thread = threading.Thread(target=self.run, daemon=True, name="IBKR-API-Thread")
            self.api_thread.start()
            
            # Wait for connection confirmation (nextValidId callback)
            timeout = 10
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                if self._connection_error:
                    logger.error(f"❌ Connection failed: {self._connection_error}")
                    return False
                time.sleep(0.1)
            
            if self.connected:
                logger.info(f"✅ IBKR connection established successfully")
                logger.info(f"   Host: {self.host}:{self.port}")
                logger.info(f"   Client ID: {self.client_id}")
                return True
            else:
                logger.error("❌ IBKR connection timeout - is TWS/Gateway running?")
                logger.error("   Troubleshooting:")
                logger.error("   1. Ensure TWS or IB Gateway is running")
                logger.error("   2. Check API settings: File > Global Configuration > API > Settings")
                logger.error("   3. Ensure 'Enable ActiveX and Socket Clients' is checked")
                logger.error(f"   4. Verify port {self.port} is correct (7497=paper, 7496=live)")
                return False
                
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            return False
    
    def disconnect_safely(self):
        """Safely disconnect from IBKR"""
        try:
            # Cancel all subscriptions first
            for ticker in list(self.active_subscriptions):
                self.unsubscribe_ticker(ticker)
            
            self.disconnect()
            self.connected = False
            logger.info("🔌 IBKR disconnected")
        except Exception as e:
            logger.error(f"Error during IBKR disconnect: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connected and self.isConnected()

    # =========================================================================
    # SUBSCRIPTION METHODS
    # =========================================================================
    
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
        """
        Subscribe to real-time market data for a ticker
        
        Args:
            ticker: Stock symbol to subscribe to
            
        Returns:
            True if subscription was successful/initiated
        """
        if not self.connected:
            logger.warning(f"Cannot subscribe to {ticker} - not connected")
            return False
        
        # Skip tickers that previously failed (e.g., no security definition)
        if ticker in self.failed_tickers:
            logger.debug(f"⛔ Skipping {ticker} - previously failed: {self.failed_tickers[ticker]}")
            return False
        
        if ticker in self.active_subscriptions:
            logger.debug(f"{ticker} already subscribed")
            return True
        
        try:
            contract = self._create_stock_contract(ticker)
            req_id = self._get_next_req_id()
            
            # Track the mapping
            self.reqid_to_ticker[req_id] = ticker
            self.ticker_to_reqid[ticker] = req_id
            self.active_subscriptions.add(ticker)
            
            # Initialize price tracking for this ticker
            with self.price_buffer_lock:
                if ticker not in self.price_buffer:
                    self.price_buffer[ticker] = []
            self.last_volume[ticker] = 0
            
            # Request streaming market data
            # genericTickList="" for basic price data
            # snapshot=False for streaming (not one-time)
            # regulatorySnapshot=False for real-time data
            self.reqMktData(req_id, contract, "", False, False, [])
            
            self.stats['subscriptions_active'] = len(self.active_subscriptions)
            logger.info(f"📡 IBKR: Subscribed to {ticker} (reqId={req_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to {ticker}: {e}")
            self.active_subscriptions.discard(ticker)
            return False
    
    def unsubscribe_ticker(self, ticker: str):
        """Cancel market data subscription for a ticker"""
        if ticker not in self.active_subscriptions:
            return
        
        req_id = self.ticker_to_reqid.get(ticker)
        if req_id:
            try:
                self.cancelMktData(req_id)
                logger.info(f"📡 IBKR: Unsubscribed from {ticker}")
            except Exception as e:
                logger.error(f"Error unsubscribing from {ticker}: {e}")
            
            # Clean up tracking
            if ticker in self.ticker_to_reqid:
                del self.ticker_to_reqid[ticker]
            if req_id in self.reqid_to_ticker:
                del self.reqid_to_ticker[req_id]
        
        self.active_subscriptions.discard(ticker)
        
        # Clean up price buffer for this ticker
        with self.price_buffer_lock:
            if ticker in self.price_buffer:
                del self.price_buffer[ticker]
        
        if ticker in self.last_volume:
            del self.last_volume[ticker]
        
        if ticker in self.last_prices:
            del self.last_prices[ticker]
        
        self.stats['subscriptions_active'] = len(self.active_subscriptions)
    
    def update_subscriptions(self, needed_tickers: Set[str]):
        """
        Update subscriptions to match the needed tickers set
        - Subscribe to new tickers
        - Unsubscribe from removed tickers
        - Skip failed tickers
        
        Args:
            needed_tickers: Set of tickers that should be subscribed
        """
        if not self.connected:
            logger.warning("Cannot update subscriptions - not connected")
            return
        
        current = self.active_subscriptions.copy()
        needed = set(needed_tickers)
        
        # Filter out failed tickers from needed set
        valid_needed = needed - set(self.failed_tickers.keys())
        skipped_failed = needed & set(self.failed_tickers.keys())
        
        if skipped_failed:
            logger.debug(f"⛔ Skipping {len(skipped_failed)} failed tickers: {skipped_failed}")
        
        # Unsubscribe from tickers no longer needed
        to_remove = current - valid_needed
        for ticker in to_remove:
            self.unsubscribe_ticker(ticker)
        
        # Subscribe to new tickers (only valid ones)
        to_add = valid_needed - current
        for ticker in to_add:
            self.subscribe_ticker(ticker)
        
        if to_add or to_remove:
            logger.info(f"📡 IBKR Subscriptions: +{len(to_add)} -{len(to_remove)} = {len(self.active_subscriptions)} active")
    
    def is_ticker_failed(self, ticker: str) -> bool:
        """Check if a ticker has permanently failed"""
        return ticker in self.failed_tickers
    
    def get_failed_tickers(self) -> Dict[str, str]:
        """Get dictionary of failed tickers and their error reasons"""
        return dict(self.failed_tickers)
    
    def clear_failed_ticker(self, ticker: str):
        """Clear a specific ticker from the failed list (allows retry)"""
        if ticker in self.failed_tickers:
            del self.failed_tickers[ticker]
            logger.info(f"🔄 Cleared {ticker} from failed list - will retry on next subscription")
    
    def clear_all_failed_tickers(self):
        """Clear all failed tickers (allows retry of all)"""
        count = len(self.failed_tickers)
        self.failed_tickers.clear()
        logger.info(f"🔄 Cleared {count} failed tickers - will retry on next subscription")

    # =========================================================================
    # BUFFER METHODS
    # =========================================================================
    
    def get_and_clear_buffer(self) -> Dict[str, List[Dict]]:
        """
        Get current price buffer contents and clear it
        Thread-safe operation
        
        Returns:
            Dictionary mapping ticker to list of price updates
        """
        with self.price_buffer_lock:
            buffer_copy = {}
            for ticker, prices in self.price_buffer.items():
                buffer_copy[ticker] = list(prices)  # Create copy of list
            self.price_buffer.clear()
        return buffer_copy
    
    def get_buffer_for_tickers(self, tickers: Set[str]) -> Dict[str, List[Dict]]:
        """
        Get buffer contents for specific tickers only and clear those entries
        Thread-safe operation
        
        Args:
            tickers: Set of tickers to retrieve
            
        Returns:
            Dictionary mapping ticker to list of price updates
        """
        result = {}
        with self.price_buffer_lock:
            for ticker in tickers:
                if ticker in self.price_buffer:
                    result[ticker] = list(self.price_buffer[ticker])  # Create copy
                    del self.price_buffer[ticker]
        return result
    
    def get_last_price(self, ticker: str) -> Optional[float]:
        """
        Get the last known price for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Last known price or None if not available
        """
        return self.last_prices.get(ticker)
    
    def get_buffer_size(self) -> int:
        """Get total number of prices in buffer across all tickers"""
        with self.price_buffer_lock:
            return sum(len(prices) for prices in self.price_buffer.values())
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            **self.stats,
            'connected': self.connected,
            'buffer_size': self.get_buffer_size(),
            'port': self.port,
            'mode': 'PAPER' if self.port == 7497 else 'LIVE',
            'failed_tickers': len(self.failed_tickers),
            'failed_ticker_list': list(self.failed_tickers.keys())
        }


# =============================================================================
# MODULE-LEVEL SINGLETON INSTANCE
# =============================================================================

_ibkr_client: Optional[IBKRClient] = None
_ibkr_client_lock = threading.Lock()


def get_ibkr_client(host: str = None, port: int = None, client_id: int = None) -> IBKRClient:
    """
    Get or create the singleton IBKR client instance
    
    Environment variables used if args not provided:
    - IBKR_HOST (default: 127.0.0.1)
    - IBKR_PORT (default: 7497 for paper, 7496 for live)
    - IBKR_CLIENT_ID (default: 10)
    
    Args:
        host: Optional host override
        port: Optional port override
        client_id: Optional client ID override
        
    Returns:
        Singleton IBKRClient instance
    """
    global _ibkr_client
    
    with _ibkr_client_lock:
        if _ibkr_client is None:
            _host = host or os.getenv('IBKR_HOST', '127.0.0.1')
            _port = port or int(os.getenv('IBKR_PORT', '7497'))
            _client_id = client_id or int(os.getenv('IBKR_CLIENT_ID', '10'))
            
            _ibkr_client = IBKRClient(host=_host, port=_port, client_id=_client_id)
            logger.info(f"Created IBKR client singleton: {_host}:{_port} (client_id={_client_id})")
        
        return _ibkr_client


def reset_ibkr_client():
    """
    Reset the singleton IBKR client (useful for testing or reconnection)
    """
    global _ibkr_client
    
    with _ibkr_client_lock:
        if _ibkr_client is not None:
            try:
                _ibkr_client.disconnect_safely()
            except:
                pass
            _ibkr_client = None
            logger.info("IBKR client singleton reset")


# =============================================================================
# MAIN - For standalone testing
# =============================================================================

if __name__ == "__main__":
    """Standalone test of IBKR client"""
    import sys
    
    # Configure logging for standalone test
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 50)
    print("IBKR Client Standalone Test")
    print("=" * 50)
    
    # Get configuration from environment
    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '7497'))
    client_id = int(os.getenv('IBKR_CLIENT_ID', '99'))  # Use 99 for testing
    
    print(f"Connecting to {host}:{port} (client_id={client_id})")
    print()
    
    # Create client
    client = IBKRClient(host=host, port=port, client_id=client_id)
    
    # Connect
    success = client.connect_and_run()
    
    if not success:
        print("❌ Connection failed!")
        sys.exit(1)
    
    # Test subscription
    test_ticker = "AAPL"
    print(f"\n📡 Subscribing to {test_ticker}...")
    client.subscribe_ticker(test_ticker)
    
    # Wait for some ticks
    print("\n⏳ Waiting 10 seconds for price data...")
    time.sleep(10)
    
    # Check buffer
    buffer = client.get_and_clear_buffer()
    if test_ticker in buffer:
        print(f"\n✅ Received {len(buffer[test_ticker])} price updates for {test_ticker}")
        for tick in buffer[test_ticker][-5:]:  # Show last 5
            print(f"   ${tick['price']:.4f} at {tick['timestamp']}")
    else:
        print(f"\n❌ No price data received for {test_ticker}")
    
    # Print stats
    stats = client.get_stats()
    print(f"\n📊 Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Disconnect
    print("\n🔌 Disconnecting...")
    client.disconnect_safely()
    
    print("\n✅ Test complete!")

