# IBKR Price Checker Implementation Prompt

## Context for Implementation

This document provides complete instructions for implementing IBKR TWS API integration into the newshead price monitoring system, replacing Polygon WebSocket as the primary data source.

---

## Project Overview

**Project:** newshead - Real-time news monitoring and price tracking system  
**Goal:** Replace Polygon WebSocket price data with IBKR TWS API streaming data  
**Current File to Modify:** `price_checker.py`  
**New File to Create:** `ibkr_client.py`

### Key Constraints

- **No Polygon Fallback:** Remove Polygon entirely - IBKR is the sole data source
- **Concurrent Tickers:** Less than 10 at any time (subscription limits not a concern)
- **Port Switching:** Must support easy switching between paper (7497) and live (7496) via environment variable
- **Simultaneous Connections:** Another project (tradehead) uses client_id=1 on the same TWS - newshead must use a different client_id (use 10)
- **Threading:** IBKR API is callback-based and requires running in a separate thread; integrate with existing asyncio event loop

---

## Current Architecture

### File: `price_checker.py` (1461 lines)

The current implementation uses Polygon WebSocket for real-time price data. Key components:

```
ContinuousPriceMonitor class:
├── __init__() - Sets up Polygon API key, WebSocket URL, buffers
├── initialize() - Creates ClickHouse connection, sets up WebSocket
├── setup_websocket_connection() - Connects to Polygon WebSocket
├── authenticate_websocket() - Sends API key to Polygon
├── websocket_listener() - Receives WebSocket messages in loop
├── handle_websocket_message() - Parses Polygon JSON messages
├── process_single_websocket_message() - Extracts price from message
├── process_websocket_prices() - Flushes buffer to ClickHouse
├── update_websocket_subscriptions() - Sends subscribe/unsubscribe to Polygon
├── track_prices_rest_fallback() - REST API fallback (remove this)
├── get_current_price_rest_fallback() - Single ticker REST call (remove this)
├── check_price_alerts_optimized() - Alert logic (KEEP UNCHANGED)
├── file_trigger_monitor_async() - File trigger handler (KEEP UNCHANGED)
├── continuous_polling_loop() - Main processing loop (KEEP - modify slightly)
├── get_tickers_within_60_second_window() - Window enforcement (KEEP UNCHANGED)
└── start() - Entry point (modify to use IBKR)
```

### Data Flow (Current - Polygon)

```
Polygon WebSocket → handle_websocket_message() → websocket_price_buffer → 
process_websocket_prices() → ClickHouse → check_price_alerts_optimized() → Alerts
```

### Data Flow (Target - IBKR)

```
IBKR TWS API → tickPrice() callback → price_buffer → 
process_prices() → ClickHouse → check_price_alerts_optimized() → Alerts
```

---

## Implementation Requirements

### 1. Create New File: `ibkr_client.py`

This module handles all IBKR-specific functionality:

#### Class Structure

```python
"""
IBKR TWS API Client for NewsHead Price Monitoring
Handles connection, market data subscriptions, and price callbacks
"""

import threading
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Callable
import pytz

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

logger = logging.getLogger(__name__)


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
        
        # Subscription tracking
        self.next_req_id = 1000
        self.reqid_to_ticker: Dict[int, str] = {}
        self.ticker_to_reqid: Dict[str, int] = {}
        self.active_subscriptions: Set[str] = set()
        
        # Price buffer (thread-safe access required)
        self.price_buffer: Dict[str, list] = {}
        self.price_buffer_lock = threading.Lock()
        
        # Volume tracking (tickSize arrives separately from tickPrice)
        self.last_volume: Dict[str, int] = {}
        
        # Message processing thread
        self.api_thread: Optional[threading.Thread] = None
```

#### Required EWrapper Callbacks to Implement

```python
    # === CONNECTION CALLBACKS ===
    
    def nextValidId(self, orderId: int):
        """Called when connection is established - signals ready state"""
        self.next_order_id = orderId
        self.connected = True
        logger.info(f"✅ IBKR Connected. Next Order ID: {orderId}")
    
    def connectionClosed(self):
        """Called when connection is lost"""
        self.connected = False
        logger.warning("❌ IBKR Connection closed")
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Handle errors from IBKR"""
        # Error codes to handle:
        # 200 = No security definition found
        # 354 = Requested market data is not subscribed
        # 10167 = Delayed market data (not real-time)
        # 2104, 2106, 2158 = Data farm connection messages (informational)
        
        if errorCode in [2104, 2106, 2158]:
            logger.debug(f"IBKR Info [{errorCode}]: {errorString}")
        elif errorCode == 200:
            ticker = self.reqid_to_ticker.get(reqId, "UNKNOWN")
            logger.error(f"❌ No security definition for {ticker}: {errorString}")
        elif errorCode == 354:
            logger.error(f"❌ Market data not subscribed: {errorString}")
        else:
            logger.error(f"IBKR Error [{errorCode}] reqId={reqId}: {errorString}")
    
    # === MARKET DATA CALLBACKS ===
    
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
        
        logger.debug(f"📈 IBKR TICK {ticker}: ${price:.4f} (vol: {volume})")
        
        # Optional callback for immediate processing
        if self.price_callback:
            self.price_callback(ticker, price, volume, current_time)
    
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
```

#### Connection and Subscription Methods

```python
    # === CONNECTION METHODS ===
    
    def connect_and_run(self):
        """Connect to TWS/Gateway and start message processing thread"""
        logger.info(f"🔌 Connecting to IBKR at {self.host}:{self.port} (client_id={self.client_id})")
        
        try:
            self.connect(self.host, self.port, self.client_id)
            
            # Start message processing in separate thread
            self.api_thread = threading.Thread(target=self.run, daemon=True)
            self.api_thread.start()
            
            # Wait for connection confirmation (nextValidId callback)
            timeout = 10
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info(f"✅ IBKR connection established successfully")
                return True
            else:
                logger.error("❌ IBKR connection timeout - is TWS/Gateway running?")
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
    
    # === SUBSCRIPTION METHODS ===
    
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
        
        Returns True if subscription was successful/initiated
        """
        if not self.connected:
            logger.warning(f"Cannot subscribe to {ticker} - not connected")
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
            
            # Request streaming market data
            # genericTickList="" for basic price data
            # snapshot=False for streaming (not one-time)
            # regulatorySnapshot=False for real-time data
            self.reqMktData(req_id, contract, "", False, False, [])
            
            logger.info(f"📡 IBKR: Subscribed to {ticker} (reqId={req_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to {ticker}: {e}")
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
            del self.ticker_to_reqid[ticker]
            del self.reqid_to_ticker[req_id]
        
        self.active_subscriptions.discard(ticker)
        
        # Clean up price buffer for this ticker
        with self.price_buffer_lock:
            if ticker in self.price_buffer:
                del self.price_buffer[ticker]
        
        if ticker in self.last_volume:
            del self.last_volume[ticker]
    
    def update_subscriptions(self, needed_tickers: Set[str]):
        """
        Update subscriptions to match the needed tickers set
        - Subscribe to new tickers
        - Unsubscribe from removed tickers
        """
        current = self.active_subscriptions.copy()
        needed = set(needed_tickers)
        
        # Unsubscribe from tickers no longer needed
        to_remove = current - needed
        for ticker in to_remove:
            self.unsubscribe_ticker(ticker)
        
        # Subscribe to new tickers
        to_add = needed - current
        for ticker in to_add:
            self.subscribe_ticker(ticker)
        
        if to_add or to_remove:
            logger.info(f"📡 IBKR Subscriptions: +{len(to_add)} -{len(to_remove)} = {len(self.active_subscriptions)} active")
    
    # === BUFFER METHODS ===
    
    def get_and_clear_buffer(self) -> Dict[str, list]:
        """
        Get current price buffer contents and clear it
        Thread-safe operation
        """
        with self.price_buffer_lock:
            buffer_copy = dict(self.price_buffer)
            self.price_buffer.clear()
        return buffer_copy
    
    def get_buffer_for_tickers(self, tickers: Set[str]) -> Dict[str, list]:
        """
        Get buffer contents for specific tickers only and clear those entries
        Thread-safe operation
        """
        result = {}
        with self.price_buffer_lock:
            for ticker in tickers:
                if ticker in self.price_buffer:
                    result[ticker] = self.price_buffer.pop(ticker)
        return result
```

#### Module-Level Helper Function

```python
# At module level - for easy instantiation

_ibkr_client: Optional[IBKRClient] = None

def get_ibkr_client(host: str = None, port: int = None, client_id: int = None) -> IBKRClient:
    """
    Get or create the singleton IBKR client instance
    
    Environment variables used if args not provided:
    - IBKR_HOST (default: 127.0.0.1)
    - IBKR_PORT (default: 7497 for paper, 7496 for live)
    - IBKR_CLIENT_ID (default: 10)
    """
    global _ibkr_client
    
    if _ibkr_client is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        _host = host or os.getenv('IBKR_HOST', '127.0.0.1')
        _port = port or int(os.getenv('IBKR_PORT', '7497'))
        _client_id = client_id or int(os.getenv('IBKR_CLIENT_ID', '10'))
        
        _ibkr_client = IBKRClient(host=_host, port=_port, client_id=_client_id)
    
    return _ibkr_client
```

---

### 2. Modify `price_checker.py`

#### Changes Summary

| Section | Action |
|---------|--------|
| Imports | Add `from ibkr_client import IBKRClient, get_ibkr_client` |
| `__init__()` | Remove Polygon vars, add IBKR vars |
| `initialize()` | Replace WebSocket setup with IBKR connection |
| `setup_websocket_connection()` | **DELETE** |
| `authenticate_websocket()` | **DELETE** |
| `close_websocket()` | Replace with `close_ibkr()` |
| `reconnect_websocket()` | Replace with `reconnect_ibkr()` |
| `update_websocket_subscriptions()` | Replace with IBKR subscription logic |
| `handle_websocket_message()` | **DELETE** (callbacks handle this) |
| `process_single_websocket_message()` | **DELETE** |
| `websocket_listener()` | **DELETE** (IBKR thread handles this) |
| `process_websocket_prices()` | Rename to `process_ibkr_prices()`, get from IBKR buffer |
| `track_prices_rest_fallback()` | **DELETE** |
| `get_current_price_rest_fallback()` | **DELETE** |
| `get_price_with_double_call()` | **DELETE** |
| `continuous_polling_loop()` | Modify to use IBKR |
| `start()` | Modify startup sequence |
| `cleanup()` | Add IBKR cleanup |

#### Detailed Changes

##### `__init__()` - Replace Polygon with IBKR

```python
def __init__(self):
    self.ch_manager = None
    self.active_tickers: Set[str] = set()
    self.ticker_timestamps: Dict[str, datetime] = {}
    self.ready_event = asyncio.Event()
    
    # IBKR client (replaces Polygon WebSocket)
    self.ibkr_client: Optional[IBKRClient] = None
    self.ibkr_connected = False
    
    # Load IBKR configuration from environment
    self.ibkr_host = os.getenv('IBKR_HOST', '127.0.0.1')
    self.ibkr_port = int(os.getenv('IBKR_PORT', '7497'))
    self.ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', '10'))
    
    logger.info(f"IBKR Config: {self.ibkr_host}:{self.ibkr_port} (client_id={self.ibkr_client_id})")
    
    # Stats
    self.stats = {
        'tickers_monitored': 0,
        'price_checks': 0,
        'alerts_triggered': 0,
        'ibkr_ticks_received': 0,
        'ibkr_reconnections': 0,
        'start_time': time.time()
    }
```

##### `initialize()` - Connect to IBKR

```python
async def initialize(self):
    """Initialize the price monitoring system with IBKR connection"""
    try:
        # Initialize ClickHouse connection
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
        
        # Create essential tables
        await self.create_essential_tables()
        
        # Load active tickers from breaking_news
        self.active_tickers = await self.get_active_tickers_from_breaking_news()
        
        # Initialize IBKR connection
        logger.info("🔌 Initializing IBKR TWS API connection...")
        await self.setup_ibkr_connection()
        
        logger.info(f"✅ IBKR Price Monitor initialized - {len(self.active_tickers)} active tickers")
        
    except Exception as e:
        logger.error(f"Error initializing price monitor: {e}")
        raise
```

##### New `setup_ibkr_connection()` Method

```python
async def setup_ibkr_connection(self):
    """Setup connection to IBKR TWS/Gateway"""
    try:
        from ibkr_client import IBKRClient
        
        self.ibkr_client = IBKRClient(
            host=self.ibkr_host,
            port=self.ibkr_port,
            client_id=self.ibkr_client_id
        )
        
        # Connect (runs in separate thread)
        success = self.ibkr_client.connect_and_run()
        
        if success:
            self.ibkr_connected = True
            logger.info(f"✅ IBKR connection established on port {self.ibkr_port}")
            logger.info(f"   Mode: {'PAPER TRADING' if self.ibkr_port == 7497 else 'LIVE TRADING'}")
        else:
            raise ConnectionError("Failed to connect to IBKR TWS/Gateway")
            
    except Exception as e:
        logger.error(f"❌ IBKR connection failed: {e}")
        logger.error("   Ensure TWS or IB Gateway is running and API is enabled")
        raise
```

##### New `update_ibkr_subscriptions()` Method

```python
async def update_ibkr_subscriptions(self):
    """Update IBKR subscriptions based on active tickers"""
    if not self.ibkr_connected or not self.ibkr_client:
        return
    
    try:
        self.ibkr_client.update_subscriptions(self.active_tickers)
    except Exception as e:
        logger.error(f"Error updating IBKR subscriptions: {e}")
```

##### New `process_ibkr_prices()` Method (replaces `process_websocket_prices`)

```python
async def process_ibkr_prices(self):
    """Process buffered IBKR prices and insert to database"""
    if not self.ibkr_client:
        return
    
    try:
        start_time = time.time()
        
        # CRITICAL: Check 60-second window BEFORE inserting price data
        valid_tickers = await self.get_tickers_within_60_second_window()
        
        if not valid_tickers:
            logger.debug("⏰ No tickers within 60-second window")
            return
        
        # Get price data from IBKR buffer (only for valid tickers)
        price_buffer = self.ibkr_client.get_buffer_for_tickers(valid_tickers)
        
        if not price_buffer:
            return
        
        # Convert buffer to database format
        price_data = []
        processed_tickers = set()
        
        for ticker, prices in price_buffer.items():
            if not prices:
                continue
            
            # Use the most recent price for each ticker
            latest_price_info = prices[-1]
            
            price_data.append((
                latest_price_info['timestamp'],
                ticker,
                latest_price_info['price'],
                latest_price_info['volume'],
                latest_price_info['source']
            ))
            processed_tickers.add(ticker)
            self.stats['ibkr_ticks_received'] += len(prices)
        
        if price_data:
            # Get sentiment data for enrichment
            sentiment_data = await self._get_sentiment_data(processed_tickers)
            
            # Prepare enriched price data with sentiment
            enriched_price_data = []
            for price_row in price_data:
                timestamp, ticker, price, volume, source = price_row
                
                ticker_sentiment = sentiment_data.get(ticker, {
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low'
                })
                
                enriched_price_data.append((
                    timestamp,
                    ticker,
                    price,
                    volume,
                    source,
                    ticker_sentiment['sentiment'],
                    ticker_sentiment['recommendation'],
                    ticker_sentiment['confidence']
                ))
            
            # Batch insert enriched price data
            self.ch_manager.client.insert(
                'News.price_tracking',
                enriched_price_data,
                column_names=['timestamp', 'ticker', 'price', 'volume', 'source', 
                             'sentiment', 'recommendation', 'confidence']
            )
            
            total_time = time.time() - start_time
            self.stats['price_checks'] += len(enriched_price_data)
            
            logger.info(f"📊 IBKR: Processed {len(enriched_price_data)} price updates in {total_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Error processing IBKR prices: {e}")
```

##### Helper Method for Sentiment Data

```python
async def _get_sentiment_data(self, tickers: Set[str]) -> Dict[str, Dict]:
    """Get latest sentiment data for tickers"""
    sentiment_data = {}
    
    if not tickers:
        return sentiment_data
    
    try:
        ticker_list = list(tickers)
        ticker_placeholders = ','.join([f"'{ticker}'" for ticker in ticker_list])
        
        sentiment_query = f"""
        SELECT 
            ticker,
            sentiment,
            recommendation,
            confidence
        FROM (
            SELECT 
                ticker,
                sentiment,
                recommendation,
                confidence,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analyzed_at DESC) as rn
            FROM News.breaking_news
            WHERE ticker IN ({ticker_placeholders})
            AND analyzed_at >= now() - INTERVAL 1 HOUR
            AND sentiment != ''
            AND recommendation != ''
        ) ranked
        WHERE rn = 1
        """
        
        result = self.ch_manager.client.query(sentiment_query)
        for row in result.result_rows:
            ticker, sentiment, recommendation, confidence = row
            sentiment_data[ticker] = {
                'sentiment': sentiment,
                'recommendation': recommendation,
                'confidence': confidence
            }
    except Exception as e:
        logger.debug(f"Error getting sentiment data: {e}")
    
    return sentiment_data
```

##### Modified `continuous_polling_loop()`

```python
async def continuous_polling_loop(self):
    """Continuous polling loop - IBKR subscription management + database operations every 2 seconds"""
    logger.info("🔄 Starting IBKR POLLING LOOP - subscription management + database operations")
    
    cycle = 0
    last_cleanup = time.time()
    
    while True:
        try:
            cycle += 1
            cycle_start = time.time()
            
            # Clean up old tickers every 5 minutes
            if time.time() - last_cleanup > 300:
                await self.cleanup_old_tickers()
                last_cleanup = time.time()
            
            # Check IBKR connection health
            if not self.ibkr_connected or not self.ibkr_client or not self.ibkr_client.connected:
                logger.warning("❌ IBKR disconnected - attempting reconnect...")
                await self.reconnect_ibkr()
                await asyncio.sleep(2.0)
                continue
            
            if self.active_tickers:
                logger.debug(f"🔄 IBKR CYCLE {cycle}: Managing {len(self.active_tickers)} active tickers")
                
                # Update IBKR subscriptions
                await self.update_ibkr_subscriptions()
                
                # Process buffered prices from IBKR
                await self.process_ibkr_prices()
                
                # Check for alerts
                await self.check_price_alerts_optimized()
                
                cycle_time = time.time() - cycle_start
                logger.info(f"✅ IBKR CYCLE {cycle}: Completed in {cycle_time:.3f}s")
            else:
                logger.debug(f"⏳ IBKR CYCLE {cycle}: No active tickers")
            
            # Wait 2 seconds before next cycle
            await asyncio.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Error in IBKR polling loop: {e}")
            await asyncio.sleep(2.0)
```

##### New `reconnect_ibkr()` Method

```python
async def reconnect_ibkr(self):
    """Reconnect to IBKR with backoff"""
    self.stats['ibkr_reconnections'] += 1
    
    # Close existing connection if any
    if self.ibkr_client:
        try:
            self.ibkr_client.disconnect_safely()
        except:
            pass
    
    self.ibkr_connected = False
    
    # Wait before reconnecting
    await asyncio.sleep(5.0)
    
    try:
        await self.setup_ibkr_connection()
    except Exception as e:
        logger.error(f"IBKR reconnection failed: {e}")
```

##### Modified `start()` Method

```python
async def start(self):
    """Start the continuous price monitoring system with IBKR"""
    try:
        logger.info("🚀 Starting IBKR Price Monitor!")
        await self.initialize()
        
        # Verify IBKR connection
        if not self.ibkr_connected:
            raise ConnectionError("IBKR connection not established")
        
        logger.info(f"⚡ IBKR Mode: Port {self.ibkr_port} ({'PAPER' if self.ibkr_port == 7497 else 'LIVE'})")
        logger.info("✅ IBKR Price Monitor operational!")
        
        # Run file trigger monitor and polling loop in parallel
        # Note: No WebSocket listener needed - IBKR callbacks run in separate thread
        await asyncio.gather(
            self.file_trigger_monitor_async(),
            self.continuous_polling_loop()
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error in IBKR price monitor: {e}")
        raise
    finally:
        await self.cleanup()
```

##### Modified `cleanup()` Method

```python
async def cleanup(self):
    """Clean up resources"""
    if self.ibkr_client:
        try:
            self.ibkr_client.disconnect_safely()
        except:
            pass
    if self.ch_manager:
        self.ch_manager.close()
    logger.info("IBKR price monitor cleanup completed")
```

##### Modified `report_stats()` Method

```python
async def report_stats(self):
    """Report monitoring statistics"""
    runtime = time.time() - self.stats['start_time']
    
    logger.info(f"📊 IBKR MONITOR STATS:")
    logger.info(f"   Runtime: {runtime/60:.1f} minutes")
    logger.info(f"   Active Tickers: {len(self.active_tickers)}")
    logger.info(f"   Price Checks: {self.stats['price_checks']}")
    logger.info(f"   Alerts Triggered: {self.stats['alerts_triggered']}")
    logger.info(f"   IBKR Ticks Received: {self.stats['ibkr_ticks_received']}")
    logger.info(f"   IBKR Reconnections: {self.stats['ibkr_reconnections']}")
    logger.info(f"   Mode: {'PAPER' if self.ibkr_port == 7497 else 'LIVE'}")
```

---

### 3. Update `requirements.txt`

Add the following line:

```
# IBKR TWS API
ibapi>=10.30.1
```

---

### 4. Update `.env` File

Add the following environment variables:

```bash
# IBKR Configuration
# Host: localhost for local TWS/Gateway
IBKR_HOST=127.0.0.1

# Port: 7497 for paper trading, 7496 for live trading
# CHANGE THIS TO 7496 FOR LIVE TRADING
IBKR_PORT=7497

# Client ID: Must be unique across all connections to same TWS/Gateway
# tradehead uses client_id=1, so newshead uses 10
IBKR_CLIENT_ID=10
```

---

### 5. Methods to DELETE from `price_checker.py`

Remove these methods entirely (they are Polygon-specific):

- `setup_websocket_connection()`
- `authenticate_websocket()`
- `close_websocket()`
- `reconnect_websocket()`
- `update_websocket_subscriptions()`
- `handle_websocket_message()`
- `process_single_websocket_message()`
- `websocket_listener()`
- `process_websocket_prices()` (replaced by `process_ibkr_prices()`)
- `track_prices_rest_fallback()`
- `get_current_price_rest_fallback()`
- `get_price_with_double_call()`
- `test_api_connectivity()`

---

### 6. Instance Variables to DELETE from `__init__`

Remove these (Polygon-specific):

- `self.polygon_api_key`
- `self.session` (aiohttp session - no longer needed)
- `self.websocket`
- `self.websocket_authenticated`
- `self.websocket_price_buffer`
- `self.websocket_url`
- `self.websocket_subscriptions`
- `self.websocket_enabled`
- `self.websocket_reconnect_delay`
- `self.websocket_max_reconnect_delay`
- `self.use_websocket_data`
- `self.base_url`

---

### 7. Imports to Remove

Remove these imports (no longer needed):

```python
# Remove these
import aiohttp  # No longer using REST API
import websockets  # No longer using Polygon WebSocket
```

---

### 8. Imports to Add

```python
# Add these
from ibkr_client import IBKRClient, get_ibkr_client
import time  # If not already imported
```

---

## Testing

### Test 1: IBKR Connection Test

Create `testfiles/test_ibkr_connection.py`:

```python
#!/usr/bin/env python3
"""Test IBKR connection"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_client import IBKRClient
import time

def main():
    print("Testing IBKR connection...")
    
    client = IBKRClient(
        host="127.0.0.1",
        port=7497,  # Paper trading
        client_id=99  # Test client ID
    )
    
    success = client.connect_and_run()
    
    if success:
        print(f"✅ Connected! Next Order ID: {client.next_order_id}")
        
        # Test subscription
        print("\nTesting AAPL subscription...")
        client.subscribe_ticker("AAPL")
        
        # Wait for some ticks
        print("Waiting 10 seconds for price data...")
        time.sleep(10)
        
        # Check buffer
        buffer = client.get_and_clear_buffer()
        if "AAPL" in buffer:
            print(f"✅ Received {len(buffer['AAPL'])} price updates for AAPL")
            for tick in buffer['AAPL'][-3:]:  # Show last 3
                print(f"   ${tick['price']:.4f} at {tick['timestamp']}")
        else:
            print("❌ No price data received")
        
        client.disconnect_safely()
    else:
        print("❌ Connection failed")

if __name__ == "__main__":
    main()
```

Run with:
```bash
source venv/bin/activate
python testfiles/test_ibkr_connection.py
```

### Test 2: Full Integration Test

After implementation, test the full system:

```bash
# Terminal 1: Ensure TWS/Gateway is running

# Terminal 2: Start the system
source venv/bin/activate
python run_system.py --skip-list --socket

# Terminal 3: Watch for price data in ClickHouse
clickhouse-client --query "SELECT * FROM News.price_tracking WHERE source = 'ibkr_last_trade' ORDER BY timestamp DESC LIMIT 10"
```

---

## Environment Configuration

### Paper Trading (Testing)
```bash
IBKR_PORT=7497
```

### Live Trading (Production)
```bash
IBKR_PORT=7496
```

No other changes needed - the code automatically uses the port from the environment variable.

---

## Multiple Connection Notes

Your setup with two projects connecting simultaneously:

| Project | Client ID | Port | Purpose |
|---------|-----------|------|---------|
| tradehead | 1 | 7496/7497 | Trading/execution |
| newshead | 10 | 7496/7497 | Price monitoring |

Both can connect to the same TWS/Gateway instance simultaneously without conflicts as long as they use different `client_id` values.

---

## Files Summary

| File | Action |
|------|--------|
| `ibkr_client.py` | **CREATE** - New IBKR client module |
| `price_checker.py` | **MODIFY** - Replace Polygon with IBKR |
| `requirements.txt` | **MODIFY** - Add `ibapi>=10.30.1` |
| `.env` | **MODIFY** - Add IBKR configuration |
| `testfiles/test_ibkr_connection.py` | **CREATE** - Connection test script |

---

## Verification Checklist

After implementation, verify:

- [ ] `python -c "import ibapi; print(ibapi.__version__)"` works
- [ ] `testfiles/test_ibkr_connection.py` connects and receives AAPL data
- [ ] `price_checker.py` starts without errors
- [ ] Price data appears in ClickHouse with `source='ibkr_last_trade'`
- [ ] Alerts trigger correctly when conditions are met
- [ ] Switching from port 7497 to 7496 works via environment variable
- [ ] Both tradehead and newshead can connect simultaneously

---

## Rollback Plan

If IBKR implementation fails:

1. The original `price_checker.py` is in git history
2. Restore with: `git checkout HEAD~1 -- price_checker.py`
3. Remove `ibkr_client.py`
4. Remove `ibapi` from requirements.txt

---

**Document Created:** December 2024  
**Target:** Complete IBKR integration, remove Polygon dependency  
**Estimated Implementation Time:** 6-10 hours

