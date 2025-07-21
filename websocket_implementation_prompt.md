# WebSocket Price Checker Implementation - Hybrid Approach

## Overview
Replace the unreliable Polygon REST API calls in `price_checker.py` with real-time WebSocket streaming while maintaining the existing proven architecture. The current system suffers from garbage data on initial API calls, causing missed price alerts (e.g., PMN ticker showing incorrect $1.64 baseline).

## Current Architecture (Keep Unchanged)
- **Process Isolation**: Price checker runs as separate subprocess
- **File Triggers**: News scraper creates `triggers/immediate_*.json` files  
- **Active Ticker Management**: `self.active_tickers` set tracks monitored symbols
- **Database Operations**: ClickHouse batch inserts every 2 seconds
- **Alert Logic**: 5% price movement detection with sentiment analysis

## Implementation Requirements

### 1. Core WebSocket Integration

**Target File**: `price_checker.py`
**Class**: `ContinuousPriceMonitor`

#### Add WebSocket Components:
```python
import websockets
import json

class ContinuousPriceMonitor:
    def __init__(self):
        # ... existing code ...
        
        # NEW: WebSocket components
        self.websocket = None
        self.websocket_authenticated = False
        self.price_buffer = {}  # Buffer prices between database writes
        self.websocket_url = "wss://socket.polygon.io/stocks"  # Real-time
        self.websocket_subscriptions = set()  # Track current subscriptions
```

#### WebSocket Connection Management:
- **Connection**: Establish WebSocket connection during `initialize()`
- **Authentication**: Use existing `self.polygon_api_key`
- **Reconnection**: Auto-reconnect on connection drops
- **Fallback**: Fall back to REST API if WebSocket fails

### 2. Dynamic Subscription Management

#### Key Requirement:
- **Subscribe** to new tickers when added to `self.active_tickers`
- **Unsubscribe** from old tickers when removed (30-minute cleanup)
- **Subscription Types**: 
  - `T.{ticker}` (Trades) - Primary price source
  - `Q.{ticker}` (Quotes/NBBO) - Fallback price source

#### Implementation Points:
- Monitor `self.active_tickers` changes in `continuous_polling_loop()`
- Send subscription/unsubscription messages as needed
- Handle subscription confirmations and errors

### 3. Real-Time Price Processing

#### Replace `track_prices_parallel()`:
- **OLD**: Make parallel REST API calls every 2 seconds
- **NEW**: Process real-time WebSocket messages as they arrive
- **Buffer**: Store prices in `self.price_buffer` until database write

#### Price Data Handling:
- **Trade Messages** (`ev: "T"`): Use `p` field for price, `s` for size
- **Quote Messages** (`ev: "Q"`): Calculate mid-price from bid/ask
- **Validation**: Apply same price validation logic as current system
- **Timestamps**: Use WebSocket timestamps, add `received_at` field

### 4. Database Integration (Unchanged)

#### Keep Existing:
- **Batch Inserts**: Every 2 seconds to `News.price_tracking` table
- **Schema**: Same columns (timestamp, ticker, price, volume, source, sentiment, recommendation, confidence)
- **Alert Logic**: No changes to `check_price_alerts_optimized()`
- **Sentiment Integration**: Same sentiment data enrichment

#### New Source Fields:
- `source: 'websocket_trade'` for trade data
- `source: 'websocket_quote'` for quote data
- `source: 'rest_fallback'` when REST API is used

### 5. Error Handling & Resilience

#### WebSocket Error Scenarios:
- **Connection Loss**: Auto-reconnect with exponential backoff
- **Authentication Failure**: Log error, attempt reconnection
- **Subscription Errors**: Fall back to REST API for affected tickers
- **Message Parsing Errors**: Log and skip malformed messages

#### Fallback Strategy:
- **Graceful Degradation**: If WebSocket fails, use existing REST API logic
- **Hybrid Mode**: Use WebSocket for some tickers, REST for others if needed
- **Status Logging**: Clear indicators of data source being used

### 6. Performance Optimization

#### Memory Management:
- **Price Buffer Size**: Limit buffer to prevent memory leaks
- **Old Data Cleanup**: Clear buffered prices after database insert
- **Connection Pooling**: Reuse WebSocket connection efficiently

#### Latency Optimization:
- **Immediate Processing**: Process WebSocket messages as they arrive
- **Async Message Handling**: Don't block other operations
- **Batch Database Writes**: Maintain 2-second batch cycle for performance

## Specific Code Changes Required

### 1. Modify `initialize()` Method
```python
async def initialize(self):
    # ... existing ClickHouse setup ...
    
    # NEW: Initialize WebSocket connection
    await self.setup_websocket_connection()
    
    # ... existing aiohttp session setup (keep for fallback) ...
```

### 2. Replace `track_prices_parallel()`
```python
async def process_websocket_prices(self):
    """Process buffered WebSocket prices and insert to database"""
    if not self.price_buffer:
        return
    
    # Convert buffer to database format
    price_data = []
    for ticker, prices in self.price_buffer.items():
        for price_info in prices:
            # Format for database insertion
            # ... existing sentiment enrichment logic ...
    
    # Batch insert (same as current system)
    # Clear buffer after insert
    self.price_buffer.clear()
```

### 3. Add WebSocket Message Handler
```python
async def handle_websocket_message(self, message):
    """Process incoming WebSocket messages"""
    try:
        data = json.loads(message)
        
        # Handle arrays and single messages
        if isinstance(data, list):
            for msg in data:
                await self.process_single_websocket_message(msg)
        else:
            await self.process_single_websocket_message(data)
            
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
```

### 4. Update `continuous_polling_loop()`
```python
async def continuous_polling_loop(self):
    # ... existing setup ...
    
    while True:
        # ... existing ticker cleanup ...
        
        if self.active_tickers:
            # NEW: Update WebSocket subscriptions
            await self.update_websocket_subscriptions()
            
            # NEW: Process buffered WebSocket prices
            await self.process_websocket_prices()
            
            # UNCHANGED: Check for alerts
            await self.check_price_alerts_optimized()
        
        await asyncio.sleep(2.0)
```

## Implementation Guidelines

### Code Quality Rules
- **File-by-file changes**: Make changes to `price_checker.py` only
- **No whitespace changes**: Focus on functional modifications
- **Preserve existing logic**: Keep all alert logic, database schema, and file trigger system
- **Descriptive variables**: Use clear names like `websocket_price_buffer` not `buf`
- **Error handling**: Robust exception handling for all WebSocket operations
- **Logging**: Comprehensive logging with clear indicators of data source

### Testing Strategy
1. **Standalone WebSocket Test**: Use existing `test_polygon_websocket.py`
2. **Hybrid Mode Testing**: Test with both WebSocket and REST fallback
3. **Price Accuracy**: Verify no more garbage data issues
4. **Performance**: Ensure 2-second database cycle maintained
5. **Resilience**: Test connection drops and recovery

### Success Criteria
- ✅ **No Garbage Data**: Eliminate incorrect initial prices (e.g., PMN $1.64 issue)
- ✅ **Real-time Updates**: Prices detected within milliseconds, not seconds
- ✅ **Reliable Alerts**: 5% price movements trigger correctly
- ✅ **Process Isolation**: Maintain separate subprocess architecture
- ✅ **Graceful Fallback**: System continues working if WebSocket fails
- ✅ **Performance**: Database operations remain efficient (2-second batches)

## Integration Points

### File Trigger System (Unchanged)
- **Trigger Files**: Continue monitoring `triggers/immediate_*.json`
- **Active Tickers**: WebSocket subscribes to tickers in `self.active_tickers`
- **Cleanup**: WebSocket unsubscribes when tickers removed

### Database Schema (Unchanged)
- **Table**: `News.price_tracking`
- **Columns**: timestamp, ticker, price, volume, source, sentiment, recommendation, confidence
- **Alert Table**: `News.news_alert` (no changes)

### Process Communication (Unchanged)
- **Subprocess**: Price checker still runs as separate process
- **Communication**: File-based triggers from news scraper
- **Isolation**: No shared memory or direct process communication

## Expected Outcome

### Before (REST API):
```
News detected → File trigger → Wait up to 2 seconds → REST API call → 
Potential garbage data → Price stored → Alert check
Total latency: 2-4 seconds + garbage data risk
```

### After (WebSocket Hybrid):
```
News detected → File trigger → WebSocket subscription → 
Real-time price stream → Buffer prices → Batch database write → Alert check
Total latency: <500ms + clean, reliable data
```

The hybrid approach eliminates the garbage data problem while maintaining the proven architecture and improving response time from 2-4 seconds to under 500ms. 