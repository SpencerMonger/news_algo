#!/usr/bin/env python3
"""
Standalone test script for Polygon WebSocket API
Tests real-time price data streaming to replace unreliable REST API calls
"""

import asyncio
import websockets
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class PolygonWebSocketTester:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY', '')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
        
        # WebSocket URLs from Polygon documentation
        self.delayed_url = "wss://delayed.polygon.io/stocks"  # 15-minute delayed
        self.realtime_url = "wss://socket.polygon.io/stocks"  # Real-time (requires subscription)
        
        # Use real-time by default, fallback to delayed if needed
        self.websocket_url = self.realtime_url
        
        self.websocket = None
        self.authenticated = False
        self.subscribed_tickers = set()
        
        # Track received prices for analysis
        self.price_data = {}
        
    async def connect(self):
        """Connect to Polygon WebSocket"""
        try:
            logger.info(f"🔌 Connecting to Polygon WebSocket: {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("✅ WebSocket connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket: {e}")
            # Try delayed feed as fallback
            if self.websocket_url == self.realtime_url:
                logger.info("🔄 Trying delayed feed as fallback...")
                self.websocket_url = self.delayed_url
                try:
                    self.websocket = await websockets.connect(self.websocket_url)
                    logger.info("✅ Connected to delayed WebSocket feed")
                    return True
                except Exception as e2:
                    logger.error(f"❌ Failed to connect to delayed feed: {e2}")
            return False
    
    async def authenticate(self):
        """Authenticate with API key"""
        if not self.websocket:
            logger.error("❌ No WebSocket connection available for authentication")
            return False
        
        try:
            # Send authentication message
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            
            logger.info("🔐 Sending authentication...")
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for authentication response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            auth_response = json.loads(response)
            
            logger.info(f"📨 Auth response: {auth_response}")
            
            # Handle both single message and array of messages
            if isinstance(auth_response, list):
                # Check if any message in the array indicates successful auth
                for msg in auth_response:
                    if msg.get("status") == "auth_success":
                        self.authenticated = True
                        logger.info("✅ Authentication successful")
                        return True
                    elif msg.get("status") == "connected":
                        logger.info("📡 Connected to WebSocket, waiting for auth confirmation...")
                        # Continue to check for auth_success in next message
                        continue
                
                # If we get here, check if we got a "connected" status and need to wait for auth_success
                if any(msg.get("status") == "connected" for msg in auth_response):
                    # Wait for the actual auth success message
                    try:
                        auth_confirm = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                        auth_confirm_data = json.loads(auth_confirm)
                        logger.info(f"📨 Auth confirmation: {auth_confirm_data}")
                        
                        if isinstance(auth_confirm_data, list):
                            for msg in auth_confirm_data:
                                if msg.get("status") == "auth_success":
                                    self.authenticated = True
                                    logger.info("✅ Authentication successful")
                                    return True
                        else:
                            if auth_confirm_data.get("status") == "auth_success":
                                self.authenticated = True
                                logger.info("✅ Authentication successful")
                                return True
                    except asyncio.TimeoutError:
                        logger.error("❌ Timeout waiting for auth confirmation")
                        return False
                
                logger.error(f"❌ Authentication failed: {auth_response}")
                return False
            else:
                # Handle single message response
                if auth_response.get("status") == "auth_success":
                    self.authenticated = True
                    logger.info("✅ Authentication successful")
                    return True
                else:
                    logger.error(f"❌ Authentication failed: {auth_response}")
                    return False
                
        except asyncio.TimeoutError:
            logger.error("❌ Authentication timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    async def subscribe_to_ticker(self, ticker: str):
        """Subscribe to real-time data for a ticker"""
        if not self.authenticated:
            logger.error("❌ Not authenticated - cannot subscribe")
            return False
        
        try:
            # Subscribe to multiple data types for comprehensive coverage
            subscriptions = [
                f"T.{ticker}",   # Trades
                f"Q.{ticker}",   # Quotes (NBBO)
                f"AM.{ticker}",  # Aggregate Per Minute
            ]
            
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(subscriptions)
            }
            
            logger.info(f"📡 Subscribing to {ticker}: {subscriptions}")
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Wait for subscription confirmation
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            sub_response = json.loads(response)
            
            logger.info(f"📨 Subscription response: {sub_response}")
            
            # Handle both single response and list of responses
            if isinstance(sub_response, list):
                # Check if any message in the list indicates successful subscription
                success_count = 0
                for msg in sub_response:
                    if msg.get("status") == "success":
                        success_count += 1
                
                if success_count > 0:
                    self.subscribed_tickers.add(ticker)
                    self.price_data[ticker] = {
                        'trades': [],
                        'quotes': [],
                        'aggregates': []
                    }
                    logger.info(f"✅ Successfully subscribed to {ticker} ({success_count} subscriptions)")
                    return True
                else:
                    logger.error(f"❌ Subscription failed for {ticker}: {sub_response}")
                    return False
            else:
                # Handle single message response
                if sub_response.get("status") == "success":
                    self.subscribed_tickers.add(ticker)
                    self.price_data[ticker] = {
                        'trades': [],
                        'quotes': [],
                        'aggregates': []
                    }
                    logger.info(f"✅ Successfully subscribed to {ticker}")
                    return True
                else:
                    logger.error(f"❌ Subscription failed for {ticker}: {sub_response}")
                    return False
                
        except asyncio.TimeoutError:
            logger.error(f"❌ Subscription timeout for {ticker}")
            return False
        except Exception as e:
            logger.error(f"❌ Subscription error for {ticker}: {e}")
            return False
    
    def process_message(self, message_data):
        """Process incoming WebSocket messages"""
        try:
            # Handle both single messages and arrays of messages
            if isinstance(message_data, list):
                for msg in message_data:
                    self.process_single_message(msg)
            else:
                self.process_single_message(message_data)
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def process_single_message(self, msg):
        """Process a single message from the WebSocket"""
        try:
            event_type = msg.get("ev")
            symbol = msg.get("sym", msg.get("S", "UNKNOWN"))
            timestamp = msg.get("t", msg.get("s", 0))
            
            if event_type == "T":  # Trade
                price = msg.get("p", 0.0)
                size = msg.get("s", 0)
                
                if symbol in self.price_data:
                    self.price_data[symbol]['trades'].append({
                        'price': price,
                        'size': size,
                        'timestamp': timestamp,
                        'received_at': datetime.now()
                    })
                
                logger.info(f"📈 TRADE {symbol}: ${price:.4f} (size: {size}) at {datetime.fromtimestamp(timestamp/1000)}")
                
            elif event_type == "Q":  # Quote (NBBO)
                bid_price = msg.get("bp", msg.get("P", 0.0))
                ask_price = msg.get("ap", msg.get("p", 0.0))
                bid_size = msg.get("bs", msg.get("S", 0))
                ask_size = msg.get("as", msg.get("s", 0))
                
                mid_price = (bid_price + ask_price) / 2.0 if bid_price > 0 and ask_price > 0 else 0.0
                
                if symbol in self.price_data:
                    self.price_data[symbol]['quotes'].append({
                        'bid_price': bid_price,
                        'ask_price': ask_price,
                        'mid_price': mid_price,
                        'bid_size': bid_size,
                        'ask_size': ask_size,
                        'timestamp': timestamp,
                        'received_at': datetime.now()
                    })
                
                logger.info(f"💬 QUOTE {symbol}: Bid=${bid_price:.4f}({bid_size}) Ask=${ask_price:.4f}({ask_size}) Mid=${mid_price:.4f}")
                
            elif event_type == "AM":  # Aggregate Per Minute
                open_price = msg.get("o", 0.0)
                close_price = msg.get("c", 0.0)
                high_price = msg.get("h", 0.0)
                low_price = msg.get("l", 0.0)
                volume = msg.get("v", 0)
                vwap = msg.get("a", 0.0)
                
                if symbol in self.price_data:
                    self.price_data[symbol]['aggregates'].append({
                        'open': open_price,
                        'close': close_price,
                        'high': high_price,
                        'low': low_price,
                        'volume': volume,
                        'vwap': vwap,
                        'timestamp': timestamp,
                        'received_at': datetime.now()
                    })
                
                logger.info(f"📊 AGGREGATE {symbol}: O=${open_price:.4f} H=${high_price:.4f} L=${low_price:.4f} C=${close_price:.4f} V={volume}")
                
            elif event_type == "status":
                logger.info(f"📡 Status message: {msg}")
                
            else:
                logger.debug(f"🔍 Unknown event type '{event_type}': {msg}")
                
        except Exception as e:
            logger.error(f"❌ Error processing single message: {e}")
            logger.debug(f"Message was: {msg}")
    
    async def listen_for_data(self, duration_seconds=60):
        """Listen for incoming data for specified duration"""
        if not self.websocket:
            logger.error("❌ No WebSocket connection available")
            return
        
        logger.info(f"👂 Listening for data for {duration_seconds} seconds...")
        start_time = datetime.now()
        message_count = 0
        
        try:
            while True:
                # Check if duration has elapsed
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration_seconds:
                    logger.info(f"⏰ Listening duration ({duration_seconds}s) completed")
                    break
                
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    message_count += 1
                    
                    # Parse and process message
                    try:
                        data = json.loads(message)
                        self.process_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON message: {e}")
                        logger.debug(f"Raw message: {message}")
                        
                except asyncio.TimeoutError:
                    # No message received in 5 seconds - continue listening
                    logger.debug("⏱️ No message received in last 5 seconds")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logger.error("❌ WebSocket connection closed unexpectedly")
        except Exception as e:
            logger.error(f"❌ Error during data listening: {e}")
        
        logger.info(f"📊 Received {message_count} messages in {elapsed:.1f} seconds")
    
    def print_summary(self):
        """Print summary of received data"""
        logger.info("📋 DATA SUMMARY:")
        logger.info("=" * 60)
        
        for ticker, data in self.price_data.items():
            trades_count = len(data['trades'])
            quotes_count = len(data['quotes'])
            agg_count = len(data['aggregates'])
            
            logger.info(f"🎯 {ticker}:")
            logger.info(f"   Trades: {trades_count}")
            logger.info(f"   Quotes: {quotes_count}")
            logger.info(f"   Aggregates: {agg_count}")
            
            # Show latest prices if available
            if data['trades']:
                latest_trade = data['trades'][-1]
                logger.info(f"   Latest Trade: ${latest_trade['price']:.4f} at {latest_trade['received_at']}")
            
            if data['quotes']:
                latest_quote = data['quotes'][-1]
                logger.info(f"   Latest Quote: Mid=${latest_quote['mid_price']:.4f} at {latest_quote['received_at']}")
            
            if data['aggregates']:
                latest_agg = data['aggregates'][-1]
                logger.info(f"   Latest Aggregate: Close=${latest_agg['close']:.4f} at {latest_agg['received_at']}")
            
            logger.info("")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 WebSocket connection closed")

async def test_websocket(tickers=None, duration=60):
    """Test WebSocket functionality with specified tickers"""
    if tickers is None:
        tickers = ["PKI", "AAPL", "MSFT"]  # Default test tickers - PKI first to test the theory
    
    logger.info("🚀 Starting Polygon WebSocket Test")
    logger.info(f"🎯 Testing tickers: {tickers}")
    logger.info(f"⏰ Duration: {duration} seconds")
    logger.info("=" * 60)
    
    tester = PolygonWebSocketTester()
    
    try:
        # Connect to WebSocket
        if not await tester.connect():
            logger.error("❌ Failed to establish WebSocket connection")
            return
        
        # Authenticate
        if not await tester.authenticate():
            logger.error("❌ Failed to authenticate")
            return
        
        # Subscribe to tickers
        for ticker in tickers:
            success = await tester.subscribe_to_ticker(ticker)
            if not success:
                logger.warning(f"⚠️ Failed to subscribe to {ticker}")
            await asyncio.sleep(0.5)  # Small delay between subscriptions
        
        # Listen for data
        await tester.listen_for_data(duration)
        
        # Print summary
        tester.print_summary()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
    finally:
        await tester.close()

async def main():
    """Main function"""
    import sys
    
    # Parse command line arguments
    tickers = ["PKI"]  # Default to PKI for testing the theory
    duration = 120  # Default 120 seconds to give more time for PKI activity
    
    if len(sys.argv) > 1:
        # First argument: comma-separated tickers
        tickers = [t.strip().upper() for t in sys.argv[1].split(',')]
    
    if len(sys.argv) > 2:
        # Second argument: duration in seconds
        try:
            duration = int(sys.argv[2])
        except ValueError:
            logger.warning(f"Invalid duration '{sys.argv[2]}', using default 120 seconds")
    
    await test_websocket(tickers, duration)

if __name__ == "__main__":
    asyncio.run(main()) 