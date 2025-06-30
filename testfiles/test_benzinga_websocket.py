#!/usr/bin/env python3
"""
Benzinga WebSocket Feed Test
Simple test script to confirm responses from Benzinga's WebSocket streaming news feed.
Tests low latency press release detection for stock tickers.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import websockets
from dotenv import load_dotenv
import re

# Setup logging
def setup_logging():
    """Setup logging for the test script"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup logger
    logger = logging.getLogger('benzinga_test')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = log_dir / f"benzinga_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class BenzingaWebSocketTest:
    """Test class for Benzinga WebSocket feed"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.api_key = None
        self.websocket_url = "wss://api.benzinga.com/api/v1/news/stream"
        self.message_count = 0
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')  # Basic ticker pattern
        
    def load_config(self):
        """Load configuration from environment"""
        load_dotenv()
        self.api_key = os.getenv('BENZINGA_API_KEY')
        
        if not self.api_key:
            self.logger.error("BENZINGA_API_KEY not found in environment variables")
            self.logger.error("Please add BENZINGA_API_KEY to your .env file")
            return False
        
        self.logger.info(f"Loaded API key: {self.api_key[:8]}...")
        return True
    
    def extract_tickers(self, text):
        """Extract potential stock tickers from text"""
        if not text:
            return []
        
        # Find all potential tickers (1-5 capital letters)
        potential_tickers = self.ticker_pattern.findall(text)
        
        # Filter out common words that aren't tickers
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'UP', 'DO', 'NO', 'IF', 'SO', 'MY', 'HE', 'ME', 'WE', 'AN', 'IS', 'AT', 'ON', 'AS', 'OF', 'TO', 'IN', 'IT', 'WITH', 'THAT', 'HAVE', 'FROM', 'OR', 'BE', 'BEEN', 'HAS', 'THEIR', 'SAID', 'EACH', 'WHICH', 'WOULD', 'THERE', 'COULD', 'OTHER', 'AFTER', 'FIRST', 'WELL', 'WAY', 'ABOUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SOME', 'HER', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'HIM', 'TIME', 'TWO', 'MORE', 'VERY', 'WHAT', 'KNOW', 'JUST', 'FIRST', 'GET', 'OVER', 'THINK', 'ALSO', 'YOUR', 'WORK', 'LIFE', 'ONLY', 'NEW', 'YEARS', 'YEAR', 'COME', 'ITS', 'PEOPLE', 'TAKE', 'GOOD', 'SEE', 'HOW', 'NOW', 'THAN', 'LOOK', 'WANT', 'GIVE', 'USE', 'FIND', 'TELL', 'ASK', 'TURN', 'END', 'WHY', 'TRY', 'BACK', 'CALL', 'CAME', 'EACH', 'PART', 'MADE', 'GREAT', 'WHERE', 'MUCH', 'STILL', 'HERE', 'OLD', 'EVERY', 'WENT', 'PLACE', 'RIGHT', 'WENT', 'NEVER', 'BEFORE', 'MUST', 'THROUGH', 'LONG', 'SOMETHING', 'BOTH', 'IMPORTANT', 'CHILDREN', 'EXAMPLE', 'BEGAN', 'THOSE', 'SCHOOL', 'HOUSE', 'NEVER', 'STARTED', 'CITY', 'EARTH', 'EYES', 'LIGHT', 'THOUGHT', 'HEAD', 'UNDER', 'STORY', 'SAW', 'LEFT', 'DONT', 'FEW', 'WHILE', 'ALONG', 'MIGHT', 'CLOSE', 'NIGHT', 'REAL', 'LIFE', 'ALMOST', 'MOVE', 'THING', 'LIVE', 'MR', 'SINCE', 'GETTING', 'ROOM', 'MADE', 'YOUNG', 'WATER', 'BOOK', 'TOOK', 'BUSINESS', 'OPEN', 'PROBLEM', 'COMPLETE', 'THOUGH', 'INFORMATION', 'NOTHING', 'EVERYTHING', 'COMMUNITY', 'BACK', 'PARENT', 'FACE', 'OTHERS', 'LEVEL', 'OFFICE', 'DOOR', 'HEALTH', 'PERSON', 'ART', 'SURE', 'SUCH', 'WAR', 'HISTORY', 'PARTY', 'WITHIN', 'GROWING', 'RESULT', 'MORNING', 'WALKING', 'PAPER', 'GROUP', 'IMPORTANT', 'MUSIC', 'THOSE', 'BOTH', 'MOST', 'OFTEN', 'UNTIL', 'BEGAN', 'STUDY', 'FOOD', 'KEEP', 'CHILDREN', 'FEET', 'LAND', 'SIDE', 'WITHOUT', 'BOY', 'ONCE', 'ANIMAL', 'LATER', 'ABOVE', 'PLANT', 'LAST', 'SCHOOL', 'FATHER', 'KEEP', 'TREE', 'NEVER', 'START', 'CITY', 'EARTH', 'EYES', 'LIGHT', 'THOUGHT', 'HEAD', 'UNDER', 'STORY', 'SAW', 'LEFT', 'DONT', 'FEW', 'WHILE', 'ALONG', 'MIGHT', 'CLOSE', 'NIGHT', 'REAL', 'LIFE', 'ALMOST', 'MOVE', 'THING', 'LIVE', 'MR', 'SINCE', 'GETTING', 'ROOM', 'MADE', 'YOUNG', 'WATER', 'BOOK', 'TOOK', 'BUSINESS', 'OPEN', 'PROBLEM', 'COMPLETE', 'THOUGH', 'INFORMATION', 'NOTHING', 'EVERYTHING', 'COMMUNITY', 'BACK', 'PARENT', 'FACE', 'OTHERS', 'LEVEL', 'OFFICE', 'DOOR', 'HEALTH', 'PERSON', 'ART', 'SURE', 'SUCH', 'WAR', 'HISTORY', 'PARTY', 'WITHIN', 'GROWING', 'RESULT', 'MORNING', 'WALKING', 'PAPER', 'GROUP', 'IMPORTANT', 'MUSIC', 'THOSE', 'BOTH', 'MOST', 'OFTEN', 'UNTIL', 'BEGAN', 'STUDY', 'FOOD', 'KEEP', 'CHILDREN', 'FEET', 'LAND', 'SIDE', 'WITHOUT', 'BOY', 'ONCE', 'ANIMAL', 'LATER', 'ABOVE', 'PLANT', 'LAST', 'SCHOOL', 'FATHER', 'KEEP', 'TREE', 'NEVER', 'START'}
        
        filtered_tickers = [ticker for ticker in potential_tickers if ticker not in common_words and len(ticker) >= 2]
        return list(set(filtered_tickers))  # Remove duplicates
    
    def analyze_message(self, message_data):
        """Analyze incoming message for relevant stock information"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'message_type': message_data.get('kind', 'unknown'),
                'api_version': message_data.get('api_version', 'unknown'),
                'tickers_found': [],
                'title': None,
                'body_preview': None,
                'channels': [],
                'securities': [],
                'url': None,
                'created_at': None,
                'action': None
            }
            
            # Extract data content
            data = message_data.get('data', {})
            content = data.get('content', {})
            
            if content:
                analysis['title'] = content.get('title', '')
                analysis['body_preview'] = content.get('body', '')[:200] if content.get('body') else None
                analysis['channels'] = content.get('channels', [])
                analysis['securities'] = content.get('securities', [])
                analysis['url'] = content.get('url', '')
                analysis['created_at'] = content.get('created_at', '')
                analysis['action'] = data.get('action', '')
                
                # Extract tickers from title and body
                title_text = analysis['title'] or ''
                body_text = analysis['body_preview'] or ''
                combined_text = f"{title_text} {body_text}"
                
                analysis['tickers_found'] = self.extract_tickers(combined_text)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing message: {e}")
            return None
    
    def log_message_summary(self, analysis):
        """Log a summary of the received message"""
        if not analysis:
            return
        
        self.message_count += 1
        
        self.logger.info(f"Message #{self.message_count} - {analysis['action']}")
        self.logger.info(f"  API Version: {analysis['api_version']}")
        self.logger.info(f"  Message Type: {analysis['message_type']}")
        self.logger.info(f"  Created At: {analysis['created_at']}")
        
        if analysis['title']:
            self.logger.info(f"  Title: {analysis['title']}")
        
        if analysis['tickers_found']:
            self.logger.info(f"  Tickers Found: {', '.join(analysis['tickers_found'])}")
        
        if analysis['channels']:
            self.logger.info(f"  Channels: {', '.join(analysis['channels'])}")
        
        if analysis['securities']:
            self.logger.info(f"  Securities: {analysis['securities']}")
        
        if analysis['url']:
            self.logger.info(f"  URL: {analysis['url']}")
        
        if analysis['body_preview']:
            self.logger.info(f"  Body Preview: {analysis['body_preview']}")
        
        self.logger.info("-" * 80)
    
    async def connect_and_listen(self, duration_minutes=5):
        """Connect to Benzinga WebSocket and listen for messages"""
        self.logger.info(f"Starting Benzinga WebSocket test for {duration_minutes} minutes")
        self.logger.info(f"Connecting to: {self.websocket_url}")
        
        # Try different authentication methods
        auth_methods = [
            # Method 1: Authorization header (most common)
            {
                "url": self.websocket_url,
                "headers": {"Authorization": f"Key {self.api_key}"},
                "name": "Authorization header"
            },
            # Method 2: Token as query parameter
            {
                "url": f"{self.websocket_url}?token={self.api_key}",
                "headers": None,
                "name": "Query parameter"
            },
            # Method 3: API key as query parameter
            {
                "url": f"{self.websocket_url}?apikey={self.api_key}",
                "headers": None,
                "name": "API key parameter"
            }
        ]
        
        for method in auth_methods:
            try:
                self.logger.info(f"Attempting to connect using {method['name']}...")
                
                # Prepare connection arguments
                connect_args = {
                    "ping_interval": 30,
                    "ping_timeout": 10
                }
                
                # Add headers if specified (for newer websockets versions)
                if method['headers']:
                    try:
                        connect_args["additional_headers"] = method['headers']
                    except:
                        # If additional_headers doesn't work, try without it
                        self.logger.info("Headers not supported, trying without authentication headers...")
                
                async with websockets.connect(method["url"], **connect_args) as websocket:
                    
                    self.logger.info(f"Successfully connected to Benzinga WebSocket using {method['name']}!")
                    self.logger.info("Listening for messages...")
                    
                    # Set timeout for the test
                    end_time = asyncio.get_event_loop().time() + (duration_minutes * 60)
                    
                    while asyncio.get_event_loop().time() < end_time:
                        try:
                            # Wait for message with timeout
                            message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                            
                            # Parse JSON message
                            try:
                                message_data = json.loads(message)
                                
                                # Analyze the message
                                analysis = self.analyze_message(message_data)
                                
                                # Log summary
                                self.log_message_summary(analysis)
                                
                                # Also log raw message for debugging (first few messages only)
                                if self.message_count <= 3:
                                    self.logger.info("Raw message data:")
                                    self.logger.info(json.dumps(message_data, indent=2))
                                    self.logger.info("=" * 80)
                            
                            except json.JSONDecodeError as e:
                                self.logger.error(f"Failed to parse JSON message: {e}")
                                self.logger.error(f"Raw message: {message}")
                        
                        except asyncio.TimeoutError:
                            self.logger.info("No message received in 30 seconds, continuing to listen...")
                            continue
                        
                        except websockets.exceptions.ConnectionClosed as e:
                            self.logger.error(f"WebSocket connection closed: {e}")
                            break
                    
                    self.logger.info(f"Test completed. Total messages received: {self.message_count}")
                    return  # Success, exit the method
            
            except websockets.exceptions.InvalidHandshake as e:
                self.logger.warning(f"Authentication method '{method['name']}' failed: {e}")
                continue  # Try next method
            
            except websockets.exceptions.WebSocketException as e:
                self.logger.warning(f"WebSocket error with '{method['name']}': {e}")
                continue  # Try next method
            
            except Exception as e:
                self.logger.warning(f"Unexpected error with '{method['name']}': {e}")
                continue  # Try next method
        
        # If we get here, all methods failed
        self.logger.error("All authentication methods failed!")
        self.logger.error("Please check:")
        self.logger.error("1. Your BENZINGA_API_KEY is correct")
        self.logger.error("2. Your API key has WebSocket streaming permissions")
        self.logger.error("3. The WebSocket URL is correct")
        self.logger.error("4. Your internet connection is working")
    
    async def run_test(self, duration_minutes=5):
        """Run the complete test"""
        self.logger.info("="*80)
        self.logger.info("BENZINGA WEBSOCKET FEED TEST")
        self.logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)
        
        # Load configuration
        if not self.load_config():
            return
        
        try:
            # Connect and listen
            await self.connect_and_listen(duration_minutes)
        
        except KeyboardInterrupt:
            self.logger.info("Test interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Test failed with error: {e}")
        
        finally:
            self.logger.info("="*80)
            self.logger.info("BENZINGA WEBSOCKET TEST COMPLETED")
            self.logger.info(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total messages processed: {self.message_count}")
            self.logger.info("="*80)

async def main():
    """Main function to run the test"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Benzinga WebSocket feed')
    parser.add_argument('--duration', type=int, default=5, 
                       help='Test duration in minutes (default: 5)')
    
    args = parser.parse_args()
    
    # Create and run test
    test = BenzingaWebSocketTest()
    await test.run_test(duration_minutes=args.duration)

if __name__ == "__main__":
    # Add websockets to requirements if not already present
    try:
        import websockets
    except ImportError:
        print("Error: websockets library not found")
        print("Please install it with: pip install websockets")
        sys.exit(1)
    
    # Run the test
    asyncio.run(main()) 