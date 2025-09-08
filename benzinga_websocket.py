#!/usr/bin/env python3
"""
Benzinga WebSocket News Feed Scraper
Real-time news monitoring using Benzinga's WebSocket streaming API
Replaces web scraping with low-latency WebSocket feed for faster ticker detection
"""

import asyncio
import json
import logging
import os
import re
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
import websockets
from dotenv import load_dotenv
from clickhouse_setup import ClickHouseManager
from bs4 import BeautifulSoup

# SENTIMENT ANALYSIS INTEGRATION
from sentiment_service import analyze_articles_with_sentiment_and_immediate_insertion

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenzingaWebSocketScraper:
    """Benzinga WebSocket scraper that matches the web_scraper.py interface"""
    
    def __init__(self, enable_old=False, process_any_ticker=False):
        self.clickhouse_manager = None
        self.websocket = None
        self.batch_queue = []
        self.ticker_list = []
        self.enable_old = enable_old  # Flag to disable freshness filtering for testing
        self.process_any_ticker = process_any_ticker  # Flag to process any ticker found
        self.api_key = None
        self.websocket_url = "wss://api.benzinga.com/api/v1/news/stream"
        self.is_running = False
        
        # 🔧 FIX: Initialize lock as None, will be created in async initialize method
        self.batch_queue_lock = None
        
        # Debug logging to confirm enable_old flag
        if self.enable_old:
            logger.info("🔓 FRESHNESS FILTER DISABLED - Will process old news articles")
        else:
            logger.info("⏰ FRESHNESS FILTER ENABLED - Will skip articles older than 2 minutes")
            
        # Debug logging for any ticker mode
        if self.process_any_ticker:
            logger.info("🎯 ANY TICKER MODE ENABLED - Will process any ticker symbols found")
        else:
            logger.info("📋 DATABASE TICKER MODE - Will only process tickers from database list")
        
        # Performance tracking (same as web_scraper.py)
        self.stats = {
            'articles_processed': 0,
            'articles_inserted': 0,
            'errors': 0,
            'total_runtime': 0,
            'websocket_messages': 0,
            'ticker_matches': 0
        }
        
    async def initialize(self):
        """Initialize the Benzinga WebSocket scraper"""
        logger.info("🚀 Initializing Benzinga WebSocket scraper...")
        
        # Load API key
        self.api_key = os.getenv('BENZINGA_API_KEY')
        if not self.api_key:
            raise Exception("BENZINGA_API_KEY not found in environment variables")
        
        # Create ClickHouse connection (same pattern as web_scraper.py)
        self.clickhouse_manager = ClickHouseManager()
        self.clickhouse_manager.connect()
        
        # Load ticker list efficiently
        await self.load_tickers()
        
        # Compile ticker patterns for faster matching
        self.compile_ticker_patterns()
        
        # 🔧 FIX: Create asyncio.Lock for atomic batch queue operations
        self.batch_queue_lock = asyncio.Lock()
        
        logger.info(f"🕥 Benzinga WebSocket scraper initialized - {len(self.ticker_list)} tickers")

    async def load_tickers(self):
        """Load ticker list from ClickHouse database (same as web_scraper.py)"""
        try:
            # Get tickers from database - NO CSV fallback allowed
            db_tickers = self.clickhouse_manager.get_active_tickers()
            
            if db_tickers:
                self.ticker_list = db_tickers
                logger.info(f"Loaded {len(self.ticker_list)} tickers from database")
            else:
                # NO CSV fallback - system should fail if database is empty
                logger.error("❌ CRITICAL ERROR: No tickers found in float_list table!")
                logger.error("❌ The system requires the Finviz scraper to populate the float_list table first")
                logger.error("❌ Run the system without --skip-list flag to update ticker list")
                raise Exception("No tickers in database - float_list table is empty")
                    
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            raise  # Re-raise the exception to stop the system

    def compile_ticker_patterns(self):
        """Simple ticker matching - just exact ticker in all caps (same as web_scraper.py)"""
        pass  # No pre-compilation needed for simple matching

    def extract_any_tickers_from_text(self, text: str) -> List[str]:
        """Extract any potential ticker symbols from text using common patterns"""
        if not text:
            return []
        
        found_tickers = []
        
        # Common ticker patterns
        patterns = [
            # Exchange patterns: "NYSE: TICKER", "NASDAQ: TICKER"
            r'(?:NYSE|NASDAQ|AMEX|OTC):\s*([A-Z]{1,5})\b',
            
            # Quoted patterns: "TICKER"
            r'"([A-Z]{2,5})"',
            
            # Parenthetical with exchange: (NYSE: TICKER)
            r'\((?:NYSE|NASDAQ|AMEX|OTC):\s*([A-Z]{1,5})\)',
            
            # Parenthetical ticker only: (TICKER) - 2-5 caps
            r'\(([A-Z]{2,5})\)',
            
            # Common financial context patterns
            r'ticker\s+([A-Z]{2,5})\b',
            r'symbol\s+([A-Z]{2,5})\b',
            
            # Stock price patterns: "TICKER at $XX.XX"
            r'([A-Z]{2,5})\s+at\s+\$\d+',
            
            # Trading patterns: "TICKER shares"
            r'([A-Z]{2,5})\s+shares?\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                ticker = match.upper()
                # Filter out common words that aren't tickers
                if ticker not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'NEW', 'INC', 'LLC', 'LTD', 'CEO', 'CFO', 'USA', 'USD', 'ETF']:
                    found_tickers.append(ticker)
        
        # Remove duplicates and return
        return list(set(found_tickers))

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract tickers from text - either from database list or any tickers based on mode"""
        if not text:
            return []
        
        if self.process_any_ticker:
            # Use any ticker extraction
            return self.extract_any_tickers_from_text(text)
        else:
            # Use original database-based extraction
            found_tickers = []
            
            for ticker in self.ticker_list:
                ticker_escaped = re.escape(ticker)
                
                # Only match tickers in proper financial contexts - NO broad word matching
                patterns = [
                    # Exchange patterns: ":TICKER" (e.g., "Nasdaq: STSS", "NYSE: AAPL")
                    rf':\s*{ticker_escaped}\b',
                    
                    # Quoted pattern: "TICKER" (e.g., "STSS" and "STSSW")
                    rf'"{ticker_escaped}"',
                    
                    # Parenthetical with exchange: (Exchange: TICKER) (e.g., "(NYSE: AAPL)")
                    rf'\([^)]*:\s*{ticker_escaped}\)',
                    
                    # Exchange with quotes: ": "TICKER"" (e.g., ': "STSS"')
                    rf':\s*"{ticker_escaped}"',
                    
                    # Parenthetical ticker only: (TICKER) - but only if 3+ chars to avoid common words
                    rf'\({ticker_escaped}\)' if len(ticker) >= 3 else None
                ]
                
                # Remove None patterns and check each remaining pattern
                valid_patterns = [p for p in patterns if p is not None]
                for pattern in valid_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        found_tickers.append(ticker)
                        logger.debug(f"Found ticker {ticker} using pattern: {pattern}")
                        break  # Found with one pattern, no need to check others for this ticker
            
            return found_tickers

    def generate_content_hash(self, title: str, url: str, ticker: str = "") -> str:
        """Generate hash for duplicate detection (same as web_scraper.py)"""
        hash_input = f"{url}_{ticker}" if ticker else url
        return hashlib.md5(hash_input.encode()).hexdigest()

    def parse_benzinga_timestamp(self, created_at: str) -> datetime:
        """Parse Benzinga timestamp format"""
        if not created_at:
            return datetime.now()
        
        try:
            # Benzinga typically uses ISO format: "2024-01-15T10:30:00Z"
            if 'T' in created_at and 'Z' in created_at:
                # Remove 'Z' and parse as ISO format
                clean_timestamp = created_at.replace('Z', '')
                parsed_time = datetime.fromisoformat(clean_timestamp)
                logger.debug(f"🕐 PARSED BENZINGA TIME: '{created_at}' -> {parsed_time}")
                return parsed_time
            
            # Fallback to current time
            logger.warning(f"⚠️ Could not parse Benzinga timestamp '{created_at}', using current time")
            return datetime.now()
            
        except Exception as e:
            logger.debug(f"Error parsing Benzinga timestamp '{created_at}': {e}")
            return datetime.now()

    def extract_ticker_from_exchange_format(self, ticker_string: str) -> str:
        """
        Extract clean ticker symbol from various exchange formats
        
        Examples:
        - "TSX:TICKER" -> "TICKER"
        - "NYSE:AAPL" -> "AAPL" 
        - "NASDAQ:MSFT" -> "MSFT"
        - "AAPL" -> "AAPL" (no change if no exchange prefix)
        """
        if not ticker_string:
            return None
            
        ticker_string = ticker_string.strip().upper()
        
        # Handle exchange-prefixed formats like "TSX:TICKER"
        if ':' in ticker_string:
            parts = ticker_string.split(':')
            if len(parts) == 2:
                exchange, ticker = parts
                # Validate that the ticker part looks like a valid ticker (2-5 uppercase letters)
                ticker = ticker.strip()
                if ticker and len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha():
                    logger.debug(f"🔄 EXTRACTED TICKER: '{ticker_string}' -> '{ticker}' (exchange: {exchange})")
                    return ticker
                else:
                    logger.warning(f"⚠️ Invalid ticker format after colon: '{ticker_string}' -> '{ticker}'")
                    return None
        else:
            # No exchange prefix, validate as regular ticker
            if ticker_string and len(ticker_string) >= 1 and len(ticker_string) <= 5 and ticker_string.isalpha():
                return ticker_string
            else:
                logger.warning(f"⚠️ Invalid ticker format: '{ticker_string}'")
                return None

    def clean_content_for_sentiment_analysis(self, content: str) -> str:
        """Clean content using the same process as the backtest for consistent 4D prompt input"""
        if not content:
            return ""
        
        try:
            # Apply the same cleaning logic as the backtest
            clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
            
            # Remove extra spaces
            clean_content = ' '.join(clean_content.split())
            
            logger.debug(f"Content cleaning: {len(content)} -> {len(clean_content)} chars")
            return clean_content
            
        except Exception as e:
            logger.warning(f"Error cleaning content for sentiment analysis: {e}")
            return content

    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content to plain text, similar to sentiment_analyzer scraping logic"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up the content
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit to 8000 characters to match sentiment service
            if len(clean_content) > 8000:
                clean_content = clean_content[:8000]
            
            return clean_content
            
        except Exception as e:
            logger.warning(f"Error cleaning HTML content: {e}")
            # Fallback to basic text extraction
            return re.sub(r'<[^>]+>', ' ', html_content).strip()[:8000]

    def process_benzinga_message(self, message_data: dict) -> List[Dict[str, Any]]:
        """Process a single Benzinga WebSocket message into article format (returns list of articles, one per ticker)"""
        try:
            # Extract data content
            data = message_data.get('data', {})
            content = data.get('content', {})
            
            if not content:
                return []
            
            title = content.get('title', '')
            body = content.get('body', '')
            url = content.get('url', '')
            created_at = content.get('created_at', '')
            channels = content.get('channels', [])
            securities = content.get('securities', [])
            
            if not title or not url:
                return []
            
            # 🎯 USE STRUCTURED SECURITIES DATA instead of parsing text!
            found_tickers = []
            
            if securities:
                logger.info(f"🔍 PROCESSING SECURITIES: {securities} (type: {type(securities)}, length: {len(securities)})")
                
                # Extract tickers from the securities field
                for i, security in enumerate(securities):
                    logger.debug(f"🔍 Security {i+1}: {security} (type: {type(security)})")
                    
                    if isinstance(security, dict):
                        # Security might have structure like {"symbol": "AAPL", "name": "Apple Inc."}
                        ticker = security.get('symbol') or security.get('ticker')
                        logger.debug(f"🔍 Dict security - symbol: {security.get('symbol')}, ticker: {security.get('ticker')} -> using: {ticker}")
                        if ticker:
                            # Handle exchange-prefixed tickers like "TSX:TICKER" or "NYSE:AAPL"
                            clean_ticker = self.extract_ticker_from_exchange_format(ticker)
                            if clean_ticker:
                                found_tickers.append(clean_ticker)
                                logger.info(f"✅ EXTRACTED from dict: '{ticker}' -> '{clean_ticker}'")
                            else:
                                logger.warning(f"⚠️ FAILED to extract from dict ticker: '{ticker}'")
                    elif isinstance(security, str):
                        # Security might be just a string ticker, possibly with exchange prefix
                        logger.debug(f"🔍 String security: '{security}'")
                        clean_ticker = self.extract_ticker_from_exchange_format(security)
                        if clean_ticker:
                            found_tickers.append(clean_ticker)
                            logger.info(f"✅ EXTRACTED from string: '{security}' -> '{clean_ticker}'")
                        else:
                            logger.warning(f"⚠️ FAILED to extract from string: '{security}'")
                    else:
                        logger.warning(f"⚠️ UNKNOWN security type: {type(security)} - {security}")
                
                logger.info(f"📊 STRUCTURED SECURITIES: {securities} -> Extracted tickers: {found_tickers}")
            else:
                logger.info("📊 NO SECURITIES DATA found in message")
            
            # Filter tickers based on mode
            if not self.process_any_ticker and found_tickers:
                # Filter to only include tickers from our database list
                filtered_tickers = [ticker for ticker in found_tickers if ticker in self.ticker_list]
                if filtered_tickers != found_tickers:
                    logger.info(f"🔽 FILTERED TICKERS: {found_tickers} -> {filtered_tickers} (database list only)")
                found_tickers = filtered_tickers
            
            if not found_tickers:
                return []  # Only process articles with ticker matches
            
            # Parse timestamp
            parsed_time = self.parse_benzinga_timestamp(created_at)
            
            # Check freshness filter (same logic as web_scraper.py)
            if not self.enable_old:
                time_diff = (datetime.now() - parsed_time).total_seconds()
                if time_diff > 120:  # More than 2 minutes old
                    logger.debug(f"⏰ SKIPPING OLD ARTICLE: {time_diff:.1f}s old - {title[:50]}...")
                    return []
            
            # Create separate article for each ticker (SAME AS WEB_SCRAPER.PY)
            articles = []
            
            for ticker in found_tickers:
                # Generate unique content hash per ticker
                content_hash = self.generate_content_hash(title, url, ticker)
                
                # Clean the content using the same process as backtest
                raw_content = self.clean_html_content(body) if body else self.clean_html_content(title)
                cleaned_content = self.clean_content_for_sentiment_analysis(raw_content)
                
                article = {
                    'source': 'Benzinga_WebSocket',
                    'ticker': ticker,  # Individual ticker per article
                    'headline': title,
                    'published_utc': created_at,  # Store raw string as per schema
                    'article_url': url,
                    'summary': title,
                    'full_content': cleaned_content,  # Use cleaned content for consistent 4D prompt input
                    'detected_at': datetime.now(),
                    'processing_latency_ms': 0,
                    'market_relevant': 1,
                    'source_check_time': datetime.now(),
                    'content_hash': content_hash,
                    'news_type': 'other',
                    'urgency_score': 5
                }
                
                articles.append(article)
                logger.info(f"✅ NEW ARTICLE: {ticker} - {title[:50]}...")
                
                # 🚀 ZERO-LAG: Create immediate trigger file for instant price checking
                try:
                    self.clickhouse_manager.create_immediate_trigger(ticker, parsed_time)
                except Exception as e:
                    logger.warning(f"⚠️ Could not create immediate trigger for {ticker}: {e}")
            
            logger.info(f"✅ TICKER MATCH: {found_tickers} in Benzinga WebSocket: {title}")
            self.stats['ticker_matches'] += len(found_tickers)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error processing Benzinga message: {e}")
            self.stats['errors'] += 1
            return []

    async def connect_websocket(self):
        """Connect to Benzinga WebSocket with proper authentication"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Benzinga WebSocket (attempt {attempt + 1})...")
                
                # Use query parameter authentication (this worked in our test)
                websocket_url_with_key = f"{self.websocket_url}?token={self.api_key}"
                
                self.websocket = await websockets.connect(
                    websocket_url_with_key,
                    ping_interval=30,
                    ping_timeout=10
                )
                
                logger.info("✅ Successfully connected to Benzinga WebSocket!")
                return True
                
            except Exception as e:
                logger.warning(f"❌ WebSocket connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        logger.error("❌ All WebSocket connection attempts failed!")
        return False

    async def websocket_listener(self):
        """Listen for WebSocket messages and process them"""
        logger.info("🚀 Starting Benzinga WebSocket listener...")
        
        # Add heartbeat logging
        last_heartbeat = time.time()
        heartbeat_interval = 10  # Log heartbeat every 10 seconds for more visibility
        message_wait_count = 0
        
        while self.is_running:
            try:
                if not self.websocket:
                    if not await self.connect_websocket():
                        logger.error("Failed to connect to WebSocket, waiting before retry...")
                        await asyncio.sleep(30)
                        continue
                
                # Log that we're actively waiting for messages
                message_wait_count += 1
                logger.info(f"👂 Actively listening for WebSocket messages (wait cycle #{message_wait_count})...")
                
                # Listen for messages
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)  # Shorter timeout for more frequent logging
                    self.stats['websocket_messages'] += 1
                    
                    # Log message reception with more detail
                    logger.info(f"📨 RECEIVED WebSocket message #{self.stats['websocket_messages']} (length: {len(message)} chars)")
                    
                    # Parse JSON message
                    try:
                        message_data = json.loads(message)
                        
                        # Log message type and basic info with more detail
                        msg_kind = message_data.get('kind', 'unknown')
                        api_version = message_data.get('api_version', 'unknown')
                        data_content = message_data.get('data', {}).get('content', {})
                        title = data_content.get('title', 'No title') if data_content else 'No content'
                        
                        logger.info(f"📰 Message details: kind={msg_kind}, api_version={api_version}")
                        logger.info(f"📰 Content title: {title[:80]}...")
                        
                        # Log raw message for first few messages
                        if self.stats['websocket_messages'] <= 3:
                            logger.info(f"🔍 Raw message #{self.stats['websocket_messages']}:")
                            logger.info(json.dumps(message_data, indent=2)[:500] + "..." if len(json.dumps(message_data)) > 500 else json.dumps(message_data, indent=2))
                        
                        # Process the message
                        articles = self.process_benzinga_message(message_data)
                        
                        if articles:
                            # 🔧 FIX: Use asyncio.Lock to ensure atomic batch queue operations
                            async with self.batch_queue_lock:
                                self.batch_queue.extend(articles)
                                self.stats['articles_processed'] += len(articles)
                                logger.info(f"✅ Added {len(articles)} articles to batch queue (total queued: {len(self.batch_queue)})")
                        else:
                            logger.info(f"⏭️ Message processed but no ticker matches found - skipped")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON message: {e}")
                        logger.error(f"Raw message (first 300 chars): {message[:300]}")
                        self.stats['errors'] += 1
                
                except asyncio.TimeoutError:
                    # More frequent heartbeat logging
                    current_time = time.time()
                    if current_time - last_heartbeat >= heartbeat_interval:
                        logger.info(f"💓 WebSocket ACTIVE - No messages in last {heartbeat_interval}s (Total: {self.stats['websocket_messages']} msgs, {self.stats['articles_processed']} processed)")
                        logger.info(f"🔗 Connection status: {'Connected' if self.websocket else 'Disconnected'}")
                        last_heartbeat = current_time
                    continue
                
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"🔌 WebSocket connection closed: {e}")
                    logger.info("🔄 Will attempt to reconnect...")
                    self.websocket = None
                    await asyncio.sleep(5)  # Brief pause before reconnecting
                    continue
                
            except Exception as e:
                logger.error(f"❌ Error in WebSocket listener: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.stats['errors'] += 1
                self.websocket = None
                await asyncio.sleep(10)  # Error recovery delay

    async def buffer_flusher(self):
        """Periodically flush buffer to ClickHouse (same as web_scraper.py)"""
        logger.info("🔄 Starting buffer flusher - checking every 250ms")
        flush_count = 0
        
        while self.is_running:
            try:
                await asyncio.sleep(0.25)  # Flush every 250ms for ULTRA-fast detection
                
                # 🔧 FIX: Use asyncio.Lock to ensure atomic batch queue operations
                articles_to_flush = []
                async with self.batch_queue_lock:
                    if self.batch_queue:
                        articles_to_flush = self.batch_queue.copy()
                        self.batch_queue.clear()
                        flush_count += 1
                        logger.info(f"🚀 Buffer flush #{flush_count} - {len(articles_to_flush)} articles ready")
                
                # Process articles outside the lock to avoid blocking WebSocket processing
                if articles_to_flush:
                    await self.flush_articles_to_clickhouse(articles_to_flush)
                
            except Exception as e:
                logger.error(f"Error in buffer flusher: {e}")

    async def flush_articles_to_clickhouse(self, articles):
        """Flush articles to ClickHouse using NEW concurrent batch processing with immediate insertion"""
        if not articles:
            return
            
        try:
            logger.info(f"🧠 Starting CONCURRENT batch processing for {len(articles)} articles (MATCHES TEST LOGIC)...")
            
            # Use the NEW sentiment service method that matches test logic
            start_time = time.time()
            result = await analyze_articles_with_sentiment_and_immediate_insertion(articles, 'breaking_news')
            processing_time = time.time() - start_time
            
            successful_count = result.get('successful_count', 0)
            total_articles = result.get('total_articles', len(articles))
            
            logger.info(f"✅ CONCURRENT BATCH PROCESSING COMPLETE: {successful_count}/{total_articles} articles successfully processed and inserted in {processing_time:.1f}s")
            logger.info(f"⚡ AVERAGE TIME PER ARTICLE: {processing_time/max(1, total_articles):.1f}s (MUCH FASTER than individual calls)")
            
            self.stats['articles_inserted'] += successful_count
            
            # Log any errors from the result
            if 'error' in result:
                logger.warning(f"⚠️ Some articles used fallback processing: {result['error']}")
            
        except Exception as e:
            logger.error(f"❌ Error in concurrent batch processing: {e}")
            logger.info(f"🔄 Falling back to old individual processing method...")
            
            # Fallback to old method if new method fails
            await self._fallback_individual_processing(articles)
    
    async def _fallback_individual_processing(self, articles):
        """Fallback to old individual processing method if new method fails"""
        successful_count = 0
        batch_size = 20
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"📦 FALLBACK Processing batch {i//batch_size + 1}: {len(batch)} articles")
            
            # Create tasks for individual processing and insertion
            tasks = []
            for j, article in enumerate(batch):
                task = asyncio.create_task(self._process_and_insert_individual_article(article, i + j + 1))
                tasks.append(task)
            
            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            for result in results:
                if not isinstance(result, Exception) and result:
                    successful_count += 1
            
            # Small delay between batches
            if i + batch_size < len(articles):
                await asyncio.sleep(0.5)
        
        logger.info(f"✅ FALLBACK processing complete: {successful_count}/{len(articles)} articles")
        self.stats['articles_inserted'] += successful_count
        self.stats['errors'] = self.stats.get('errors', 0) + 1
    
    async def _process_and_insert_individual_article(self, article: Dict[str, Any], index: int) -> bool:
        """Process a single article with sentiment analysis and immediately insert to database (FALLBACK METHOD)"""
        ticker = article.get('ticker', 'UNKNOWN')
        
        try:
            logger.debug(f"🧠 #{index:2d} ANALYZING (FALLBACK): {ticker}")
            
            # STEP 1: Analyze sentiment for this single article  
            from sentiment_service import get_sentiment_service
            sentiment_service = await get_sentiment_service()
            sentiment_result = await sentiment_service.analyze_article_sentiment(article)
            
            # STEP 2: Enrich article with sentiment data
            if isinstance(sentiment_result, dict) and 'error' not in sentiment_result:
                article.update({
                    'sentiment': sentiment_result.get('sentiment', 'neutral'),
                    'recommendation': sentiment_result.get('recommendation', 'HOLD'),
                    'confidence': sentiment_result.get('confidence', 'low'),
                    'explanation': sentiment_result.get('explanation', 'No explanation'),
                    'analysis_time_ms': sentiment_result.get('analysis_time_ms', 0),
                    'analyzed_at': sentiment_result.get('analyzed_at', datetime.now())
                })
                logger.debug(f"✅ #{index:2d} SENTIMENT (FALLBACK): {ticker} -> {sentiment_result.get('recommendation', 'HOLD')}")
            else:
                # Use default sentiment for failed analysis
                error_msg = sentiment_result.get('error', 'Unknown error') if sentiment_result else 'No response'
                article.update({
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Analysis failed: {error_msg}',
                    'analysis_time_ms': 0,
                    'analyzed_at': datetime.now()
                })
                logger.warning(f"⚠️ #{index:2d} DEFAULT (FALLBACK): {ticker} -> Using default sentiment")
            
            # STEP 3: Immediately insert to database
            inserted_count = self.clickhouse_manager.insert_articles([article])
            
            if inserted_count > 0:
                logger.debug(f"💾 #{index:2d} INSERTED (FALLBACK): {ticker} -> Database")
                return True
            elif inserted_count == -1:
                logger.debug(f"📝 #{index:2d} DUPLICATE SKIPPED (FALLBACK): {ticker} -> Database (already exists)")
                return True  # Duplicate skip is considered successful
            else:
                logger.error(f"❌ #{index:2d} INSERT FAILED (FALLBACK): {ticker}")
                return False
                
        except Exception as e:
            logger.error(f"❌ #{index:2d} EXCEPTION (FALLBACK): {ticker} -> {str(e)}")
            
            # ZERO LOSS GUARANTEE: Insert with default sentiment even on exception
            try:
                article.update({
                    'sentiment': 'neutral',
                    'recommendation': 'HOLD',
                    'confidence': 'low',
                    'explanation': f'Processing exception: {str(e)}',
                    'analysis_time_ms': 0,
                    'analyzed_at': datetime.now()
                })
                
                inserted_count = self.clickhouse_manager.insert_articles([article])
                if inserted_count > 0:
                    logger.warning(f"🛡️ #{index:2d} ZERO LOSS (FALLBACK): {ticker} -> Inserted with default sentiment")
                    return True
                else:
                    logger.error(f"❌ #{index:2d} ZERO LOSS FAILED (FALLBACK): {ticker}")
                    return False
                    
            except Exception as fallback_error:
                logger.error(f"❌ #{index:2d} FALLBACK FAILED: {ticker} -> {str(fallback_error)}")
                return False

    async def stats_reporter(self):
        """Report performance statistics (same as web_scraper.py)"""
        start_time = time.time()
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                runtime = time.time() - start_time
                ws_rate = self.stats['websocket_messages'] / runtime if runtime > 0 else 0
                process_rate = self.stats['articles_processed'] / runtime if runtime > 0 else 0
                
                logger.info(f"BENZINGA WEBSOCKET STATS - Runtime: {runtime:.1f}s, "
                          f"WS Messages: {self.stats['websocket_messages']}, "
                          f"Processed: {self.stats['articles_processed']}, "
                          f"Ticker Matches: {self.stats['ticker_matches']}, "
                          f"Inserted: {self.stats['articles_inserted']}, "
                          f"Errors: {self.stats['errors']}, "
                          f"WS Rate: {ws_rate:.2f}/sec, Process Rate: {process_rate:.2f}/sec")
                          
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")

    async def start_scraping(self):
        """Start scraping using WebSocket (same interface as web_scraper.py)"""
        logger.info("🚀 Starting Benzinga WebSocket scraping...")
        
        # Initialize
        await self.initialize()
        
        # Set running flag
        self.is_running = True
        
        # Create tasks (same pattern as web_scraper.py)
        tasks = []
        
        # WebSocket listener task (replaces bulk scraping)
        websocket_task = asyncio.create_task(self.websocket_listener())
        tasks.append(websocket_task)
        
        # Buffer flusher task
        buffer_task = asyncio.create_task(self.buffer_flusher())
        tasks.append(buffer_task)
        
        # Stats reporter task  
        stats_task = asyncio.create_task(self.stats_reporter())
        tasks.append(stats_task)
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Benzinga WebSocket scraping stopped by user")
        finally:
            # Clean up
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources (same as web_scraper.py)"""
        try:
            # Stop running flag
            self.is_running = False
            
            # Final flush before shutdown - use the new method with proper locking
            async with self.batch_queue_lock:
                if self.batch_queue:
                    articles_to_flush = self.batch_queue.copy()
                    self.batch_queue.clear()
                    if articles_to_flush:
                        await self.flush_articles_to_clickhouse(articles_to_flush)
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                
            # Close ClickHouse connection
            if self.clickhouse_manager:
                self.clickhouse_manager.close()
                
            logger.info("Benzinga WebSocket scraper cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Maintain same interface as web_scraper.py for drop-in replacement
# This allows run_system.py to use BenzingaWebSocketScraper instead of Crawl4AIScraper
Crawl4AIScraper = BenzingaWebSocketScraper

async def main():
    """Main function for standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benzinga WebSocket News Scraper')
    parser.add_argument('--enable-old', action='store_true', 
                       help='Process old news articles (disable freshness filter)')
    parser.add_argument('--any', '--process-any-ticker', action='store_true', 
                       help='Process any ticker symbols found (bypass ticker list filtering)')
    parser.add_argument('--duration', type=int, default=0,
                       help='Test duration in minutes (0 = run indefinitely)')
    
    args = parser.parse_args()
    
    # Create and run scraper
    scraper = BenzingaWebSocketScraper(enable_old=args.enable_old, process_any_ticker=args.any)
    
    if args.duration > 0:
        # Run for specific duration
        logger.info(f"Running Benzinga WebSocket scraper for {args.duration} minutes...")
        try:
            await asyncio.wait_for(scraper.start_scraping(), timeout=args.duration * 60)
        except asyncio.TimeoutError:
            logger.info(f"Test completed after {args.duration} minutes")
    else:
        # Run indefinitely
        await scraper.start_scraping()

if __name__ == "__main__":
    asyncio.run(main()) 