# NewsHead - Zero-Lag News & Price Monitoring System Architecture

## Overview

NewsHead is a high-performance real-time stock market news monitoring and price tracking system designed to achieve **sub-10-second** news-to-alert latency. The system implements a zero-lag architecture through process isolation, file-based triggers, and aggressive optimization techniques. The system now supports **dual news collection modes**: traditional web scraping and real-time WebSocket streaming via Benzinga's API.

### Overview
The NewsHead system includes a comprehensive sentiment analysis engine that analyzes news articles using Claude API before database insertion. This enables intelligent price alerts that only trigger when both price movements AND favorable sentiment conditions are met. The sentiment analysis now features **native load balancing** across multiple API keys without requiring external gateway servers.

## Core Architecture Principles

### 1. Process Isolation Design
- **Separate Processes**: News monitoring and price checking run in completely isolated processes
- **Zero Resource Contention**: Browser automation doesn't interfere with API calls
- **File-Based Communication**: Inter-process communication via filesystem triggers
- **Independent Scaling**: Each component can be optimized independently

### 2. Zero-Lag Data Flow
```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN PROCESS                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   News Source   │───▶│ File Triggers   │                │
│  │ (Web/WebSocket) │    │   (triggers/)   │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (File System)
┌─────────────────────────────────────────────────────────────┐
│                  ISOLATED PROCESS                           │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ File Trigger    │───▶│ Price Monitor   │                │
│  │   Monitor       │    │  (Polygon API)  │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### 3. Performance-First Design
- **Ultra-Fast Buffers**: 250ms flush intervals for immediate data processing
- **Aggressive Timeouts**: 2-second API timeouts matching polling intervals
- **Parallel Processing**: Concurrent news source monitoring and price checking
- **Smart Caching**: 10-second database query cache to reduce load

## System Components

### Core Entry Points

#### `run_system.py` - Process Orchestrator
**Purpose**: Main system controller that manages process isolation, startup sequence, and news source selection

**Key Responsibilities**:
- Database initialization and table setup
- Sequential startup (price checker first, then news monitor)
- Process lifecycle management with proper cleanup
- News source mode selection (Web Scraper vs WebSocket)
- Graceful shutdown handling
- Comprehensive logging setup

**Process Flow**:
1. Setup ClickHouse database with fresh tables
2. Optional Finviz ticker list update (unless `--skip-list`)
3. Start price checker in isolated subprocess
4. Wait for price checker initialization (10s)
5. Start selected news monitor (Web Scraper or WebSocket)
6. Handle shutdown signals and cleanup

**Command Line Options and System Configurations**:

For comprehensive command variations and detailed usage examples, see `commands.txt`. The main command line options are:

**Core Arguments**:
- `--skip-list`: Skip Finviz ticker list update (recommended for most operations)
- `--enable-old`: Disable freshness filtering, process articles older than 2 minutes
- `--socket`: **Use Benzinga WebSocket instead of web scraper** (recommended for production)
- `--any`: Process any ticker symbols found (WebSocket only, bypasses database filtering)
- `--no-sentiment`: Skip sentiment analysis initialization (for testing/debugging)

**Recommended Production Configurations**:

1. **High-Performance WebSocket Mode** (recommended):
   ```bash
   python3 run_system.py --skip-list --socket
   ```
   - Uses Benzinga WebSocket for sub-second news detection
   - Database ticker filtering enabled
   - Sentiment analysis enabled
   - Freshness filtering enabled (2-minute window)

2. **High-Volume Testing Mode**:
   ```bash
   python3 run_system.py --skip-list --socket --any
   ```
   - Processes any ticker symbols found (not just database tickers)
   - Useful for discovering new tickers and testing system capacity

3. **Debug/Testing Mode**:
   ```bash
   python3 run_system.py --skip-list --socket --no-sentiment
   ```
   - Disables sentiment analysis for faster processing
   - Useful for debugging news detection pipeline

4. **Legacy Web Scraper Mode** (slower, fallback only):
   ```bash
   python3 run_system.py --skip-list
   ```
   - Uses traditional web scraping instead of WebSocket
   - 5-second polling intervals vs sub-second WebSocket detection

**News Source Selection Logic**:
```python
if args.socket:
    # WebSocket Mode - Real-time streaming (RECOMMENDED)
    from benzinga_websocket import Crawl4AIScraper
    scraper = Crawl4AIScraper(enable_old=enable_old, process_any_ticker=process_any_ticker)
else:
    # Web Scraper Mode - Traditional scraping (LEGACY)
    from web_scraper import Crawl4AIScraper  
    scraper = Crawl4AIScraper(enable_old=enable_old)
```

**System Initialization with Sentiment Analysis**:
The system automatically initializes sentiment analysis unless `--no-sentiment` is specified:
```python
# Sentiment service initialization
if not args.no_sentiment:
    sentiment_initialized = await initialize_sentiment_service()
    if sentiment_initialized:
        logger.info("🤖 AI-powered sentiment analysis is ACTIVE")
    else:
        logger.warning("⚠️ Sentiment analysis initialization failed - continuing without sentiment")
```

### News Collection Layer

#### `benzinga_websocket.py` - **NEW** Real-Time WebSocket Engine
**Technology**: WebSocket streaming with Benzinga's real-time news API
**Purpose**: Ultra-low latency news monitoring via persistent WebSocket connection

**Key Features**:
- **Real-Time Streaming**: Sub-second news detection via WebSocket
- **Structured Data**: Uses Benzinga's structured securities data instead of text parsing
- **Drop-in Replacement**: Maintains same interface as `web_scraper.py`
- **Dual Ticker Modes**: Database-filtered or any-ticker processing
- **Zero-Lag Triggers**: Immediate file trigger creation on ticker matches

**WebSocket Connection**:
- **Endpoint**: `wss://api.benzinga.com/api/v1/news/stream`
- **Authentication**: Query parameter token authentication
- **Reliability**: Auto-reconnection with exponential backoff
- **Heartbeat**: 30-second ping intervals with connection monitoring

**Ticker Extraction Strategy**:
```python
# Structured Securities Processing (vs text parsing)
securities = content.get('securities', [])
for security in securities:
    if isinstance(security, dict):
        ticker = security.get('symbol') or security.get('ticker')
    elif isinstance(security, str):
        ticker = security  # Handle exchange-prefixed format
    
    # Extract clean ticker from "TSX:TICKER" or "NYSE:AAPL" formats
    clean_ticker = extract_ticker_from_exchange_format(ticker)
```

**Dual Processing Modes**:
1. **Database Mode** (default): Only process tickers from float_list table
2. **Any Ticker Mode** (`--any` flag): Process any ticker symbols found using pattern matching

**Performance Characteristics**:
- **Message Processing**: ~0.1-0.5s from WebSocket message to database insert
- **Buffer Flushing**: 250ms intervals for ultra-fast detection
- **Connection Resilience**: Automatic reconnection with detailed logging
- **Structured Data**: No regex parsing needed, uses API's securities field

**WebSocket Message Flow**:
```
WebSocket Message → JSON Parse → Securities Extraction → Ticker Validation → 
Article Creation → Batch Queue → Database Insert → File Trigger
```

#### `web_scraper.py` - Traditional Web Scraping Engine
**Technology**: Crawl4AI with Chromium browser automation
**Purpose**: Multi-source newswire monitoring with CPU optimization

**News Sources**:
- GlobeNewswire (24-hour feed)
- BusinessWire (current releases)
- PR Newswire (latest releases)
- AccessNewswire (breaking news)

**Performance Optimizations**:
- **CPU-Efficient Browser Flags**: Disabled GPU, images, JavaScript for speed
- **Fast Cycle Times**: 5-second polling intervals
- **Parallel Source Processing**: All sources scraped simultaneously
- **Immediate Triggers**: File-based notifications on ticker matches

**Ticker Extraction Logic**:
```python
# Only match tickers in proper financial contexts
patterns = [
    r':\s*{ticker}\b',           # Exchange: TICKER
    r'"{ticker}"',               # Quoted: "TICKER"
    r'\([^)]*:\s*{ticker}\)',    # Parenthetical: (NYSE: TICKER)
    r':\s*"{ticker}"',           # Exchange quoted: : "TICKER"
    r'\({ticker}\)'              # Simple parenthetical (3+ chars only)
]
```

**Freshness Filtering**:
- **2-minute window**: Only processes news published within last 2 minutes
- **Timezone-agnostic**: Compares minute:second portions only
- **Current-day filter**: Additional date validation for certain sources

#### `rss_news_monitor.py` - RSS Comparison Engine
**Purpose**: RSS feed monitoring for comparison and validation
**Technology**: Native feedparser with aiohttp

**RSS Sources**:
- GlobeNewswire RSS
- BusinessWire RSS
- PR Newswire RSS

**Features**:
- **Identical Logic**: Uses same ticker extraction as web scraper
- **Parallel Processing**: All RSS feeds processed simultaneously
- **Comparison Logging**: Tracks differences between web and RSS detection

#### `newswire_monitor.py` - Alternative News Engine
**Purpose**: Lightweight RSS-only monitoring system
**Use Case**: Fallback or standalone RSS monitoring

### Price Monitoring Layer

#### `price_checker.py` - Zero-Lag Price Monitor
**Purpose**: Process-isolated price tracking with sub-second response
**Technology**: Polygon.io API with proxy support

**Architecture Components**:
- **File Trigger Monitor**: Processes trigger files for immediate ticker addition
- **Continuous Polling**: 2-second cycles for all active tickers
- **Dual System**: File triggers for notifications + polling for price inserts

**Performance Features**:
- **Aggressive Timeouts**: 2s total, 0.5s connect, 1.5s read
- **Parallel Processing**: Bulk API calls for multiple tickers
- **Smart Caching**: 10-second database query cache
- **Clean Failure Handling**: Skip failed requests, maintain 2-second intervals

**Price Data Strategy**:
- **Single Endpoint**: `/v2/last/trade/{ticker}` - Uses only actual executed trade prices
- **Double-Call Fix**: NEW - For newly added tickers (≤10 seconds old), makes two API calls and discards the first (often garbage) response
- **Garbage Data Protection**: Polygon API sometimes returns stale/incorrect prices on first call for new tickers, causing false alerts
- **Smart Source Marking**: New tickers use `trade_verified` source to indicate double-call verification
- **No Fallbacks**: Unreliable NBBO calculations removed entirely
- **Skip on Failure**: Failed requests are skipped to maintain clean intervals
- **Reliability Priority**: Only real trade prices prevent phantom price alerts (e.g., FEAM $4.05, CAPS $4.06 issues)

**Double-Call Implementation**:
```python
# NEW: Double API call logic for new tickers
async def get_price_with_double_call(self, ticker: str):
    """Make double API call for new tickers - discard first (garbage), use second (correct)"""
    
    # First call - expect garbage data, discard it
    async with self.session.get(url, params=params) as response1:
        if response1.status == 200:
            garbage_data = await response1.json()
            garbage_price = garbage_data['results'].get('p', 0.0)
            logger.info(f"🗑️ {ticker}: Discarding first call garbage price: ${garbage_price:.4f}")
    
    # Small delay between calls
    await asyncio.sleep(0.1)
    
    # Second call - expect correct data, use this one
    async with self.session.get(url, params=params) as response2:
        if response2.status == 200:
            data = await response2.json()
            if 'results' in data and data['results']:
                correct_price = data['results'].get('p', 0.0)
                logger.info(f"✅ {ticker}: Using second call verified price: ${correct_price:.4f}")
                return {
                    'ticker': ticker,
                    'price': correct_price,
                    'source': 'trade_verified',  # Mark as double-call verified
                    'timestamp': datetime.now()
                }
```

**Garbage Data Issue Analysis**:
- **Problem**: Polygon API returns incorrect prices (e.g., $0.28 for BCTX, $0.35 for TTEC) on first API call for newly tracked tickers
- **Impact**: Caused massive false alerts (e.g., +1391% for TTEC, +984% for BCTX) when second call returned correct price
- **Root Cause**: API cache issues, stale data, or uninitialized values in Polygon's system
- **Solution**: Double-call pattern discards first response, uses second verified response
- **Performance**: Minimal impact (~0.1s delay) only for new tickers, existing tickers use single calls

**Price Alert Logic**:
```python
# Alert triggers based on price movement thresholds
PRICE_CHANGE_THRESHOLD = 0.05  # 5% change
VOLUME_SPIKE_THRESHOLD = 2.0   # 2x average volume
```

### Data Management Layer

#### `clickhouse_setup.py` - Database Management
**Purpose**: Optimized database operations with performance focus
**Technology**: ClickHouse with clickhouse-connect driver

**Key Tables**:

**`News.breaking_news`** - News Storage
```sql
CREATE TABLE News.breaking_news (
    id UUID DEFAULT generateUUIDv4(),
    timestamp DateTime64(3) DEFAULT now64(),
    source String,
    ticker String,
    headline String,
    published_utc String,
    article_url String,
    summary String,
    full_content String,
    detected_at DateTime64(3) DEFAULT now64(),
    processing_latency_ms UInt32,
    market_relevant UInt8 DEFAULT 0,
    source_check_time DateTime64(3),
    content_hash String,
    news_type String DEFAULT 'other',
    urgency_score UInt8 DEFAULT 0,
    
    -- Sentiment analysis fields
    sentiment String DEFAULT 'neutral',
    recommendation String DEFAULT 'HOLD',
    confidence String DEFAULT 'low',
    explanation String DEFAULT '',
    analysis_time_ms UInt32 DEFAULT 0,
    analyzed_at DateTime64(3) DEFAULT now64(),
    
    INDEX idx_sentiment (sentiment) TYPE set(10) GRANULARITY 1,
    INDEX idx_recommendation (recommendation) TYPE set(10) GRANULARITY 1
) ENGINE = ReplacingMergeTree(detected_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (content_hash, article_url)
```

**`News.price_tracking`** - Price Data
```sql
CREATE TABLE News.price_tracking (
    timestamp DateTime DEFAULT now(),
    ticker String,
    price Float64,
    volume UInt64,
    source String DEFAULT 'polygon',
    
    -- Sentiment analysis fields (from associated news)
    sentiment String DEFAULT 'neutral',
    recommendation String DEFAULT 'HOLD',
    confidence String DEFAULT 'low'
) ENGINE = MergeTree()
ORDER BY (ticker, timestamp)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 7 DAY
```

**`News.news_alert`** - Alert Generation
```sql
CREATE TABLE News.news_alert (
    ticker String,
    timestamp DateTime DEFAULT now(),
    alert UInt8 DEFAULT 1,
    price Float64
) ENGINE = MergeTree()
ORDER BY (ticker, timestamp)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 30 DAY
```

**Performance Features**:
- **Pipeline Reset**: Complete table clearing for fresh starts
- **Trigger File Creation**: File-based notification system
- **Optimized Queries**: No FINAL clauses, limited time windows
- **Batch Operations**: Efficient bulk inserts

## Sentiment Analysis Integration

### Overview
The NewsHead system includes a comprehensive sentiment analysis engine that analyzes news articles using Claude API before database insertion. This enables intelligent price alerts that only trigger when both price movements AND favorable sentiment conditions are met.

**NEW**: The sentiment analysis system now includes **native Python load balancing** that automatically distributes requests across multiple API keys without requiring external servers like Portkey Gateway.

### Architecture Components

#### `sentiment_service.py` - AI-Powered Sentiment Analysis Service with Native Load Balancing
**Purpose**: Real-time sentiment analysis of news articles using Claude API with automatic load balancing
**Technology**: Anthropic Claude 3.5 Sonnet API with native Python load balancing and async processing

**Key Features**:
- **Native Load Balancing**: Automatically distributes requests across multiple API keys without external dependencies
- **Zero Infrastructure**: No external servers or Node.js dependencies required
- **Automatic Failover**: Seamlessly switches between API keys when rate limits are hit
- **Backward Compatibility**: Falls back to single-key mode if only one API key is available
- **Cloud AI Processing**: Uses Claude API for high-quality sentiment analysis
- **Batch Processing**: Analyzes multiple articles simultaneously (up to 20 concurrent)
- **Intelligent Caching**: Avoids re-analyzing identical content
- **Fallback Handling**: Graceful degradation when API analysis fails
- **Performance Tracking**: Detailed statistics and timing metrics including per-key analytics
- **Rate Limit Management**: Automatic retry with exponential backoff and key rotation

**Native Load Balancing Architecture**:
```python
# Multi-Key Configuration (automatic detection)
ANTHROPIC_API_KEY=sk-ant-key1     # Primary key
ANTHROPIC_API_KEY2=sk-ant-key2    # Secondary key  
ANTHROPIC_API_KEY3=sk-ant-key3    # Tertiary key
# ... up to ANTHROPIC_API_KEY9
```

**Load Balancing Features**:
- **Round-Robin Distribution**: Evenly distributes requests across available keys
- **Rate Limit Awareness**: Automatically excludes rate-limited keys from rotation
- **Smart Recovery**: Automatically re-enables keys after rate limit cooldown
- **Per-Key Statistics**: Tracks usage, success rates, and failures for each key
- **Automatic Key Switching**: Seamlessly switches to next available key on 429 errors
- **Fallback Logic**: Uses least recently rate-limited key when all keys are limited

**AI Model Integration**:
- **API Endpoint**: `https://api.anthropic.com/v1/messages`
- **Model Type**: Claude 3.5 Sonnet (claude-3-5-sonnet-20240620)
- **Analysis Prompt**: Financial news sentiment analysis with BUY/SELL recommendations
- **Response Format**: Structured JSON with sentiment, recommendation, confidence, and explanation
- **Rate Limits**: 40,000 tokens/minute per key (multiplied by number of keys)
- **Concurrent Requests**: Up to 20 simultaneous API calls distributed across keys

**Load Balancing vs Single Key Performance**:

| Feature | Single Key | Native Load Balancing |
|---------|------------|----------------------|
| **Throughput** | ~40 requests/minute | ~40 × N keys requests/minute |
| **Rate Limit Resilience** | Fails on 429 errors | Automatic key switching |
| **Infrastructure** | Simple | Zero additional complexity |
| **Failover** | Manual retry only | Automatic failover |
| **Monitoring** | Basic stats | Per-key detailed analytics |
| **Setup** | 1 environment variable | N environment variables |

**Analysis Results Structure**:
```json
{
    "ticker": "AAPL",
    "sentiment": "positive",
    "recommendation": "BUY", 
    "confidence": "high",
    "explanation": "Strong quarterly earnings beat expectations",
    "analysis_time_ms": 1500,
    "analyzed_at": "2025-01-15T10:30:00Z"
}
```

**Load Balancing Statistics**:
```json
{
    "load_balancing_enabled": true,
    "load_balancing_stats": {
        "total_keys": 3,
        "available_keys": 2,
        "rate_limited_keys": 1,
        "total_requests": 150,
        "successful_requests": 142,
        "key_switches": 8,
        "success_rate": 94.7,
        "key_details": [
            {
                "key_id": "Key_1",
                "last_8_chars": "abc123ef",
                "request_count": 52,
                "success_count": 48,
                "success_rate": 92.3,
                "rate_limit_count": 2,
                "is_rate_limited": false
            }
        ]
    }
}
```

**Possible Label Values**:
- **Sentiment**: `"positive"`, `"negative"`, `"neutral"`
- **Recommendation**: `"BUY"`, `"SELL"`, `"HOLD"`
- **Confidence**: `"high"`, `"medium"`, `"low"`

### Integration Points

#### News Pipeline Integration
Both news collection systems now include native load balancing sentiment analysis:

**Web Scraper Integration** (`web_scraper.py`):
```python
async def flush_buffer_to_clickhouse(self):
    """Flush article buffer to ClickHouse WITH native load balancing sentiment analysis"""
    if self.batch_queue:
        # SENTIMENT ANALYSIS WITH NATIVE LOAD BALANCING
        try:
            enriched_articles = await analyze_articles_with_sentiment(list(self.batch_queue))
            inserted_count = self.ch_manager.insert_articles(enriched_articles)
        except Exception as e:
            # Fallback: Insert without sentiment analysis
            inserted_count = self.ch_manager.insert_articles(list(self.batch_queue))
```

**WebSocket Integration** (`benzinga_websocket.py`):
```python
async def flush_buffer_to_clickhouse(self):
    """Flush article buffer to ClickHouse WITH native load balancing sentiment analysis"""
    if self.batch_queue:
        # SENTIMENT ANALYSIS WITH NATIVE LOAD BALANCING
        try:
            enriched_articles = await analyze_articles_with_sentiment(list(self.batch_queue))
            inserted_count = self.ch_manager.insert_articles(enriched_articles)
        except Exception as e:
            # Fallback: Insert without sentiment analysis
            inserted_count = self.ch_manager.insert_articles(list(self.batch_queue))
```

#### Price Alert Enhancement
The price monitoring system now includes sentiment-based filtering:

**Enhanced Price Alert Logic** (`price_checker.py`):
```sql
-- SENTIMENT-ENHANCED PRICE ALERTS WITH NATIVE LOAD BALANCING
-- Only trigger alerts when:
-- 1. Price moves 5%+ within 30 seconds (existing logic)
-- 2. AND sentiment is 'BUY' with 'high' confidence (NEW requirement)

SELECT 
    p.ticker,
    p.current_price,
    p.first_price,
    ((p.current_price - p.first_price) / p.first_price) * 100 as change_pct,
    -- Sentiment analysis data (processed via native load balancing)
    n.sentiment,
    n.recommendation,
    n.confidence,
    n.explanation,
    n.headline
FROM price_data p
LEFT JOIN (
    -- Get the most recent sentiment analysis for each ticker
    SELECT ticker, sentiment, recommendation, confidence, explanation, headline,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analyzed_at DESC) as rn
    FROM News.breaking_news
    WHERE analyzed_at >= now() - INTERVAL 1 HOUR
    AND sentiment != '' AND recommendation != ''
) n ON p.ticker = n.ticker AND n.rn = 1
WHERE ((p.current_price - p.first_price) / p.first_price) * 100 >= 5.0 
AND dateDiff('second', p.first_timestamp, p.current_timestamp) <= 30
-- CRITICAL: Only trigger if recommendation is 'BUY' with 'high' confidence
AND (n.recommendation = 'BUY' AND n.confidence = 'high')
```

### System Initialization
Sentiment analysis with native load balancing is initialized as part of the main system startup:

**System Startup Integration** (`run_system.py`):
```python
async def initialize_sentiment_service():
    """Initialize the sentiment analysis service with native load balancing"""
    logging.info("🧠 Initializing sentiment analysis service with native load balancing...")
    
    try:
        # Get sentiment service instance (now with native load balancing)
        service = await get_sentiment_service()
        
        # Test connection to Claude API (automatically uses load balancing if available)
        is_connected = await service.test_connection()
        
        if is_connected:
            stats = service.get_stats()
            if stats['load_balancing_enabled']:
                lb_stats = stats['load_balancing_stats']
                logging.info("✅ Sentiment analysis service initialized with NATIVE LOAD BALANCING")
                logging.info(f"🔑 Load balancing across {lb_stats['total_keys']} API keys")
                logging.info("🤖 AI-powered sentiment analysis with automatic failover is ACTIVE")
            else:
                logging.info("✅ Sentiment analysis service initialized with SINGLE API KEY")
                logging.info("🤖 AI-powered sentiment analysis is ACTIVE")
            return True
        else:
            logging.error("❌ Sentiment analysis service failed to initialize")
            return False
            
    except Exception as e:
        logging.error(f"Error initializing sentiment service: {e}")
        return False
```

### Performance Characteristics

#### Native Load Balancing Performance
- **Analysis Time**: ~2-5 seconds per article (depends on API latency)
- **Batch Processing**: Multiple articles analyzed in parallel (up to 20 concurrent requests)
- **Load Distribution**: Automatic round-robin across available keys
- **Rate Limit Handling**: Immediate key switching on 429 errors
- **Cache Hit Rate**: ~30-50% for similar content
- **Fallback Time**: <100ms when analysis fails
- **Memory Usage**: ~60MB additional for load balancer (vs ~50MB for single key)
- **Rate Limits**: 40,000 tokens/minute × number of API keys
- **Concurrent Requests**: Up to 20 simultaneous API calls distributed across keys

#### Database Impact
- **Schema Enhancement**: Added 6 sentiment fields to `breaking_news` table
- **Index Optimization**: New indexes on sentiment and recommendation fields
- **Storage Increase**: ~20% additional storage for sentiment data
- **Query Performance**: Optimized sentiment-based queries with proper indexing

### Data Flow with Native Load Balancing

#### Enhanced News Processing Flow
```
News Source → Article Extraction → Native Load Balancing → Sentiment Analysis → Database Insert → File Trigger
     0s              ~1s                    ~0.1s               ~2s              ~2.5s         ~3s
```

#### Load Balancing Request Flow
```
Article → Load Balancer → Key Selection → API Request → Response → Next Article (Different Key)
  ~0s         ~0.01s          ~0.01s         ~2s          ~0.1s           ~0s
```

#### Sentiment-Enhanced Alert Flow
```
File Trigger → Price Check → Sentiment Lookup → Alert Decision → Database Insert
     0s           ~2s            ~0.1s           ~0.1s          ~2.5s
```

### Error Handling and Resilience

#### Native Load Balancing Fallbacks
- **Key Rate Limited**: Automatically switch to next available key
- **All Keys Rate Limited**: Use least recently rate-limited key
- **API Service Unavailable**: Articles processed without sentiment (default values)
- **Analysis Timeout**: 180-second timeout with graceful fallback
- **Malformed Responses**: JSON parsing errors handled gracefully
- **Network Issues**: Retry logic with exponential backoff per key
- **Load Balancer Failure**: Automatic fallback to single-key legacy mode

#### Alert System Resilience
- **Missing Sentiment Data**: Backward compatibility maintained
- **Confidence Degradation**: Only high-confidence BUY signals trigger alerts
- **Data Quality**: Sentiment analysis results validated before use

### Configuration and Tuning

#### Native Load Balancing Configuration
```python
# Environment Variables (automatic detection)
ANTHROPIC_API_KEY=sk-ant-primary-key      # Required: Primary key
ANTHROPIC_API_KEY2=sk-ant-secondary-key   # Optional: Secondary key
ANTHROPIC_API_KEY3=sk-ant-tertiary-key    # Optional: Tertiary key
# ... up to ANTHROPIC_API_KEY9

# Load Balancer Settings
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
CLAUDE_TIMEOUT = 180  # seconds
CLAUDE_MAX_WORKERS = 20  # concurrent requests
CLAUDE_CACHE_SIZE = 1000  # cached analyses
RATE_LIMIT_RESET_DELAY = 60  # seconds to wait after rate limit
```

#### Alert Threshold Configuration
```python
# Price Alert Thresholds (with sentiment)
PRICE_CHANGE_THRESHOLD = 0.05  # 5% price change required
SENTIMENT_REQUIREMENT = "BUY"  # Only BUY recommendations
CONFIDENCE_REQUIREMENT = "high"  # Only high confidence analyses
SENTIMENT_LOOKBACK = 1  # hour (how far back to look for sentiment)
```

### Monitoring and Metrics

#### Native Load Balancing Metrics
- **Total API Keys**: Count of configured API keys
- **Available Keys**: Count of non-rate-limited keys
- **Rate Limited Keys**: Count of currently rate-limited keys
- **Key Switches**: Number of automatic key switches due to rate limits
- **Per-Key Statistics**: Individual success rates, request counts, and failure rates
- **Load Distribution**: Request distribution across keys
- **Failover Events**: Count of automatic failovers
- **Recovery Events**: Count of rate-limit recoveries

#### Enhanced Alert Metrics
- **Sentiment-Filtered Alerts**: Alerts triggered with sentiment conditions
- **Blocked Alerts**: Price moves blocked by sentiment filter
- **Sentiment Distribution**: Distribution of BUY/SELL/HOLD recommendations
- **Confidence Levels**: Distribution of high/medium/low confidence analyses
- **Load Balancing Efficiency**: Success rate improvement from load balancing

### Command Line Options

#### Sentiment-Related Flags
```bash
# Skip sentiment analysis initialization (for testing)
python3 run_system.py --skip-list --no-sentiment

# Normal operation with native load balancing sentiment analysis (default)
python3 run_system.py --skip-list

# WebSocket mode with native load balancing sentiment analysis
python3 run_system.py --skip-list --socket
```

### Migration from External Gateways

#### Benefits Over External Gateway Solutions
1. **Zero Infrastructure**: No external servers, no Node.js dependencies
2. **No Deployment Complexity**: No additional processes to manage
3. **Better Performance**: Direct API calls without proxy overhead
4. **Enhanced Monitoring**: Detailed per-key statistics and health tracking
5. **Automatic Failover**: Built-in intelligence for key management
6. **Cost Effective**: No additional infrastructure costs
7. **Simpler Maintenance**: Single Python codebase

#### Migration Steps
1. **Add Additional API Keys**: Set `ANTHROPIC_API_KEY2`, `ANTHROPIC_API_KEY3`, etc.
2. **Restart System**: Native load balancing auto-detects and enables
3. **Monitor Performance**: Check load balancing stats in logs
4. **Remove External Dependencies**: No longer need Portkey Gateway or similar services

### Future Enhancements

#### Planned Load Balancing Improvements
1. **Weighted Load Balancing**: Different weights for different key tiers
2. **Geographic Distribution**: Route requests based on key geographic regions
3. **Cost Optimization**: Automatically use least expensive keys first
4. **Health Monitoring**: Advanced key health checks and performance monitoring
5. **Dynamic Key Management**: Hot-swap keys without service restart

#### Advanced Alert Logic
1. **Sentiment Momentum**: Consider sentiment changes over time
2. **Multi-Factor Alerts**: Combine sentiment with technical indicators
3. **Confidence Weighting**: Different thresholds for different confidence levels
4. **Sector-Specific Models**: Specialized sentiment models by industry

This native load balancing integration transforms NewsHead's sentiment analysis from a single-point-of-failure system into a robust, high-throughput, and highly available service that can handle enterprise-scale news processing without external dependencies.

### Ticker Universe Management

#### `finviz_scraper.py` - Ticker Universe Maintenance
**Purpose**: Maintains low-float stock universe from Finviz Elite
**Technology**: aiohttp with BeautifulSoup parsing

**Screening Criteria**:
- **Geographic**: Global (USA, Asia, Latin America, Europe, BRIC countries, and many other regions)
- **Sectors**: Healthcare, Technology, Industrials, Consumer, Communications, Energy, Basic Materials
- **Float**: Under 100M shares
- **Price**: Under $10

**Features**:
- **Elite Login**: Automated Finviz Elite authentication
- **Multi-page Scraping**: Handles paginated results
- **Data Validation**: Parses financial metrics (float, price, volume)
- **Database Integration**: Updates ClickHouse ticker tables

### Logging and Monitoring

#### `log_manager.py` - Comprehensive Logging
**Purpose**: System-wide logging with automatic rotation and cleanup

**Features**:
- **Daily Rotation**: Automatic log file rotation
- **5-Day Retention**: Automatic cleanup of old logs
- **Multi-Handler**: File and console output
- **System Information**: Logs platform, memory, CPU details
- **Startup/Shutdown Banners**: Clear system lifecycle tracking

**Log Structure**:
```
logs/
├── run_system.log.YYYY-MM-DD     # Main system logs
├── articles/
│   ├── article_tracking.log      # Article processing logs
│   └── article_events.log        # Article event logs
└── clickhouse_operations.log     # Database operation logs
```

### Supporting Components

#### `news_checker.py` - News Validation
**Purpose**: Validates news detection and timing
**Features**:
- **60-second window**: Checks for news in last minute
- **Timezone handling**: Proper ET/UTC conversion
- **Batch processing**: Handles multiple tickers efficiently

#### `main.py` - Legacy Entry Point
**Purpose**: Alternative entry point for RSS-only monitoring
**Status**: Legacy component, superseded by `run_system.py`

## Data Flow Architecture

### 1. WebSocket News Detection Flow (NEW)
```
Benzinga WebSocket → Message Parse → Securities Extract → Ticker Filter → Database Insert → File Trigger
```

**Timing**: ~100-500ms from WebSocket message to trigger file creation

### 2. Traditional News Detection Flow
```
Newswire Sources → Web Scraper → Ticker Extraction → Freshness Filter → Database Insert → File Trigger
```

**Timing**: ~2-5s from detection to trigger file creation

### 3. Price Monitoring Flow (Same for Both)
```
File Trigger → Price Checker → API Call → Price Analysis → Alert Generation → Database Insert
```

**Timing**: ~2-3 seconds from trigger to price data

### 4. End-to-End Latency Comparison

**WebSocket Mode**:
```
News Published → WebSocket Receive → Ticker Extract → Trigger Create → Price Check → Alert Generate
     0s               ~0.1s              ~0.2s          ~0.3s         ~2.5s        ~3-5s
```