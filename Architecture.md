# NewsHead - Zero-Lag News & Price Monitoring System Architecture

## Overview

NewsHead is a high-performance real-time stock market news monitoring and price tracking system designed to achieve **sub-10-second** news-to-alert latency. The system implements a zero-lag architecture through process isolation, file-based triggers, and aggressive optimization techniques. The system now supports **dual news collection modes**: traditional web scraping and real-time WebSocket streaming via Benzinga's API.

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

**Command Line Options**:
- `--skip-list`: Skip Finviz ticker list update
- `--enable-old`: Disable freshness filtering for testing
- `--socket`: **NEW** - Use Benzinga WebSocket instead of web scraper
- `--any`: **NEW** - Process any ticker symbols found (WebSocket only, bypasses database filtering)

**News Source Selection Logic**:
```python
if args.socket:
    # WebSocket Mode - Real-time streaming
    from benzinga_websocket import Crawl4AIScraper
    scraper = Crawl4AIScraper(enable_old=enable_old, process_any_ticker=process_any_ticker)
else:
    # Web Scraper Mode - Traditional scraping
    from web_scraper import Crawl4AIScraper  
    scraper = Crawl4AIScraper(enable_old=enable_old)
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
The NewsHead system includes a comprehensive sentiment analysis engine that analyzes news articles using LM Studio's local AI models before database insertion. This enables intelligent price alerts that only trigger when both price movements AND favorable sentiment conditions are met.

### Architecture Components

#### `sentiment_service.py` - AI-Powered Sentiment Analysis Service
**Purpose**: Real-time sentiment analysis of news articles using local LM Studio API
**Technology**: LM Studio local AI models with async processing

**Key Features**:
- **Local AI Processing**: Uses LM Studio API for privacy and speed
- **Batch Processing**: Analyzes multiple articles simultaneously
- **Intelligent Caching**: Avoids re-analyzing identical content
- **Fallback Handling**: Graceful degradation when AI analysis fails
- **Performance Tracking**: Detailed statistics and timing metrics

**Sentiment Analysis Workflow**:
```python
# Article Analysis Process
Article → Content Hash Check → AI Analysis → Response Parsing → Database Enrichment
```

**AI Model Integration**:
- **API Endpoint**: `http://localhost:1234/v1/chat/completions`
- **Model Type**: Local LM Studio model (configurable)
- **Analysis Prompt**: Financial news sentiment analysis with BUY/SELL recommendations
- **Response Format**: Structured JSON with sentiment, recommendation, confidence, and explanation

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

**Possible Label Values**:
- **Sentiment**: `"positive"`, `"negative"`, `"neutral"`
- **Recommendation**: `"BUY"`, `"SELL"`, `"HOLD"`
- **Confidence**: `"high"`, `"medium"`, `"low"`

### Integration Points

#### News Pipeline Integration
Both news collection systems now include sentiment analysis:

**Web Scraper Integration** (`web_scraper.py`):
```python
async def flush_buffer_to_clickhouse(self):
    """Flush article buffer to ClickHouse WITH sentiment analysis"""
    if self.batch_queue:
        # SENTIMENT ANALYSIS BEFORE DATABASE INSERTION
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
    """Flush article buffer to ClickHouse WITH sentiment analysis (same as web_scraper.py)"""
    if self.batch_queue:
        # SENTIMENT ANALYSIS BEFORE DATABASE INSERTION
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
-- SENTIMENT-ENHANCED PRICE ALERTS
-- Only trigger alerts when:
-- 1. Price moves 5%+ within 30 seconds (existing logic)
-- 2. AND sentiment is 'BUY' with 'high' confidence (NEW requirement)

SELECT 
    p.ticker,
    p.current_price,
    p.first_price,
    ((p.current_price - p.first_price) / p.first_price) * 100 as change_pct,
    -- Sentiment analysis data
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
Sentiment analysis is initialized as part of the main system startup:

**System Startup Integration** (`run_system.py`):
```python
async def initialize_sentiment_service():
    """Initialize the sentiment analysis service"""
    logging.info("🧠 Initializing sentiment analysis service...")
    
    try:
        # Get sentiment service instance
        service = await get_sentiment_service()
        
        # Test connection to LM Studio
        is_connected = await service.test_connection()
        
        if is_connected:
            logging.info("✅ Sentiment analysis service initialized successfully")
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

#### Sentiment Analysis Performance
- **Analysis Time**: ~1-3 seconds per article (depends on model size)
- **Batch Processing**: Multiple articles analyzed in parallel
- **Cache Hit Rate**: ~30-50% for similar content
- **Fallback Time**: <100ms when analysis fails
- **Memory Usage**: ~100MB additional for AI service

#### Database Impact
- **Schema Enhancement**: Added 6 sentiment fields to `breaking_news` table
- **Index Optimization**: New indexes on sentiment and recommendation fields
- **Storage Increase**: ~20% additional storage for sentiment data
- **Query Performance**: Optimized sentiment-based queries with proper indexing

### Data Flow with Sentiment Analysis

#### Enhanced News Processing Flow
```
News Source → Article Extraction → Sentiment Analysis → Database Insert → File Trigger
     0s              ~1s                 ~2s              ~2.5s         ~3s
```

#### Sentiment-Enhanced Alert Flow
```
File Trigger → Price Check → Sentiment Lookup → Alert Decision → Database Insert
     0s           ~2s            ~0.1s           ~0.1s          ~2.5s
```

### Error Handling and Resilience

#### Sentiment Analysis Fallbacks
- **AI Service Unavailable**: Articles processed without sentiment (default values)
- **Analysis Timeout**: 30-second timeout with graceful fallback
- **Malformed Responses**: JSON parsing errors handled gracefully
- **Network Issues**: Retry logic with exponential backoff

#### Alert System Resilience
- **Missing Sentiment Data**: Backward compatibility maintained
- **Confidence Degradation**: Only high-confidence BUY signals trigger alerts
- **Data Quality**: Sentiment analysis results validated before use

### Configuration and Tuning

#### Sentiment Service Configuration
```python
# LM Studio API Configuration
SENTIMENT_SERVICE_URL = "http://localhost:1234/v1/chat/completions"
SENTIMENT_TIMEOUT = 30  # seconds
SENTIMENT_BATCH_SIZE = 5  # articles per batch
SENTIMENT_CACHE_SIZE = 1000  # cached analyses
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

#### Sentiment Analysis Metrics
- **Total Articles Analyzed**: Count of articles processed
- **Analysis Success Rate**: Percentage of successful analyses
- **Average Analysis Time**: Time per article analysis
- **Cache Hit Rate**: Percentage of cache hits vs new analyses
- **AI Service Uptime**: Availability of LM Studio API

#### Enhanced Alert Metrics
- **Sentiment-Filtered Alerts**: Alerts triggered with sentiment conditions
- **Blocked Alerts**: Price moves blocked by sentiment filter
- **Sentiment Distribution**: Distribution of BUY/SELL/HOLD recommendations
- **Confidence Levels**: Distribution of high/medium/low confidence analyses

### Command Line Options

#### Sentiment-Related Flags
```bash
# Skip sentiment analysis initialization (for testing)
python3 run_system.py --skip-list --no-sentiment

# Normal operation with sentiment analysis (default)
python3 run_system.py --skip-list

# WebSocket mode with sentiment analysis
python3 run_system.py --skip-list --socket
```

### Future Enhancements

#### Planned Sentiment Improvements
1. **Multi-Model Support**: Support for different AI models
2. **Custom Prompts**: Configurable analysis prompts
3. **Sentiment Scoring**: Numerical sentiment scores vs categorical
4. **Historical Analysis**: Sentiment trend analysis over time
5. **Model Fine-tuning**: Custom model training on financial news

#### Advanced Alert Logic
1. **Sentiment Momentum**: Consider sentiment changes over time
2. **Multi-Factor Alerts**: Combine sentiment with technical indicators
3. **Confidence Weighting**: Different thresholds for different confidence levels
4. **Sector-Specific Models**: Specialized sentiment models by industry

This sentiment analysis integration transforms NewsHead from a simple news monitoring system into an intelligent trading signal generator that combines real-time news detection with AI-powered sentiment analysis to produce high-quality, actionable alerts.

### Ticker Universe Management

#### `finviz_scraper.py` - Ticker Universe Maintenance
**Purpose**: Maintains low-float stock universe from Finviz Elite
**Technology**: aiohttp with BeautifulSoup parsing

**Screening Criteria**:
- **Geographic**: USA only
- **Sectors**: Healthcare, Technology, Industrials, Consumer, Communications, Energy, Basic Materials
- **Float**: Under 50M shares
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
**Total WebSocket Latency**: 3-5 seconds

**Web Scraper Mode**:
```
News Published → News Detected → Ticker Matched → Trigger Created → Price Checked → Alert Generated
     0s              ~2s            ~2.5s          ~3s            ~5s           ~6-10s
```
**Total Web Scraper Latency**: 6-10 seconds

## News Source Comparison

| Feature | Web Scraper | Benzinga WebSocket |
|---------|-------------|-------------------|
| **Latency** | 6-10 seconds | 3-5 seconds |
| **Data Quality** | Text parsing | Structured API data |
| **Reliability** | Browser dependent | WebSocket connection |
| **Resource Usage** | High (Chromium) | Low (WebSocket) |
| **News Sources** | 4 newswire sites | Benzinga feed |
| **Ticker Detection** | Regex patterns | API securities field |
| **Cost** | Free | API key required |
| **Freshness Filter** | 2-minute window | 2-minute window |
| **Any Ticker Mode** | No | Yes (--any flag) |

## File System Architecture

### Trigger System
```
triggers/
├── {TICKER}_{TIMESTAMP}.json     # Individual ticker triggers
└── immediate_triggers/           # High-priority triggers
```

**Trigger File Format**:
```json
{
  "ticker": "AAPL",
  "timestamp": "2025-01-15T10:30:00",
  "source": "Benzinga_WebSocket",  // or source website
  "headline": "Apple Announces...",
  "article_url": "https://..."
}
```

### Data Files
```
data_files/
└── FV_master_u50float_u10price.csv  # Ticker universe from Finviz
```

### Log Files
```
logs/
├── run_system.log.{date}         # Main system logs
├── articles/                     # Article tracking
├── clickhouse_operations.log     # Database logs
└── price_checker.log            # Price monitoring logs
```

## Configuration Management

### Environment Variables (.env)
```bash
# ClickHouse Database
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password

# Finviz Elite (optional)
FINVIZ_EMAIL=your_email@example.com
FINVIZ_PASSWORD=your_password

# Polygon API
POLYGON_API_KEY=your_api_key
PROXY_URL=your_proxy_url  # Optional

# Benzinga WebSocket API (NEW)
BENZINGA_API_KEY=your_benzinga_api_key_here

# LM Studio API for Sentiment Analysis (NEW)
LM_STUDIO_URL=http://localhost:1234/v1/chat/completions
LM_STUDIO_TIMEOUT=30  # seconds
LM_STUDIO_BATCH_SIZE=5  # articles per batch
LM_STUDIO_CACHE_SIZE=1000  # cached analyses

# Performance Tuning
CHECK_INTERVAL=1
MAX_AGE_SECONDS=90
BATCH_SIZE=50
LOG_LEVEL=INFO
```

## Performance Optimizations

### Latency Improvements
| Component | Web Scraper | WebSocket | Improvement |
|-----------|-------------|-----------|-------------|
| News Buffer | 0.5s | 0.25s | **2x faster** |
| News Detection | 2-5s | 0.1-0.5s | **10x faster** |
| Ticker Extraction | Regex parsing | Structured data | **Instant** |
| Overall Latency | 6-10s | 3-5s | **2x faster** |

### Resource Usage Optimization
| Resource | Web Scraper | WebSocket | Improvement |
|----------|-------------|-----------|-------------|
| CPU Usage | High (Chromium) | Low (WebSocket) | **10x lower** |
| Memory Usage | ~500MB | ~50MB | **10x lower** |
| Network | HTTP requests | Persistent connection | **More efficient** |
| Dependencies | Playwright, Chromium | websockets only | **Simpler** |

### API Performance
- **WebSocket**: Persistent connection, sub-second message delivery
- **Structured Data**: No regex processing overhead
- **Bulk Operations**: Parallel ticker processing maintained
- **Timeout Optimization**: Matches polling interval
- **Proxy Support**: Available for both modes

## Deployment Architecture

### System Requirements
- **Python 3.8+**
- **ClickHouse Server** (running and accessible)
- **LM Studio** (for sentiment analysis) - Local AI model server
- **Chromium Browser** (via Playwright) - Web Scraper mode only
- **WebSocket Support** - WebSocket mode only
- **4GB+ RAM** (Web Scraper) / **1GB+ RAM** (WebSocket) / **8GB+ RAM** (with sentiment analysis)
- **2+ CPU cores** (recommended) / **4+ CPU cores** (with AI sentiment analysis)

### Process Architecture
```
┌─────────────────────────────────────────┐
│              Host System                │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │         Main Process                │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │      run_system.py              │ │ │
│  │  │  ┌─────────────────────────────┐ │ │ │
│  │  │  │   News Monitor              │ │ │ │
│  │  │  │ ┌─────────────────────────┐ │ │ │ │
│  │  │  │ │ web_scraper.py          │ │ │ │ │
│  │  │  │ │ (Chromium Browser)      │ │ │ │ │
│  │  │  │ └─────────────────────────┘ │ │ │ │
│  │  │  │           OR                │ │ │ │
│  │  │  │ ┌─────────────────────────┐ │ │ │ │
│  │  │  │ │ benzinga_websocket.py   │ │ │ │ │
│  │  │  │ │ (WebSocket Client)      │ │ │ │ │
│  │  │  │ └─────────────────────────┘ │ │ │ │
│  │  │  └─────────────────────────────┘ │ │
│  │  └─────────────────────────────────┘ │
│  └─────────────────────────────────────┘ │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │       Isolated Process              │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │     price_checker.py            │ │ │
│  │  │   (Polygon API Client)          │ │ │
│  │  └─────────────────────────────────┘ │ │
│  └─────────────────────────────────────┘ │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │       ClickHouse Database           │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │     News Schema                 │ │ │
│  │  │  • breaking_news               │ │ │
│  │  │  • price_tracking              │ │ │
│  │  │  • news_alert                  │ │ │
│  │  │  • float_list                  │ │ │
│  │  │  └─────────────────────────────────┘ │ │
│  │  └─────────────────────────────────────┘ │
│  └─────────────────────────────────────────┘
```

## Usage Examples

### WebSocket Mode (Recommended for Speed)
```bash
# Real-time WebSocket with database ticker filtering
python run_system.py --socket

# WebSocket with any ticker processing (bypass database filter)
python run_system.py --socket --any

# WebSocket with old news processing for testing
python run_system.py --socket --enable-old

# Skip ticker list update and use WebSocket
python run_system.py --socket --skip-list
```

### Web Scraper Mode (Traditional)
```bash
# Traditional web scraping (default)
python run_system.py

# Web scraper with old news processing
python run_system.py --enable-old

# Skip ticker list update and use web scraper
python run_system.py --skip-list
```

### Standalone Testing
```bash
# Test Benzinga WebSocket directly
python benzinga_websocket.py --any --duration 5

# Test web scraper directly  
python web_scraper.py --enable-old
```

## Error Handling and Resilience

### WebSocket-Specific Error Handling
- **Connection Failures**: Automatic reconnection with exponential backoff
- **Message Parsing**: Graceful handling of malformed JSON
- **API Rate Limits**: Built-in respect for API limitations
- **Authentication Issues**: Clear error messages for API key problems

### Graceful Degradation
- **API Failures**: Multiple fallback strategies for price data
- **Browser Crashes**: Automatic restart and recovery (Web Scraper mode)
- **WebSocket Disconnections**: Auto-reconnection with state preservation
- **Database Issues**: Local buffering and retry logic
- **Network Issues**: Timeout handling and reconnection

### Monitoring and Alerting
- **Performance Stats**: Real-time processing metrics for both modes
- **Error Tracking**: Comprehensive error logging with mode-specific details
- **Health Checks**: System component status monitoring
- **Resource Monitoring**: CPU, memory, and network usage

## Development Patterns

### Async/Await Pattern
- **Concurrent Processing**: All I/O operations use async/await
- **Parallel Tasks**: Multiple news sources and price checks
- **Resource Management**: Proper cleanup with context managers

### WebSocket Pattern
```python
async def websocket_listener(self):
    while self.is_running:
        try:
            if not self.websocket:
                await self.connect_websocket()
            
            message = await self.websocket.recv()
            articles = self.process_benzinga_message(json.loads(message))
            
            if articles:
                self.batch_queue.extend(articles)
                
        except websockets.exceptions.ConnectionClosed:
            self.websocket = None
            await asyncio.sleep(5)  # Reconnection delay
```

### Error Handling Pattern
```python
try:
    # Operation
    result = await operation()
except SpecificException as e:
    logger.error(f"Specific error: {e}")
    # Specific handling
except Exception as e:
    logger.error(f"General error: {e}")
    # General handling
finally:
    # Cleanup
    await cleanup()
```

### Logging Pattern
```python
logger.info(f"Operation started: {context}")
try:
    result = await operation()
    logger.info(f"Operation completed: {result}")
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## Testing and Debugging

### Debug Modes
- `--enable-old`: Process historical news for testing (both modes)
- `--skip-list`: Skip ticker list updates for faster startup
- `--socket`: Use WebSocket mode instead of web scraper
- `--any`: Process any ticker symbols (WebSocket mode only)
- Debug logging levels for detailed operation tracking

### Testing Components
- `test_benzinga_websocket.py`: WebSocket connection and message testing
- `debug_*.py` files for specific component testing
- `check_db.py` for database validation
- Individual component testing via direct execution

### WebSocket Testing
```bash
# Test WebSocket connection and messages
python testfiles/test_benzinga_websocket.py --duration 5

# Test WebSocket with any ticker processing
python benzinga_websocket.py --any --duration 10
```

## Legacy Components

### `old_code/` Directory
Contains previous implementations and experimental code:
- `news_sources_scraper.py`: Original scraping implementation
- `price_check.py`: Legacy price checking logic
- `main_trigger.py`: Original trigger system

These are kept for reference but not used in production.

## Future Enhancements

### Planned Improvements
1. **Multiple WebSocket Sources**: Add more real-time news APIs
2. **Machine Learning Integration**: News sentiment analysis
3. **Advanced Alerting**: Webhook and notification systems
4. **Real-time Dashboard**: Web-based monitoring interface
5. **Multi-Exchange Support**: Expand beyond US markets
6. **High Availability**: Clustering and failover support

### WebSocket Enhancements
1. **Message Filtering**: Server-side filtering by ticker or sector
2. **Multiple Connections**: Parallel WebSocket connections for redundancy
3. **Custom Channels**: Subscribe to specific news channels
4. **Rate Limiting**: Smart throttling for high-volume periods

### Scalability Considerations
- **Horizontal Scaling**: Multiple scraper instances
- **Database Sharding**: Partition by ticker or time
- **Caching Layer**: Redis for frequently accessed data
- **Load Balancing**: Distribute API calls across proxies
- **WebSocket Clustering**: Multiple WebSocket connections with load balancing

This architecture document provides a comprehensive overview of the NewsHead system, including the new Benzinga WebSocket functionality that enables ultra-low latency news monitoring. The dual-mode architecture allows users to choose between traditional web scraping and real-time WebSocket streaming based on their latency requirements, resource constraints, and API access. 