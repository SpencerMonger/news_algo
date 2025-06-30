# NewsHead - Zero-Lag News & Price Monitoring System Architecture

## Overview

NewsHead is a high-performance real-time stock market news monitoring and price tracking system designed to achieve **sub-10-second** news-to-alert latency. The system implements a zero-lag architecture through process isolation, file-based triggers, and aggressive optimization techniques.

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
│  │   Web Scraper   │───▶│ File Triggers   │                │
│  │  (Chromium)     │    │   (triggers/)   │                │
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
- **Ultra-Fast Buffers**: 500ms flush intervals for immediate data processing
- **Aggressive Timeouts**: 2-second API timeouts matching polling intervals
- **Parallel Processing**: Concurrent news source monitoring and price checking
- **Smart Caching**: 10-second database query cache to reduce load

## System Components

### Core Entry Points

#### `run_system.py` - Process Orchestrator
**Purpose**: Main system controller that manages process isolation and startup sequence

**Key Responsibilities**:
- Database initialization and table setup
- Sequential startup (price checker first, then browser)
- Process lifecycle management
- Graceful shutdown handling
- Comprehensive logging setup

**Process Flow**:
1. Setup ClickHouse database with fresh tables
2. Start price checker in isolated subprocess
3. Wait for price checker initialization (10s)
4. Start news monitor with Chromium browser
5. Handle shutdown signals and cleanup

**Command Line Options**:
- `--skip-list`: Skip Finviz ticker list update
- `--enable-old`: Disable freshness filtering for testing

### News Collection Layer

#### `web_scraper.py` - Primary News Engine
**Technology**: Crawl4AI with Chromium browser automation
**Purpose**: Real-time newswire monitoring with CPU optimization

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
- **No Fallbacks**: Unreliable NBBO calculations removed entirely
- **Skip on Failure**: Failed requests are skipped to maintain clean intervals
- **Reliability Priority**: Only real trade prices prevent phantom price alerts (e.g., FEAM $4.05, CAPS $4.06 issues)

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
    urgency_score UInt8 DEFAULT 0
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
    source String DEFAULT 'polygon'
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

### 1. News Detection Flow
```
Newswire Sources → Web Scraper → Ticker Extraction → Freshness Filter → Database Insert → File Trigger
```

**Timing**: ~500ms from detection to trigger file creation

### 2. Price Monitoring Flow
```
File Trigger → Price Checker → API Call → Price Analysis → Alert Generation → Database Insert
```

**Timing**: ~2-3 seconds from trigger to price data

### 3. End-to-End Latency
```
News Published → News Detected → Ticker Matched → Trigger Created → Price Checked → Alert Generated
     0s              ~2s            ~2.5s          ~3s            ~5s           ~6-10s
```

**Total Latency**: 6-10 seconds (vs 60+ seconds previously)

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
  "source": "GlobeNewswire",
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

# Performance Tuning
CHECK_INTERVAL=1
MAX_AGE_SECONDS=90
BATCH_SIZE=50
LOG_LEVEL=INFO
```

## Performance Optimizations

### Latency Improvements
| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| News Buffer | 3s | 0.5s | **6x faster** |
| Ticker Notifications | 20s database scan | <1ms file trigger | **20,000x faster** |
| Price Monitoring | 5s intervals | 2s intervals | **2.5x faster** |
| Alert Generation | 60+ seconds | 6-10 seconds | **10x faster** |

### CPU Usage Optimization
- **Browser Efficiency**: Optimized flags without speed throttling
- **Process Isolation**: Eliminates resource contention
- **Smart Resource Limits**: 512MB memory, 2 concurrent sessions
- **Efficient Delays**: Strategic delays for CPU breathing

### API Performance
- **Consistent Speed**: 0.137-0.188s API calls
- **Bulk Operations**: Parallel ticker processing
- **Timeout Optimization**: Matches polling interval
- **Proxy Support**: High-performance proxy integration

## Deployment Architecture

### System Requirements
- **Python 3.8+**
- **ClickHouse Server** (running and accessible)
- **Chromium Browser** (via Playwright)
- **4GB+ RAM** (recommended)
- **2+ CPU cores** (recommended)

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
│  │  │  │     web_scraper.py          │ │ │ │
│  │  │  │   (Chromium Browser)        │ │ │ │
│  │  │  └─────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────┘ │ │
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

## Error Handling and Resilience

### Graceful Degradation
- **API Failures**: Multiple fallback strategies for price data
- **Browser Crashes**: Automatic restart and recovery
- **Database Issues**: Local buffering and retry logic
- **Network Issues**: Timeout handling and reconnection

### Monitoring and Alerting
- **Performance Stats**: Real-time processing metrics
- **Error Tracking**: Comprehensive error logging
- **Health Checks**: System component status monitoring
- **Resource Monitoring**: CPU, memory, and disk usage

## Development Patterns

### Async/Await Pattern
- **Concurrent Processing**: All I/O operations use async/await
- **Parallel Tasks**: Multiple news sources and price checks
- **Resource Management**: Proper cleanup with context managers

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
- `--enable-old`: Process historical news for testing
- `--skip-list`: Skip ticker list updates for faster startup
- Debug logging levels for detailed operation tracking

### Testing Components
- `debug_*.py` files for specific component testing
- `check_db.py` for database validation
- Individual component testing via direct execution

## Legacy Components

### `old_code/` Directory
Contains previous implementations and experimental code:
- `news_sources_scraper.py`: Original scraping implementation
- `price_check.py`: Legacy price checking logic
- `main_trigger.py`: Original trigger system

These are kept for reference but not used in production.

## Future Enhancements

### Planned Improvements
1. **Machine Learning Integration**: News sentiment analysis
2. **Advanced Alerting**: Webhook and notification systems
3. **Real-time Dashboard**: Web-based monitoring interface
4. **Multi-Exchange Support**: Expand beyond US markets
5. **High Availability**: Clustering and failover support

### Scalability Considerations
- **Horizontal Scaling**: Multiple scraper instances
- **Database Sharding**: Partition by ticker or time
- **Caching Layer**: Redis for frequently accessed data
- **Load Balancing**: Distribute API calls across proxies

This architecture document provides a comprehensive overview of the NewsHead system, enabling any developer or LLM to understand the codebase structure, data flow, and design decisions that enable the zero-lag news monitoring capability. 