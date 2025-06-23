# News & Price Monitoring System - Zero-Lag Architecture

A high-performance real-time stock market news monitoring and price tracking system that achieves **sub-10-second** news-to-alert latency through process isolation, file-based triggers, and aggressive optimization techniques.

## 🚀 System Overview

This system implements an ultra-fast news-to-trading workflow optimized for minimal latency:

1. **Ticker Universe Management**: Maintains low-float stock universe from Finviz Elite screener
2. **Real-Time News Monitoring**: Scrapes major newswires with sub-second detection
3. **Zero-Lag Ticker Matching**: Instant ticker extraction with file-based triggers
4. **Process-Isolated Price Tracking**: Separate process eliminates resource contention
5. **Ultra-Fast Alert Generation**: Sub-10-second end-to-end latency

## 🏗️ Zero-Lag Architecture

### **Process Isolation Design**
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

### **Data Flow - Optimized for Speed**
1. **News Detection** → Web scraper finds ticker matches (500ms buffer)
2. **File Trigger** → Creates immediate trigger file (< 1ms)
3. **Process Communication** → Isolated price checker detects trigger (< 100ms)
4. **Price Monitoring** → Starts 2-second polling cycle (immediate)
5. **Alert Generation** → Creates alerts on price movements (< 2s)

**Total Latency: 6-10 seconds** (previously 60+ seconds)

## 📁 Core System Files

### `run_system.py` - Process Orchestrator
- **Purpose**: Manages process isolation and system startup
- **Key Features**:
  - **Process Isolation**: Runs price checker in separate subprocess
  - **Sequential Startup**: Price checker first, then browser (eliminates resource contention)
  - **Graceful Shutdown**: Proper cleanup of all processes
  - **Database Reset**: Complete pipeline table clearing on startup

### `web_scraper.py` - News Collection Engine
- **Purpose**: Real-time newswire monitoring with CPU optimization
- **Technology**: Crawl4AI with Chromium browser
- **News Sources**:
  - GlobeNewswire, BusinessWire, PR Newswire, AccessNewswire
- **Key Optimizations**:
  - **Ultra-Fast Buffer**: 500ms flush intervals (vs 3s previously)
  - **Freshness Filter**: 2-minute timezone-agnostic filtering
  - **CPU Efficiency**: Optimized browser flags without throttling speed
  - **File Triggers**: Immediate trigger file creation on ticker matches

### `price_checker.py` - Zero-Lag Price Monitor
- **Purpose**: Process-isolated price tracking with sub-second response
- **Data Source**: Polygon.io API with proxy support
- **Architecture**:
  - **File Trigger Monitor**: Processes trigger files for immediate ticker addition
  - **Continuous Polling**: 2-second cycles for all active tickers
  - **Dual System**: File triggers for notifications + polling for price inserts
- **Performance Features**:
  - **Aggressive Timeouts**: 2s total, 0.5s connect, 1.5s read
  - **Parallel Processing**: Bulk API calls for multiple tickers
  - **Smart Caching**: 10-second database query cache
  - **Zero Database Interference**: File-based ticker notifications only

### `clickhouse_setup.py` - Database Management
- **Purpose**: Optimized database operations with performance focus
- **Key Features**:
  - **Pipeline Reset**: Complete table clearing for fresh starts
  - **Trigger File Creation**: File-based notification system
  - **Optimized Queries**: No FINAL clauses, limited time windows
  - **Batch Operations**: Efficient bulk inserts

## 🎯 Performance Optimizations

### **Latency Improvements**
| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| News Buffer | 3s | 0.5s | **6x faster** |
| Ticker Notifications | 20s database scan | <1ms file trigger | **20,000x faster** |
| Price Monitoring | 5s intervals | 2s intervals | **2.5x faster** |
| Alert Generation | 60+ seconds | 6-10 seconds | **10x faster** |

### **CPU Usage Optimization**
- **Browser Efficiency**: Optimized flags without speed throttling
- **Process Isolation**: Eliminates resource contention between browser and API
- **Smart Resource Limits**: 512MB memory, 2 concurrent sessions
- **Efficient Delays**: Strategic delays for CPU breathing without speed loss

### **API Performance**
- **Consistent Speed**: 0.137-0.188s API calls (vs 16+ second timeouts)
- **Bulk Operations**: Parallel ticker processing
- **Timeout Optimization**: Matches polling interval to prevent blocking
- **Proxy Support**: High-performance proxy integration

## 🗄️ Database Schema

### News.breaking_news
```sql
CREATE TABLE News.breaking_news (
    id UUID DEFAULT generateUUIDv4(),
    timestamp DateTime64(3) DEFAULT now(),
    source String,
    ticker String,
    headline String,
    published_utc String,
    article_url String,
    summary String,
    full_content String,
    detected_at DateTime64(3) DEFAULT now(),
    processing_latency_ms UInt32,
    content_hash String
) ENGINE = MergeTree()
ORDER BY (ticker, timestamp)
PARTITION BY toYYYYMM(timestamp)
```

### News.price_tracking
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

### News.news_alert
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

## 🔧 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **ClickHouse Server** (running and accessible)
- **Finviz Elite** subscription (optional for ticker updates)
- **Polygon.io API** key (for price data)

### Installation Steps

1. **Clone Repository**
```bash
git clone [repository-url]
cd newshead
```

2. **Virtual Environment**
```bash
python -m venv news-env
source news-env/bin/activate  # Linux/Mac
# or
news-env\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Browser Setup (Crawl4AI)**
```bash
playwright install chromium
```

5. **Environment Configuration**
```bash
cp env_template.txt .env
# Edit .env with your credentials
```

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

## 🚀 Usage

### **Full System Launch (Recommended)**
```bash
python run_system.py
```
- Starts both news monitoring and price checking with process isolation
- Automatically sets up database and creates necessary tables
- Optimal for production use

### **Skip Ticker List Update**
```bash
python run_system.py --skip-list
```
- Uses existing ticker data without Finviz update
- Faster startup for testing

### **Enable Old News Processing (Testing)**
```bash
python run_system.py --enable-old
```
- Disables 2-minute freshness filter
- Useful for testing with historical data

### **Individual Components**

#### Price Monitor Only
```bash
python price_checker.py
```
- Runs price monitoring in isolation
- Useful for testing API connectivity

#### News Monitor Only
```bash
python web_scraper.py
```
- Runs news scraping without price monitoring
- Useful for testing news detection

#### Database Management
```bash
python clickhouse_setup.py  # Setup/reset all tables
python check_db.py         # View database contents
python drop_table.py       # Clean specific tables
```

## 📊 System Monitoring

### **Real-Time Performance Metrics**
- **News Processing**: Articles/minute, detection latency
- **Price Monitoring**: API response times, active tickers
- **Alert Generation**: Alerts/hour, end-to-end latency
- **Resource Usage**: CPU, memory, network utilization

### **Log Analysis**
```bash
# Monitor system performance
tail -f logs/system.log

# Check API performance
grep "API call completed" logs/price_checker.log

# Monitor news detection
grep "TICKER MATCH" logs/web_scraper.log
```

### **Health Checks**
```bash
# Database connectivity
python check_db.py

# API connectivity
python -c "from price_checker import ContinuousPriceMonitor; import asyncio; asyncio.run(ContinuousPriceMonitor().test_api_connectivity())"

# Trigger file system
ls -la triggers/
```

## 🎯 File Trigger System

### **Trigger File Structure**
```json
{
  "ticker": "EXAMPLE",
  "timestamp": "2024-01-01T12:00:00",
  "news_headline": "Breaking News Headline",
  "article_url": "https://...",
  "source": "GlobeNewswire",
  "detected_at": "2024-01-01 12:00:00.123"
}
```

### **Trigger Directory**
- **Location**: `triggers/immediate_*.json`
- **Creation**: Automatic on ticker detection
- **Processing**: Sub-100ms by price monitor
- **Cleanup**: Automatic after processing

## 🔍 Testing & Debugging

### **Performance Testing**
```bash
# Test polling loop performance
python testfiles/test_polling_loop.py

# Test file trigger system
python testfiles/test_file_triggers.py

# End-to-end latency test
python testfiles/test_end_to_end_lag.py
```

### **Debug Tools**
```bash
# Test ticker extraction
python debug_ticker_extraction.py

# Test news feed access
python debug_feeds.py

# Test database connectivity
python check_db.py
```

## ⚡ Performance Characteristics

### **Throughput**
- **News Processing**: 2000+ articles/hour
- **Ticker Matching**: <10ms per article
- **Price Updates**: 200+ tickers/minute
- **Database Writes**: 10,000+ records/minute

### **Latency**
- **News Detection**: 0.5-2 seconds from publication
- **Trigger Creation**: <1ms after detection
- **Price Response**: 2-second polling cycle
- **Alert Generation**: 6-10 seconds total

### **Resource Usage**
- **Memory**: 800MB typical, 1.2GB peak
- **CPU**: 20-40% of single core (optimized)
- **Network**: 50MB/hour (including price data)
- **Storage**: 2GB/week (with TTL policies)

## 🔧 Troubleshooting

### **Common Issues**

#### High CPU Usage
- **Symptom**: >80% CPU usage
- **Solution**: System is optimized for speed, not CPU usage
- **Note**: High CPU is expected for maximum news detection speed

#### API Timeouts
- **Symptom**: "BULK TIMEOUT" warnings
- **Solution**: Check proxy configuration and network connectivity
- **Debug**: Test with `python price_checker.py` in isolation

#### Missing Trigger Files
- **Symptom**: No price monitoring for detected news
- **Solution**: Check `triggers/` directory permissions
- **Debug**: Monitor with `ls -la triggers/` during operation

#### Database Connection Issues
- **Symptom**: ClickHouse connection errors
- **Solution**: Verify ClickHouse server status and credentials
- **Debug**: Test with `python check_db.py`

### **Performance Tuning**

#### For Maximum Speed
- Use process isolation (default in `run_system.py`)
- Ensure adequate memory (>2GB available)
- Use SSD storage for database and trigger files
- Configure high-performance proxy if using Polygon API

#### For Resource Conservation
- Reduce browser concurrent sessions in `web_scraper.py`
- Increase polling intervals in `price_checker.py`
- Implement ticker filtering to reduce monitoring scope

## 🔄 System Architecture Evolution

### **Previous Architecture Issues**
- **Sequential Processing**: 60+ second latencies
- **Resource Contention**: Browser and API competing for resources
- **Database Bottlenecks**: Complex queries blocking operations
- **Stale Data Processing**: Old news triggering unnecessary alerts

### **Current Zero-Lag Architecture**
- **Process Isolation**: Complete separation of concerns
- **File-Based Communication**: Ultra-fast inter-process messaging
- **Optimized Queries**: Minimal database overhead
- **Freshness Filtering**: Only process actionable news

### **Key Innovations**
1. **File Trigger System**: Sub-millisecond notification mechanism
2. **Process Isolation**: Eliminates all resource contention
3. **Aggressive Optimization**: Speed-first design philosophy
4. **Smart Caching**: Reduces database load without sacrificing speed

## 📈 Success Metrics

### **Latency Achievements**
- **News to Database**: 0.5 seconds (vs 3+ seconds)
- **Ticker Notification**: <1ms (vs 20+ seconds)
- **Price Monitoring Start**: 2 seconds (vs 30+ seconds)
- **Total Alert Time**: 6-10 seconds (vs 60+ seconds)

### **Reliability Improvements**
- **API Success Rate**: >99% (vs <80% with timeouts)
- **News Detection Rate**: >95% of relevant articles
- **System Uptime**: 24/7 operation capability
- **Resource Stability**: Consistent performance under load

## 📄 License & Disclaimer

This system is for educational and research purposes. Users are responsible for:
- Complying with data provider terms of service
- Ensuring appropriate trading risk management
- Verifying all trading decisions independently
- Monitoring system performance and accuracy

**No warranty is provided for trading accuracy or system reliability.**

---

## 🚀 Quick Start Guide

1. **Setup**: `cp env_template.txt .env` and configure credentials
2. **Install**: `pip install -r requirements.txt && playwright install chromium`
3. **Run**: `python run_system.py`
4. **Monitor**: Watch console output for news detection and price monitoring
5. **Test**: Check `triggers/` directory for trigger file creation

The system will automatically detect news, create trigger files, and begin price monitoring with sub-10-second latency. 