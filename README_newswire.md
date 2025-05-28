# Newswire Breaking News Monitor

High-speed monitoring system for capturing breaking news from primary newswire sources (GlobeNewswire, BusinessWire, PRNewswire) with ClickHouse database storage.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup ClickHouse Database
- Install ClickHouse server locally or use cloud instance
- Update `.env` file with ClickHouse connection details (see `env_template.txt`)

### 3. Configure Environment
Copy `env_template.txt` to `.env` and update:
```bash
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password
```

### 4. Test Setup
```bash
python test_setup.py
```

### 5. Start Monitoring
```bash
python newswire_monitor.py
```

## Architecture

### Components
- **`clickhouse_setup.py`**: Database schema and connection management
- **`newswire_monitor.py`**: Core monitoring engine with async RSS polling
- **`test_setup.py`**: Validation tests for all components

### Performance Features
- **1-2 second polling intervals** for primary sources
- **Async concurrent monitoring** of all sources
- **Pre-compiled regex patterns** for fast ticker extraction
- **Batch ClickHouse inserts** every 5 seconds
- **Duplicate detection** via content hashing
- **Real-time performance tracking**

### Data Storage
Database: `News.breaking_news`
- Optimized for time-series queries
- Indexed on ticker, timestamp, source
- Automatic partitioning by month
- Performance tracking columns

## Monitoring Sources

### Primary (1-second polling)
- GlobeNewswire RSS
- BusinessWire RSS  
- MarketWatch Bulletins

### Secondary (2-second polling)
- PRNewswire RSS

## Expected Performance

### Speed Targets
- RSS publish → Database: **< 5 seconds**
- Ticker extraction: **< 100ms per article**
- Database write: **< 1 second per batch**

### Coverage
- **99%+ capture rate** for breaking news
- **< 0.1% duplicate** articles
- **95%+ uptime** per source

## Example: SPRO Case Study

The system would capture the SPRO announcement from [this morning's GlobeNewswire release](https://elite.finviz.com/news/67597/spero-therapeutics-and-gsk-announce-pivot-po-phase-3-study-for-tebipenem-hbr-stopped-early-for-efficacy-following-review-by-independent-data-monitoring-committee) within 1-3 seconds of RSS publication.

Stored data includes:
- Full headline and content
- Publication timestamp
- Source tracking
- Processing latency metrics
- News classification (clinical_trial)
- Urgency scoring

## Monitoring

View real-time stats in logs:
```
STATS - Runtime: 300.0s, Processed: 45, Inserted: 42, Duplicates: 3, Errors: 0, Rate: 0.15 articles/sec
```

Query ClickHouse for analysis:
```sql
SELECT source, COUNT(*) as articles, AVG(processing_latency_ms) as avg_latency
FROM News.breaking_news 
WHERE timestamp >= now() - INTERVAL 1 HOUR
GROUP BY source;
```

## Next Steps

1. **Price Integration**: Connect to price checking logic
2. **Enhanced Classification**: ML-based news categorization  
3. **Alert System**: Real-time notifications for high-urgency news
4. **Web Dashboard**: Real-time monitoring interface 