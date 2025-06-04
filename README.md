# News & Price Monitoring System

A comprehensive real-time stock market news monitoring and price tracking system that scans newswires for breaking news, matches articles to relevant stock tickers, and monitors subsequent price movements to generate trading alerts.

## System Overview

This system implements a complete news-to-trading workflow:

1. **Ticker Universe Management**: Pulls and maintains a list of low-float stocks from Finviz Elite screener
2. **Real-Time News Monitoring**: Scrapes major newswires (GlobeNewswire, BusinessWire, PR Newswire) for breaking news
3. **Intelligent Ticker Matching**: Identifies relevant stock symbols mentioned in news articles
4. **Price Movement Tracking**: Monitors price movements for news-mentioned stocks
5. **Alert Generation**: Creates trading alerts when news correlates with significant price action

## Architecture Components

### Core System Files

#### `run_system.py` - Main Orchestrator
- **Purpose**: Entry point that coordinates all system components
- **Key Functions**: 
  - Initializes ClickHouse database
  - Updates ticker list from Finviz (optional with `--skip-list` flag)
  - Launches news and price monitoring systems concurrently
  - Handles graceful shutdown

#### `finviz_scraper.py` - Ticker Universe Manager
- **Purpose**: Maintains the list of tradeable stocks to monitor
- **Data Source**: Finviz Elite screener (low-float, sub-$10 price stocks)
- **Key Features**:
  - Authenticates with Finviz Elite account
  - Scrapes screener results across multiple pages
  - Parses stock data (ticker, price, float, volume, etc.)
  - Updates ClickHouse ticker database
  - Handles rate limiting and anti-detection measures

#### `web_scraper.py` - News Collection Engine
- **Purpose**: Real-time news article collection from major newswires
- **Technology**: Uses Crawl4AI for reliable web scraping
- **News Sources**:
  - GlobeNewswire: `https://www.globenewswire.com/en/search/date/24HOURS`
  - BusinessWire: `https://www.businesswire.com/newsroom`
  - PR Newswire: `https://www.prnewswire.com/news-releases/news-releases-list/`
- **Key Features**:
  - Concurrent scraping of multiple news sources
  - Intelligent ticker extraction (exact match, case-sensitive)
  - Duplicate detection via content hashing
  - Precise timestamp parsing from article metadata
  - Batch processing for database efficiency

#### `price_checker.py` - Price Movement Monitor
- **Purpose**: Tracks real-time price movements for news-mentioned stocks
- **Data Source**: Polygon.io API (with proxy support)
- **Key Features**:
  - Continuous monitoring of recently mentioned tickers
  - Real-time price data collection
  - Alert generation based on price movement thresholds
  - Performance statistics tracking
  - Automatic cleanup of stale monitoring data

#### `clickhouse_setup.py` - Database Management
- **Purpose**: Manages all database operations and schema
- **Database Tables**:
  - `breaking_news`: Stores all collected news articles with metadata
  - `float_list`: Maintains the ticker universe from Finviz
  - `monitored_tickers`: Tracks stocks currently being price-monitored
  - `price_tracking`: Real-time price data
  - `news_alert`: Generated trading alerts
- **Key Features**:
  - Optimized schema with proper indexing
  - Batch insert operations
  - Duplicate detection and deduplication
  - Data retention policies (TTL)
  - Performance monitoring

### Legacy/Alternative Components

#### `main.py` - Alternative RSS-Based System
- **Purpose**: RSS feed-based news monitoring (alternative to web scraping)
- **Status**: Legacy system, replaced by direct web scraping
- **Features**: RSS feed parsing, concurrent processing, CSV logging

#### `rss_news_monitor.py` - RSS Feed Handler
- **Purpose**: Handles RSS feed-based news collection
- **Status**: Backup system for web scraping
- **Features**: Multi-feed RSS parsing, ticker extraction, duplicate detection

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Finviz Elite  │───▶│ Ticker Database │───▶│  News Scanner   │
│   (Screener)    │    │   (ClickHouse)  │    │  (Web Scraper)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Trading Alerts │◄───│ Price Monitor   │◄───│ Breaking News   │
│   (Triggers)    │    │  (Real-time)    │    │   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Detailed Data Flow

1. **Ticker Collection**: 
   - `finviz_scraper.py` authenticates with Finviz Elite
   - Scrapes screener for low-float stocks (<50M float, <$10 price)
   - Stores ticker data in `News.float_list` table

2. **News Monitoring**:
   - `web_scraper.py` continuously scrapes major newswires
   - Extracts article metadata (headline, timestamp, URL)
   - Performs exact ticker matching against known ticker list
   - Stores matches in `News.breaking_news` table

3. **Price Tracking**:
   - `price_checker.py` scans for newly mentioned tickers
   - Adds them to `News.monitored_tickers` for tracking
   - Continuously fetches real-time prices via Polygon API
   - Stores price data in `News.price_tracking` table

4. **Alert Generation**:
   - System compares current prices to historical baselines
   - Generates alerts for significant price movements
   - Creates trigger files in `/triggers` directory
   - Logs alerts to `News.news_alert` table

## Database Schema

### News.breaking_news
```sql
- id: UUID (Primary Key)
- timestamp: DateTime64(3) (Article timestamp)
- source: String (News source)
- ticker: String (Matched ticker symbol)
- headline: String (Article title)
- published_utc: String (Original publication time)
- article_url: String (Source URL)
- summary: String (Article summary)
- full_content: String (Complete article text)
- detected_at: DateTime64(3) (System detection time)
- processing_latency_ms: UInt32 (Processing speed)
- content_hash: String (Duplicate detection)
```

### News.float_list
```sql
- ticker: String (Stock symbol)
- company: String (Company name)
- sector: String (Business sector)
- industry: String (Industry classification)
- country: String (Geographic location)
- market_cap: Float64 (Market capitalization)
- price: Float64 (Current price)
- change: Float64 (Price change)
- volume: UInt64 (Trading volume)
- float_shares: Float64 (Shares outstanding)
- last_updated: DateTime (Data freshness)
```

### News.monitored_tickers
```sql
- ticker: String (Stock being monitored)
- first_seen: DateTime (When first mentioned)
- news_headline: String (Triggering news headline)
- news_url: String (Source article URL)
- active: UInt8 (Monitoring status)
- last_updated: DateTime (Last update time)
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- ClickHouse database server
- Finviz Elite subscription
- Polygon.io API key (optional for price data)

### Installation

1. **Clone Repository**
```bash
git clone [repository-url]
cd news2
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

4. **Environment Configuration**
```bash
cp env_template.txt .env
# Edit .env with your credentials:
# - CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
# - FINVIZ_EMAIL, FINVIZ_PASSWORD
# - POLYGON_API_KEY (optional)
# - PROXY_URL (optional)
```

5. **Database Setup**
```bash
python clickhouse_setup.py
```

### Browser Dependencies (for Crawl4AI)
```bash
# Install Playwright browsers
playwright install chromium
```

## Usage

### Full System Launch
```bash
python run_system.py
```

### Skip Ticker Update (Use Cached Data)
```bash
python run_system.py --skip-list
```

### Individual Components

#### Update Ticker List Only
```bash
python finviz_scraper.py
```

#### News Monitoring Only
```bash
python web_scraper.py
```

#### Price Monitoring Only
```bash
python price_checker.py
```

#### Database Management
```bash
python clickhouse_setup.py  # Setup/verify tables
python check_db.py         # View database contents
python drop_table.py       # Clean database
```

## System Monitoring & Logs

### Log Files
- **System logs**: Console output with timestamps
- **Database logs**: ClickHouse query performance
- **Scraping logs**: Article processing statistics
- **Price logs**: API call performance

### Performance Metrics
- Articles processed per hour
- Ticker matching accuracy
- Price API response times
- Database insertion rates
- Memory and CPU usage

### Health Checks
```bash
python check_db.py  # Verify database connectivity and data
```

## Key Configuration Options

### Environment Variables
```bash
# Monitoring frequency
CHECK_INTERVAL=1          # Seconds between news checks
MAX_AGE_SECONDS=90        # Only process recent articles
BATCH_SIZE=50            # Database batch size

# Performance tuning
LOG_LEVEL=INFO           # Logging verbosity
```

### Screener Criteria (Finviz)
- **Geography**: USA only
- **Sectors**: Healthcare, Technology, Industrials, Consumer Defensive, Communication Services, Energy, Consumer Cyclical, Basic Materials
- **Float**: Under 50 million shares
- **Price**: Under $10 per share

### News Sources Priority
1. **GlobeNewswire**: Primary source for press releases
2. **BusinessWire**: Secondary corporate news
3. **PR Newswire**: Additional press release coverage

## Alert System

### Trigger Generation
- Alerts saved to `/triggers` directory as JSON files
- Contains complete news and price data
- Timestamped for chronological analysis
- Ready for downstream trading systems

### Alert Structure
```json
{
  "symbol": "TICKER",
  "timestamp": "2024-01-01T12:00:00",
  "news": {
    "title": "Breaking News Headline",
    "published_utc": "2024-01-01T11:59:30Z",
    "article_url": "https://...",
    "news_detected_time": "2024-01-01 12:00:00"
  },
  "price_data": {
    "current_price": 5.25,
    "previous_close": 4.80,
    "price_change_percentage": 9.375
  }
}
```

## Performance Characteristics

### Throughput
- **News Processing**: ~1000 articles/hour
- **Ticker Matching**: <50ms per article
- **Price Updates**: ~100 tickers/minute
- **Database Writes**: ~5000 records/minute

### Latency
- **News Detection**: 15-60 seconds from publication
- **Price Response**: 1-5 seconds from news detection
- **Alert Generation**: <10 seconds end-to-end

### Resource Usage
- **Memory**: ~500MB typical, ~1GB peak
- **CPU**: ~25% of single core
- **Network**: ~10MB/hour (excluding price data)
- **Storage**: ~1GB/week (with retention policies)

## Troubleshooting

### Common Issues

#### Finviz Authentication
- Verify credentials in `.env` file
- Check for account suspension/limits
- Monitor for CAPTCHA requirements

#### Database Connection
- Ensure ClickHouse server is running
- Verify network connectivity and credentials
- Check disk space for database storage

#### News Source Access
- Monitor for website changes/blocking
- Adjust User-Agent headers if needed
- Implement proxy rotation if IP blocked

#### Price Data Issues
- Verify Polygon API key and limits
- Check for market hours/weekend limitations
- Monitor for rate limiting

### Debug Tools
```bash
python debug_ticker_extraction.py  # Test ticker matching
python debug_feeds.py             # Test news source access
python test_setup.py              # Comprehensive system test
```

## Development Notes

### Code Organization
- Modular design with clear separation of concerns
- Async/await patterns for concurrent operations
- Comprehensive error handling and logging
- Type hints for code clarity
- Configurable via environment variables

### Testing
- Unit tests for core functions
- Integration tests for database operations
- Performance benchmarks for critical paths
- Mock data for development/testing

### Extensibility
- Easy to add new news sources
- Pluggable ticker matching algorithms
- Configurable alert criteria
- Multiple price data providers supported

## License & Disclaimer

This system is for educational and research purposes. Users are responsible for:
- Complying with data provider terms of service
- Ensuring appropriate trading risk management
- Verifying all trading decisions independently
- Monitoring system performance and accuracy

**No warranty is provided for trading accuracy or system reliability.** 