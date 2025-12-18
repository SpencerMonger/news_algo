# NewsHead - Real-Time News & Price Monitoring System

## Quick Start

**Start the main monitoring system:**
```bash
./start_newshead.sh
```

**Start the stock analysis pipeline (generates strength scores):**
```bash
./stockanalysis/start.sh
```

## System Overview

NewsHead is a real-time stock market news monitoring and price tracking system. It detects breaking news via Benzinga WebSocket, analyzes sentiment using Claude AI, tracks prices via IBKR TWS API, and generates alerts when sentiment, price, and strength score conditions are met.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN MONITORING SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │ Benzinga WebSocket│────▶│ Sentiment Service│────▶│   ClickHouse DB  │    │
│  │  (News Detection) │     │  (Claude AI)     │     │  (Data Storage)  │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│           │                                                   ▲              │
│           │ File Triggers (triggers/*.json)                   │              │
│           ▼                                                   │              │
│  ┌──────────────────┐     ┌──────────────────┐               │              │
│  │  Price Checker   │────▶│  Alert Generator │───────────────┘              │
│  │ (IBKR TWS API)   │     │ (Price+Sentiment │                              │
│  │                  │     │  +Strength Score)│                              │
│  └──────────────────┘     └──────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Entry Points

| File | Purpose |
|------|---------|
| `start_newshead.sh` | Main startup script - runs `run_system.py` in a screen session |
| `run_system.py` | Python orchestrator that starts all monitoring processes |
| `kill_newshead.sh` | Stops the monitoring system |

### Command Line Options

```bash
python3 run_system.py --socket              # Use Benzinga WebSocket (recommended)
python3 run_system.py --skip-list           # Skip Finviz ticker list update
python3 run_system.py --socket --any        # Process any ticker found
python3 run_system.py --no-sentiment        # Disable sentiment analysis (testing)
python3 run_system.py --enable-old          # Process articles older than 2 minutes
```

## Key Files

### News Collection

| File | Purpose |
|------|---------|
| `benzinga_websocket.py` | Real-time news via Benzinga WebSocket API (**only news source used**) |
| `web_scraper.py` | **DEPRECATED** - Legacy web scraping, not used in production |

**Benzinga WebSocket** connects to `wss://api.benzinga.com/api/v1/news/stream` and processes incoming news in real-time. It extracts tickers from the `securities` field and creates immediate file triggers for price tracking.

### Price Monitoring

| File | Purpose |
|------|---------|
| `price_checker.py` | Real-time price monitoring via IBKR TWS API |
| `ibkr_client.py` | IBKR TWS API client module with callbacks |

**Price Checker** uses IBKR TWS API:
- **Primary**: IBKR TWS API for real-time trade data (tickPrice callback with tickType=4 LAST)
- **Threading**: IBKR API runs in separate thread, integrated with asyncio event loop
- **Subscriptions**: Dynamic subscription management based on active tickers
- **Port Configuration**: 7497 for paper trading, 7496 for live trading (via IBKR_PORT env var)
- **Client ID**: Uses client_id=10 to avoid conflict with tradehead (client_id=1)

### Sentiment Analysis

| File | Purpose |
|------|---------|
| `sentiment_service.py` | Claude API with native load balancing across multiple keys |

**Key Features**:
- Uses Claude claude-sonnet-4-20250514 with 4D timing analysis prompt
- Supports multiple API keys via `ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY2`, etc.
- Automatic failover when keys hit rate limits
- Returns BUY/HOLD/SELL with confidence (high/medium/low)

### Database

| File | Purpose |
|------|---------|
| `clickhouse_setup.py` | ClickHouse database management and table creation |

**Key Tables**:
- `News.breaking_news` - News articles with sentiment data
- `News.price_tracking` - Price data with sentiment enrichment
- `News.news_alert` - Generated alerts with priority levels
- `News.float_list` - Ticker universe from Finviz
- `News.float_list_detailed_dedup` - Detailed stock statistics with strength scores

### Ticker Management

| File | Purpose |
|------|---------|
| `finviz_scraper.py` | Scrapes low-float stock list from Finviz Elite |

**Criteria**: Under 100M float, under $10 price, global markets, multiple sectors

## Alert Trigger Conditions

An alert is generated when ALL conditions are met:

1. **Price Movement**: ≥5% increase from baseline price
2. **Baseline Calculation**: Uses the **LOWEST of 2nd or 3rd price** (not just 2nd price)
3. **Time Window**: Within 60 seconds of first price timestamp
4. **Price Range**: Current price between $0.40 and $11.00
5. **Data Quality**: At least 3 price records available
6. **Sentiment**: AI recommendation = `BUY` with `high` confidence

### Alert Priority (Strength Score)

The alert value is determined by the stock's **strength score** from `float_list_detailed_dedup`:

| Strength Score | Alert Value | Priority |
|----------------|-------------|----------|
| ≥ 4 | 1 | High Priority |
| < 4 | 2 | Lower Priority |
| NULL (no data) | 3 | Unknown |

The strength score must be generated by running the Stock Analysis Pipeline (see below).

## Process Isolation

The system uses **process isolation** to ensure price checking is not affected by news processing:

```
Main Process (run_system.py)
    ├── Subprocess: price_checker.py (isolated)
    └── Async Tasks: benzinga_websocket.py
```

**Communication**: File-based triggers in `triggers/` directory for immediate ticker notifications.

## Stock Analysis Pipeline (`stockanalysis/`)

**This pipeline MUST be run to populate strength scores for alert prioritization.**

Scrapes detailed financial statistics from StockAnalysis.com and generates strength scores (1-10) using a weighted formula based on key financial metrics.

```bash
./stockanalysis/start.sh                    # Run complete pipeline in screen
python3 stockanalysis/run_analysis.py       # Run directly
python3 stockanalysis/run_analysis.py --limit 5  # Test with 5 tickers
python3 stockanalysis/run_analysis.py --skip-scraper  # Only run analyzer
python3 stockanalysis/run_analysis.py --reanalyze     # Force re-analyze all
```

**Pipeline Steps**:
1. `stockanalysis_scraper.py` - Scrapes 115+ statistics per ticker from StockAnalysis.com
2. `stock_strength_analyzer.py` - Generates strength scores (1-10) using weighted formula

**Output**: Populates `strength_score` column in `News.float_list_detailed_dedup` table.

**Alert Impact**: 
- Stocks with `strength_score >= 4` get **Alert Type 1** (High Priority)
- Stocks with `strength_score < 4` get **Alert Type 2** (Lower Priority)
- Stocks without strength scores get **Alert Type 3** (Unknown)

## Backtesting System (`Backtesting/`)

Historical backtesting for news-based trading strategies.

```bash
cd Backtesting && python run_backtest.py    # Run complete backtest
python run_backtest.py --limit 10           # Test with 10 tickers
python run_backtest.py --ticker AAPL        # Single ticker
```

**Pipeline Steps**:
1. `create_tables.py` - Set up ClickHouse tables
2. `finviz_pages.py` - Scrape historical news from Finviz
3. `fetch_historical_prices.py` - Download 10-second bars from Polygon
4. `sentiment_historical.py` - Analyze sentiment with Claude
5. `trade_simulation.py` - Simulate trades
6. `export_csv.py` - Export to Tradervue format

## Environment Variables

Required in `.env` file (see `env_template.txt`):

```bash
# IBKR Configuration (replaces Polygon)
IBKR_HOST=127.0.0.1           # TWS/Gateway host
IBKR_PORT=7497                # 7497=paper, 7496=live
IBKR_CLIENT_ID=10             # Must be unique (tradehead uses 1)

# Database
CLICKHOUSE_HOST=""
CLICKHOUSE_HTTP_PORT=""
CLICKHOUSE_USER=""
CLICKHOUSE_PASSWORD=""
CLICKHOUSE_DATABASE=""
CLICKHOUSE_SECURE=""

# Finviz Elite (for ticker list updates)
FINVIZ_EMAIL=""
FINVIZ_PASSWORD=""

# Benzinga (for WebSocket news)
BENZINGA_EMAIL=""
BENZINGA_PASSWORD=""
BENZINGA_API_KEY=""

# Dow Jones API (optional)
DOW_JONES_API_KEY=""

# Sentiment Analysis (supports multiple keys for load balancing)
ANTHROPIC_API_KEY=""
ANTHROPIC_API_KEY2=""  # optional
ANTHROPIC_API_KEY3=""  # optional
```

## Logging

Log files are stored in `logs/`:
- `logs/run_system.log.YYYY-MM-DD` - Main system logs
- `logs/articles/article_tracking.log` - Article processing logs
- `logs/clickhouse_operations.log` - Database operation logs

## Key Design Decisions

1. **Benzinga WebSocket Only**: Web scraping is deprecated; all news comes via WebSocket for sub-second detection
2. **Process Isolation**: Price checker runs separately to avoid resource contention
3. **File-Based Triggers**: Simple, reliable inter-process communication
4. **IBKR TWS API**: Replaced Polygon with IBKR for more reliable, lower-latency price data
5. **Lowest of 2nd/3rd Price Baseline**: Uses `least(2nd_price, 3rd_price)` for percentage calculations (1st often stale, using lowest of 2nd/3rd is more conservative)
6. **60-Second Window**: Strict enforcement at both price insertion and alert generation levels
7. **Native Load Balancing**: Multiple Claude API keys without external infrastructure
8. **Strength Score Priority**: Alerts are prioritized based on pre-computed stock strength scores from financial analysis
9. **Thread-Safe IBKR Integration**: IBKR callbacks run in separate thread with thread-safe price buffer
