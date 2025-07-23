# NewsHead Backtesting System

A comprehensive backtesting system for testing news-based trading strategies. This system scrapes historical news from Finviz, analyzes sentiment using Claude API, simulates trades using Polygon API, and exports results to Tradervue for analysis.

## Overview

The backtesting system consists of 5 main components that run sequentially:

1. **Create Tables** - Sets up ClickHouse database tables
2. **Scrape News** - Scrapes 6 months of newswire articles from Finviz
3. **Analyze Sentiment** - Uses Claude API for sentiment analysis
4. **Simulate Trades** - Simulates trades using Polygon API price data
5. **Export CSV** - Exports results to Tradervue CSV format

## Prerequisites

### Required API Keys

You need the following API keys configured in your `.env` file:

```bash
# Required for sentiment analysis
ANTHROPIC_API_KEY=your_claude_api_key_here

# Required for trade simulation
POLYGON_API_KEY=your_polygon_api_key_here

# Optional: Use proxy URL for Polygon API if needed
PROXY_URL=your_proxy_url_here
```

### Dependencies

Make sure you have all required Python packages installed:

```bash
pip install aiohttp beautifulsoup4 clickhouse-driver python-dotenv pytz
```

### ClickHouse Database

Ensure your ClickHouse database is running and accessible via the `clickhouse_setup.py` configuration.

## Quick Start

### Run Complete Backtesting Pipeline

```bash
cd Backtesting
python run_backtest.py
```

This will run all 5 steps sequentially and create a Tradervue-ready CSV file.

### Run Individual Steps

You can run specific steps or ranges of steps:

```bash
# Run only steps 1-3 (setup through sentiment analysis)
python run_backtest.py --start-step 1 --end-step 3

# Run only the trade simulation step
python run_backtest.py --start-step 4 --end-step 4

# Run only the CSV export step
python run_backtest.py --start-step 5 --end-step 5
```

### Custom Options

```bash
# Specify custom CSV filename
python run_backtest.py --csv-filename my_backtest_results.csv

# Dry run to see execution plan
python run_backtest.py --dry-run
```

## File Structure

```
Backtesting/
├── run_backtest.py          # Main orchestration script
├── create_tables.py         # ClickHouse table creation
├── finviz_pages.py         # Historical news scraping
├── sentiment_historical.py  # Sentiment analysis
├── trade_simulation.py     # Trade simulation
├── export_csv.py           # CSV export for Tradervue
├── exports/                # Generated CSV files and reports
├── logs/                   # Log files from backtesting runs
└── README.md               # This file
```

## Detailed Component Description

### 1. Create Tables (`create_tables.py`)

Creates the necessary ClickHouse tables:
- `historical_news` - Stores scraped news articles
- `historical_sentiment` - Stores sentiment analysis results
- `backtest_trades` - Stores simulated trade results
- `ticker_master_backtest` - Stores ticker metadata

### 2. Finviz News Scraper (`finviz_pages.py`)

**What it does:**
- Scrapes ticker lists from Finviz screener URLs (under $3 and $3-$10 stocks)
- Navigates to individual ticker pages to scrape news
- Filters for newswire articles only (PRNewswire, BusinessWire, GlobeNewswire, Accesswire)
- Only includes articles published between 5am-9am EST
- Scrapes 6 months of historical data

**Key Features:**
- Duplicate detection using content hashes
- Rate limiting to avoid getting blocked
- Comprehensive error handling and retry logic

### 3. Sentiment Analysis (`sentiment_historical.py`)

**What it does:**
- Processes scraped articles using Claude API
- Scrapes full article content from URLs
- Applies country-specific analysis (Bitcoin/crypto consideration for USA tickers)
- Generates BUY/SELL/HOLD recommendations with confidence levels

**Trading Predicates:**
- Only processes articles that haven't been analyzed
- Provides detailed explanations for each recommendation
- Caches content to avoid re-scraping

### 4. Trade Simulation (`trade_simulation.py`)

**What it does:**
- Simulates trades based on sentiment analysis
- Uses Polygon API for historical bid/ask quotes
- **Entry:** BUY on ask price 30 seconds after article publication
- **Exit:** SELL on bid price at exactly 9:28 AM EST
- Only trades articles with `recommendation='BUY'` and `confidence='high'`

**Trade Logic:**
- 100 shares per trade (configurable)
- Calculates P&L and percentage returns
- Tracks trade duration and spreads
- Comprehensive error handling for missing price data

### 5. CSV Export (`export_csv.py`)

**What it does:**
- Exports trade results to Tradervue generic CSV format
- Creates entry and exit rows for each trade
- Includes detailed notes with sentiment analysis
- Generates comprehensive summary report

**Export Features:**
- Enhanced format with P&L and commission columns
- Automatic filename generation with timestamps  
- Summary statistics and performance metrics
- Ready-to-import format for Tradervue

## Trading Strategy Details

### Entry Criteria
- Article sentiment analysis = 'BUY'
- Confidence level = 'high'
- Article published between 5am-9am EST
- From approved newswire sources only

### Trade Execution
- **Entry Time:** 30 seconds after article publication
- **Entry Price:** Ask price (market buy)
- **Exit Time:** 9:28 AM EST (same day)
- **Exit Price:** Bid price (market sell)
- **Position Size:** 100 shares

### Article Filtering
- **Time Window:** 5am-9am EST only
- **Newswire Sources:** PRNewswire, BusinessWire, GlobeNewswire, Accesswire
- **Ticker Universe:** Low float stocks under $10 from Finviz screeners
- **Geography:** Global tickers (with USA-specific Bitcoin/crypto considerations)

## Usage Examples

### Full Backtesting Run

```bash
# Complete pipeline with confirmation prompt
python run_backtest.py

# Skip confirmation (for automation)
echo "y" | python run_backtest.py
```

### Partial Runs

```bash
# Just scrape news (if tables already exist)
python run_backtest.py --start-step 2 --end-step 2

# Re-run sentiment analysis (if news already scraped)
python run_backtest.py --start-step 3 --end-step 3

# Generate new CSV export (if trades already simulated)
python run_backtest.py --start-step 5 --end-step 5
```

### Individual Component Testing

```bash
# Test table creation
python create_tables.py

# Test news scraping
python finviz_pages.py

# Test sentiment analysis
python sentiment_historical.py

# Test trade simulation  
python trade_simulation.py

# Test CSV export
python export_csv.py
```

## Output Files

### Generated Files
- `exports/newshead_backtest_tradervue_YYYYMMDD_HHMMSS.csv` - Tradervue import file
- `exports/newshead_backtest_tradervue_YYYYMMDD_HHMMSS_summary.txt` - Performance summary
- `logs/backtest_YYYYMMDD_HHMMSS.log` - Detailed execution log

### CSV Import to Tradervue
1. Log into your Tradervue account
2. Go to Import > Generic CSV
3. Upload the generated CSV file
4. Verify column mappings (should auto-detect)
5. Complete the import

## Performance Monitoring

The system provides comprehensive logging and statistics:

- **News Scraping:** Articles found, filtered, and stored
- **Sentiment Analysis:** Success/failure rates, API response times
- **Trade Simulation:** Win rate, total P&L, API call statistics
- **Overall:** Step-by-step success tracking, total runtime

## Troubleshooting

### Common Issues

**API Key Errors:**
```bash
❌ ANTHROPIC_API_KEY not found in environment variables
```
Solution: Add your Claude API key to the `.env` file

**Rate Limiting:**
```bash
⚠️ Rate limit hit, waiting before retry...
```
Solution: The system handles this automatically with exponential backoff

**No Trade Data:**
```bash
No articles found matching trading criteria
```
Solution: Check that sentiment analysis completed and found BUY/high confidence articles

**Database Connection:**
```bash
Error connecting to ClickHouse
```
Solution: Verify ClickHouse is running and `clickhouse_setup.py` is configured correctly

### Debug Mode

For detailed debugging, check the generated log files in the `logs/` directory or run individual components:

```bash
# Enable debug logging for specific component
python finviz_pages.py  # Check news scraping
python sentiment_historical.py  # Check sentiment analysis
```

## Configuration

### Customizable Parameters

Edit the respective Python files to customize:

- **Trade size:** Modify `default_quantity` in `trade_simulation.py`
- **Exit time:** Modify `exit_time_est` in `trade_simulation.py`  
- **Sentiment batch size:** Modify `batch_size` in `sentiment_historical.py`
- **News filtering:** Modify time ranges and newswire sources in `finviz_pages.py`

### Advanced Usage

The system is modular and can be extended:

- Add new sentiment analysis models
- Implement different trading strategies
- Export to other trading platforms
- Add risk management rules
- Integrate with live trading systems

## Support

For issues or questions:
1. Check the generated log files in the `logs/` directory for detailed error messages
2. Verify all API keys are correctly configured
3. Ensure ClickHouse database connectivity
4. Review individual component outputs for debugging

## License

This backtesting system is part of the NewsHead trading project. 