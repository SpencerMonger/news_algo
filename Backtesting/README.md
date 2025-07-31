# NewsHead Backtesting System

A comprehensive backtesting system for testing news-based trading strategies. This system scrapes historical news from Finviz, analyzes sentiment using Claude API, simulates trades using Polygon API, and exports results to Tradervue for analysis.

## Overview

The backtesting system consists of 6 main components that run sequentially:

1. **Create Tables** - Sets up ClickHouse database tables
2. **Scrape News** - Scrapes 6 months of newswire articles from Finviz
3. **Fetch Price Data** - Downloads 10-second aggregate bars from Polygon API
4. **Analyze Sentiment** - Uses Claude API for sentiment analysis
5. **Simulate Trades** - Simulates trades using pre-fetched price data
6. **Export CSV** - Exports results to Tradervue CSV format

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
pip install aiohttp beautifulsoup4 clickhouse-driver python-dotenv pytz crawl4ai
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

# Limit ticker processing for testing (e.g., first 10 tickers)
python run_backtest.py --limit 10

# Process a single specific ticker
python run_backtest.py --ticker AAPL
```

### Single Ticker Processing

You can run the complete pipeline or individual steps for a single ticker:

```bash
# Complete pipeline for single ticker
python run_backtest.py --ticker AAPL

# Just scrape news for single ticker
python run_backtest.py --start-step 2 --end-step 2 --ticker TSLA

# Fetch price data for single ticker
python fetch_historical_prices.py --ticker NVDA

# Run sentiment analysis and trading simulation for single ticker
python run_backtest.py --start-step 3 --end-step 4 --ticker MSFT
```

### Custom Options

```bash
# Specify custom CSV filename
python run_backtest.py --csv-filename my_backtest_results.csv

# Skip sentiment requirements for testing (price movement only)
python run_backtest.py --skip-sentiment-check

# Dry run to see execution plan
python run_backtest.py --dry-run

# Combine options for comprehensive testing
python run_backtest.py --limit 10 --skip-sentiment-check

# Single ticker with custom options
python run_backtest.py --ticker AAPL --skip-sentiment-check --csv-filename aapl_backtest.csv
```

## Testing Mode

### Skip Sentiment Analysis for Testing

You can bypass the sentiment analysis requirement to test the price movement detection logic independently:

```bash
# Test with price movement only (no sentiment requirements)
python run_backtest.py --skip-sentiment-check --limit 10

# Run only the trade simulation step in testing mode
python run_backtest.py --start-step 4 --end-step 4 --skip-sentiment-check

# Run steps 1, 2, and 4 only (skip sentiment analysis entirely)
python run_backtest.py --start-step 1 --end-step 4 --skip-sentiment-check
```

**Testing Mode Features:**
- **Skips Step 3 Entirely:** Sentiment analysis step is automatically bypassed
- **Bypasses Sentiment Requirements:** Uses all articles regardless of sentiment analysis
- **Price Movement Only:** Tests pure price-based trading logic (5% increase in 40 seconds)
- **Faster Testing:** No dependency on Claude API or sentiment analysis step
- **Debugging Tool:** Isolates price movement detection from sentiment analysis

**Use Cases:**
- Testing Polygon API integration and price data retrieval
- Validating 10-second bar aggregation logic
- Debugging price movement detection algorithms
- Performance testing without API rate limits from sentiment analysis

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

**Note:** Tables are automatically dropped and recreated on each backtest run to ensure fresh data.

### 2. Finviz News Scraper (`finviz_pages.py`)

**What it does:**
- Gets ticker list from existing `News.float_list` table (faster than scraping screeners)
- Navigates to individual ticker pages to scrape news using Crawl4AI
- Filters for newswire articles only (PRNewswire, BusinessWire, GlobeNewswire, Accesswire, TipRanks)
- Only includes articles published between 7am-9:30am EST (updated from 5am-9am)
- Scrapes complete historical data (no date filtering)

**Key Features:**
- **Crawl4AI Integration:** Uses AsyncWebCrawler for reliable web scraping with browser automation
- **Proximity-Based Timestamp Matching:** Advanced algorithm that finds the timestamp closest to each article link in the HTML
- **Comprehensive Article Detection:** Finds all newswire articles on ticker pages (typically 80-100+ articles per ticker)
- **Duplicate Detection:** Uses content hashes to prevent duplicate articles
- **Rate Limiting:** Respectful scraping with delays to avoid getting blocked

#### Finviz HTML Structure & Parsing Logic

Through extensive debugging and analysis, we discovered the **actual HTML structure** used by Finviz and developed a precise parsing system that correctly extracts newswire articles with perfect timestamp accuracy.

**🎯 KEY DISCOVERY: Finviz uses a simple table structure where each row contains exactly two cells:**
- **Cell 1:** Timestamp (e.g., `Jul-29-25 08:00AM`, `08:30AM`)  
- **Cell 2:** Article link and newswire attribution

**1. Correct Table Structure Understanding:**
```python
# Find the news table - it contains both timestamps and newswire links
news_table = None
for table in soup.find_all('table'):
    table_text = table.get_text()
    has_timestamps = bool(re.search(timestamp_pattern, table_text))
    has_newswires = any(wire in table_text for wire in target_newswires)
    
    if has_timestamps and has_newswires:
        news_table = table
        break

# Process table rows: each row has timestamp cell and article cell
news_rows = news_table.find_all('tr')
for row in news_rows:
    cells = row.find_all(['td', 'th'])
    
    # Skip rows that don't have exactly 2 cells (timestamp + article)
    if len(cells) != 2:
        continue
    
    timestamp_cell = cells[0]  # First cell contains timestamp
    article_cell = cells[1]    # Second cell contains article link
```

**2. Timestamp Handling Logic:**
```python
# Handle both full timestamps and time-only patterns
timestamp_text = timestamp_cell.get_text().strip()
is_full_timestamp = bool(re.search(r'\w{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}[AP]M', timestamp_text))
is_time_only = bool(re.match(r'^\d{1,2}:\d{2}[AP]M$', timestamp_text))

if is_full_timestamp:
    # Extract full timestamp (e.g., "Jul-29-25 08:00AM")
    current_timestamp = timestamp_match.group()
    last_full_date = current_timestamp.split()[0]  # Remember date part
    
elif is_time_only and last_full_date:
    # Combine time-only with last known date (e.g., "08:30AM" + "Jul-29-25")
    current_timestamp = f"{last_full_date} {timestamp_text}"
```

**3. Newswire Detection:**
```python
# Complete list of newswire variations to handle case sensitivity
target_newswires = [
    'GlobeNewswire', 'Globe Newswire', 'GLOBENEWSWIRE', 'GLOBE NEWSWIRE',
    'PRNewswire', 'PR Newswire', 'PRNEWSWIRE', 'PR NEWSWIRE', 
    'BusinessWire', 'Business Wire', 'BUSINESSWIRE', 'BUSINESS WIRE',
    'Accesswire', 'AccessWire', 'ACCESSWIRE', 'ACCESS WIRE'
]

def find_newswire_for_link(self, link, article_cell, target_newswires) -> str:
    """Find specific newswire type for a given link"""
    # Search in the article cell and parent elements for newswire indicators
    search_elements = [link]
    current = link
    for level in range(3):  # Search up to 3 levels up
        if current.parent:
            current = current.parent
            search_elements.append(current)
    
    # Look for parenthetical indicators first (most reliable)
    for element in search_elements:
        element_text = element.get_text()
        for newswire in target_newswires:
            if f'({newswire})' in element_text:
                return self.normalize_newswire_name(newswire)
    
    # Check for non-parenthetical mentions with word boundaries
    for element in search_elements:
        element_text = element.get_text()
        for newswire in target_newswires:
            if re.search(rf'\b{re.escape(newswire)}\b', element_text, re.IGNORECASE):
                return self.normalize_newswire_name(newswire)
```

**4. Enhanced Timestamp Parsing:**
```python
def parse_finviz_timestamp(self, time_text: str) -> datetime:
    """Parse various Finviz timestamp formats with high accuracy"""
    
    # Handle Finviz-specific format: "Jul-21-25 08:20AM"
    finviz_match = re.search(r'(\w{3})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})([AP]M)', time_text)
    if finviz_match:
        month_str = finviz_match.group(1)
        day = int(finviz_match.group(2))
        year = int(f"20{finviz_match.group(3)}")  # Convert 25 to 2025
        hour = int(finviz_match.group(4))
        minute = int(finviz_match.group(5))
        ampm = finviz_match.group(6)
        
        # Convert to 24-hour format
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        
        # Parse month abbreviation
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month = month_map.get(month_str, 1)
        
        return datetime(year, month, day, hour, minute)
```

**❌ PREVIOUS APPROACH (Incorrect):**
- ~~Proximity-based timestamp matching~~
- ~~Complex container detection strategies~~  
- ~~URL-based date extraction (Finviz URLs don't contain dates)~~
- ~~Multiple fallback methods for timestamp association~~

**✅ CURRENT APPROACH (Correct):**
- **Simple Row-Based Processing:** Each table row = one timestamp + one article
- **Direct Cell Pairing:** First cell is timestamp, second cell is article
- **Time-Only Handling:** Combines time-only timestamps with last full date
- **Perfect Accuracy:** 100% match with Finviz display, verified against screenshots

**Key Improvements:**

1. **Eliminated Guesswork:** No more proximity matching or complex algorithms
   - **Before:** Tried to guess which timestamp belongs to which article
   - **After:** Direct 1:1 mapping from table structure

2. **Perfect Timestamp Accuracy:** Every article gets the correct timestamp
   - **Before:** Articles could get wrong timestamps from proximity matching
   - **After:** Timestamps match exactly what's shown on Finviz

3. **Simplified Logic:** Much cleaner and more maintainable code
   - **Before:** 6 different timestamp association methods with fallbacks
   - **After:** Single row-based processing with time-only combination

4. **Handles Time-Only Patterns:** Correctly processes partial timestamps
   - **Example:** Row with "08:30AM" uses date from previous "Jul-29-25 08:00AM" row
   - **Result:** "Jul-29-25 08:30AM" (perfectly accurate)

**Verification Results:**
- **ACCO Articles:** 100% match with Finviz screenshot (3/3 newswire articles)
- **ABEO Articles:** 100% match with Finviz screenshot (7/7 newswire articles)  
- **Timestamp Precision:** Every timestamp matches exactly what's displayed
- **Newswire Detection:** All Business Wire, PR Newswire, and GlobeNewswire articles captured

### 3. Price Data Fetching (`fetch_historical_prices.py`)

**What it does:**
- Fetches 10-second aggregate bars from Polygon API for all tickers in `float_list` table
- Only retrieves data during trading hours (7am-9:30am EST)
- Stores data in `News.historical_price` table for efficient backtesting
- Processes 180 days of historical data by default (configurable)
- Handles rate limiting and API errors gracefully

**Key Features:**
- **Trading Hours Only:** Filters data to 7am-9:30am EST window automatically
- **10-Second Granularity:** High-resolution price data for accurate backtesting
- **Batch Processing:** Efficiently processes multiple tickers and dates
- **Rate Limiting:** Respects Polygon API limits with controlled concurrency
- **Data Persistence:** Stores all data locally for fast trade simulation

**Usage:**
```bash
# Fetch price data for all tickers (180 days back)
python -m Backtesting.fetch_historical_prices

# Test with limited tickers
python -m Backtesting.fetch_historical_prices --limit 10

# Fetch different time range
python -m Backtesting.fetch_historical_prices --days 90
```

### 4. Sentiment Analysis (`sentiment_historical.py`)

**What it does:**
- Processes scraped articles using Claude API
- Scrapes full article content from URLs
- Applies country-specific analysis (Bitcoin/crypto consideration for USA tickers)
- Generates BUY/SELL/HOLD recommendations with confidence levels

**Trading Predicates:**
- Only processes articles that haven't been analyzed
- Provides detailed explanations for each recommendation
- Caches content to avoid re-scraping

### 5. Trade Simulation (`trade_simulation.py`)

**What it does:**
- Simulates trades based on sentiment analysis and price movement conditions
- Uses Polygon API 10-second aggregate bars for historical price data
- **Entry:** BUY 30 seconds after initial timestamp if all conditions are met
- **Exit:** SELL at exactly 9:28 AM EST
- Only trades articles that meet ALL the following conditions:
  1. Published between 7am-9:30am EST
  2. Sentiment analysis shows 'BUY' with 'high' confidence
  3. Price increases 5%+ within first 40 seconds of publication timestamp

**Enhanced Trade Logic:**
- **Time Filtering:** Only processes articles published between 7am-9:30am EST
- **Price Movement Detection:** Uses 10-second aggregate bars to detect 5% price increase within 40 seconds
- **Dual Condition Requirement:** Both sentiment (BUY/high) AND price movement (5%+) must be met
- **Historical Bar Analysis:** Gets complete price data from publication time until 9:30am EST
- **Realistic Entry/Exit:** Entry at 30 seconds after trigger, exit at 9:28am EST using closest bar prices

### 6. CSV Export (`export_csv.py`)

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

### Entry Criteria (ALL must be met)
1. **Time Window:** Article published between 7am-9:30am EST
2. **Sentiment Analysis:** Article sentiment = 'BUY' with 'high' confidence
3. **Price Movement:** Stock price increases 5%+ within first 40 seconds of publication
4. **Newswire Source:** From approved sources (PRNewswire, BusinessWire, GlobeNewswire, Accesswire, TipRanks)

### Trade Execution
- **Price Data Source:** Polygon API 10-second aggregate bars
- **Entry Time:** Exactly 30 seconds after article publication timestamp
- **Entry Price:** Close price from closest 10-second bar to entry time
- **Exit Time:** Exactly 9:28 AM EST (same day)
- **Exit Price:** Close price from closest 10-second bar to exit time
- **Position Size:** 100 shares per trade

### Enhanced Logic Flow
1. **Article Analysis:** Process articles with BUY/high sentiment
2. **Time Filtering:** Reject articles outside 7am-9:30am EST window
3. **Price Data Retrieval:** Get 10-second bars from publication to 9:30am EST
4. **Movement Detection:** Check for 5%+ price increase within first 40 seconds
5. **Trade Execution:** If all conditions met, simulate entry at +30s and exit at 9:28am
6. **P&L Calculation:** Calculate profit/loss using bar close prices

## Usage Examples

### Full Backtesting Run

```bash
# Complete pipeline with confirmation prompt
python run_backtest.py

# Skip confirmation (for automation)
echo "y" | python run_backtest.py

# Test with limited tickers
python run_backtest.py --limit 10
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

# Test news scraping for single ticker
python finviz_pages.py --ticker AAPL

# Test price data fetching
python fetch_historical_prices.py

# Test price data fetching for single ticker
python fetch_historical_prices.py --ticker TSLA

# Test price data fetching with custom parameters
python fetch_historical_prices.py --ticker NVDA --days 90

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

- **News Scraping:** Articles found, filtered, and stored (typically 80-100+ articles per ticker)
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

**Timestamp Issues:**
```bash
⚠️ Could not parse time 'invalid_format', using current time
```
Solution: The proximity-based matching should handle this automatically. Check debug logs for parsing issues.

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
- **Ticker limit:** Use `--limit N` flag for testing with fewer tickers

### Advanced Usage

The system is modular and can be extended:

- Add new sentiment analysis models
- Implement different trading strategies
- Export to other trading platforms
- Add risk management rules
- Integrate with live trading systems

## Technical Notes

### Finviz Scraping Architecture

The news scraper uses a sophisticated approach developed through extensive debugging:

1. **Crawl4AI Integration:** Browser-based scraping for reliable JavaScript-rendered content
2. **Smart Table Detection:** Identifies news containers by analyzing timestamp density and link count
3. **Proximity Matching:** Associates timestamps with articles based on HTML position distance
4. **Robust Parsing:** Handles multiple timestamp formats and newswire label variations
5. **Comprehensive Coverage:** Finds 10-20x more articles than simple table-row approaches

This architecture ensures high accuracy and completeness in historical news data collection.

## Support

For issues or questions:
1. Check the generated log files in the `logs/` directory for detailed error messages
2. Verify all API keys are correctly configured
3. Ensure ClickHouse database connectivity
4. Review individual component outputs for debugging
5. For timestamp parsing issues, check the proximity-based matching debug output

## License

This backtesting system is part of the NewsHead trading project. 