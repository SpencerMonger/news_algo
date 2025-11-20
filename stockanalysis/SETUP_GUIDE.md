# StockAnalysis.com Scraper - Setup & Quick Start Guide

## 🎯 Overview

This new service scrapes detailed stock statistics from StockAnalysis.com for all tickers in your `float_list` table. It uses Crawl4AI (same as your existing web scraper) and stores comprehensive data in a new ClickHouse table called `float_list_detailed`.

## 📁 Files Created

### 1. `/stockanalysis/stockanalysis_scraper.py`
Main scraper script that:
- Fetches ticker list from `float_list` table
- Scrapes statistics from `https://stockanalysis.com/stocks/{ticker}/statistics/`
- Parses 45+ data points per ticker
- Inserts data into ClickHouse immediately after each ticker
- Implements respectful rate limiting (1.5s between requests)

### 2. `/clickhouse_setup.py` (Modified)
Added two new methods:
- `create_float_list_detailed_table()` - Creates the database table
- `insert_float_list_detailed()` - Inserts scraped statistics

### 3. `/stockanalysis/example_queries.py`
Demonstrates useful queries:
- Latest statistics for a ticker
- High beta (volatile) stocks
- Dividend growth stocks
- Undervalued stocks (low P/E, P/B)
- High profitability stocks
- Database summary statistics

### 4. `/stockanalysis/README.md`
Comprehensive documentation with:
- Feature descriptions
- Usage examples
- Query examples
- Architecture details
- Troubleshooting tips

## 🚀 Quick Start

### Step 1: Create the Database Table

```bash
cd /home/synk/Development/newshead
python3 stockanalysis/stockanalysis_scraper.py --setup-table
```

This creates the `News.float_list_detailed` table with 45+ columns for all the statistics.

### Step 2: Test with a Few Tickers

```bash
# Test with just 5 tickers to verify everything works
python3 stockanalysis/stockanalysis_scraper.py --limit 5
```

Expected output:
```
🚀 Initializing StockAnalysis.com scraper...
✅ Crawl4AI browser started successfully
✅ StockAnalysis scraper initialized successfully
📊 Found 150 tickers to process
🎯 Processing limited set of 5 tickers
🔍 Scraping statistics for AAPL from https://stockanalysis.com/stocks/aapl/statistics/
✅ Extracted 42 statistics for AAPL
✅ Inserted statistics for AAPL into database
...
```

### Step 3: Run Full Scrape

```bash
# Scrape all tickers in your float_list table
python3 stockanalysis/stockanalysis_scraper.py
```

This will:
- Process all tickers in batches of 10
- Take ~1.5 seconds per ticker (respectful rate limiting)
- Insert each ticker immediately after scraping
- Show progress and summary statistics

## 📊 Data Fields Extracted

The scraper extracts **45+ statistics** organized into 8 categories:

### Valuation Metrics (7 fields)
- Market Cap, Enterprise Value
- P/E, P/S, P/B Ratios
- EV/Sales, EV/EBITDA

### Price Metrics (7 fields)
- Beta (5Y)
- 52-Week High/Low/Change
- 50-Day & 200-Day Moving Averages

### Share Statistics (6 fields)
- Shares Outstanding, Float
- Insider/Institutional Ownership %
- Short Interest %, Short Ratio

### Dividend Metrics (5 fields)
- Dividend Per Share, Yield
- Payout Ratio, Growth YoY
- Years of Dividend Growth

### Profitability Metrics (4 fields)
- Profit Margin, Operating Margin
- Return on Assets, Return on Equity

### Financial Metrics (8 fields)
- Revenue (TTM), Revenue Per Share
- Quarterly Revenue Growth
- Gross Profit, EBITDA, Net Income
- EPS, Quarterly Earnings Growth

### Balance Sheet (5 fields)
- Total Cash, Total Debt
- Debt/Equity, Current Ratio
- Book Value Per Share

### Cash Flow (2 fields)
- Operating Cash Flow
- Free Cash Flow

## 💻 Usage Examples

### View Example Queries

```bash
python3 stockanalysis/example_queries.py
```

This runs 6 example queries showing:
1. Database summary statistics
2. Latest stats for AAPL
3. Top 10 high beta stocks
4. Top 10 dividend growth stocks
5. Top 10 potentially undervalued stocks
6. Top 10 high profitability stocks

### Query from Python

```python
from clickhouse_setup import ClickHouseManager

ch_manager = ClickHouseManager()
ch_manager.connect()

# Get latest statistics for a ticker
query = """
SELECT ticker, market_cap, pe_ratio, dividend_yield, beta_5y
FROM News.float_list_detailed
WHERE ticker = 'AAPL'
ORDER BY scraped_at DESC
LIMIT 1
"""

result = ch_manager.client.query(query)
for row in result.result_rows:
    print(f"Ticker: {row[0]}")
    print(f"Market Cap: ${row[1]:,.0f}")
    print(f"P/E Ratio: {row[2]:.2f}")
    print(f"Dividend Yield: {row[3]:.2f}%")
    print(f"Beta (5Y): {row[4]:.2f}")
```

### Query from ClickHouse CLI

```sql
-- Find high beta stocks
SELECT ticker, beta_5y, market_cap, `52_week_change`
FROM News.float_list_detailed
WHERE beta_5y > 1.5
ORDER BY beta_5y DESC
LIMIT 10;

-- Find dividend growth stocks
SELECT ticker, dividend_yield, dividend_growth_yoy, years_dividend_growth
FROM News.float_list_detailed
WHERE dividend_yield > 2.0
  AND dividend_growth_yoy > 5.0
  AND years_dividend_growth >= 5
ORDER BY dividend_yield DESC;
```

**Note:** Column names starting with numbers (like `52_week_high`) must be backtick-escaped in queries.

## ⚙️ Configuration

### Rate Limiting
The scraper is configured to be respectful:
- **1.5 seconds** between individual requests
- **5 seconds** between batches (10 tickers per batch)
- **2 second** page load delay for JavaScript rendering

### Browser Settings
Uses the same Crawl4AI configuration as your existing web scraper:
- Headless Chromium
- Disabled GPU, images, JavaScript (for speed)
- 512MB memory limit
- 30-second timeout per page

## 🔄 Automation

### Daily Updates via Cron

```bash
# Edit crontab
crontab -e

# Add this line to run daily at 2 AM
0 2 * * * cd /home/synk/Development/newshead && python3 stockanalysis/stockanalysis_scraper.py >> logs/stockanalysis.log 2>&1
```

### Weekly Updates via Cron

```bash
# Run every Sunday at 3 AM
0 3 * * 0 cd /home/synk/Development/newshead && python3 stockanalysis/stockanalysis_scraper.py >> logs/stockanalysis.log 2>&1
```

## 📋 Database Schema

```sql
CREATE TABLE News.float_list_detailed (
    id UUID DEFAULT generateUUIDv4(),
    ticker String,
    scraped_at DateTime64(3) DEFAULT now64(),
    source_url String,
    
    -- 45+ statistics columns organized by category
    -- See README.md for complete list
    
    INDEX idx_ticker (ticker) TYPE bloom_filter GRANULARITY 1,
    INDEX idx_scraped_at (scraped_at) TYPE minmax GRANULARITY 3
) 
ENGINE = ReplacingMergeTree(scraped_at)
ORDER BY (ticker, scraped_at)
PARTITION BY toYYYYMM(scraped_at);
```

**Key Features:**
- `ReplacingMergeTree` - Automatically deduplicates based on latest `scraped_at`
- Partitioned by month for efficient queries
- Indexes on ticker and timestamp for fast lookups

## 🐛 Troubleshooting

### "No tickers found in float_list table"
**Solution:** Run the Finviz scraper first to populate the float_list table.

```bash
python3 finviz_scraper.py
```

### "Failed to scrape ticker"
**Solution:** Some tickers may not have statistics pages on StockAnalysis.com. This is normal - the scraper will continue with other tickers.

### Many Extraction Failures
**Solution:** The website structure may have changed. Update the regex patterns in the `extract_statistics()` method.

### ClickHouse Connection Error
**Solution:** Verify your `.env` file has correct credentials:
```
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password
```

## 📈 Performance

Expected performance:
- **Processing Time:** ~1.5-2 seconds per ticker
- **Total Time:** ~250 tickers = ~6-8 minutes
- **Success Rate:** 90-95% (some tickers may not have statistics pages)
- **Memory Usage:** ~512MB (browser automation)
- **Database Size:** ~50KB per ticker (~12.5MB for 250 tickers)

## 🔐 Security Notes

- Uses existing ClickHouse credentials from your `.env` file
- No API keys required (public web scraping)
- Respectful rate limiting to avoid overwhelming the website
- User agent and browser automation properly configured

## 📚 Additional Resources

- **Main Documentation:** See `/stockanalysis/README.md`
- **Example Queries:** Run `python3 stockanalysis/example_queries.py`
- **Architecture:** Follows patterns from `/Architecture.md`
- **Existing Web Scraper:** Based on `/web_scraper.py`

## ✅ Checklist

- [x] Create database table
- [ ] Test with limited tickers (`--limit 5`)
- [ ] Run full scrape
- [ ] Verify data in ClickHouse
- [ ] Run example queries
- [ ] Set up cron job (optional)

## 🎉 You're Ready!

The StockAnalysis.com scraper is now ready to use. Start with the Quick Start section above and refer to the README.md for more detailed documentation.

