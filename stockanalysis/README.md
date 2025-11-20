# StockAnalysis.com Statistics Scraper

This service scrapes detailed stock statistics from [StockAnalysis.com](https://stockanalysis.com) for all tickers in the `float_list` table and stores the data in the `float_list_detailed` table in ClickHouse.

## Overview

The scraper extracts **115+ comprehensive statistics** for each ticker including:

- **Total Valuation** (2): Market Cap, Enterprise Value
- **Important Dates** (2): Earnings Date, Ex-Dividend Date
- **Stock Price Statistics** (8): Beta (5Y), 52-Week High/Low/Change, 50/200-Day MA, RSI, Average Volume (20D)
- **Share Statistics** (7): Current Share Class, Shares Outstanding, Shares Change (YoY/QoQ), Insider/Institutional %, Float
- **Short Selling** (5): Short Interest, Previous Month, % of Shares/Float, Short Ratio
- **Valuation Ratios** (9): PE, Forward PE, PS, Forward PS, PB, P/TBV, P/FCF, P/OCF, PEG
- **Enterprise Valuation** (5): EV/Earnings, EV/Sales, EV/EBITDA, EV/EBIT, EV/FCF
- **Financial Position** (6): Current Ratio, Quick Ratio, Debt/Equity, Debt/EBITDA, Debt/FCF, Interest Coverage
- **Financial Efficiency** (9): ROE, ROA, ROIC, ROCE, Revenue/Profits Per Employee, Employee Count, Asset/Inventory Turnover
- **Taxes** (2): Income Tax, Effective Tax Rate
- **Income Statement** (8): Revenue, Gross Profit, Operating Income, Pretax Income, Net Income, EBITDA, EBIT, EPS
- **Balance Sheet** (7): Cash & Equivalents, Total Debt, Net Cash, Net Cash Per Share, Equity, Book Value Per Share, Working Capital
- **Cash Flow** (4): Operating CF, Capital Expenditures, Free CF, FCF Per Share
- **Margins** (7): Gross, Operating, Pretax, Profit, EBITDA, EBIT, FCF Margins
- **Dividends & Yields** (9): Dividend Per Share/Yield, Growth YoY, Years of Growth, Payout/Buyback/Shareholder/Earnings/FCF Yields
- **Stock Splits** (3): Last Split Date, Split Type, Split Ratio
- **Scores** (2): Altman Z-Score, Piotroski F-Score

## Setup

### 1. Create the Database Table

Before running the scraper for the first time, create the `float_list_detailed` table:

```bash
python3 stockanalysis/stockanalysis_scraper.py --setup-table
```

This will create the ClickHouse table with all required columns for storing the detailed statistics.

### 2. Verify Database Connection

Ensure your `.env` file has the ClickHouse connection details:

```
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password
```

## Usage

### Scrape All Tickers

To scrape statistics for all tickers in the `float_list` table:

```bash
python3 stockanalysis/stockanalysis_scraper.py
```

### Test with Limited Tickers

For testing purposes, you can limit the number of tickers to process:

```bash
# Process only the first 5 tickers
python3 stockanalysis/stockanalysis_scraper.py --limit 5

# Process first 10 tickers
python3 stockanalysis/stockanalysis_scraper.py --limit 10
```

## Features

### Respectful Scraping
- **Rate Limiting**: 1.5 second delay between requests
- **Batch Processing**: Processes tickers in batches of 10
- **Batch Delays**: 5 second delay between batches
- **Error Handling**: Graceful handling of failures without stopping the entire process

### Data Parsing
- **JSON Extraction**: Extracts data directly from embedded JavaScript objects (faster and more reliable)
- **Smart Number Parsing**: Handles various formats (35.23M, 1.5B, $1,234.56, 15.3%)
- **Null Handling**: Properly handles n/a, missing, or invalid data
- **Fallback Regex**: Uses regex patterns as fallback if JSON extraction fails

### Performance Tracking
- **Real-time Statistics**: Tracks processed, successful, and failed tickers
- **Detailed Logging**: Comprehensive logging of all operations
- **Immediate Insertion**: Data is inserted to database immediately after scraping each ticker

## Database Schema

The `float_list_detailed` table uses a `ReplacingMergeTree` engine with the following key features:

- **Primary Key**: `(ticker, scraped_at)` - Allows tracking statistics over time
- **Partitioning**: By month (`toYYYYMM(scraped_at)`)
- **Deduplication**: Automatically replaces old records with newer ones for the same ticker
- **Indexes**: Bloom filter on ticker, minmax on scraped_at

## Query Examples

### Get Latest Statistics for a Ticker

```sql
SELECT *
FROM News.float_list_detailed
WHERE ticker = 'AAPL'
ORDER BY scraped_at DESC
LIMIT 1;
```

### Get Statistics for All Tickers (Latest Only)

```sql
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY scraped_at DESC) as rn
    FROM News.float_list_detailed
) WHERE rn = 1;
```

### Find High Beta Stocks

```sql
SELECT ticker, beta_5y, market_cap, 52_week_change
FROM News.float_list_detailed
WHERE beta_5y > 1.5
ORDER BY beta_5y DESC;
```

### Find Dividend Growth Stocks

```sql
SELECT ticker, dividend_yield, dividend_growth_yoy, years_dividend_growth
FROM News.float_list_detailed
WHERE dividend_yield > 2.0
  AND dividend_growth_yoy > 5.0
  AND years_dividend_growth >= 5
ORDER BY dividend_yield DESC;
```

## Architecture

The scraper follows the existing NewsHead architecture patterns:

1. **Crawl4AI Integration**: Uses the same browser automation setup as `web_scraper.py`
2. **ClickHouse Integration**: Uses the existing `ClickHouseManager` from `clickhouse_setup.py`
3. **Async Processing**: Fully async implementation for efficient processing
4. **Error Resilience**: Graceful error handling and recovery

## Monitoring

The scraper provides real-time progress updates:

```
🔍 Scraping statistics for AAPL from https://stockanalysis.com/stocks/aapl/statistics/
✅ Extracted 42 statistics for AAPL
✅ Inserted statistics for AAPL into database
```

And a summary at the end:

```
================================================================================
📊 SCRAPING SUMMARY:
   Total Tickers Processed: 150
   ✅ Successful: 142
   ❌ Failed: 8
   Errors: 0
================================================================================
```

## Maintenance

### Update Statistics Periodically

You can set up a cron job to update statistics daily or weekly:

```bash
# Add to crontab (daily at 2 AM)
0 2 * * * cd /home/synk/Development/newshead && python3 stockanalysis/stockanalysis_scraper.py >> logs/stockanalysis.log 2>&1
```

### Clean Old Data

The table is partitioned by month, so you can drop old partitions if needed:

```sql
-- Drop data older than 6 months
ALTER TABLE News.float_list_detailed DROP PARTITION '202406';
```

## Troubleshooting

### "No tickers found in float_list table"

Ensure you've run the Finviz scraper first to populate the `float_list` table:

```bash
python3 finviz_scraper.py
```

### "Failed to scrape ticker"

Some tickers may not have a statistics page on StockAnalysis.com. This is normal and the scraper will continue with other tickers.

### High Error Rate

If you're seeing many errors, check:
1. Your internet connection
2. Whether StockAnalysis.com is accessible
3. Whether the website structure has changed (may need to update regex patterns)

## Future Enhancements

Potential improvements:
- Add historical tracking to detect changes over time
- Implement alerts for significant statistic changes
- Add more detailed financial statement data
- Include analyst ratings and price targets
- Add comparison metrics vs. sector averages

