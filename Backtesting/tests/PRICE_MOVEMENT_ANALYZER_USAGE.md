# Price Movement Analyzer Usage Guide

## Overview

The **Price Movement Analyzer** is a standalone script that analyzes historical news articles to identify those that preceded significant price movements and detect "false pump" scenarios. It creates a new table `News.price_movement_analysis` with comprehensive price movement data including:

- **30% Sustained Increases**: Articles followed by 30%+ price increase maintained until 9:28 AM EST
- **False Pumps**: Articles where price increased 20%+ but then fell back to within 10% of initial price by 9:28 AM EST
- **Complete Price Tracking**: Entry price, exit price, and maximum price reached during the analysis period

## Algorithm

For each article published between **6:00 AM to 9:01 AM UTC** in the **last 6 months**:

1. **Entry Price**: Get stock price at `published_utc + 30 seconds`
2. **Exit Price**: Get stock price at `9:28 AM EST` on the same date  
3. **Max Price**: Track highest price reached during the entire analysis period
4. **Calculate Ratios**: 
   - `price_increase_ratio = exit_price / entry_price`
   - `max_price_ratio = max_price / entry_price`
5. **Classify Movement**:
   - `has_30pct_increase = 1` if exit price ≥ 130% of entry price
   - `is_false_pump = 1` if max price ≥ 120% of entry price BUT exit price fell back to 90-110% of entry price

## Enhanced Database Schema

The analyzer creates a new table `News.price_movement_analysis` with these columns:

```sql
CREATE TABLE News.price_movement_analysis (
    ticker String,
    headline String,
    article_url String,
    published_utc DateTime,
    newswire_type String,
    content_hash String,
    entry_time DateTime,
    exit_time DateTime,
    entry_price Float64,
    exit_price Float64,
    max_price Float64,
    price_increase_ratio Float64,
    max_price_ratio Float64,
    has_30pct_increase UInt8,
    is_false_pump UInt8,
    analysis_date DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (ticker, published_utc)
PARTITION BY toYYYYMM(published_utc)
```

## Prerequisites

1. **Database Tables**: Ensure these tables exist and have data:
   - `News.historical_news` - Contains scraped news articles (last 6 months)
   - `News.historical_price` - Contains 10-second price bars from Polygon API

2. **Dependencies**: Same as other backtesting scripts (ClickHouse, pytz, etc.)

## Basic Usage

### Run Complete Analysis

```bash
cd Backtesting
python price_movement_analyzer.py
```

This will:
- Create a fresh `News.price_movement_analysis` table
- Process all articles from the last 6 months published between 6:00-9:01 AM UTC
- Track sustained increases and false pump scenarios
- Provide comprehensive statistics

## Expected Output

### During Processing
```
2024-01-15 10:30:00 - INFO - 🚀 Starting Price Movement Analysis with False Pump Detection...
2024-01-15 10:30:01 - INFO - ✅ Price Movement Analyzer initialized successfully
2024-01-15 10:30:01 - INFO - ✅ Created fresh price_movement_analysis table with false pump detection
2024-01-15 10:30:02 - INFO - 📄 Found 1,250 articles to analyze (last 6 months, 6:00-9:01 AM)
2024-01-15 10:30:05 - INFO - 📈 30%+ increase found: AAPL - Apple Announces Revolutionary New Product... (1.45x)
2024-01-15 10:30:08 - INFO - 🎢 False pump detected: TSLA - Tesla Reports Strong Q4... (max: 1.35x, final: 1.05x)
2024-01-15 10:30:10 - INFO - 📈 PROGRESS: 100/1250 articles analyzed
2024-01-15 10:30:10 - INFO -   • With price data: 87
2024-01-15 10:30:10 - INFO -   • With 30%+ increase: 12
2024-01-15 10:30:10 - INFO -   • With false pumps: 8
2024-01-15 10:30:10 - INFO -   • No price data: 13
```

### Final Statistics
```
🎉 PRICE MOVEMENT ANALYSIS COMPLETE!
📊 FINAL STATS:
  • Total articles analyzed: 1,250
  • Articles with price data: 1,089
  • Articles with 30%+ increase: 87
  • Articles with false pumps: 156
  • Articles without price data: 161
  • 30%+ increase rate: 7.99%
  • False pump rate: 14.32%
  • Time elapsed: 0:15:32
```

## Querying Results

After running the analyzer, you can query the comprehensive results:

### Basic Queries

```sql
-- Get articles with sustained 30%+ increases
SELECT ticker, headline, published_utc, entry_price, exit_price, price_increase_ratio
FROM News.price_movement_analysis 
WHERE has_30pct_increase = 1
ORDER BY price_increase_ratio DESC;

-- Get articles with false pumps
SELECT ticker, headline, published_utc, entry_price, max_price, exit_price, 
       max_price_ratio, price_increase_ratio
FROM News.price_movement_analysis 
WHERE is_false_pump = 1
ORDER BY max_price_ratio DESC;

-- Get comprehensive summary statistics
SELECT 
    has_30pct_increase,
    is_false_pump,
    COUNT(*) as article_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM News.price_movement_analysis
GROUP BY has_30pct_increase, is_false_pump
ORDER BY has_30pct_increase DESC, is_false_pump DESC;
```

### Advanced Analysis Queries

```sql
-- Compare false pumps vs sustained increases by ticker
SELECT 
    ticker,
    COUNT(*) as total_articles,
    SUM(has_30pct_increase) as sustained_increases,
    SUM(is_false_pump) as false_pumps,
    AVG(has_30pct_increase) * 100 as sustained_rate_percent,
    AVG(is_false_pump) * 100 as false_pump_rate_percent
FROM News.price_movement_analysis
GROUP BY ticker
HAVING total_articles >= 10
ORDER BY sustained_rate_percent DESC;

-- Analyze false pump characteristics
SELECT 
    ticker,
    headline,
    entry_price,
    max_price,
    exit_price,
    max_price_ratio,
    price_increase_ratio,
    (max_price - entry_price) as max_gain_dollars,
    (exit_price - entry_price) as final_gain_dollars
FROM News.price_movement_analysis
WHERE is_false_pump = 1
ORDER BY max_price_ratio DESC
LIMIT 20;

-- Time-based analysis of movement patterns
SELECT 
    EXTRACT(HOUR FROM published_utc) as hour,
    COUNT(*) as total_articles,
    AVG(has_30pct_increase) * 100 as sustained_rate,
    AVG(is_false_pump) * 100 as false_pump_rate,
    AVG(max_price_ratio) as avg_max_ratio
FROM News.price_movement_analysis
GROUP BY hour
ORDER BY hour;
```

## Time Filtering Logic

The script analyzes articles published between **6:00 AM to 9:01 AM UTC** in the **last 6 months** because:

1. **Price Data Availability**: Uses 6 months of historical price data
2. **Pre-Market Context**: Captures the critical pre-market period when news impacts opening prices
3. **Analysis Window**: Ensures sufficient time between publication (+30 seconds) and market open (9:30 AM EST)
4. **Extended Coverage**: Now includes articles published up to 9:01 AM UTC for comprehensive analysis

## False Pump Detection Logic

The enhanced analyzer identifies **false pump scenarios** using these criteria:

1. **Initial Pump**: Stock price must reach at least **120% of entry price** during the analysis period
2. **Subsequent Fall**: Final price (at 9:28 AM EST) must fall back to **90-110% of entry price**
3. **Classification**: This indicates initial excitement/momentum that ultimately failed before market open

**Example False Pump**:
- Entry Price: $10.00
- Max Price Reached: $13.50 (135% of entry = pump detected)
- Exit Price: $10.50 (105% of entry = fell back to initial range)
- Result: `is_false_pump = 1`

## Performance Considerations

- **Batch Processing**: Processes articles in batches (default 20) to manage memory usage
- **Rate Limiting**: 100ms delay between articles to avoid overwhelming the database
- **Fresh Analysis**: Creates new table each run for clean, up-to-date results
- **Efficient Queries**: Uses indexed columns for fast price lookups

## Integration with Backtesting System

This analyzer provides valuable pre-filtering data for the backtesting pipeline:

1. **Step 1-2**: Run normal backtesting (scrape news, fetch prices)
2. **Step 2.5**: Run price movement analyzer → `python price_movement_analyzer.py`
3. **Step 3-6**: Continue with sentiment analysis and trade simulation

The enhanced price movement data can be used to:
- **Pre-filter articles**: Focus sentiment analysis on high-potential articles
- **Avoid false pumps**: Exclude articles that historically showed false pump patterns
- **Strategy validation**: Compare predicted vs actual price movements
- **Source analysis**: Identify which newswires produce the most reliable signals
- **Risk management**: Understand false pump frequency for better position sizing

## Expected Results

Based on enhanced analysis, typical results might show:
- **Overall 30%+ Sustained Rate**: 2-8% of pre-market news articles
- **False Pump Rate**: 10-20% of articles (more common than sustained increases)
- **High-Impact Sources**: Certain newswires may show higher sustained rates and lower false pump rates
- **Sector Variations**: Biotech, small-cap stocks may show higher rates of both patterns
- **Risk Insights**: False pumps provide crucial risk management data

This comprehensive analysis provides deep insights into both successful price movements and failed momentum, enabling more sophisticated trading strategies and risk management approaches. 