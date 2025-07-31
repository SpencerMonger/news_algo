# Price Movement Analyzer Usage Guide

## Overview

The **Price Movement Analyzer** is a standalone script that analyzes historical news articles to identify those that preceded significant price movements (30%+ increase). It adds a new column `price_movement_30pct` to the `historical_news` table with the following values:

- `1` (True) - Article was followed by a 30%+ price increase
- `0` (False) - Article was NOT followed by a 30%+ price increase  
- `NULL` - No price data available for analysis

## Algorithm

For each article published between **6:00 AM to 9:00 AM EST** in **2025 onwards** (matching price data availability):

1. **Entry Price**: Get stock price at `published_utc + 30 seconds`
2. **Exit Price**: Get stock price at `9:28 AM EST` on the same date
3. **Calculate**: Check if `exit_price >= 1.30 × entry_price` (30%+ increase)
4. **Label**: Store `True`/`False`/`NULL` in the `price_movement_30pct` column

## Prerequisites

1. **Database Tables**: Ensure these tables exist and have data:
   - `News.historical_news` - Contains scraped news articles (2025+ for analysis)
   - `News.historical_price` - Contains 10-second price bars from Polygon API (6 months back)

2. **Dependencies**: Same as other backtesting scripts (ClickHouse, pytz, etc.)

## Basic Usage

### Run Complete Analysis

```bash
cd Backtesting
python price_movement_analyzer.py
```

This will:
- Add the new column to `historical_news` table (if it doesn't exist)
- Process all articles that don't have price movement labels yet
- Update the database with calculated results
- Provide comprehensive statistics

### Test with Limited Articles

```bash
# Test with dry run (no database updates)
python test_price_movement.py --dry-run --limit 100

# Test with small batch size
python test_price_movement.py --batch-size 10 --limit 50

# Run actual analysis with smaller batches
python test_price_movement.py --batch-size 25
```

## Expected Output

### During Processing
```
2024-01-15 10:30:00 - INFO - 🚀 Starting Price Movement Analysis...
2024-01-15 10:30:01 - INFO - ✅ Price Movement Analyzer initialized successfully
2024-01-15 10:30:01 - INFO - 📊 price_movement_30pct column already exists
2024-01-15 10:30:02 - INFO - 📄 Found 1,250 articles needing analysis (2025+ only, filtered for 6:00-9:00 AM EST)
2024-01-15 10:30:02 - INFO - 📊 Processing batch of 50 articles...
2024-01-15 10:30:05 - INFO - 📈 30%+ increase found: AAPL - Apple Announces Revolutionary New Product...
2024-01-15 10:30:10 - INFO - 📈 PROGRESS: 100 articles analyzed
2024-01-15 10:30:10 - INFO -   • With price data: 87
2024-01-15 10:30:10 - INFO -   • With 30%+ increase: 12
2024-01-15 10:30:10 - INFO -   • No price data: 13
```

### Final Statistics
```
🎉 PRICE MOVEMENT ANALYSIS COMPLETE!
📊 FINAL STATS:
  • Total articles analyzed: 1,250
  • Articles with price data: 1,089
  • Articles with 30%+ increase: 87
  • Articles without price data: 161
  • 30%+ increase rate: 7.99%
  • Time elapsed: 0:15:32
```

## Database Schema Changes

The script automatically adds this column to `historical_news`:

```sql
ALTER TABLE News.historical_news 
ADD COLUMN price_movement_30pct Nullable(UInt8) DEFAULT NULL
```

## Querying Results

After running the analyzer, you can query the results:

```sql
-- Get articles with 30%+ price increases
SELECT ticker, headline, published_utc, price_movement_30pct
FROM News.historical_news 
WHERE price_movement_30pct = 1
ORDER BY published_utc DESC;

-- Get summary statistics
SELECT 
    price_movement_30pct,
    COUNT(*) as article_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM News.historical_news 
WHERE price_movement_30pct IS NOT NULL
GROUP BY price_movement_30pct;

-- Get results by ticker
SELECT 
    ticker,
    COUNT(*) as total_articles,
    SUM(price_movement_30pct) as articles_with_30pct_increase,
    AVG(price_movement_30pct) * 100 as success_rate_percent
FROM News.historical_news 
WHERE price_movement_30pct IS NOT NULL
GROUP BY ticker
HAVING total_articles >= 10
ORDER BY success_rate_percent DESC;
```

## Time Filtering Logic

The script only analyzes articles published between **6:00 AM to 9:00 AM EST** in **2025 onwards** because:

1. **Price Data Availability**: Only 6 months of historical price data is available (2025 onwards)
2. **Market Context**: This is the pre-market period when news can significantly impact opening prices
3. **Analysis Window**: Ensures there's enough time between publication (+ 30 seconds) and market open (9:30 AM EST)
4. **Realistic Trading**: Aligns with the backtesting system's focus on early morning newswire articles

## Performance Considerations

- **Batch Processing**: Processes articles in batches (default 50) to manage memory usage
- **Rate Limiting**: 100ms delay between articles to avoid overwhelming the database
- **Incremental**: Only processes articles without existing labels (resumable)
- **Efficient Queries**: Uses indexed columns (ticker, timestamp) for fast price lookups

## Troubleshooting

### No Articles Found
```
📄 Found 0 articles needing analysis (2025+ only, filtered for 6:00-9:00 AM EST)
```
**Solution**: Check that `historical_news` table has articles published in the 6:00-9:00 AM EST window from 2025 onwards.

### No Price Data
```
• Articles without price data: 500
```
**Solution**: Ensure `historical_price` table has 10-second bars for the relevant tickers and dates.

### Column Already Exists Error
The script handles this automatically by checking if the column exists before adding it.

### Database Connection Issues
Verify ClickHouse is running and `clickhouse_setup.py` is configured correctly.

## Integration with Backtesting System

This analyzer complements the existing backtesting pipeline:

1. **Step 1-2**: Run normal backtesting (scrape news, fetch prices)
2. **Step 2.5**: Run price movement analyzer → `python price_movement_analyzer.py`
3. **Step 3-6**: Continue with sentiment analysis and trade simulation

The price movement labels can be used to:
- Pre-filter articles for sentiment analysis (focus on high-potential articles)
- Validate trading strategy performance
- Research correlation between news sentiment and actual price movements
- Identify the most effective newswire sources

## Advanced Usage

### Custom Batch Sizes
```bash
# Larger batches (faster, more memory)
python price_movement_analyzer.py --batch-size 100

# Smaller batches (slower, less memory)
python test_price_movement.py --batch-size 10
```

### Integration with Other Scripts
```python
from price_movement_analyzer import PriceMovementAnalyzer

# Use within other scripts
analyzer = PriceMovementAnalyzer()
await analyzer.initialize()
result = await analyzer.analyze_article(article_data)
await analyzer.cleanup()
```

## Expected Results

Based on financial markets research, typical results might show:
- **Overall 30%+ Rate**: 2-8% of pre-market news articles
- **High-Impact Sources**: Certain newswires may show higher success rates
- **Sector Variations**: Biotech, small-cap stocks may show higher rates
- **Seasonal Patterns**: Earnings seasons may show different patterns

This data provides valuable insights into which news articles historically preceded significant price movements, helping improve trading strategy development and backtesting accuracy. 