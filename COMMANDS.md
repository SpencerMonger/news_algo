# NewsHead System - Complete Command Reference

## System Operations

### Basic Operations

```bash
# Normal operation (freshness filter enabled, web scraper, with sentiment analysis)
python3 run_system.py --skip-list

# Testing mode (freshness filter disabled, processes old articles)
python3 run_system.py --skip-list --enable-old
```

### WebSocket Operations

```bash
# Use Benzinga WebSocket instead of web scraper (normal filtering)
python3 run_system.py --skip-list --socket

# WebSocket with old articles processing
python3 run_system.py --skip-list --socket --enable-old

# WebSocket with ANY ticker processing (bypasses ticker list filtering)
python3 run_system.py --skip-list --socket --any

# WebSocket with ANY ticker + old articles
python3 run_system.py --skip-list --socket --any --enable-old
```

### Sentiment Analysis Variations

```bash
# Skip sentiment analysis initialization (for testing)
python3 run_system.py --skip-list --no-sentiment

# WebSocket without sentiment analysis
python3 run_system.py --skip-list --socket --no-sentiment

# Any ticker processing without sentiment analysis
python3 run_system.py --skip-list --socket --any --no-sentiment
```

### Full System Operations (with Finviz ticker list update)

```bash
# Full system with ticker list update (web scraper)
python3 run_system.py

# Full system with ticker list update (websocket)
python3 run_system.py --socket

# Full system with ticker list update (websocket + any ticker)
python3 run_system.py --socket --any
```

---

## Test Commands

### RAG Comparison Tests

```bash
# Basic RAG vs Traditional comparison
python3 tests/rag_comparison_test.py

# With custom sample size
python3 tests/rag_comparison_test.py --sample-size 50

# Test modes: traditional only, rag only, or parallel (both)
python3 tests/rag_comparison_test.py --test-mode traditional
python3 tests/rag_comparison_test.py --test-mode rag
python3 tests/rag_comparison_test.py --test-mode parallel

# With PnL calculations (requires POLYGON_API_KEY)
python3 tests/rag_comparison_test.py --sample-size 30 --test-mode parallel --pnl

# Custom confidence thresholds
python3 tests/rag_comparison_test.py --buy-high-threshold 0.8 --buy-medium-threshold 0.5

# Disable result saving
python3 tests/rag_comparison_test.py --no-save-results

# Complete example with all options
python3 tests/rag_comparison_test.py --sample-size 50 --test-mode parallel --pnl --buy-high-threshold 0.8
```

### Breaking News RAG Tests

```bash
# Basic breaking news analysis
python3 tests/breaking_news_rag_test.py

# With custom parameters
python3 tests/breaking_news_rag_test.py --limit 20 --hours-back 24

# Filter by specific ticker
python3 tests/breaking_news_rag_test.py --ticker AAPL --limit 10

# Custom confidence threshold
python3 tests/breaking_news_rag_test.py --buy-high-threshold 0.8

# Complete example
python3 tests/breaking_news_rag_test.py --limit 15 --hours-back 12 --ticker NVDA --buy-high-threshold 0.85
```

### Vector Generation and Management

```bash
# Generate E5 embedding vectors for training data
python3 tests/generate_vectors.py

# With custom batch size
python3 tests/generate_vectors.py --batch-size 25

# Verify existing vectors only (no generation)
python3 tests/generate_vectors.py --verify-only
```

### RAG Embedding Tests

```bash
# Test RAG embedding functionality
python3 tests/rag_embedding_test.py

# With custom sample size
python3 tests/rag_embedding_test.py --sample-size 20

# Save detailed results
python3 tests/rag_embedding_test.py --save-results
```

### RAG Similarity Tests

```bash
# Test similarity matching
python3 tests/rag_similarity_test.py

# With custom parameters
python3 tests/rag_similarity_test.py --sample-size 15 --top-k 5

# Test specific ticker
python3 tests/rag_similarity_test.py --ticker AAPL --top-k 3
```

### Training Data Management

```bash
# Create train/test data split
python3 tests/create_train_test_split.py

# With custom split ratios
python3 tests/create_train_test_split.py --train-ratio 0.8 --test-ratio 0.2

# Scrape training content from URLs
python3 tests/scrape_training_content.py

# With custom batch size and delay
python3 tests/scrape_training_content.py --batch-size 10 --delay 2
```

---

## Backtesting Commands

### Trade Simulation

```bash
# Normal trade simulation (requires BUY+high sentiment)
python3 Backtesting/trade_simulation.py

# Testing mode (price movement only, bypasses sentiment requirements)
python3 Backtesting/trade_simulation.py --skip-sentiment
```

### Price Movement Analysis

```bash
# Analyze historical price movements
python3 Backtesting/price_movement_analyzer.py

# With custom date range
python3 Backtesting/price_movement_analyzer.py --start-date 2025-01-01 --end-date 2025-07-31

# Specific ticker analysis
python3 Backtesting/price_movement_analyzer.py --ticker AAPL
```

---

## Argument Reference

### System Arguments
- `--skip-list`: Skip Finviz ticker list update
- `--enable-old`: Process old news articles (disable freshness filter)
- `--socket`: Use Benzinga WebSocket instead of web scraper
- `--any`: Process any ticker symbols found (only works with --socket)
- `--no-sentiment`: Skip sentiment analysis initialization (for testing)

### RAG Comparison Test Arguments
- `--sample-size N`: Number of test articles to analyze (default: 30)
- `--test-mode MODE`: Test mode - `traditional`, `rag`, or `parallel` (default: parallel)
- `--pnl`: Calculate actual PnL based on BUY+high recommendations
- `--buy-high-threshold N`: Confidence threshold for BUY+high (default: 0.8)
- `--buy-medium-threshold N`: Confidence threshold for BUY+medium (default: 0.5)
- `--save-results`: Save test results (default: true)

### Breaking News Test Arguments
- `--limit N`: Number of articles to analyze (default: 20)
- `--hours-back N`: Hours back to look for articles (default: 24)
- `--ticker SYMBOL`: Filter by specific ticker
- `--buy-high-threshold N`: Confidence threshold for BUY+high (default: 0.8)

### Vector Generation Arguments
- `--batch-size N`: Batch size for processing (default: 25)
- `--verify-only`: Only verify existing vectors, don't generate new ones

### Trade Simulation Arguments
- `--skip-sentiment`: Skip sentiment requirements for testing (price movement only)

---

## Recommended Configurations

### Production Usage
```bash
# Most common production setup
python3 run_system.py --skip-list

# High-volume testing with WebSocket
python3 run_system.py --skip-list --socket --any
```

### Development & Testing
```bash
# RAG system testing with PnL analysis
python3 tests/rag_comparison_test.py --sample-size 30 --test-mode parallel --pnl

# Breaking news analysis
python3 tests/breaking_news_rag_test.py --limit 15 --hours-back 12

# Vector generation for new training data
python3 tests/generate_vectors.py --batch-size 25
```

### Debugging
```bash
# System without sentiment analysis
python3 run_system.py --skip-list --socket --no-sentiment

# Verify vectors only
python3 tests/generate_vectors.py --verify-only
```

---

## Utility Commands

### Log Analysis
```bash
# Find tickers blocked by sentiment filter
grep "blocked by sentiment filter" logs/run_system.log.2025-07-24 | grep -o '[A-Z][A-Z][A-Z][A-Z]*:' | cut -d':' -f1 | sort | uniq
```

### Environment Setup
```bash
# Install required packages
pip install sentence-transformers
pip install aiohttp pytz python-dotenv clickhouse-connect

# Set environment variables
export POLYGON_API_KEY="your_api_key_here"
export PROXY_URL="your_proxy_url_here"  # Optional
``` 