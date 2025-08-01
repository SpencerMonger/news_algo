# RAG Test Framework for NewsHead Sentiment Analysis

This test framework is designed to validate the RAG (Retrieval-Augmented Generation) system before integrating it into the live NewsHead sentiment analysis pipeline.

## Overview

The test framework compares traditional sentiment analysis against RAG-enhanced analysis using labeled data from the `price_movement_analysis` table to validate improvements in accuracy and trading precision.

## Test Files

### 1. `rag_comparison_test.py` - Main Comparison Test
**Purpose**: Compare traditional vs RAG sentiment analysis using labeled historical data

**Key Features**:
- Uses real labeled data from `price_movement_analysis` table
- Calculates accuracy scores based on actual price movement outcomes
- Measures BUY+high precision (critical for trading decisions)
- Provides comprehensive performance metrics
- Saves detailed comparison results

**Usage**:
```bash
# Run basic comparison test with 50 articles
python3 tests/rag_comparison_test.py --sample-size 50

# Run traditional-only baseline test
python3 tests/rag_comparison_test.py --test-mode traditional --sample-size 100

# Run full parallel comparison
python3 tests/rag_comparison_test.py --test-mode parallel --sample-size 100 --save-results
```

**Key Metrics**:
- **Traditional vs RAG Accuracy**: Overall prediction accuracy
- **BUY+high Precision**: How many BUY+high predictions were actually TRUE_BULLISH
- **False Positive Reduction**: Improvement in avoiding FALSE_PUMP scenarios
- **Analysis Time**: Performance impact of RAG enhancement

### 2. `rag_embedding_test.py` - Embedding Generation Test
**Purpose**: Test embedding generation and basic similarity search functionality

**Key Features**:
- Tests OpenAI embedding API integration
- Validates embedding generation performance
-Creates test ClickHouse tables for RAG data
- Tests vector similarity search in ClickHouse
- Measures embedding generation latency

**Usage**:
```bash
# Test basic embedding generation
python3 tests/rag_embedding_test.py --sample-size 20

# Test with similarity search
python3 tests/rag_embedding_test.py --test-similarity --sample-size 20

# Create and populate test tables
python3 tests/rag_embedding_test.py --create-table --populate-table --sample-size 50

# Test ClickHouse vector similarity
python3 tests/rag_embedding_test.py --test-clickhouse --sample-size 20
```

**Key Metrics**:
- **Embedding Success Rate**: Percentage of successful embedding generations
- **Generation Time**: Average time to generate embeddings
- **Similarity Correlation**: How well embeddings correlate with outcome types
- **ClickHouse Performance**: Vector similarity search performance

### 3. `rag_similarity_test.py` - Similarity Search Validation
**Purpose**: Validate that similarity search properly retrieves relevant historical examples

**Key Features**:
- Tests outcome correlation (similar articles have similar outcomes)
- Validates RAG precision for trading decisions
- Measures cross-contamination between outcome types
- Analyzes separation quality between TRUE_BULLISH and FALSE_PUMP

**Usage**:
```bash
# Run full similarity validation
python3 tests/rag_similarity_test.py

# Test only outcome correlation
python3 tests/rag_similarity_test.py --test-outcome-correlation --no-validate-precision

# Test only RAG precision validation
python3 tests/rag_similarity_test.py --validate-precision --no-test-outcome-correlation
```

**Key Metrics**:
- **Outcome Precision**: How often similar articles have same outcome
- **Top-K Accuracy**: Percentage of top similar articles with same outcome
- **Separation Quality**: How well TRUE_BULLISH and FALSE_PUMP are distinguished
- **Contamination Risk**: Risk of FALSE_PUMP examples influencing TRUE_BULLISH decisions

## Expected Test Results

### Success Criteria

**Primary Goals**:
1. **Reduce False Positives**: RAG should reduce BUY+high recommendations for FALSE_PUMP articles
2. **Improve True Positives**: RAG should increase BUY+high recommendations for TRUE_BULLISH articles
3. **Maintain Performance**: RAG should add <300ms to analysis time
4. **High Precision**: BUY+high precision should be >80% (80% of BUY+high should be TRUE_BULLISH)

**Technical Benchmarks**:
- RAG accuracy improvement: >10% over traditional analysis
- False positive reduction: >15% reduction in FALSE_PUMP → BUY+high errors
- TRUE_BULLISH recall: >90% of TRUE_BULLISH articles should get BUY recommendation
- Similarity precision: >70% of similar articles should have same outcome type

### Data Classification

The test framework uses three outcome types from `price_movement_analysis`:

1. **TRUE_BULLISH**: `has_30pt_increase = 1`
   - Articles that led to actual 30%+ price increases
   - **Expected**: BUY with high confidence
   - **RAG Goal**: Provide similar examples to support BUY+high decisions

2. **FALSE_PUMP**: `is_false_pump = 1`  
   - Articles that appeared bullish but were fake pumps
   - **Expected**: BUY with medium confidence or HOLD
   - **RAG Goal**: Provide cautionary examples to prevent BUY+high decisions

3. **NEUTRAL**: Both columns = 0
   - Articles with no significant price movement
   - **Expected**: HOLD or SELL
   - **RAG Goal**: Provide examples showing minimal market impact

## Test Workflow

### Phase 1: Baseline Testing (Traditional Only)
```bash
# Establish baseline performance
python3 tests/rag_comparison_test.py --test-mode traditional --sample-size 100
```

### Phase 2: Embedding System Testing
```bash
# Test embedding generation and similarity
python3 tests/rag_embedding_test.py --test-similarity --create-table --populate-table
```

### Phase 3: Similarity Validation
```bash
# Validate similarity search quality
python3 tests/rag_similarity_test.py --test-outcome-correlation --validate-precision
```

### Phase 4: RAG Comparison Testing
```bash
# Compare traditional vs RAG performance
python3 tests/rag_comparison_test.py --test-mode parallel --sample-size 100
```

## Results Analysis

### Key Files Generated
- `tests/results/comparison_results_YYYYMMDD_HHMMSS.json`: Detailed comparison results
- `tests/results/performance_metrics_YYYYMMDD_HHMMSS.json`: Summary performance metrics
- `tests/results/embedding_test_results_YYYYMMDD_HHMMSS.json`: Embedding test results
- `tests/results/similarity_test_results_YYYYMMDD_HHMMSS.json`: Similarity validation results

### Critical Metrics to Monitor

1. **BUY+high Precision** (Most Important):
   - Traditional: Baseline precision rate
   - RAG: Should be significantly higher
   - **Goal**: >80% precision (80% of BUY+high should be TRUE_BULLISH)

2. **FALSE_PUMP Avoidance**:
   - Traditional: Rate of FALSE_PUMP → BUY+high errors
   - RAG: Should be much lower
   - **Goal**: <10% of FALSE_PUMP articles should get BUY+high

3. **TRUE_BULLISH Recall**:
   - Traditional: Percentage of TRUE_BULLISH articles that get BUY recommendation
   - RAG: Should maintain or improve
   - **Goal**: >90% of TRUE_BULLISH should get BUY (any confidence)

4. **Performance Impact**:
   - Analysis time increase due to RAG
   - **Goal**: <300ms additional latency per article

## Environment Setup

### Required Environment Variables
```bash
# For embedding generation (optional - will use dummy embeddings if not set)
export OPENAI_API_KEY_EMBEDDING=your_openai_api_key

# For sentiment analysis (existing)
export ANTHROPIC_API_KEY=your_claude_api_key
# Additional keys for load balancing
export ANTHROPIC_API_KEY2=your_second_key
export ANTHROPIC_API_KEY3=your_third_key
```

### Prerequisites
- ClickHouse database with `price_movement_analysis` table populated
- Python dependencies: `asyncio`, `aiohttp`, `numpy`, `pandas`
- Access to existing sentiment analysis service

## Integration Decision Criteria

**Proceed with RAG integration if**:
- BUY+high precision improves by >15%
- FALSE_PUMP avoidance improves by >20%
- Analysis latency increase <300ms
- Overall accuracy improvement >10%

**Do not integrate if**:
- RAG reduces TRUE_BULLISH recall
- Analysis time increases >500ms
- System reliability decreases
- No significant accuracy improvement

## Next Steps After Testing

1. **If tests pass**: Integrate RAG into `sentiment_service.py`
2. **If tests show promise but need tuning**: Adjust similarity thresholds and example counts
3. **If tests fail**: Investigate embedding quality, similarity algorithms, or data quality issues

## Troubleshooting

### Common Issues
- **No test articles found**: Check `price_movement_analysis` table has data
- **Embedding generation fails**: Verify `OPENAI_API_KEY_EMBEDDING` is set
- **ClickHouse connection errors**: Ensure database is running and accessible
- **Memory issues**: Reduce sample sizes for initial testing

### Debug Mode
Add `--verbose` or set logging level to DEBUG for detailed output:
```python
logging.basicConfig(level=logging.DEBUG)
```

This test framework provides comprehensive validation of the RAG system before production deployment, ensuring that the enhancement will actually improve trading decision quality. 