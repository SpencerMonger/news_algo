# RAG Implementation Guide for NewsHead Sentiment Analysis

## Overview
This guide provides a complete implementation plan for integrating Retrieval-Augmented Generation (RAG) into the existing NewsHead sentiment analysis system. The RAG system will enhance Claude's sentiment analysis by providing relevant historical examples from the `price_movement_analysis` table that contain **labeled outcomes** to improve BUY/SELL/HOLD recommendations with appropriate confidence levels.

## Architecture Integration

### Current Flow
```
News Article → Claude Sentiment Analysis → Database Insert → Price Alert (BUY+high only)
```

### Enhanced RAG Flow
```
News Article → Claude Feature Extraction → Find Similar Labeled Articles → Enhanced Claude Prompt → Database Insert → Price Alert (BUY+high only)
```

## Labeled Data Strategy

### Data Classification from `price_movement_analysis` Table

**High-Confidence BUY Signals** (`has_30pt_increase = 1`):
- Articles that led to actual 30%+ price increases
- Should be used as positive examples to support BUY+high recommendations
- These represent "real" bullish signals that produced desired outcomes

**Medium-Confidence BUY Signals** (`is_false_pump = 1`):
- Articles that appeared bullish but were fake pump-and-dumps
- Should be used as cautionary examples to prevent BUY+high recommendations
- These represent "false positives" that should result in BUY+medium or HOLD

**Neutral/Hold Signals** (both columns = 0):
- Articles that produced no significant price movement
- Should be used as examples supporting HOLD or SELL recommendations
- These represent market noise or truly neutral events

### RAG Context Strategy

**For potentially bullish articles**, the RAG system will provide:
1. **2-3 similar articles with `has_30pt_increase = 1`** → Support BUY+high confidence
2. **1-2 similar articles with `is_false_pump = 1`** → Warn against overconfidence, suggest BUY+medium
3. **1 similar article with both = 0** → Show neutral baseline

**Goal**: Help Claude distinguish between articles that will produce real moves vs. fake pumps

## Phase 1: Database Setup and Historical Data Preparation

### 1.1 RAG Training Data Extraction

**Source Table**: `News.price_movement_analysis` (already exists with labeled data)

**Key Columns**:
- `ticker`, `headline`, `article_url`, `published_est`
- `has_30pct_increase` (1 = true bullish signal)
- `is_false_pump` (1 = false bullish signal)
- `content_hash` (for deduplication)

**Data Quality Requirements**:
- Focus on articles with clear binary outcomes (not both columns = 1)
- Prioritize articles with rich content (headline + full article text)
- Ensure diverse ticker representation across different market conditions

### 1.2 Create RAG Vector Storage Table

**ClickHouse Table Schema:**
```sql
CREATE TABLE News.rag_article_features (
    id UUID DEFAULT generateUUIDv4(),
    original_content_hash String,
    ticker String,
    headline String,
    full_content String,  -- Scraped/full article content
    
    -- Claude-generated features (instead of embeddings)
    features Array(Float32),  -- Feature vector from Claude analysis
    feature_model String DEFAULT 'claude-3-5-sonnet-20240620',
    
    -- Outcome labels from price_movement_analysis
    outcome_type String,  -- 'TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL'
    has_30pt_increase UInt8,
    is_false_pump UInt8,
    price_increase_ratio Float64,
    max_price_ratio Float64,
    
    -- Original article metadata
    published_est DateTime,
    article_url String,
    created_at DateTime DEFAULT now(),
    
    -- Indexing for fast retrieval
    INDEX idx_outcome_type (outcome_type) TYPE set(10) GRANULARITY 1,
    INDEX idx_ticker (ticker) TYPE set(1000) GRANULARITY 1,
    INDEX idx_has_30pt (has_30pt_increase) TYPE set(10) GRANULARITY 1,
    INDEX idx_false_pump (is_false_pump) TYPE set(10) GRANULARITY 1
) ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(published_est)
ORDER BY (ticker, original_content_hash)
```

### 1.3 Environment Configuration

```bash
# Use existing Claude API keys (no additional keys needed)
# RAG will use the same Claude API as sentiment analysis
RAG_ENABLED=true
RAG_SIMILARITY_THRESHOLD=0.75
RAG_MAX_TRUE_BULLISH_EXAMPLES=2  # Articles with has_30pt_increase=1
RAG_MAX_FALSE_PUMP_EXAMPLES=1    # Articles with is_false_pump=1  
RAG_MAX_NEUTRAL_EXAMPLES=1       # Articles with both=0
RAG_FEATURE_CACHE_SIZE=10000
```

## Phase 2: Core RAG Components Implementation

### 2.1 RAG Service Architecture

**New Module: `rag_service.py`**

**Key Components**:
1. **ClaudeFeatureExtractor**: Uses Claude to extract semantic features from articles
2. **LabeledArticleRetriever**: Finds similar articles by outcome type using feature similarity
3. **ContextBuilder**: Constructs enhanced prompts with historical examples
4. **OutcomeClassifier**: Maps price_movement_analysis labels to context types

### 2.2 Claude-Based Feature Extraction

**Feature Extraction Strategy:**

```python
async def extract_article_features(self, article_content: str) -> List[float]:
    """Extract semantic features using Claude analysis"""
    
    analysis_prompt = f"""
Analyze the following news article and extract key features for similarity comparison.
Focus on: sentiment, topic, urgency, market impact, company type, and news type.

Article: {article_content[:2000]}

Respond with a JSON object containing numerical scores (0.0 to 1.0) for these features:
{{
    "sentiment_score": 0.0-1.0,
    "bullish_score": 0.0-1.0, 
    "urgency_score": 0.0-1.0,
    "financial_impact_score": 0.0-1.0,
    "earnings_related": 0.0-1.0,
    "partnership_related": 0.0-1.0,
    "product_related": 0.0-1.0,
    "regulatory_related": 0.0-1.0,
    "market_general": 0.0-1.0,
    "biotech_pharma": 0.0-1.0,
    "tech_software": 0.0-1.0,
    "finance_banking": 0.0-1.0,
    "energy_commodities": 0.0-1.0,
    "retail_consumer": 0.0-1.0,
    "manufacturing": 0.0-1.0
}}
"""
    
    # Use existing Claude API through sentiment service
    result = await self.sentiment_service.load_balancer.make_claude_request(analysis_prompt)
    
    if result and isinstance(result, dict):
        # Convert Claude's analysis to feature vector
        feature_vector = [
            result.get('sentiment_score', 0.5),
            result.get('bullish_score', 0.5),
            result.get('urgency_score', 0.5),
            result.get('financial_impact_score', 0.5),
            result.get('earnings_related', 0.0),
            result.get('partnership_related', 0.0),
            result.get('product_related', 0.0),
            result.get('regulatory_related', 0.0),
            result.get('market_general', 0.0),
            result.get('biotech_pharma', 0.0),
            result.get('tech_software', 0.0),
            result.get('finance_banking', 0.0),
            result.get('energy_commodities', 0.0),
            result.get('retail_consumer', 0.0),
            result.get('manufacturing', 0.0)
        ]
        
        # Add derived features
        while len(feature_vector) < 50:
            if len(feature_vector) < 30:
                feature_vector.append(feature_vector[0] * feature_vector[1])  # sentiment * bullish
            else:
                feature_vector.append(0.5)  # neutral padding
        
        return feature_vector
    else:
        # Return neutral feature vector on failure
        return [0.5] * 50
```

### 2.3 Enhanced Prompt Engineering

**RAG-Enhanced Prompt Structure:**

```python
async def create_rag_enhanced_prompt(self, article: Dict[str, Any], similar_articles: Dict[str, List]) -> str:
    """Create enhanced prompt with labeled historical examples"""
    
    ticker = article.get('ticker', 'UNKNOWN')
    content = article.get('content_to_analyze', '')
    
    prompt = f"""
Analyze the following news article about {ticker} for sentiment and trading recommendation.

CURRENT ARTICLE TO ANALYZE:
{content}

HISTORICAL CONTEXT - SIMILAR ARTICLES THAT LED TO REAL GAINS:
"""
    
    # Add TRUE_BULLISH examples (has_30pt_increase=1)
    for i, example in enumerate(similar_articles.get('TRUE_BULLISH', []), 1):
        prompt += f"""
Example {i} - TICKER: {example['ticker']} (GAINED {example['price_increase_ratio']:.1%})
Headline: {example['headline']}
Outcome: This article led to a {example['price_increase_ratio']:.1%} price increase, confirming strong bullish sentiment.
"""

    prompt += """
HISTORICAL CONTEXT - SIMILAR ARTICLES THAT WERE FALSE PUMPS:
"""
    
    # Add FALSE_PUMP examples (is_false_pump=1)  
    for i, example in enumerate(similar_articles.get('FALSE_PUMP', []), 1):
        prompt += f"""
Example {i} - TICKER: {example['ticker']} (FAKE PUMP)
Headline: {example['headline']}
Outcome: This appeared bullish but was a false pump - peaked at {example['max_price_ratio']:.1%} then fell back to {example['price_increase_ratio']:.1%}.
"""

    prompt += f"""
ANALYSIS INSTRUCTIONS:
Based on the historical patterns above:

1. If the current article resembles the TRUE BULLISH examples → Consider BUY with HIGH confidence
2. If the current article resembles the FALSE PUMP examples → Consider BUY with MEDIUM confidence or HOLD
3. Analyze sentiment (positive, negative, neutral)
4. Consider potential for false pump vs real sustained move
5. Provide recommendation: BUY (high confidence for real moves, medium for uncertain), SELL, or HOLD

Respond in this exact JSON format:
{{
    "ticker": "{ticker}",
    "sentiment": "positive/negative/neutral", 
    "recommendation": "BUY/SELL/HOLD",
    "confidence": "high/medium/low",
    "explanation": "Brief explanation comparing to historical examples"
}}

Important: Only use BUY+high for articles that strongly resemble the TRUE BULLISH historical examples.
"""
    
    return prompt
```

### 2.4 Similarity Search Implementation

**Claude Feature-Based Similarity:**

```python
async def find_similar_labeled_articles(self, article_features: List[float], ticker: str) -> Dict[str, List]:
    """Find similar articles grouped by outcome type using Claude features"""
    
    # Use ClickHouse vector similarity (cosine similarity)
    query = """
    SELECT 
        ticker, headline, full_content, article_url,
        has_30pt_increase, is_false_pump,
        price_increase_ratio, max_price_ratio,
        cosineDistance(features, %(features)s) as similarity
    FROM News.rag_article_features
    WHERE similarity < %(threshold)s
    ORDER BY similarity ASC
    LIMIT 20
    """
    
    results = await self.ch_manager.query(query, {
        'features': article_features,
        'threshold': 0.25  # Cosine distance threshold
    })
    
    # Group by outcome type
    grouped_results = {
        'TRUE_BULLISH': [],  # has_30pt_increase=1
        'FALSE_PUMP': [],    # is_false_pump=1
        'NEUTRAL': []        # both=0
    }
    
    for row in results:
        if row['has_30pt_increase'] == 1:
            grouped_results['TRUE_BULLISH'].append(row)
        elif row['is_false_pump'] == 1:
            grouped_results['FALSE_PUMP'].append(row)
        else:
            grouped_results['NEUTRAL'].append(row)
    
    # Limit examples per category
    return {
        'TRUE_BULLISH': grouped_results['TRUE_BULLISH'][:2],
        'FALSE_PUMP': grouped_results['FALSE_PUMP'][:1], 
        'NEUTRAL': grouped_results['NEUTRAL'][:1]
    }
```

## Phase 3: Integration with Existing Architecture

### 3.1 Modify `sentiment_service.py`

**Enhanced Analysis Method:**

```python
async def analyze_article_sentiment_with_rag(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze sentiment with RAG enhancement using labeled historical data"""
    
    try:
        # Check if RAG is enabled
        if not self.rag_enabled:
            return await self.analyze_article_sentiment(article)
        
        # Extract features for current article using Claude
        content_to_analyze = self.extract_content_for_analysis(article)
        article_features = await self.rag_service.extract_article_features(content_to_analyze)
        
        # Find similar labeled articles
        similar_articles = await self.rag_service.find_similar_labeled_articles(
            article_features, article.get('ticker', '')
        )
        
        # Create enhanced prompt with historical context
        enhanced_prompt = await self.create_rag_enhanced_prompt(article, similar_articles)
        
        # Analyze with Claude using enhanced prompt (uses existing load balancer)
        result = await self.load_balancer.make_claude_request(enhanced_prompt)
        
        # Add RAG metadata to result
        result['rag_used'] = True
        result['similar_articles_found'] = {
            'true_bullish': len(similar_articles.get('TRUE_BULLISH', [])),
            'false_pump': len(similar_articles.get('FALSE_PUMP', [])),
            'neutral': len(similar_articles.get('NEUTRAL', []))
        }
        
        return result
        
    except Exception as e:
        logger.warning(f"RAG analysis failed, falling back to traditional: {e}")
        return await self.analyze_article_sentiment(article)
```

### 3.2 A/B Testing Framework

**Parallel Analysis for Comparison:**

```python
async def analyze_article_with_comparison(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """Run both traditional and RAG analysis for comparison"""
    
    # Run both analyses in parallel
    traditional_task = self.analyze_article_sentiment(article)
    rag_task = self.analyze_article_sentiment_with_rag(article)
    
    traditional_result, rag_result = await asyncio.gather(
        traditional_task, rag_task, return_exceptions=True
    )
    
    # Store comparison results
    comparison = {
        'traditional': traditional_result,
        'rag_enhanced': rag_result,
        'agreement': {
            'recommendation': traditional_result.get('recommendation') == rag_result.get('recommendation'),
            'confidence': traditional_result.get('confidence') == rag_result.get('confidence'),
            'sentiment': traditional_result.get('sentiment') == rag_result.get('sentiment')
        }
    }
    
    # Log comparison for analysis
    await self.log_comparison_result(article, comparison)
    
    # Return RAG result for actual use, traditional for comparison
    return rag_result if not isinstance(rag_result, Exception) else traditional_result
```

## Phase 4: Test Framework Implementation

### 4.1 Test File Structure

**Create dedicated test files before integration**:

```
tests/
├── rag_comparison_test.py          # Main comparison test
├── rag_embedding_test.py           # Test Claude feature extraction  
├── rag_similarity_test.py          # Test similarity search
├── run_all_tests.py               # Automated test runner
└── results/
    ├── comparison_results.json      # A/B test results
    └── performance_metrics.json     # Performance comparison
```

### 4.2 Key Test Scenarios

**Test Cases**:
1. **True Bullish Articles**: Articles similar to `has_30pt_increase=1` examples
2. **False Pump Articles**: Articles similar to `is_false_pump=1` examples  
3. **Neutral Articles**: Articles similar to both=0 examples
4. **Edge Cases**: Articles with no similar historical examples

**Success Metrics**:
- RAG system identifies true bullish signals with BUY+high confidence
- RAG system avoids false pumps (prevents BUY+high, suggests BUY+medium or HOLD)
- Traditional vs RAG accuracy comparison
- Latency impact analysis

## Phase 5: Implementation Timeline

### Week 1-2: Data Preparation & Feature Storage
- [ ] Extract and clean data from `price_movement_analysis` table
- [ ] Create `rag_article_features` table
- [ ] Generate Claude-based features for all labeled articles
- [ ] Implement similarity search functionality using ClickHouse

### Week 3-4: RAG Service Development  
- [ ] Build `rag_service.py` with Claude feature extraction and similarity components
- [ ] Implement enhanced prompt generation with historical examples
- [ ] Create outcome classification logic (TRUE_BULLISH, FALSE_PUMP, NEUTRAL)
- [ ] Add caching and error handling

### Week 5-6: Test Framework & Comparison
- [ ] Create comprehensive test suite
- [ ] Implement A/B testing framework
- [ ] Run parallel analysis (traditional vs RAG)
- [ ] Analyze performance differences and accuracy improvements

### Week 7-8: Integration & Production Rollout
- [ ] Integrate RAG into `sentiment_service.py` 
- [ ] Deploy with gradual rollout (10% → 50% → 100%)
- [ ] Monitor BUY+high vs BUY+medium distribution changes
- [ ] Validate trading performance improvements

## Success Criteria

**Primary Goals**:
- **Reduce False Positives**: Fewer BUY+high recommendations for articles similar to false pumps
- **Improve True Positives**: More BUY+high recommendations for articles similar to real gainers
- **Maintain Performance**: RAG adds <300ms to analysis latency
- **Trading Improvements**: Measurable improvement in trading success rate

**Technical Metrics**:
- RAG accuracy > traditional analysis accuracy by 10%+
- False positive rate reduction of 15%+
- True positive rate improvement of 10%+
- System uptime and reliability maintained

## Key Advantages of Claude-Based RAG

### 1. **Unified Architecture**
- Uses same Claude API as existing sentiment analysis
- Leverages existing native load balancing across multiple API keys
- No additional API dependencies or costs

### 2. **Semantic Understanding**
- Claude extracts meaningful semantic features, not just statistical patterns
- Features are interpretable and domain-specific (financial news)
- Better understanding of context and nuance

### 3. **Cost Efficiency**
- No separate embedding API costs
- Reuses existing Claude API quota
- Feature extraction integrated with sentiment analysis

### 4. **Reliability**
- Uses proven Claude API infrastructure
- Benefits from existing error handling and failover
- Consistent with system's reliability patterns

This updated RAG implementation leverages your existing Claude 3.5 Sonnet infrastructure to improve the critical BUY+high vs BUY+medium distinction, which directly impacts your trading filter. The system uses Claude for both feature extraction and enhanced prompting, maintaining architectural consistency while adding powerful contextual intelligence. 