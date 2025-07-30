# RAG Implementation Guide for NewsHead Sentiment Analysis

## Overview
This guide provides a complete implementation plan for integrating Retrieval-Augmented Generation (RAG) into the existing NewsHead sentiment analysis system. The RAG system will enhance Claude's sentiment analysis by providing relevant historical examples of articles that led to known outcomes.

## Architecture Integration

### Current Flow
```
News Article → Claude Sentiment Analysis → Database Insert → Price Alert
```

### Enhanced RAG Flow
```
News Article → Generate Embedding → Find Similar Historical Articles → Enhanced Claude Prompt → Database Insert → Price Alert
```

## Phase 1: Database Setup and Historical Data Preparation

### 1.1 Create RAG Training Table

**ClickHouse Table Schema:**
```sql
CREATE TABLE News.rag_training_articles (
    id UUID DEFAULT generateUUIDv4(),
    original_article_id String,
    ticker String,
    headline String,
    content String,
    content_hash String,
    
    -- Embedding storage
    embedding Array(Float32),  -- 1536 dimensions for OpenAI text-embedding-3-small
    embedding_model String DEFAULT 'text-embedding-3-small',
    
    -- Outcome classification
    outcome_label String,      -- 'POSITIVE_OUTCOME', 'NEGATIVE_OUTCOME', 'NEUTRAL_OUTCOME'
    outcome_confidence String, -- 'high', 'medium', 'low'
    outcome_description String, -- Human-readable description of what happened
    
    -- Metadata
    original_date DateTime,
    added_to_rag DateTime DEFAULT now(),
    data_source String,       -- 'manual_review', 'automated_screening', etc.
    
    -- Indexing for fast retrieval
    INDEX idx_ticker (ticker) TYPE set(1000) GRANULARITY 1,
    INDEX idx_outcome (outcome_label) TYPE set(10) GRANULARITY 1,
    INDEX idx_confidence (outcome_confidence) TYPE set(10) GRANULARITY 1
) ENGINE = ReplacingMergeTree(added_to_rag)
PARTITION BY toYYYYMM(original_date)
ORDER BY (ticker, content_hash)
```

### 1.2 Historical Data Collection Strategy

**Data Sources:**
1. **Existing NewsHead Database**: Mine `News.breaking_news` table for articles with known outcomes
2. **Manual Curation**: Hand-pick clear positive/negative examples
3. **Automated Screening**: Use price movement data to identify clear winners/losers

**Recommended Data Distribution:**
- **60% Positive Examples**: Articles that led to desired outcomes
- **30% Negative Examples**: Articles that looked promising but failed
- **10% Neutral Examples**: Articles with no significant impact

**Quality Criteria:**
- Clear, unambiguous outcomes
- High-quality article content (not just headlines)
- Diverse ticker representation
- Various market conditions represented

### 1.3 Embedding Generation Setup

**Environment Variables:**
```bash
# Add to .env file
OPENAI_API_KEY_EMBEDDING=your_openai_key_for_embeddings
RAG_ENABLED=true
RAG_SIMILARITY_THRESHOLD=0.75
RAG_MAX_POSITIVE_EXAMPLES=2
RAG_MAX_NEGATIVE_EXAMPLES=1
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_CACHE_SIZE=10000
```

## Phase 2: Core RAG Components Implementation

### 2.1 Embedding Service Integration

**New Module: `rag_service.py`**

**Key Components:**
1. **EmbeddingGenerator**: Handles OpenAI embedding API calls with caching
2. **SimilaritySearch**: Efficient vector similarity search in ClickHouse
3. **ContextBuilder**: Constructs enhanced prompts with historical examples
4. **RAGCache**: Caches embeddings and similarity results

**Integration Points:**
- Modify `sentiment_service.py` to include RAG context
- Add RAG initialization to `run_system.py`
- Extend native load balancing to include embedding API calls

### 2.2 Enhanced Sentiment Analysis Flow

**Modified `SentimentService.analyze_article_sentiment()` Method:**

**Steps:**
1. **Check RAG Cache**: Look for cached similar articles
2. **Generate Embedding**: Create vector representation of current article
3. **Similarity Search**: Find top-K similar historical articles
4. **Context Assembly**: Build enhanced prompt with historical examples
5. **Claude Analysis**: Send enhanced prompt to Claude API
6. **Result Caching**: Cache results for future similarity searches

**Performance Optimizations:**
- Parallel embedding generation and similarity search
- Ticker-first filtering to reduce search space
- Embedding caching with LRU eviction
- Batch processing for multiple articles

### 2.3 Enhanced Prompt Engineering

**RAG-Enhanced Prompt Structure:**

```
SYSTEM CONTEXT:
You are analyzing news sentiment with access to historical examples.

CURRENT ARTICLE:
[Current article content]

HISTORICAL CONTEXT - POSITIVE EXAMPLES:
[2-3 similar articles that led to positive outcomes]

HISTORICAL CONTEXT - NEGATIVE EXAMPLES:  
[1 similar article that failed to produce results]

ANALYSIS INSTRUCTIONS:
Based on the historical patterns above, analyze the current article...
```

**Dynamic Context Selection:**
- Prioritize same-ticker examples
- Include cross-ticker examples for pattern recognition
- Balance positive and negative historical context
- Adjust similarity thresholds based on available examples

## Phase 3: Performance and Reliability Enhancements

### 3.1 Caching Strategy

**Multi-Level Caching:**
1. **Embedding Cache**: Store article embeddings (content_hash → embedding)
2. **Similarity Cache**: Store similarity search results (embedding_hash → similar_articles)
3. **Context Cache**: Store assembled contexts (context_hash → enhanced_prompt)

**Cache Management:**
- LRU eviction with configurable size limits
- Periodic cache warming for active tickers
- Cache persistence across system restarts

### 3.2 Error Handling and Fallbacks

**Graceful Degradation:**
1. **Embedding API Failure**: Fall back to traditional sentiment analysis
2. **Similarity Search Failure**: Use cached results or skip RAG
3. **Empty Results**: Proceed with traditional analysis
4. **Timeout Handling**: Set aggressive timeouts for RAG components

**Monitoring and Alerting:**
- Track RAG success/failure rates
- Monitor embedding API usage and costs
- Alert on degraded RAG performance

### 3.3 A/B Testing Framework

**Parallel Analysis System:**
- Run both traditional and RAG-enhanced analysis
- Compare results and track performance metrics
- Gradual rollout based on performance data

**Metrics to Track:**
- Sentiment prediction accuracy
- Recommendation precision/recall
- Analysis latency impact
- API cost implications

## Phase 4: Integration with Existing Architecture

### 4.1 Modify `sentiment_service.py`

**Key Changes:**
1. Add RAG initialization to `SentimentService.__init__()`
2. Modify `analyze_article_sentiment()` to include RAG context
3. Update `analyze_batch_articles()` for RAG processing
4. Extend statistics tracking for RAG metrics

**Backward Compatibility:**
- RAG can be disabled via environment variable
- Fallback to traditional analysis on RAG failures
- No changes to existing API interfaces

### 4.2 Update `run_system.py`

**System Initialization:**
1. Initialize RAG service after sentiment service
2. Populate embedding cache during startup
3. Add RAG health checks to system monitoring

**Configuration Options:**
- `--disable-rag`: Disable RAG for testing
- `--rag-rebuild-cache`: Rebuild embedding cache on startup
- `--rag-similarity-threshold`: Adjust similarity matching

### 4.3 Database Integration

**ClickHouse Extensions:**
1. Add RAG training table to `clickhouse_setup.py`
2. Create indexes for efficient similarity search
3. Add RAG metrics to existing monitoring queries

**Data Management:**
- Automated cleanup of old embeddings
- Periodic recomputation of embeddings with new models
- Data quality monitoring for RAG training set

## Phase 5: Deployment and Monitoring

### 5.1 Deployment Strategy

**Staged Rollout:**
1. **Week 1**: Deploy RAG infrastructure, populate training data
2. **Week 2**: Enable RAG for 10% of articles (A/B test)
3. **Week 3**: Increase to 50% if metrics improve
4. **Week 4**: Full rollout or rollback based on results

**Rollback Plan:**
- Environment variable to disable RAG instantly
- Automatic fallback on high error rates
- Database rollback procedures for schema changes

### 5.2 Performance Monitoring

**Key Metrics:**
- **Latency Impact**: RAG vs traditional analysis time
- **Accuracy Improvement**: Precision/recall of recommendations
- **API Costs**: Embedding generation costs vs accuracy gains
- **Cache Hit Rates**: Effectiveness of caching strategies

**Alerting Thresholds:**
- RAG failure rate > 5%
- Analysis latency increase > 20%
- Embedding API error rate > 2%
- Cache hit rate < 60%

### 5.3 Continuous Improvement

**Ongoing Optimization:**
1. **Monthly Training Data Updates**: Add new positive/negative examples
2. **Similarity Threshold Tuning**: Optimize based on performance data
3. **Embedding Model Updates**: Migrate to newer/better embedding models
4. **Prompt Engineering**: Refine RAG prompts based on results

**Quality Assurance:**
- Regular audit of RAG training data quality
- Manual review of RAG-influenced decisions
- Feedback loop from trading results to RAG training

## Implementation Timeline

### Week 1-2: Foundation
- [ ] Create ClickHouse RAG training table
- [ ] Implement basic embedding generation
- [ ] Populate initial training dataset (500+ examples)
- [ ] Build similarity search functionality

### Week 3-4: Integration
- [ ] Modify sentiment_service.py for RAG integration
- [ ] Implement caching and error handling
- [ ] Add RAG configuration to run_system.py
- [ ] Create A/B testing framework

### Week 5-6: Testing and Optimization
- [ ] Deploy in parallel mode (both traditional and RAG)
- [ ] Compare performance metrics
- [ ] Tune similarity thresholds and example counts
- [ ] Optimize for latency and accuracy

### Week 7-8: Production Rollout
- [ ] Gradual rollout to production traffic
- [ ] Monitor performance and costs
- [ ] Full deployment or rollback decision
- [ ] Documentation and training

## Success Criteria

**Technical Metrics:**
- RAG adds < 200ms to analysis latency
- Embedding cache hit rate > 70%
- RAG system availability > 99.5%
- Zero impact on existing system reliability

**Business Metrics:**
- 10%+ improvement in sentiment prediction accuracy
- 15%+ reduction in false positive recommendations
- Positive ROI within 3 months of deployment
- Improved trader confidence in system recommendations

## Risk Mitigation

**Technical Risks:**
- **Embedding API Limits**: Implement rate limiting and fallbacks
- **Vector Search Performance**: Optimize indexes and caching
- **Memory Usage**: Monitor and limit cache sizes
- **Integration Complexity**: Maintain backward compatibility

**Business Risks:**
- **Cost Overruns**: Monitor API usage and set budgets
- **Accuracy Regression**: Comprehensive A/B testing
- **Operational Complexity**: Thorough documentation and training
- **Data Quality Issues**: Regular audits and validation

## Conclusion

This RAG implementation will enhance the NewsHead sentiment analysis system by providing historical context to Claude's decision-making process. The phased approach ensures minimal risk while maximizing the potential for improved accuracy and reduced false positives.

The system is designed to integrate seamlessly with the existing architecture while providing clear fallback mechanisms and comprehensive monitoring. Success will be measured through both technical performance metrics and business impact on trading decisions. 