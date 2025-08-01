# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 15:45:15
- **Total Articles Analyzed**: 15
- **TRUE_BULLISH Articles in Test Set**: 5

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 40.0%
- **BUY+High Precision**: 33.3% (3 signals)
- **BUY+High Recall**: 20.0%
- **BUY (Any) Recall**: 80.0%

### RAG Model
- **Overall Accuracy**: 66.7%
- **BUY+High Precision**: 100.0% (4 signals)
- **BUY+High Recall**: 80.0%
- **BUY (Any) Recall**: 100.0%

### Performance Improvements
- **Accuracy**: +26.7%
- **BUY+High Precision**: +66.7%
- **BUY+High Recall**: +60.0%
- **BUY (Any) Recall**: +20.0%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (3 total)


#### ✅ Correct Predictions (1/3 = 33.3%)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: high)

#### ❌ Incorrect Predictions (2/3 = 66.7%)
- **BMGL**: Basel Medical Group Subsidiary Awarded S$375 Million Contract to Supply Healthcare Products; Group t... (Confidence: high, Actual: False Pump)
- **BCTX**: BriaCell Presents Benchmark Beating Survival and Clinical Benefit at AACR 2025; Advancements in Next... (Confidence: high, Actual: False Pump)

### RAG Model BUY+High Predictions (4 total)


#### ✅ Correct Predictions (4/4 = 100.0%)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Confidence: 0.95, Similar Examples: 3, Embed: 79ms, Search: 29ms, LLM: 2928ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 3, Embed: 260ms, Search: 28ms, LLM: 2128ms)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Confidence: 0.95, Similar Examples: 3, Embed: 187ms, Search: 29ms, LLM: 4168ms)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: 0.95, Similar Examples: 3, Embed: 227ms, Search: 25ms, LLM: 3114ms)

#### ❌ Incorrect Predictions (0/4 = 0.0%)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (4 missed)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (1 missed)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: 0.70)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 33.3% ❌
  - RAG: 100.0% ✅

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 80.0% ❌
  - RAG: 100.0% ✅

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 20.0%
  - RAG: 80.0%

### Integration Recommendation
⚠️ **CONDITIONAL INTEGRATION** - RAG shows improvement but may need tuning

**Analysis Time Overhead**: 0.46s per article ❌


## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: 7
- **Total P&L**: $29709.00
- **Total Investment**: $81272.00
- **Overall Return**: 36.56%

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | 3 | $8862.00 | $43741.00 | 20.26% |
| RAG | 4 | $20847.00 | $37531.00 | 55.55% |

---

### Performance Breakdown by Ticker

| Ticker | Traditional P&L | RAG P&L | Total P&L | Traditional Trades | RAG Trades |
|--------|-----------------|---------|-----------|-------------------|------------|
| **ACXP** | $6082.00 | $6082.00 | $12164.00 | 1 | 1 |
| **ANEB** | $0.00 | $8160.00 | $8160.00 | 0 | 1 |
| **APM** | $0.00 | $3800.00 | $3800.00 | 0 | 1 |
| **ATXG** | $0.00 | $2805.00 | $2805.00 | 0 | 1 |
| **BCTX** | $2300.00 | $0.00 | $2300.00 | 1 | 0 |
| **BMGL** | $480.00 | $0.00 | $480.00 | 1 | 0 |

### Performance Breakdown by Publication Hour (EST)

| Hour (EST) | Traditional P&L | RAG P&L | Total P&L | Traditional Trades | RAG Trades |
|------------|-----------------|---------|-----------|-------------------|------------|
| **03:00** | $8382.00 | $6082.00 | $14464.00 | 2 | 1 |
| **04:00** | $0.00 | $11960.00 | $11960.00 | 0 | 2 |
| **05:00** | $480.00 | $2805.00 | $3285.00 | 1 | 1 |

### Performance Breakdown by Price Bracket

| Price Bracket | Position Size | Traditional P&L | RAG P&L | Total P&L | Traditional Trades | RAG Trades |
|---------------|---------------|-----------------|---------|-----------|-------------------|------------|
| **$0.01-$1.00** | 10,000 | $6082.00 | $12687.00 | $18769.00 | 1 | 3 |
| **$1.00-$3.00** | 8,000 | $480.00 | $8160.00 | $8640.00 | 1 | 1 |
| **$3.00-$5.00** | 5,000 | $2300.00 | $0.00 | $2300.00 | 1 | 0 |
