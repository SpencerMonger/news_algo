# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 15:27:12
- **Total Articles Analyzed**: 30
- **TRUE_BULLISH Articles in Test Set**: 10

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 36.7%
- **BUY+High Precision**: 40.0% (5 signals)
- **BUY+High Recall**: 20.0%
- **BUY (Any) Recall**: 80.0%

### RAG Model
- **Overall Accuracy**: 66.7%
- **BUY+High Precision**: 87.5% (8 signals)
- **BUY+High Recall**: 70.0%
- **BUY (Any) Recall**: 100.0%

### Performance Improvements
- **Accuracy**: +30.0%
- **BUY+High Precision**: +47.5%
- **BUY+High Recall**: +50.0%
- **BUY (Any) Recall**: +20.0%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (5 total)


#### ✅ Correct Predictions (2/5 = 40.0%)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: high)
- **GCTK**: Glucotrack and OneTwo Analytics Present Positive Final Results of First-In-Human Study for Continuou... (Confidence: high)

#### ❌ Incorrect Predictions (3/5 = 60.0%)
- **BCTX**: BriaCell Presents Benchmark Beating Survival and Clinical Benefit at AACR 2025; Advancements in Next... (Confidence: high, Actual: False Pump)
- **CYBN**: Cybin Announces Additional Strategic Clinical Site Partnerships to Support PARADIGM, a Multinational... (Confidence: high, Actual: Neutral)
- **BMGL**: Basel Medical Group Subsidiary Awarded S$375 Million Contract to Supply Healthcare Products; Group t... (Confidence: high, Actual: False Pump)

### RAG Model BUY+High Predictions (8 total)


#### ✅ Correct Predictions (7/8 = 87.5%)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Confidence: 0.95, Similar Examples: 3, Embed: 194ms, Search: 30ms, LLM: 3256ms)
- **CGBS**: Crown LNG Signs Gas Sales MOU with India Gas Exchange (Confidence: 0.95, Similar Examples: 3, Embed: 241ms, Search: 30ms, LLM: 2451ms)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Confidence: 0.95, Similar Examples: 3, Embed: 107ms, Search: 30ms, LLM: 3429ms)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Confidence: 0.95, Similar Examples: 3, Embed: 292ms, Search: 31ms, LLM: 3707ms)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: 0.95, Similar Examples: 3, Embed: 222ms, Search: 36ms, LLM: 3930ms)
- **FTRK**: FAST TRACK GROUP Responds to Inaccurate and Misleading Online Rumors Regarding Alleged Registered Di... (Confidence: 0.95, Similar Examples: 3, Embed: 2478ms, Search: 25ms, LLM: 3296ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 3, Embed: 231ms, Search: 31ms, LLM: 2961ms)

#### ❌ Incorrect Predictions (1/8 = 12.5%)
- **CNSP**: CNS Pharmaceuticals Announces Virtual Investor KOL Connect Segment Discussing Glioblastoma Multiform... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 266ms, Search: 30ms, LLM: 3513ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (8 missed)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **CGBS**: Crown LNG Signs Gas Sales MOU with India Gas Exchange (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)
- **FTRK**: FAST TRACK GROUP Responds to Inaccurate and Misleading Online Rumors Regarding Alleged Registered Di... (Predicted: HOLD, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (3 missed)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: 0.70)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: 0.70)
- **GCTK**: Glucotrack and OneTwo Analytics Present Positive Final Results of First-In-Human Study for Continuou... (Predicted: BUY, Confidence: 0.70)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 40.0% ❌
  - RAG: 87.5% ✅

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 80.0% ❌
  - RAG: 100.0% ✅

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 20.0%
  - RAG: 70.0%

### Integration Recommendation
⚠️ **CONDITIONAL INTEGRATION** - RAG shows improvement but may need tuning

**Analysis Time Overhead**: 0.85s per article 🚫

