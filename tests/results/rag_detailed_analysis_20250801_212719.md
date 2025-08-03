# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 21:27:19
- **Total Articles Analyzed**: 30
- **TRUE_BULLISH Articles in Test Set**: 10

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 43.3%
- **BUY+High Precision**: 50.0% (4 signals)
- **BUY+High Recall**: 20.0%
- **BUY (Any) Recall**: 80.0%

### RAG Model
- **Overall Accuracy**: 43.3%
- **BUY+High Precision**: 34.8% (23 signals)
- **BUY+High Recall**: 80.0%
- **BUY (Any) Recall**: 80.0%

### Performance Improvements
- **Accuracy**: +0.0%
- **BUY+High Precision**: -15.2%
- **BUY+High Recall**: +60.0%
- **BUY (Any) Recall**: +0.0%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (4 total)


#### ✅ Correct Predictions (2/4 = 50.0%)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: high)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: high)

#### ❌ Incorrect Predictions (2/4 = 50.0%)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: high, Actual: Neutral)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: high, Actual: Neutral)

### RAG Model BUY+High Predictions (23 total)


#### ✅ Correct Predictions (8/23 = 34.8%)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3692ms)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 12ms, LLM: 4068ms)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 12ms, LLM: 2977ms)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 12ms, LLM: 2773ms)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 12ms, LLM: 3781ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3232ms)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3951ms)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Confidence: 0.95, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3422ms)

#### ❌ Incorrect Predictions (15/23 = 65.2%)
- **CPOP**: CPOP Announces Plans to Enter Cryptocurrency Market (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 2938ms)
- **DVLT**: Datavault AI Announces Strategic and Operational Objectives for 3Q 2025 (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3226ms)
- **ADTX**: Aditxt's Subsidiary Pearsanta Receives an Invitation to Submit Full Proposal for the U.S. Department... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3522ms)
- **APDN**: Applied DNA Announces New Follow-On LineaDNA Order from Global IVD Manufacturer for Use in Cancer Di... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 13ms, LLM: 3549ms)
- **CTOR**: Citius Oncology Expands Distribution Network for LYMPHIR with Execution of Distribution Services Agr... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 12ms, LLM: 3039ms)
- **GAME**: GameSquare Announces Pricing of Underwritten Public Offering to Launch Ethereum Treasury Strategy (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 2552ms)
- **CING**: Cingulate Receives $4.3M Waiver from FDA Ahead of Imminent Filing for Marketing Approval of Lead ADH... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3712ms)
- **AERT**: Aeries Technology, Inc. (NASDAQ: AERT) Partners with Skydda.ai to Bring AI-Enabled SOC Operations to... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3683ms)
- **CRGX**: CARGO Therapeutics Enters into Agreement to Be Acquired by Concentra Biosciences for $4.379 in Cash ... (Confidence: 0.95, Actual: Neutral, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3561ms)
- **FLUX**: Flux Power Recognized Among Financial Times' Fastest Growing Companies in the Americas 2025 (Confidence: 0.95, Actual: Neutral, Similar Examples: 5, Embed: 0ms, Search: 13ms, LLM: 2686ms)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: 0.95, Actual: Neutral, Similar Examples: 5, Embed: 0ms, Search: 10ms, LLM: 3554ms)
- **CXDO**: Crexendo Announces Completion of Key Oracle Cloud Infrastructure (OCI) Migration Milestones (Confidence: 0.95, Actual: Neutral, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3023ms)
- **ASTI**: Ascent Solar Technologies to Deliver Thin-Film Solar Technology to a Colorado-based Space Solar Arra... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 2699ms)
- **CELZ**: Creative Medical Technology Holdings Receives Notice of Allowance for ImmCelz for Treatment of Heart... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5, Embed: 0ms, Search: 12ms, LLM: 3420ms)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: 0.95, Actual: Neutral, Similar Examples: 5, Embed: 0ms, Search: 14ms, LLM: 3878ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (8 missed)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: medium)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: high)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (2 missed)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: 0.95)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: 0.95)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 50.0% ❌
  - RAG: 34.8% ❌

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 80.0% ❌
  - RAG: 80.0% ❌

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 20.0%
  - RAG: 80.0%

### Integration Recommendation
❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria

**Analysis Time Overhead**: 1.08s per article 🚫

