# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 16:00:29
- **Total Articles Analyzed**: 99
- **TRUE_BULLISH Articles in Test Set**: 33

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 43.4%
- **BUY+High Precision**: 56.2% (16 signals)
- **BUY+High Recall**: 27.3%
- **BUY (Any) Recall**: 87.9%

### RAG Model
- **Overall Accuracy**: 61.6%
- **BUY+High Precision**: 90.0% (20 signals)
- **BUY+High Recall**: 54.5%
- **BUY (Any) Recall**: 100.0%

### Performance Improvements
- **Accuracy**: +18.2%
- **BUY+High Precision**: +33.8%
- **BUY+High Recall**: +27.3%
- **BUY (Any) Recall**: +12.1%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (16 total)


#### ✅ Correct Predictions (9/16 = 56.2%)
- **KNW**: Greg Kidd to Acquire Controlling Interest in Know Labs and Introduce Bitcoin Treasury Strategy (Confidence: high)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: high)
- **GCTK**: Glucotrack and OneTwo Analytics Present Positive Final Results of First-In-Human Study for Continuou... (Confidence: high)
- **NVNI**: Nvni Group Limited Reports Record 2024 Financial Results (Confidence: high)
- **XLO**: Xilio Therapeutics Announces Multiple Masked T Cell Engager Programs (Confidence: high)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: high)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: high)
- **WGRX**: CORRECTION FROM SOURCE: Wellgistics Health Secures $50M ELOC Facility for XRP Treasury Reserve and R... (Confidence: high)
- **INAB**: IN8bio Reports Updated Positive Results from Phase 1 Trial of INB-100 in Leukemia Patients (Confidence: high)

#### ❌ Incorrect Predictions (7/16 = 43.8%)
- **ECDA**: ECD Automotive Design Wins Dual Awards for Custom Jaguar E-Type and Mustang Builds (Confidence: high, Actual: False Pump)
- **BMGL**: Basel Medical Group Subsidiary Awarded S$375 Million Contract to Supply Healthcare Products; Group t... (Confidence: high, Actual: False Pump)
- **BCTX**: BriaCell Presents Benchmark Beating Survival and Clinical Benefit at AACR 2025; Advancements in Next... (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: high, Actual: False Pump)
- **CYBN**: Cybin Announces Additional Strategic Clinical Site Partnerships to Support PARADIGM, a Multinational... (Confidence: high, Actual: Neutral)
- **HURA**: TuHURA Biosciences Initiates Its Phase 3 Accelerated Approval Trial of IFx-2.0 as an Adjunctive Ther... (Confidence: high, Actual: False Pump)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: high, Actual: False Pump)

### RAG Model BUY+High Predictions (20 total)


#### ✅ Correct Predictions (18/20 = 90.0%)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Confidence: 0.95, Similar Examples: 3, Embed: 81ms, Search: 27ms, LLM: 2894ms)
- **CGBS**: Crown LNG Signs Gas Sales MOU with India Gas Exchange (Confidence: 0.95, Similar Examples: 3, Embed: 230ms, Search: 28ms, LLM: 2588ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 3, Embed: 262ms, Search: 27ms, LLM: 2313ms)
- **VVPR**: Tembo E-LV Progresses Business Combination Agreement with CCTS at a Combined Enterprise Value Of US$... (Confidence: 0.95, Similar Examples: 3, Embed: 791ms, Search: 24ms, LLM: 3496ms)
- **OSRH**: Vaximm AG, an OSR Company, Announces Results from Phase 2a Trial of VXM01 and Avelumab Combination T... (Confidence: 0.95, Similar Examples: 3, Embed: 282ms, Search: 28ms, LLM: 3765ms)
- **KNW**: Greg Kidd to Acquire Controlling Interest in Know Labs and Introduce Bitcoin Treasury Strategy (Confidence: 0.95, Similar Examples: 3, Embed: 191ms, Search: 29ms, LLM: 2825ms)
- **FTRK**: FAST TRACK GROUP Responds to Inaccurate and Misleading Online Rumors Regarding Alleged Registered Di... (Confidence: 0.95, Similar Examples: 3, Embed: 268ms, Search: 28ms, LLM: 3028ms)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: 0.95, Similar Examples: 3, Embed: 199ms, Search: 28ms, LLM: 3281ms)
- **SRXH**: SRx Health Solutions Announces Intention to Create Subsidiary for Crypto-Based Borrowing (Confidence: 0.95, Similar Examples: 3, Embed: 264ms, Search: 22ms, LLM: 2634ms)
- **VVPR**: VivoPower in Advanced Bilateral Negotiations on All-Cash Takeover Offer at Enterprise Value of US$12... (Confidence: 0.95, Similar Examples: 3, Embed: 280ms, Search: 28ms, LLM: 3354ms)
- **XLO**: Xilio Therapeutics Announces Multiple Masked T Cell Engager Programs (Confidence: 0.95, Similar Examples: 3, Embed: 285ms, Search: 29ms, LLM: 3222ms)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: 0.95, Similar Examples: 3, Embed: 297ms, Search: 27ms, LLM: 3431ms)
- **LYRA**: Lyra Therapeutics Reports Positive Results from the ENLIGHTEN 2 Phase 3 Trial of LYR-210 Achieving S... (Confidence: 0.95, Similar Examples: 3, Embed: 353ms, Search: 27ms, LLM: 2061ms)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Confidence: 0.95, Similar Examples: 3, Embed: 254ms, Search: 28ms, LLM: 3249ms)
- **WGRX**: CORRECTION FROM SOURCE: Wellgistics Health Secures $50M ELOC Facility for XRP Treasury Reserve and R... (Confidence: 0.95, Similar Examples: 3, Embed: 261ms, Search: 28ms, LLM: 2723ms)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Confidence: 0.95, Similar Examples: 3, Embed: 254ms, Search: 27ms, LLM: 2968ms)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Confidence: 0.95, Similar Examples: 3, Embed: 275ms, Search: 29ms, LLM: 2871ms)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Confidence: 0.95, Similar Examples: 3, Embed: 259ms, Search: 29ms, LLM: 4180ms)

#### ❌ Incorrect Predictions (2/20 = 10.0%)
- **ECDA**: ECD Automotive Design Wins Dual Awards for Custom Jaguar E-Type and Mustang Builds (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 398ms, Search: 23ms, LLM: 2328ms)
- **CNSP**: CNS Pharmaceuticals Announces Virtual Investor KOL Connect Segment Discussing Glioblastoma Multiform... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 249ms, Search: 24ms, LLM: 2666ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (24 missed)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **CGBS**: Crown LNG Signs Gas Sales MOU with India Gas Exchange (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **VVPR**: Tembo E-LV Progresses Business Combination Agreement with CCTS at a Combined Enterprise Value Of US$... (Predicted: HOLD, Confidence: medium)
- **OSRH**: Vaximm AG, an OSR Company, Announces Results from Phase 2a Trial of VXM01 and Avelumab Combination T... (Predicted: BUY, Confidence: medium)
- **OMEX**: Odyssey Marine Exploration Confirms Sufficient Operational Funding and Welcomes New Executive Order (Predicted: BUY, Confidence: medium)
- **FTRK**: FAST TRACK GROUP Responds to Inaccurate and Misleading Online Rumors Regarding Alleged Registered Di... (Predicted: HOLD, Confidence: medium)
- **SRXH**: SRx Health Solutions Announces Intention to Create Subsidiary for Crypto-Based Borrowing (Predicted: HOLD, Confidence: medium)
- **VVPR**: VivoPower in Advanced Bilateral Negotiations on All-Cash Takeover Offer at Enterprise Value of US$12... (Predicted: BUY, Confidence: medium)
- **MIRA**: MIRA Pharmaceuticals Announces Positive Results for Ketamir-2 in Diabetic Neuropathy Animal Model, R... (Predicted: BUY, Confidence: medium)
- **MBRX**: Moleculin Receives Positive FDA Guidance for Acceleration of its Registration-Enabling MIRACLE Trial... (Predicted: BUY, Confidence: medium)
- **LYRA**: Lyra Therapeutics Reports Positive Results from the ENLIGHTEN 2 Phase 3 Trial of LYR-210 Achieving S... (Predicted: BUY, Confidence: medium)
- **KIDZ**: Classover Holdings Enters into $400 Million Equity Purchase Facility Agreement to Launch SOL-Based T... (Predicted: BUY, Confidence: medium)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: medium)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: medium)
- **WINT**: Windtree Announces License and Supply Agreement to Become Sourcing Partner for a Small Biotech with ... (Predicted: BUY, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **UOKA**: MDJM Announces the Introduction of OpenAI's ChatGPT Team to Promote Cultural Business Development an... (Predicted: BUY, Confidence: medium)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: medium)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: medium)
- **IXHL**: Incannex Healthcare Inc. Provides Clinical Program Update on IHL-42X, an Oral Once-Daily Treatment f... (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (15 missed)
- **OMEX**: Odyssey Marine Exploration Confirms Sufficient Operational Funding and Welcomes New Executive Order (Predicted: BUY, Confidence: 0.70)
- **GCTK**: Glucotrack and OneTwo Analytics Present Positive Final Results of First-In-Human Study for Continuou... (Predicted: BUY, Confidence: 0.70)
- **NVNI**: Nvni Group Limited Reports Record 2024 Financial Results (Predicted: BUY, Confidence: 0.70)
- **MIRA**: MIRA Pharmaceuticals Announces Positive Results for Ketamir-2 in Diabetic Neuropathy Animal Model, R... (Predicted: BUY, Confidence: 0.70)
- **MBRX**: Moleculin Receives Positive FDA Guidance for Acceleration of its Registration-Enabling MIRACLE Trial... (Predicted: BUY, Confidence: 0.70)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Predicted: BUY, Confidence: 0.50)
- **KIDZ**: Classover Holdings Enters into $400 Million Equity Purchase Facility Agreement to Launch SOL-Based T... (Predicted: BUY, Confidence: 0.70)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: 0.70)
- **WINT**: Windtree Announces License and Supply Agreement to Become Sourcing Partner for a Small Biotech with ... (Predicted: BUY, Confidence: 0.70)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: 0.70)
- **UOKA**: MDJM Announces the Introduction of OpenAI's ChatGPT Team to Promote Cultural Business Development an... (Predicted: BUY, Confidence: 0.70)
- **INAB**: IN8bio Reports Updated Positive Results from Phase 1 Trial of INB-100 in Leukemia Patients (Predicted: BUY, Confidence: 0.70)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: 0.70)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: 0.70)
- **IXHL**: Incannex Healthcare Inc. Provides Clinical Program Update on IHL-42X, an Oral Once-Daily Treatment f... (Predicted: BUY, Confidence: 0.70)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 56.2% ❌
  - RAG: 90.0% ✅

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 87.9% ❌
  - RAG: 100.0% ✅

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 27.3%
  - RAG: 54.5%

### Integration Recommendation
⚠️ **CONDITIONAL INTEGRATION** - RAG shows improvement but may need tuning

**Analysis Time Overhead**: 0.87s per article 🚫


## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: 36
- **Total P&L**: $282707.90
- **Total Investment**: $452563.00
- **Overall Return**: 62.47%

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | 16 | $93119.80 | $221977.00 | 41.95% |
| RAG | 20 | $189588.10 | $230586.00 | 82.22% |

---

### Performance Breakdown by Ticker

#### Traditional Model Performance by Ticker

| Ticker | P&L | Investment | Return % | Trades |
|--------|-----|------------|----------|--------|
| **LIDR** | $22880.80 | $9040.00 | 253.11% | 1 |
| **GCTK** | $20820.00 | $21030.00 | 99.00% | 1 |
| **WGRX** | $11412.00 | $22750.00 | 50.16% | 1 |
| **HSDT** | $9040.00 | $19000.00 | 47.58% | 1 |
| **INAB** | $7188.00 | $16206.00 | 44.35% | 1 |
| **KNW** | $6500.00 | $5000.00 | 130.00% | 1 |
| **XLO** | $6200.00 | $7200.00 | 86.11% | 1 |
| **ACXP** | $6082.00 | $3071.00 | 198.05% | 1 |
| **BCTX** | $2300.00 | $23150.00 | 9.94% | 1 |
| **NVNI** | $2097.00 | $5029.00 | 41.70% | 1 |
| **NXXT** | $1040.00 | $18000.00 | 5.78% | 1 |
| **BMGL** | $480.00 | $17520.00 | 2.74% | 1 |
| **ICU** | $428.00 | $7070.00 | 6.05% | 1 |
| **ECDA** | $142.00 | $2551.00 | 5.57% | 1 |
| **CYBN** | $30.00 | $21600.00 | 0.14% | 1 |
| **HURA** | $-3520.00 | $23760.00 | -14.81% | 1 |

#### RAG Model Performance by Ticker

| Ticker | P&L | Investment | Return % | Trades |
|--------|-----|------------|----------|--------|
| **LYRA** | $70410.30 | $17190.00 | 409.60% | 1 |
| **OSRH** | $20000.00 | $13360.00 | 149.70% | 1 |
| **HTOO** | $14520.00 | $15450.00 | 93.98% | 1 |
| **VVPR** | $13971.00 | $24050.00 | 58.09% | 2 |
| **WGRX** | $11412.00 | $22750.00 | 50.16% | 1 |
| **HSDT** | $9040.00 | $19000.00 | 47.58% | 1 |
| **ANEB** | $8160.00 | $17600.00 | 46.36% | 1 |
| **SMX** | $7279.20 | $15760.00 | 46.19% | 1 |
| **SRXH** | $6600.00 | $5400.00 | 122.22% | 1 |
| **KNW** | $6500.00 | $5000.00 | 130.00% | 1 |
| **XLO** | $6200.00 | $7200.00 | 86.11% | 1 |
| **ACXP** | $6082.00 | $3071.00 | 198.05% | 1 |
| **APM** | $3800.00 | $9700.00 | 39.18% | 1 |
| **ATXG** | $2805.00 | $7160.00 | 39.18% | 1 |
| **CLSD** | $1727.00 | $3514.00 | 49.15% | 1 |
| **CGBS** | $730.00 | $3190.00 | 22.88% | 1 |
| **ECDA** | $142.00 | $2551.00 | 5.57% | 1 |
| **CNSP** | $117.60 | $18240.00 | 0.64% | 1 |
| **FTRK** | $92.00 | $20400.00 | 0.45% | 1 |

### Performance Breakdown by Publication Hour (EST)

#### Traditional Model Performance by Hour

| Hour (EST) | P&L | Investment | Return % | Trades |
|------------|-----|------------|----------|--------|
| **02:00** | $6200.00 | $7200.00 | 86.11% | 1 |
| **03:00** | $24640.00 | $83027.00 | 29.68% | 5 |
| **04:00** | $43887.80 | $86480.00 | 50.75% | 7 |
| **05:00** | $18392.00 | $45270.00 | 40.63% | 3 |

#### RAG Model Performance by Hour

| Hour (EST) | P&L | Investment | Return % | Trades |
|------------|-----|------------|----------|--------|
| **02:00** | $6200.00 | $7200.00 | 86.11% | 1 |
| **03:00** | $107989.30 | $59325.00 | 182.03% | 6 |
| **04:00** | $26714.00 | $65701.00 | 40.66% | 5 |
| **05:00** | $48684.80 | $98360.00 | 49.50% | 8 |

### Performance Breakdown by Price Bracket

#### Traditional Model Performance by Price Bracket

| Price Bracket | Position Size | P&L | Investment | Return % | Trades |
|---------------|---------------|-----|------------|----------|--------|
| **$0.01-$1.00** | 10,000 | $21449.00 | $29921.00 | 71.69% | 6 |
| **$1.00-$3.00** | 8,000 | $20880.80 | $68320.00 | 30.56% | 4 |
| **$3.00-$5.00** | 5,000 | $13712.00 | $45900.00 | 29.87% | 2 |
| **$5.00-$8.00** | 3,000 | $20850.00 | $42630.00 | 48.91% | 2 |
| **$8.00+** | 2,000 | $16228.00 | $35206.00 | 46.09% | 2 |

#### RAG Model Performance by Price Bracket

| Price Bracket | Position Size | P&L | Investment | Return % | Trades |
|---------------|---------------|-----|------------|----------|--------|
| **$0.01-$1.00** | 10,000 | $39677.00 | $55396.00 | 71.62% | 10 |
| **$1.00-$3.00** | 8,000 | $44411.20 | $82560.00 | 53.79% | 5 |
| **$3.00-$5.00** | 5,000 | $11412.00 | $22750.00 | 50.16% | 1 |
| **$5.00-$8.00** | 3,000 | $84930.30 | $32640.00 | 260.20% | 2 |
| **$8.00+** | 2,000 | $9157.60 | $37240.00 | 24.59% | 2 |
