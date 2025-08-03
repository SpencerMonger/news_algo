# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 17:22:14
- **Total Articles Analyzed**: 98
- **TRUE_BULLISH Articles in Test Set**: 33

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 45.9%
- **BUY+High Precision**: 42.9% (21 signals)
- **BUY+High Recall**: 27.3%
- **BUY (Any) Recall**: 87.9%

### RAG Model
- **Overall Accuracy**: 35.7%
- **BUY+High Precision**: 25.0% (4 signals)
- **BUY+High Recall**: 3.0%
- **BUY (Any) Recall**: 75.8%

### Performance Improvements
- **Accuracy**: -10.2%
- **BUY+High Precision**: -17.9%
- **BUY+High Recall**: -24.2%
- **BUY (Any) Recall**: -12.1%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (21 total)


#### ✅ Correct Predictions (9/21 = 42.9%)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: high)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Confidence: high)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: high)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Confidence: high)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Confidence: high)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: high)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Confidence: high)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Confidence: high)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: high)

#### ❌ Incorrect Predictions (12/21 = 57.1%)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: high, Actual: False Pump)
- **MGRM**: Zimmer Biomet Announces Definitive Agreement to Acquire Monogram Technologies, Expanding Robotics Su... (Confidence: high, Actual: False Pump)
- **OKYO**: OKYO Pharma Unveils Strong Phase 2 Clinical Trial Results for Urcosimod to Treat Neuropathic Corneal... (Confidence: high, Actual: False Pump)
- **STXS**: Stereotaxis Receives U.S. FDA Clearance for MAGiC Sweep Catheter (Confidence: high, Actual: False Pump)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: high, Actual: False Pump)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: high, Actual: Neutral)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: high, Actual: False Pump)
- **TELO**: Telomir Demonstrates Telomir-1 Reverses Epigenetic Gene Silencing of STAT1, Restoring Tumor Suppress... (Confidence: high, Actual: False Pump)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: high, Actual: Neutral)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: high, Actual: False Pump)
- **NIVF**: NewGen Announces Strategic Acquisition of Cytometry Technology and Assets to Support Planned U.S. Ex... (Confidence: high, Actual: Neutral)

### RAG Model BUY+High Predictions (4 total)


#### ✅ Correct Predictions (1/4 = 25.0%)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: 0.90, Similar Examples: 3, Embed: 203ms, Search: 0ms, LLM: 2139ms)

#### ❌ Incorrect Predictions (3/4 = 75.0%)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: 0.91, Actual: False Pump, Similar Examples: 3, Embed: 269ms, Search: 0ms, LLM: 2451ms)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 180ms, Search: 0ms, LLM: 2732ms)
- **KIDZ**: Classover Increases Solana (SOL) Holdings by 295%, Surpasses 50,000 SOL Tokens in Treasury Reserve (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 195ms, Search: 0ms, LLM: 2347ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (24 missed)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: HOLD, Confidence: medium)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Predicted: BUY, Confidence: medium)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Predicted: BUY, Confidence: medium)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: medium)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: BUY, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: medium)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: medium)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Predicted: BUY, Confidence: medium)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Predicted: BUY, Confidence: medium)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: medium)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: medium)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: high)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: medium)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: medium)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Predicted: BUY, Confidence: medium)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (32 missed)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: 0.89)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: 0.89)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: HOLD, Confidence: 0.88)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Predicted: HOLD, Confidence: 0.91)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: HOLD, Confidence: 0.89)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Predicted: HOLD, Confidence: 0.90)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Predicted: BUY, Confidence: 0.89)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Predicted: BUY, Confidence: 0.89)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Predicted: BUY, Confidence: 0.88)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: 0.89)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: 0.89)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Predicted: BUY, Confidence: 0.88)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: 0.89)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: BUY, Confidence: 0.79)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: 0.89)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Predicted: HOLD, Confidence: 0.89)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: 0.80)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: 0.89)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Predicted: BUY, Confidence: 0.89)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Predicted: BUY, Confidence: 0.88)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Predicted: BUY, Confidence: 0.89)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: HOLD, Confidence: 0.89)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Predicted: HOLD, Confidence: 0.90)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: 0.89)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: BUY, Confidence: 0.89)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: 0.89)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Predicted: BUY, Confidence: 0.89)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: 0.89)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Predicted: BUY, Confidence: 0.89)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Predicted: BUY, Confidence: 0.89)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Predicted: BUY, Confidence: 0.88)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Predicted: BUY, Confidence: 0.89)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 42.9% ❌
  - RAG: 25.0% ❌

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 87.9% ❌
  - RAG: 75.8% ❌

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 27.3%
  - RAG: 3.0%

### Integration Recommendation
❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria

**Analysis Time Overhead**: 0.58s per article 🚫


## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: 25
- **Total P&L**: $118403.20
- **Total Investment**: $336499.00
- **Overall Return**: 35.19%

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | 21 | $108874.20 | $280028.00 | 38.88% |
| RAG | 4 | $9529.00 | $56471.00 | 16.87% |

---

### Performance Breakdown by Ticker

#### Traditional Model Performance by Ticker

| Ticker | P&L | Investment | Return % | Trades |
|--------|-----|------------|----------|--------|
| **LIDR** | $22880.80 | $9040.00 | 253.11% | 1 |
| **BGLC** | $18200.00 | $21500.00 | 84.65% | 1 |
| **SAFX** | $15600.80 | $12160.00 | 128.30% | 1 |
| **CVM** | $12900.00 | $19100.00 | 67.54% | 1 |
| **HSDT** | $9040.00 | $19000.00 | 47.58% | 1 |
| **PROK** | $6002.00 | $6998.00 | 85.77% | 1 |
| **AMOD** | $5280.00 | $10320.00 | 51.16% | 1 |
| **SUGP** | $4451.00 | $4980.00 | 89.38% | 1 |
| **UAVS** | $4405.60 | $11760.00 | 37.46% | 1 |
| **TELO** | $2240.00 | $22880.00 | 9.79% | 1 |
| **OKYO** | $1760.00 | $22400.00 | 7.86% | 1 |
| **STXS** | $1440.00 | $18400.00 | 7.83% | 1 |
| **NXXT** | $1040.00 | $18000.00 | 5.78% | 1 |
| **ICU** | $1008.00 | $12190.00 | 8.27% | 2 |
| **NIVF** | $727.00 | $5299.00 | 13.72% | 1 |
| **OSTX** | $640.00 | $15280.00 | 4.19% | 1 |
| **MGRM** | $540.00 | $16230.00 | 3.33% | 1 |
| **HYPR** | $409.00 | $7991.00 | 5.12% | 1 |
| **AREC** | $160.00 | $8800.00 | 1.82% | 1 |
| **ABEO** | $150.00 | $17700.00 | 0.85% | 1 |

#### RAG Model Performance by Ticker

| Ticker | P&L | Investment | Return % | Trades |
|--------|-----|------------|----------|--------|
| **AMOD** | $5280.00 | $10320.00 | 51.16% | 1 |
| **KIDZ** | $3200.00 | $22880.00 | 13.99% | 1 |
| **OSTX** | $640.00 | $15280.00 | 4.19% | 1 |
| **HYPR** | $409.00 | $7991.00 | 5.12% | 1 |

### Performance Breakdown by Publication Hour (EST)

#### Traditional Model Performance by Hour

| Hour (EST) | P&L | Investment | Return % | Trades |
|------------|-----|------------|----------|--------|
| **03:00** | $37863.80 | $110748.00 | 34.19% | 8 |
| **04:00** | $55623.40 | $122481.00 | 45.41% | 10 |
| **05:00** | $15387.00 | $46799.00 | 32.88% | 3 |

#### RAG Model Performance by Hour

| Hour (EST) | P&L | Investment | Return % | Trades |
|------------|-----|------------|----------|--------|
| **03:00** | $640.00 | $15280.00 | 4.19% | 1 |
| **04:00** | $8889.00 | $41191.00 | 21.58% | 3 |

### Performance Breakdown by Price Bracket

#### Traditional Model Performance by Price Bracket

| Price Bracket | Position Size | P&L | Investment | Return % | Trades |
|---------------|---------------|-----|------------|----------|--------|
| **$0.01-$1.00** | 10,000 | $12597.00 | $37458.00 | 33.63% | 6 |
| **$1.00-$3.00** | 8,000 | $55447.20 | $149040.00 | 37.20% | 10 |
| **$3.00-$5.00** | 5,000 | $31100.00 | $40600.00 | 76.60% | 2 |
| **$5.00-$8.00** | 3,000 | $690.00 | $33930.00 | 2.03% | 2 |
| **$8.00+** | 2,000 | $9040.00 | $19000.00 | 47.58% | 1 |

#### RAG Model Performance by Price Bracket

| Price Bracket | Position Size | P&L | Investment | Return % | Trades |
|---------------|---------------|-----|------------|----------|--------|
| **$0.01-$1.00** | 10,000 | $409.00 | $7991.00 | 5.12% | 1 |
| **$1.00-$3.00** | 8,000 | $9120.00 | $48480.00 | 18.81% | 3 |
