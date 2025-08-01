# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 16:42:03
- **Total Articles Analyzed**: 98
- **TRUE_BULLISH Articles in Test Set**: 33

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 45.9%
- **BUY+High Precision**: 45.5% (22 signals)
- **BUY+High Recall**: 30.3%
- **BUY (Any) Recall**: 87.9%

### RAG Model
- **Overall Accuracy**: 45.9%
- **BUY+High Precision**: 34.8% (23 signals)
- **BUY+High Recall**: 24.2%
- **BUY (Any) Recall**: 78.8%

### Performance Improvements
- **Accuracy**: +0.0%
- **BUY+High Precision**: -10.7%
- **BUY+High Recall**: -6.1%
- **BUY (Any) Recall**: -9.1%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (22 total)


#### ✅ Correct Predictions (10/22 = 45.5%)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Confidence: high)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Confidence: high)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: high)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Confidence: high)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: high)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Confidence: high)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Confidence: high)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Confidence: high)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: high)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: high)

#### ❌ Incorrect Predictions (12/22 = 54.5%)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: high, Actual: False Pump)
- **MGRM**: Zimmer Biomet Announces Definitive Agreement to Acquire Monogram Technologies, Expanding Robotics Su... (Confidence: high, Actual: False Pump)
- **OKYO**: OKYO Pharma Unveils Strong Phase 2 Clinical Trial Results for Urcosimod to Treat Neuropathic Corneal... (Confidence: high, Actual: False Pump)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: high, Actual: Neutral)
- **STXS**: Stereotaxis Receives U.S. FDA Clearance for MAGiC Sweep Catheter (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: high, Actual: False Pump)
- **TELO**: Telomir Demonstrates Telomir-1 Reverses Epigenetic Gene Silencing of STAT1, Restoring Tumor Suppress... (Confidence: high, Actual: False Pump)
- **NIVF**: NewGen Announces Strategic Acquisition of Cytometry Technology and Assets to Support Planned U.S. Ex... (Confidence: high, Actual: Neutral)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: high, Actual: False Pump)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: high, Actual: False Pump)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: high, Actual: Neutral)

### RAG Model BUY+High Predictions (23 total)


#### ✅ Correct Predictions (8/23 = 34.8%)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Confidence: 0.95, Similar Examples: 3, Embed: 275ms, Search: 28ms, LLM: 2958ms)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: 0.95, Similar Examples: 3, Embed: 268ms, Search: 28ms, LLM: 3503ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 3, Embed: 196ms, Search: 26ms, LLM: 2889ms)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Confidence: 0.95, Similar Examples: 3, Embed: 259ms, Search: 29ms, LLM: 2842ms)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Confidence: 0.95, Similar Examples: 3, Embed: 302ms, Search: 27ms, LLM: 2491ms)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Confidence: 0.95, Similar Examples: 3, Embed: 273ms, Search: 24ms, LLM: 3105ms)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Confidence: 0.95, Similar Examples: 3, Embed: 195ms, Search: 23ms, LLM: 4528ms)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Confidence: 0.95, Similar Examples: 3, Embed: 122ms, Search: 29ms, LLM: 3703ms)

#### ❌ Incorrect Predictions (15/23 = 65.2%)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 249ms, Search: 30ms, LLM: 3306ms)
- **MOGO**: Mogo Acquires 9% Stake in Bitcoin & Gold Treasury Company Digital Commodities Capital Corp. (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 82ms, Search: 29ms, LLM: 3492ms)
- **OSRH**: OSR Holdings Enters into Term Sheet to Acquire Woori IO, a Pioneer in Noninvasive Glucose Monitoring... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 201ms, Search: 27ms, LLM: 2938ms)
- **XAIR**: Beyond Air Awarded Therapeutic Gases Agreement with Premier, Inc. (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 260ms, Search: 28ms, LLM: 3405ms)
- **NTWK**: NETSOL Technologies China Signs Strategic Agreement at the SCO Summit 2025 (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 187ms, Search: 21ms, LLM: 2153ms)
- **KIDZ**: Classover Increases Solana (SOL) Holdings by 295%, Surpasses 50,000 SOL Tokens in Treasury Reserve (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 237ms, Search: 28ms, LLM: 3182ms)
- **CING**: Cingulate Receives $4.3M Waiver from FDA Ahead of Imminent Filing for Marketing Approval of Lead ADH... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 1668ms, Search: 23ms, LLM: 3768ms)
- **SCOR**: U.S. Joint Industry Committee Completes Audit of Certified Currencies to Validate Transactability of... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 266ms, Search: 28ms, LLM: 2558ms)
- **CXDO**: Crexendo Announces Completion of Key Oracle Cloud Infrastructure (OCI) Migration Milestones (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 263ms, Search: 28ms, LLM: 2343ms)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 244ms, Search: 28ms, LLM: 2733ms)
- **CPOP**: CPOP Announces Plans to Enter Cryptocurrency Market (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 228ms, Search: 28ms, LLM: 4045ms)
- **GAME**: GameSquare Announces Pricing of Underwritten Public Offering to Launch Ethereum Treasury Strategy (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 199ms, Search: 28ms, LLM: 2338ms)
- **KITT**: Nauticus Robotics Announces an Excellent Start to the 2025 Offshore Season (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 259ms, Search: 29ms, LLM: 2444ms)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 219ms, Search: 21ms, LLM: 3229ms)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 212ms, Search: 28ms, LLM: 2904ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (23 missed)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Predicted: BUY, Confidence: medium)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: BUY, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: medium)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Predicted: BUY, Confidence: medium)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: medium)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: HOLD, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: medium)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Predicted: BUY, Confidence: medium)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: medium)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: medium)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: medium)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Predicted: BUY, Confidence: medium)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: high)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (25 missed)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Predicted: BUY, Confidence: 0.70)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Predicted: BUY, Confidence: 0.70)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: HOLD, Confidence: 0.70)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Predicted: BUY, Confidence: 0.70)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: 0.70)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Predicted: HOLD, Confidence: 0.70)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: HOLD, Confidence: 0.70)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: HOLD, Confidence: 0.70)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Predicted: BUY, Confidence: 0.70)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Predicted: BUY, Confidence: 0.70)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Predicted: BUY, Confidence: 0.70)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Predicted: BUY, Confidence: 0.70)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: 0.70)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: 0.70)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: BUY, Confidence: 0.70)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Predicted: BUY, Confidence: 0.70)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: 0.70)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: 0.70)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: 0.70)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Predicted: BUY, Confidence: 0.50)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Predicted: BUY, Confidence: 0.70)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: HOLD, Confidence: 0.70)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: 0.95)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: 0.70)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: 0.70)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 45.5% ❌
  - RAG: 34.8% ❌

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 87.9% ❌
  - RAG: 78.8% ❌

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 30.3%
  - RAG: 24.2%

### Integration Recommendation
❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria

**Analysis Time Overhead**: 0.92s per article 🚫


## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: 45
- **Total P&L**: $246221.80
- **Total Investment**: $607003.00
- **Overall Return**: 40.56%

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | 22 | $117461.20 | $284243.00 | 41.32% |
| RAG | 23 | $128760.60 | $322760.00 | 39.89% |

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
| **PMN** | $8587.00 | $4215.00 | 203.72% | 1 |
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
| **DARE** | $43680.00 | $20160.00 | 216.67% | 1 |
| **MBIO** | $18240.80 | $9520.00 | 191.61% | 1 |
| **HTOO** | $14520.00 | $15450.00 | 93.98% | 1 |
| **HOLO** | $11430.00 | $17910.00 | 63.82% | 1 |
| **MEIP** | $11040.00 | $15300.00 | 72.16% | 1 |
| **AEHL** | $6550.00 | $20600.00 | 31.80% | 1 |
| **AMOD** | $5280.00 | $10320.00 | 51.16% | 1 |
| **APM** | $3800.00 | $9700.00 | 39.18% | 1 |
| **KIDZ** | $3200.00 | $22880.00 | 13.99% | 1 |
| **CPOP** | $2500.00 | $6410.00 | 39.00% | 1 |
| **XAIR** | $1450.00 | $16650.00 | 8.71% | 1 |
| **MOGO** | $1200.00 | $15840.00 | 7.58% | 1 |
| **CING** | $1110.00 | $15090.00 | 7.36% | 1 |
| **NXXT** | $1040.00 | $18000.00 | 5.78% | 1 |
| **GAME** | $902.00 | $8900.00 | 10.13% | 1 |
| **OSRH** | $716.80 | $8160.00 | 8.78% | 1 |
| **OSTX** | $640.00 | $15280.00 | 4.19% | 1 |
| **ICU** | $580.00 | $5120.00 | 11.33% | 1 |
| **NTWK** | $500.00 | $19000.00 | 2.63% | 1 |
| **HYPR** | $409.00 | $7991.00 | 5.12% | 1 |
| **KITT** | $242.00 | $9049.00 | 2.67% | 1 |
| **SCOR** | $0.00 | $15930.00 | 0.00% | 1 |
| **CXDO** | $-270.00 | $19500.00 | -1.38% | 1 |

### Performance Breakdown by Publication Hour (EST)

#### Traditional Model Performance by Hour

| Hour (EST) | P&L | Investment | Return % | Trades |
|------------|-----|------------|----------|--------|
| **03:00** | $37863.80 | $110748.00 | 34.19% | 8 |
| **04:00** | $64210.40 | $126696.00 | 50.68% | 11 |
| **05:00** | $15387.00 | $46799.00 | 32.88% | 3 |

#### RAG Model Performance by Hour

| Hour (EST) | P&L | Investment | Return % | Trades |
|------------|-----|------------|----------|--------|
| **03:00** | $4006.80 | $55930.00 | 7.16% | 4 |
| **04:00** | $117973.80 | $207730.00 | 56.79% | 16 |
| **05:00** | $6780.00 | $59100.00 | 11.47% | 3 |

### Performance Breakdown by Price Bracket

#### Traditional Model Performance by Price Bracket

| Price Bracket | Position Size | P&L | Investment | Return % | Trades |
|---------------|---------------|-----|------------|----------|--------|
| **$0.01-$1.00** | 10,000 | $21184.00 | $41673.00 | 50.83% | 7 |
| **$1.00-$3.00** | 8,000 | $55447.20 | $149040.00 | 37.20% | 10 |
| **$3.00-$5.00** | 5,000 | $31100.00 | $40600.00 | 76.60% | 2 |
| **$5.00-$8.00** | 3,000 | $690.00 | $33930.00 | 2.03% | 2 |
| **$8.00+** | 2,000 | $9040.00 | $19000.00 | 47.58% | 1 |

#### RAG Model Performance by Price Bracket

| Price Bracket | Position Size | P&L | Investment | Return % | Trades |
|---------------|---------------|-----|------------|----------|--------|
| **$0.01-$1.00** | 10,000 | $8433.00 | $47170.00 | 17.88% | 6 |
| **$1.00-$3.00** | 8,000 | $73997.60 | $120160.00 | 61.58% | 8 |
| **$3.00-$5.00** | 5,000 | $8500.00 | $56250.00 | 15.11% | 3 |
| **$5.00-$8.00** | 3,000 | $37830.00 | $99180.00 | 38.14% | 6 |
