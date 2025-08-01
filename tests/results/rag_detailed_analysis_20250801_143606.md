# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 14:36:06
- **Total Articles Analyzed**: 99
- **TRUE_BULLISH Articles in Test Set**: 33

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 42.4%
- **BUY+High Precision**: 56.2% (16 signals)
- **BUY+High Recall**: 27.3%
- **BUY (Any) Recall**: 87.9%

### RAG Model
- **Overall Accuracy**: 46.5%
- **BUY+High Precision**: 50.0% (20 signals)
- **BUY+High Recall**: 30.3%
- **BUY (Any) Recall**: 72.7%

### Performance Improvements
- **Accuracy**: +4.0%
- **BUY+High Precision**: -6.2%
- **BUY+High Recall**: +3.0%
- **BUY (Any) Recall**: -15.2%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (16 total)


#### ✅ Correct Predictions (9/16 = 56.2%)
- **WGRX**: CORRECTION FROM SOURCE: Wellgistics Health Secures $50M ELOC Facility for XRP Treasury Reserve and R... (Confidence: high)
- **KNW**: Greg Kidd to Acquire Controlling Interest in Know Labs and Introduce Bitcoin Treasury Strategy (Confidence: high)
- **XLO**: Xilio Therapeutics Announces Multiple Masked T Cell Engager Programs (Confidence: high)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Confidence: high)
- **NVNI**: Nvni Group Limited Reports Record 2024 Financial Results (Confidence: high)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: high)
- **GCTK**: Glucotrack and OneTwo Analytics Present Positive Final Results of First-In-Human Study for Continuou... (Confidence: high)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: high)
- **INAB**: IN8bio Reports Updated Positive Results from Phase 1 Trial of INB-100 in Leukemia Patients (Confidence: high)

#### ❌ Incorrect Predictions (7/16 = 43.8%)
- **BMGL**: Basel Medical Group Subsidiary Awarded S$375 Million Contract to Supply Healthcare Products; Group t... (Confidence: high, Actual: False Pump)
- **CYBN**: Cybin Announces Additional Strategic Clinical Site Partnerships to Support PARADIGM, a Multinational... (Confidence: high, Actual: Neutral)
- **ECDA**: ECD Automotive Design Wins Dual Awards for Custom Jaguar E-Type and Mustang Builds (Confidence: high, Actual: False Pump)
- **HURA**: TuHURA Biosciences Initiates Its Phase 3 Accelerated Approval Trial of IFx-2.0 as an Adjunctive Ther... (Confidence: high, Actual: False Pump)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: high, Actual: False Pump)
- **BCTX**: BriaCell Presents Benchmark Beating Survival and Clinical Benefit at AACR 2025; Advancements in Next... (Confidence: high, Actual: False Pump)

### RAG Model BUY+High Predictions (20 total)


#### ✅ Correct Predictions (10/20 = 50.0%)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 5)
- **WGRX**: CORRECTION FROM SOURCE: Wellgistics Health Secures $50M ELOC Facility for XRP Treasury Reserve and R... (Confidence: 0.95, Similar Examples: 5)
- **KNW**: Greg Kidd to Acquire Controlling Interest in Know Labs and Introduce Bitcoin Treasury Strategy (Confidence: 0.95, Similar Examples: 5)
- **XLO**: Xilio Therapeutics Announces Multiple Masked T Cell Engager Programs (Confidence: 0.95, Similar Examples: 5)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Confidence: 0.95, Similar Examples: 5)
- **SRXH**: SRx Health Solutions Announces Intention to Create Subsidiary for Crypto-Based Borrowing (Confidence: 0.95, Similar Examples: 5)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Confidence: 0.95, Similar Examples: 5)
- **NVNI**: Nvni Group Limited Reports Record 2024 Financial Results (Confidence: 0.95, Similar Examples: 5)
- **VVPR**: VivoPower in Advanced Bilateral Negotiations on All-Cash Takeover Offer at Enterprise Value of US$12... (Confidence: 0.95, Similar Examples: 5)
- **KIDZ**: Classover Holdings Enters into $400 Million Equity Purchase Facility Agreement to Launch SOL-Based T... (Confidence: 0.95, Similar Examples: 5)

#### ❌ Incorrect Predictions (10/20 = 50.0%)
- **BMGL**: Basel Medical Group Subsidiary Awarded S$375 Million Contract to Supply Healthcare Products; Group t... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **CYBN**: Cybin Announces Additional Strategic Clinical Site Partnerships to Support PARADIGM, a Multinational... (Confidence: 0.95, Actual: Neutral, Similar Examples: 5)
- **ALZN**: Alzamend Neuro Announces Completion of a Novel Head Coil by Tesla for Measuring Brain Structure Lith... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **IPW**: iPower Announces Strategic Shift Toward Crypto Treasury and Blockchain Infrastructure Services (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **MBRX**: Moleculin Receives European Medicines Agency Approval to Expand Phase 3 MIRACLE Clinical Trial (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **HURA**: TuHURA Biosciences Initiates Its Phase 3 Accelerated Approval Trial of IFx-2.0 as an Adjunctive Ther... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **TRNR**: Interactive Strength Inc. (NASDAQ 'TRNR') Signs Binding Agreement to Acquire Wattbike, a $15M+, Omni... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **VCIG**: VCI Global Secures US$12 Million Contract With Datanex for AI-Powered Digital Marketing Solutions Th... (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)
- **WINT**: Windtree Announces Publication of Istaroxime Positive Phase 2 SEISMiC B Study (Confidence: 0.95, Actual: False Pump, Similar Examples: 5)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (24 missed)
- **MIRA**: MIRA Pharmaceuticals Announces Positive Results for Ketamir-2 in Diabetic Neuropathy Animal Model, R... (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **MBRX**: Moleculin Receives Positive FDA Guidance for Acceleration of its Registration-Enabling MIRACLE Trial... (Predicted: BUY, Confidence: medium)
- **VVPR**: Tembo E-LV Progresses Business Combination Agreement with CCTS at a Combined Enterprise Value Of US$... (Predicted: HOLD, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)
- **OMEX**: Odyssey Marine Exploration Confirms Sufficient Operational Funding and Welcomes New Executive Order (Predicted: BUY, Confidence: medium)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: medium)
- **OSRH**: Vaximm AG, an OSR Company, Announces Results from Phase 2a Trial of VXM01 and Avelumab Combination T... (Predicted: BUY, Confidence: medium)
- **SRXH**: SRx Health Solutions Announces Intention to Create Subsidiary for Crypto-Based Borrowing (Predicted: HOLD, Confidence: medium)
- **LYRA**: Lyra Therapeutics Reports Positive Results from the ENLIGHTEN 2 Phase 3 Trial of LYR-210 Achieving S... (Predicted: BUY, Confidence: medium)
- **WINT**: Windtree Announces License and Supply Agreement to Become Sourcing Partner for a Small Biotech with ... (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: medium)
- **VVPR**: VivoPower in Advanced Bilateral Negotiations on All-Cash Takeover Offer at Enterprise Value of US$12... (Predicted: BUY, Confidence: medium)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: medium)
- **UOKA**: MDJM Announces the Introduction of OpenAI's ChatGPT Team to Promote Cultural Business Development an... (Predicted: BUY, Confidence: medium)
- **CGBS**: Crown LNG Signs Gas Sales MOU with India Gas Exchange (Predicted: BUY, Confidence: medium)
- **IXHL**: Incannex Healthcare Inc. Provides Clinical Program Update on IHL-42X, an Oral Once-Daily Treatment f... (Predicted: BUY, Confidence: medium)
- **KIDZ**: Classover Holdings Enters into $400 Million Equity Purchase Facility Agreement to Launch SOL-Based T... (Predicted: BUY, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: medium)
- **FTRK**: FAST TRACK GROUP Responds to Inaccurate and Misleading Online Rumors Regarding Alleged Registered Di... (Predicted: HOLD, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (23 missed)
- **MIRA**: MIRA Pharmaceuticals Announces Positive Results for Ketamir-2 in Diabetic Neuropathy Animal Model, R... (Predicted: HOLD, Confidence: 0.70)
- **MBRX**: Moleculin Receives Positive FDA Guidance for Acceleration of its Registration-Enabling MIRACLE Trial... (Predicted: BUY, Confidence: 0.70)
- **VVPR**: Tembo E-LV Progresses Business Combination Agreement with CCTS at a Combined Enterprise Value Of US$... (Predicted: BUY, Confidence: 0.70)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: 0.70)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: HOLD, Confidence: 0.70)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: HOLD, Confidence: 0.70)
- **OMEX**: Odyssey Marine Exploration Confirms Sufficient Operational Funding and Welcomes New Executive Order (Predicted: HOLD, Confidence: 0.70)
- **OSRH**: Vaximm AG, an OSR Company, Announces Results from Phase 2a Trial of VXM01 and Avelumab Combination T... (Predicted: BUY, Confidence: 0.70)
- **LYRA**: Lyra Therapeutics Reports Positive Results from the ENLIGHTEN 2 Phase 3 Trial of LYR-210 Achieving S... (Predicted: HOLD, Confidence: 0.70)
- **ACXP**: Acurx Announces Publication in Lancet Microbe of Phase 2b Clinical Trial Data for Ibezapolstat in CD... (Predicted: BUY, Confidence: 0.70)
- **WINT**: Windtree Announces License and Supply Agreement to Become Sourcing Partner for a Small Biotech with ... (Predicted: HOLD, Confidence: 0.70)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: HOLD, Confidence: 0.70)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Predicted: HOLD, Confidence: 0.70)
- **BCTX**: BriaCell Confirms 100% Resolution of Lung Metastasis with Bria-OTS (Predicted: BUY, Confidence: 0.70)
- **UOKA**: MDJM Announces the Introduction of OpenAI's ChatGPT Team to Promote Cultural Business Development an... (Predicted: BUY, Confidence: 0.70)
- **GCTK**: Glucotrack and OneTwo Analytics Present Positive Final Results of First-In-Human Study for Continuou... (Predicted: BUY, Confidence: 0.70)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Predicted: BUY, Confidence: 0.70)
- **CGBS**: Crown LNG Signs Gas Sales MOU with India Gas Exchange (Predicted: BUY, Confidence: 0.70)
- **INAB**: IN8bio Reports Updated Positive Results from Phase 1 Trial of INB-100 in Leukemia Patients (Predicted: BUY, Confidence: 0.70)
- **IXHL**: Incannex Healthcare Inc. Provides Clinical Program Update on IHL-42X, an Oral Once-Daily Treatment f... (Predicted: BUY, Confidence: 0.70)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: 0.70)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: 0.70)
- **FTRK**: FAST TRACK GROUP Responds to Inaccurate and Misleading Online Rumors Regarding Alleged Registered Di... (Predicted: HOLD, Confidence: 0.70)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 56.2% ❌
  - RAG: 50.0% ❌

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 87.9% ❌
  - RAG: 72.7% ❌

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 27.3%
  - RAG: 30.3%

### Integration Recommendation
❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria

**Analysis Time Overhead**: 6.30s per article 🚫

