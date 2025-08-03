# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 17:07:11
- **Total Articles Analyzed**: 98
- **TRUE_BULLISH Articles in Test Set**: 33

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 46.9%
- **BUY+High Precision**: 45.5% (22 signals)
- **BUY+High Recall**: 30.3%
- **BUY (Any) Recall**: 90.9%

### RAG Model
- **Overall Accuracy**: 34.7%
- **BUY+High Precision**: 35.5% (76 signals)
- **BUY+High Recall**: 81.8%
- **BUY (Any) Recall**: 90.9%

### Performance Improvements
- **Accuracy**: -12.2%
- **BUY+High Precision**: -9.9%
- **BUY+High Recall**: +51.5%
- **BUY (Any) Recall**: +0.0%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (22 total)


#### ✅ Correct Predictions (10/22 = 45.5%)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: high)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: high)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: high)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Confidence: high)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Confidence: high)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Confidence: high)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Confidence: high)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Confidence: high)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: high)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Confidence: high)

#### ❌ Incorrect Predictions (12/22 = 54.5%)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: high, Actual: False Pump)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: high, Actual: False Pump)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: high, Actual: False Pump)
- **OKYO**: OKYO Pharma Unveils Strong Phase 2 Clinical Trial Results for Urcosimod to Treat Neuropathic Corneal... (Confidence: high, Actual: False Pump)
- **MGRM**: Zimmer Biomet Announces Definitive Agreement to Acquire Monogram Technologies, Expanding Robotics Su... (Confidence: high, Actual: False Pump)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: high, Actual: Neutral)
- **NIVF**: NewGen Announces Strategic Acquisition of Cytometry Technology and Assets to Support Planned U.S. Ex... (Confidence: high, Actual: Neutral)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: high, Actual: Neutral)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: high, Actual: False Pump)
- **TELO**: Telomir Demonstrates Telomir-1 Reverses Epigenetic Gene Silencing of STAT1, Restoring Tumor Suppress... (Confidence: high, Actual: False Pump)
- **STXS**: Stereotaxis Receives U.S. FDA Clearance for MAGiC Sweep Catheter (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: high, Actual: False Pump)

### RAG Model BUY+High Predictions (76 total)


#### ✅ Correct Predictions (27/76 = 35.5%)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Confidence: 0.95, Similar Examples: 3, Embed: 273ms, Search: 23ms, LLM: 2844ms)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: 0.95, Similar Examples: 3, Embed: 64ms, Search: 27ms, LLM: 2615ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.95, Similar Examples: 3, Embed: 2276ms, Search: 23ms, LLM: 2356ms)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Confidence: 0.95, Similar Examples: 3, Embed: 260ms, Search: 28ms, LLM: 2920ms)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: 0.95, Similar Examples: 3, Embed: 294ms, Search: 28ms, LLM: 2552ms)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: 0.95, Similar Examples: 3, Embed: 270ms, Search: 29ms, LLM: 4185ms)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Confidence: 0.95, Similar Examples: 3, Embed: 272ms, Search: 29ms, LLM: 2819ms)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Confidence: 0.95, Similar Examples: 3, Embed: 224ms, Search: 29ms, LLM: 3012ms)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Confidence: 0.95, Similar Examples: 3, Embed: 256ms, Search: 22ms, LLM: 2620ms)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Confidence: 0.95, Similar Examples: 3, Embed: 274ms, Search: 29ms, LLM: 2891ms)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Confidence: 0.95, Similar Examples: 3, Embed: 274ms, Search: 28ms, LLM: 2970ms)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Confidence: 0.95, Similar Examples: 3, Embed: 121ms, Search: 27ms, LLM: 2790ms)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Confidence: 0.95, Similar Examples: 3, Embed: 1266ms, Search: 24ms, LLM: 3514ms)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Confidence: 0.95, Similar Examples: 3, Embed: 198ms, Search: 27ms, LLM: 3005ms)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Confidence: 0.95, Similar Examples: 3, Embed: 253ms, Search: 27ms, LLM: 3228ms)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Confidence: 0.95, Similar Examples: 3, Embed: 192ms, Search: 28ms, LLM: 3788ms)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Confidence: 0.95, Similar Examples: 3, Embed: 259ms, Search: 28ms, LLM: 3037ms)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Confidence: 0.95, Similar Examples: 3, Embed: 203ms, Search: 27ms, LLM: 4245ms)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: 0.95, Similar Examples: 3, Embed: 330ms, Search: 28ms, LLM: 3933ms)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Confidence: 0.95, Similar Examples: 3, Embed: 208ms, Search: 22ms, LLM: 2701ms)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Confidence: 0.95, Similar Examples: 3, Embed: 204ms, Search: 28ms, LLM: 2951ms)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Confidence: 0.95, Similar Examples: 3, Embed: 226ms, Search: 27ms, LLM: 3832ms)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Confidence: 0.95, Similar Examples: 3, Embed: 190ms, Search: 29ms, LLM: 3227ms)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Confidence: 0.95, Similar Examples: 3, Embed: 255ms, Search: 28ms, LLM: 2851ms)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Confidence: 0.95, Similar Examples: 3, Embed: 201ms, Search: 28ms, LLM: 3709ms)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Confidence: 0.95, Similar Examples: 3, Embed: 228ms, Search: 29ms, LLM: 2961ms)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Confidence: 0.95, Similar Examples: 3, Embed: 178ms, Search: 28ms, LLM: 3610ms)

#### ❌ Incorrect Predictions (49/76 = 64.5%)
- **AERT**: Aeries Technology, Inc. (NASDAQ: AERT) Partners with Skydda.ai to Bring AI-Enabled SOC Operations to... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 167ms, Search: 28ms, LLM: 3414ms)
- **MOGO**: Mogo Acquires 9% Stake in Bitcoin & Gold Treasury Company Digital Commodities Capital Corp. (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 64ms, Search: 28ms, LLM: 3092ms)
- **WINT**: Kraken and Windtree Therapeutics Announce Strategic Partnership for BNB Custody, Trading, and OTC Se... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 274ms, Search: 29ms, LLM: 4321ms)
- **RXT**: Enterprises Enhance Privacy, Security and Control with Rackspace Technology's OpenStack Business Pri... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 243ms, Search: 29ms, LLM: 2768ms)
- **KIDZ**: Classover Increases Solana (SOL) Holdings by 295%, Surpasses 50,000 SOL Tokens in Treasury Reserve (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 277ms, Search: 29ms, LLM: 4430ms)
- **SCOR**: U.S. Joint Industry Committee Completes Audit of Certified Currencies to Validate Transactability of... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 269ms, Search: 29ms, LLM: 2591ms)
- **CXDO**: Crexendo Announces Completion of Key Oracle Cloud Infrastructure (OCI) Migration Milestones (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 191ms, Search: 28ms, LLM: 2278ms)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 1213ms, Search: 24ms, LLM: 2967ms)
- **CING**: Cingulate Receives $4.3M Waiver from FDA Ahead of Imminent Filing for Marketing Approval of Lead ADH... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 287ms, Search: 27ms, LLM: 2863ms)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 254ms, Search: 30ms, LLM: 2989ms)
- **OSRH**: OSR Holdings Enters into Term Sheet to Acquire Woori IO, a Pioneer in Noninvasive Glucose Monitoring... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 205ms, Search: 28ms, LLM: 2870ms)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 285ms, Search: 28ms, LLM: 2180ms)
- **KITT**: Nauticus Robotics Announces an Excellent Start to the 2025 Offshore Season (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 199ms, Search: 22ms, LLM: 3193ms)
- **CPOP**: CPOP Announces Plans to Enter Cryptocurrency Market (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 189ms, Search: 28ms, LLM: 4292ms)
- **KITT**: Nauticus Robotics Signs Master Services Agreement with Advanced Ocean Systems (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 248ms, Search: 28ms, LLM: 4185ms)
- **FLUX**: Flux Power Recognized Among Financial Times' Fastest Growing Companies in the Americas 2025 (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 205ms, Search: 28ms, LLM: 3489ms)
- **NAOV**: NanoVibronix Announces Financing of up to $50 Million Private Placement of Preferred Stock (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 96ms, Search: 27ms, LLM: 2560ms)
- **MGRM**: Zimmer Biomet Announces Definitive Agreement to Acquire Monogram Technologies, Expanding Robotics Su... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 203ms, Search: 28ms, LLM: 2841ms)
- **CRGX**: CARGO Therapeutics Enters into Agreement to Be Acquired by Concentra Biosciences for $4.379 in Cash ... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 208ms, Search: 22ms, LLM: 2663ms)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 290ms, Search: 27ms, LLM: 3542ms)
- **DVLT**: Datavault AI Announces Strategic and Operational Objectives for 3Q 2025 (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 109ms, Search: 28ms, LLM: 3140ms)
- **JSPR**: Jasper Therapeutics Reports Clinical Data Update from Briquilimab Studies in Chronic Spontaneous Urt... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 2829ms, Search: 24ms, LLM: 3781ms)
- **NIVF**: NewGen Announces Strategic Acquisition of Cytometry Technology and Assets to Support Planned U.S. Ex... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 258ms, Search: 29ms, LLM: 2913ms)
- **LEXX**: Lexaria's DehydraTECH Technology Has the Potential to Unlock Accelerated Revenue Growth in the GLP-1... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 222ms, Search: 28ms, LLM: 3063ms)
- **ISPC**: iSpecimen Inc. Announces Pricing of $4 Million Underwritten Offering (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 217ms, Search: 30ms, LLM: 2393ms)
- **CLAR**: Clarus Corporation Completes Sale of PIEPS Snow Safety Brand (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 187ms, Search: 28ms, LLM: 2691ms)
- **VCIG**: VCI Global Appoints Award-Winning Cybersecurity Leader Jane Teh as Chief AI Security Officer (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 187ms, Search: 21ms, LLM: 2765ms)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 198ms, Search: 28ms, LLM: 3335ms)
- **CELZ**: Creative Medical Technology Holdings Receives Notice of Allowance for ImmCelz for Treatment of Heart... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 262ms, Search: 28ms, LLM: 3645ms)
- **XAIR**: Beyond Air Awarded Therapeutic Gases Agreement with Premier, Inc. (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 268ms, Search: 28ms, LLM: 2835ms)
- **GAME**: GameSquare Announces Pricing of Underwritten Public Offering to Launch Ethereum Treasury Strategy (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 182ms, Search: 27ms, LLM: 2418ms)
- **NTWK**: NETSOL Technologies China Signs Strategic Agreement at the SCO Summit 2025 (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 245ms, Search: 29ms, LLM: 3858ms)
- **IMNN**: IMUNON Announces First Patient Dosed in Phase 3 OVATION 3 Study of IMNN-001 in Newly Diagnosed Advan... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 263ms, Search: 29ms, LLM: 2523ms)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 184ms, Search: 31ms, LLM: 3220ms)
- **TELO**: Telomir Demonstrates Telomir-1 Reverses Epigenetic Gene Silencing of STAT1, Restoring Tumor Suppress... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 268ms, Search: 22ms, LLM: 3423ms)
- **LASE**: Laser Photonics Subsidiary CMS Laser Wins New Order From Electrical Automation Professionals (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 235ms, Search: 28ms, LLM: 2951ms)
- **NVVE**: Nuvve Holding Corp. Announces Pricing of Public Offering of Common Stock to Launch HYPE Treasury Str... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 93ms, Search: 29ms, LLM: 2626ms)
- **GTN**: Gray Telemundo Stations to Air Carolina Panthers Preseason Games  in Spanish for the First Time (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 247ms, Search: 28ms, LLM: 2523ms)
- **ASTI**: Ascent Solar Technologies to Deliver Thin-Film Solar Technology to a Colorado-based Space Solar Arra... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 2849ms, Search: 23ms, LLM: 2477ms)
- **JAMF**: Jamf Announces Strategic Reinvestment Plan (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 252ms, Search: 29ms, LLM: 3517ms)
- **GTI**: Graphjet Technology Discloses Filing of Annual Report (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 181ms, Search: 28ms, LLM: 2411ms)
- **MAIA**: MAIA Biotechnology Announces First Patient Dosed in Expansion of Phase 2 Trial for Ateganosine in Ad... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 277ms, Search: 29ms, LLM: 3359ms)
- **GCL**: GCL Schedules Fiscal Year 2025 Earnings Release and Conference Call Date (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 485ms, Search: 24ms, LLM: 2891ms)
- **STXS**: Stereotaxis Receives U.S. FDA Clearance for MAGiC Sweep Catheter (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 200ms, Search: 28ms, LLM: 3175ms)
- **SLRX**: Salarius Pharmaceuticals' Seclidemstat Demonstrates Supporting Role in Inhibiting Validated Oncology... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 274ms, Search: 27ms, LLM: 2568ms)
- **MGRM**: Monogram Technologies Announces the Mandatory Conversion of 8.00% Series D Convertible Cumulative Pr... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 203ms, Search: 28ms, LLM: 2903ms)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 194ms, Search: 28ms, LLM: 3482ms)
- **PETS**: PetMeds Partners with myBalto Foundation to Raise Nearly $45,000 to Alleviate Financial Stress Pet O... (Confidence: 0.95, Actual: Neutral, Similar Examples: 3, Embed: 196ms, Search: 28ms, LLM: 2723ms)
- **APDN**: Applied DNA Announces New Follow-On LineaDNA Order from Global IVD Manufacturer for Use in Cancer Di... (Confidence: 0.95, Actual: False Pump, Similar Examples: 3, Embed: 280ms, Search: 27ms, LLM: 4353ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (23 missed)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: high)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Predicted: BUY, Confidence: medium)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: medium)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: medium)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: medium)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Predicted: BUY, Confidence: medium)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Predicted: BUY, Confidence: medium)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Predicted: BUY, Confidence: medium)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: BUY, Confidence: medium)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: BUY, Confidence: medium)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: medium)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: medium)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (6 missed)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: 0.95)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: 0.70)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: 0.70)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: HOLD, Confidence: 0.70)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: BUY, Confidence: 0.70)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: BUY, Confidence: 0.70)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 45.5% ❌
  - RAG: 35.5% ❌

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 90.9% ✅
  - RAG: 90.9% ✅

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 30.3%
  - RAG: 81.8%

### Integration Recommendation
❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria

**Analysis Time Overhead**: 1.07s per article 🚫


## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: 98
- **Total P&L**: $444294.50
- **Total Investment**: $1301438.80
- **Overall Return**: 34.14%

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | 22 | $117461.20 | $284243.00 | 41.32% |
| RAG | 76 | $326833.30 | $1017195.80 | 32.13% |

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
| **EVOK** | $27900.00 | $15000.00 | 186.00% | 1 |
| **HTOO** | $24820.00 | $39350.00 | 63.07% | 2 |
| **LIDR** | $22880.80 | $9040.00 | 253.11% | 1 |
| **AIM** | $20160.00 | $17000.00 | 118.59% | 1 |
| **MBIO** | $18240.80 | $9520.00 | 191.61% | 1 |
| **BGLC** | $18200.00 | $21500.00 | 84.65% | 1 |
| **SAFX** | $15600.80 | $12160.00 | 128.30% | 1 |
| **SMX** | $13918.40 | $32160.00 | 43.28% | 2 |
| **CVM** | $12900.00 | $19100.00 | 67.54% | 1 |
| **HOLO** | $11430.00 | $17910.00 | 63.82% | 1 |
| **MEIP** | $11040.00 | $15300.00 | 72.16% | 1 |
| **KAPA** | $10290.00 | $6710.00 | 153.35% | 1 |
| **HSDT** | $9040.00 | $19000.00 | 47.58% | 1 |
| **PMN** | $8587.00 | $4215.00 | 203.72% | 1 |
| **AEHL** | $6550.00 | $20600.00 | 31.80% | 1 |
| **PROK** | $6002.00 | $6998.00 | 85.77% | 1 |
| **AMOD** | $5280.00 | $10320.00 | 51.16% | 1 |
| **CGTX** | $4465.50 | $5200.00 | 85.88% | 1 |
| **SUGP** | $4451.00 | $4980.00 | 89.38% | 1 |
| **UAVS** | $4405.60 | $11760.00 | 37.46% | 1 |
| **APM** | $3800.00 | $9700.00 | 39.18% | 1 |
| **CLDI** | $3756.00 | $5349.00 | 70.22% | 1 |
| **KIDZ** | $3200.00 | $22880.00 | 13.99% | 1 |
| **ATXG** | $2805.00 | $7160.00 | 39.18% | 1 |
| **CPOP** | $2500.00 | $6410.00 | 39.00% | 1 |
| **TELO** | $2240.00 | $22880.00 | 9.79% | 1 |
| **CLSD** | $1727.00 | $3514.00 | 49.15% | 1 |
| **IMNN** | $1470.00 | $23550.00 | 6.24% | 1 |
| **XAIR** | $1450.00 | $16650.00 | 8.71% | 1 |
| **CELZ** | $1440.00 | $22240.00 | 6.47% | 1 |
| **STXS** | $1440.00 | $18400.00 | 7.83% | 1 |
| **AERT** | $1400.00 | $9000.00 | 15.56% | 1 |
| **APDN** | $1200.00 | $16050.00 | 7.48% | 1 |
| **MOGO** | $1200.00 | $15840.00 | 7.58% | 1 |
| **CING** | $1110.00 | $15090.00 | 7.36% | 1 |
| **NXXT** | $1040.00 | $18000.00 | 5.78% | 1 |
| **ICU** | $1008.00 | $12190.00 | 8.27% | 2 |
| **MAIA** | $954.40 | $15040.00 | 6.35% | 1 |
| **GAME** | $902.00 | $8900.00 | 10.13% | 1 |
| **GCL** | $900.00 | $16950.00 | 5.31% | 1 |
| **NIVF** | $727.00 | $5299.00 | 13.72% | 1 |
| **OSRH** | $716.80 | $8160.00 | 8.78% | 1 |
| **SLRX** | $700.00 | $7900.00 | 8.86% | 1 |
| **OSTX** | $640.00 | $15280.00 | 4.19% | 1 |
| **JAMF** | $620.00 | $16580.00 | 3.74% | 1 |
| **WINT** | $590.00 | $9710.00 | 6.08% | 1 |
| **MGRM** | $540.00 | $39990.00 | 1.35% | 2 |
| **NTWK** | $500.00 | $19000.00 | 2.63% | 1 |
| **GTN** | $450.00 | $16500.00 | 2.73% | 1 |
| **HYPR** | $409.00 | $7991.00 | 5.12% | 1 |
| **NAOV** | $362.00 | $9738.00 | 3.72% | 1 |
| **VCIG** | $320.00 | $10960.00 | 2.92% | 1 |
| **RXT** | $303.20 | $10640.80 | 2.85% | 1 |
| **KITT** | $243.00 | $19048.00 | 1.28% | 2 |
| **ASTI** | $240.00 | $14800.00 | 1.62% | 1 |
| **NVVE** | $211.00 | $9300.00 | 2.27% | 1 |
| **AREC** | $160.00 | $8800.00 | 1.82% | 1 |
| **ABEO** | $150.00 | $17700.00 | 0.85% | 1 |
| **GTI** | $9.00 | $721.00 | 1.25% | 1 |
| **LEXX** | $8.00 | $9700.00 | 0.08% | 1 |
| **SCOR** | $0.00 | $15930.00 | 0.00% | 1 |
| **FLUX** | $0.00 | $16000.00 | 0.00% | 1 |
| **PETS** | $0.00 | $18050.00 | 0.00% | 1 |
| **LASE** | $-100.00 | $15950.00 | -0.63% | 1 |
| **CXDO** | $-270.00 | $19500.00 | -1.38% | 1 |
| **CLAR** | $-450.00 | $18450.00 | -2.44% | 1 |
| **DVLT** | $-670.00 | $8682.00 | -7.72% | 1 |
| **CRGX** | $-1020.00 | $15000.00 | -6.80% | 1 |
| **ISPC** | $-1280.00 | $8480.00 | -15.09% | 1 |
| **JSPR** | $-12660.00 | $19560.00 | -64.72% | 1 |

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
| **02:00** | $5619.20 | $31400.00 | 17.90% | 2 |
| **03:00** | $33993.10 | $204304.00 | 16.64% | 16 |
| **04:00** | $231393.40 | $537035.80 | 43.09% | 40 |
| **05:00** | $55827.60 | $244456.00 | 22.84% | 18 |

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
| **$0.01-$1.00** | 10,000 | $54282.50 | $178415.00 | 30.42% | 25 |
| **$1.00-$3.00** | 8,000 | $136620.80 | $357280.80 | 38.24% | 24 |
| **$3.00-$5.00** | 5,000 | $78150.00 | $205150.00 | 38.09% | 11 |
| **$5.00-$8.00** | 3,000 | $27960.00 | $223770.00 | 12.49% | 13 |
| **$8.00+** | 2,000 | $29820.00 | $52580.00 | 56.71% | 3 |
