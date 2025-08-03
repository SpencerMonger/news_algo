# RAG vs Traditional Sentiment Analysis - Detailed BUY+High Analysis

## Test Summary
- **Test Date**: 2025-08-01 21:12:11
- **Total Articles Analyzed**: 98
- **TRUE_BULLISH Articles in Test Set**: 33

## Overall Performance Metrics

### Traditional Model
- **Overall Accuracy**: 46.9%
- **BUY+High Precision**: 45.5% (22 signals)
- **BUY+High Recall**: 30.3%
- **BUY (Any) Recall**: 90.9%

### RAG Model
- **Overall Accuracy**: 35.7%
- **BUY+High Precision**: 30.3% (76 signals)
- **BUY+High Recall**: 69.7%
- **BUY (Any) Recall**: 75.8%

### Performance Improvements
- **Accuracy**: -11.2%
- **BUY+High Precision**: -15.2%
- **BUY+High Recall**: +39.4%
- **BUY (Any) Recall**: -15.2%

---

## Detailed BUY+High Analysis

### Traditional Model BUY+High Predictions (22 total)


#### ✅ Correct Predictions (10/22 = 45.5%)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Confidence: high)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Confidence: high)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: high)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Confidence: high)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Confidence: high)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Confidence: high)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: high)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Confidence: high)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: high)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Confidence: high)

#### ❌ Incorrect Predictions (12/22 = 54.5%)
- **MGRM**: Zimmer Biomet Announces Definitive Agreement to Acquire Monogram Technologies, Expanding Robotics Su... (Confidence: high, Actual: False Pump)
- **TELO**: Telomir Demonstrates Telomir-1 Reverses Epigenetic Gene Silencing of STAT1, Restoring Tumor Suppress... (Confidence: high, Actual: False Pump)
- **STXS**: Stereotaxis Receives U.S. FDA Clearance for MAGiC Sweep Catheter (Confidence: high, Actual: False Pump)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: high, Actual: False Pump)
- **NIVF**: NewGen Announces Strategic Acquisition of Cytometry Technology and Assets to Support Planned U.S. Ex... (Confidence: high, Actual: Neutral)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: high, Actual: Neutral)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: high, Actual: False Pump)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: high, Actual: False Pump)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: high, Actual: False Pump)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: high, Actual: False Pump)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: high, Actual: Neutral)
- **OKYO**: OKYO Pharma Unveils Strong Phase 2 Clinical Trial Results for Urcosimod to Treat Neuropathic Corneal... (Confidence: high, Actual: False Pump)

### RAG Model BUY+High Predictions (76 total)


#### ✅ Correct Predictions (23/76 = 30.3%)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Confidence: 0.89, Similar Examples: 3, Embed: 213ms, Search: 0ms, LLM: 2827ms)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Confidence: 0.88, Similar Examples: 3, Embed: 242ms, Search: 0ms, LLM: 2235ms)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Confidence: 0.89, Similar Examples: 3, Embed: 184ms, Search: 0ms, LLM: 2462ms)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Confidence: 0.89, Similar Examples: 3, Embed: 257ms, Search: 0ms, LLM: 2123ms)
- **LIDR**: Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform, Paving the Way for Significan... (Confidence: 0.89, Similar Examples: 3, Embed: 380ms, Search: 0ms, LLM: 2328ms)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Confidence: 0.88, Similar Examples: 3, Embed: 223ms, Search: 0ms, LLM: 2353ms)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Confidence: 0.89, Similar Examples: 3, Embed: 239ms, Search: 0ms, LLM: 2315ms)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Confidence: 0.89, Similar Examples: 3, Embed: 211ms, Search: 0ms, LLM: 2446ms)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Confidence: 0.89, Similar Examples: 3, Embed: 228ms, Search: 0ms, LLM: 2363ms)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Confidence: 0.89, Similar Examples: 3, Embed: 236ms, Search: 0ms, LLM: 2213ms)
- **SUGP**: SU Group Secures Record-Breaking US$11.3 Million Hospital Contract in Hong Kong (Confidence: 0.88, Similar Examples: 3, Embed: 227ms, Search: 0ms, LLM: 2797ms)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Confidence: 0.89, Similar Examples: 3, Embed: 205ms, Search: 0ms, LLM: 2293ms)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Confidence: 0.89, Similar Examples: 3, Embed: 227ms, Search: 0ms, LLM: 2213ms)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Confidence: 0.89, Similar Examples: 3, Embed: 210ms, Search: 0ms, LLM: 2295ms)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Confidence: 0.89, Similar Examples: 3, Embed: 220ms, Search: 0ms, LLM: 3082ms)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Confidence: 0.88, Similar Examples: 3, Embed: 112ms, Search: 0ms, LLM: 2567ms)
- **AMOD**: Alpha Modus (NASDAQ: AMOD) Secures Exclusive National Rights to Deploy CashXAI's AI-Driven Financial... (Confidence: 0.90, Similar Examples: 3, Embed: 202ms, Search: 0ms, LLM: 2067ms)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Confidence: 0.89, Similar Examples: 3, Embed: 236ms, Search: 0ms, LLM: 2332ms)
- **CVM**: CEL-SCI to Sign Partnership Agreement With Leading Saudi Arabian Pharma Company for Multikine in the... (Confidence: 0.89, Similar Examples: 3, Embed: 1324ms, Search: 0ms, LLM: 2848ms)
- **BGLC**: BioNexus Gene Lab Corp. and Fidelion Diagnostics Announce Landmark Alliance-Touted as a new "DeepSee... (Confidence: 0.89, Similar Examples: 3, Embed: 279ms, Search: 0ms, LLM: 2484ms)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Confidence: 0.89, Similar Examples: 3, Embed: 248ms, Search: 0ms, LLM: 3454ms)
- **SAFX**: XCF Global Outlines Plan to Build Multiple SAF Production Facilities and Invest Nearly $1 Billion in... (Confidence: 0.89, Similar Examples: 3, Embed: 179ms, Search: 0ms, LLM: 2337ms)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Confidence: 0.89, Similar Examples: 3, Embed: 236ms, Search: 0ms, LLM: 3273ms)

#### ❌ Incorrect Predictions (53/76 = 69.7%)
- **CELZ**: Creative Medical Technology Holdings Receives Notice of Allowance for ImmCelz for Treatment of Heart... (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 174ms, Search: 0ms, LLM: 2576ms)
- **MGRM**: Zimmer Biomet Announces Definitive Agreement to Acquire Monogram Technologies, Expanding Robotics Su... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 224ms, Search: 0ms, LLM: 2437ms)
- **OSRH**: OSR Holdings Enters into Term Sheet to Acquire Woori IO, a Pioneer in Noninvasive Glucose Monitoring... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 223ms, Search: 0ms, LLM: 2522ms)
- **GCL**: GCL Schedules Fiscal Year 2025 Earnings Release and Conference Call Date (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 203ms, Search: 0ms, LLM: 2233ms)
- **CING**: Cingulate Receives $4.3M Waiver from FDA Ahead of Imminent Filing for Marketing Approval of Lead ADH... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 251ms, Search: 0ms, LLM: 2172ms)
- **TELO**: Telomir Demonstrates Telomir-1 Reverses Epigenetic Gene Silencing of STAT1, Restoring Tumor Suppress... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 181ms, Search: 0ms, LLM: 2004ms)
- **STXS**: Stereotaxis Receives U.S. FDA Clearance for MAGiC Sweep Catheter (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 237ms, Search: 0ms, LLM: 2205ms)
- **KIDZ**: Classover Increases Solana (SOL) Holdings by 295%, Surpasses 50,000 SOL Tokens in Treasury Reserve (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 198ms, Search: 0ms, LLM: 3000ms)
- **SCOR**: U.S. Joint Industry Committee Completes Audit of Certified Currencies to Validate Transactability of... (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 1098ms, Search: 0ms, LLM: 2267ms)
- **VSTM**: Verastem Oncology Announces Inducement Grants Under Nasdaq Listing Rule 5635(c)(4) (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 107ms, Search: 0ms, LLM: 2311ms)
- **KITT**: Nauticus Robotics Announces an Excellent Start to the 2025 Offshore Season (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 172ms, Search: 0ms, LLM: 3111ms)
- **PETS**: PetMeds Partners with myBalto Foundation to Raise Nearly $45,000 to Alleviate Financial Stress Pet O... (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 257ms, Search: 0ms, LLM: 2551ms)
- **VCIG**: VCI Global Appoints Award-Winning Cybersecurity Leader Jane Teh as Chief AI Security Officer (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 168ms, Search: 0ms, LLM: 3269ms)
- **CRGX**: CARGO Therapeutics Enters into Agreement to Be Acquired by Concentra Biosciences for $4.379 in Cash ... (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 209ms, Search: 0ms, LLM: 2282ms)
- **MAIA**: MAIA Biotechnology Announces First Patient Dosed in Expansion of Phase 2 Trial for Ateganosine in Ad... (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 228ms, Search: 0ms, LLM: 2242ms)
- **WVVI**: Willamette Valley Vineyards Expands Ownership Access with New Preferred Stock Offering (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 219ms, Search: 0ms, LLM: 2444ms)
- **NXXT**: NextNRG Reports Preliminary June 2025 Revenue Growth of 231% Year-Over-Year (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 222ms, Search: 0ms, LLM: 2127ms)
- **CPOP**: CPOP Announces Plans to Enter Cryptocurrency Market (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 172ms, Search: 0ms, LLM: 2460ms)
- **GAME**: GameSquare Announces Pricing of Underwritten Public Offering to Launch Ethereum Treasury Strategy (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 205ms, Search: 0ms, LLM: 2732ms)
- **RXT**: Enterprises Enhance Privacy, Security and Control with Rackspace Technology's OpenStack Business Pri... (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 173ms, Search: 0ms, LLM: 2089ms)
- **NTWK**: NETSOL Technologies China Signs Strategic Agreement at the SCO Summit 2025 (Confidence: 0.88, Actual: False Pump, Similar Examples: 3, Embed: 237ms, Search: 0ms, LLM: 2395ms)
- **NIVF**: NewGen Announces Strategic Acquisition of Cytometry Technology and Assets to Support Planned U.S. Ex... (Confidence: 0.90, Actual: Neutral, Similar Examples: 3, Embed: 204ms, Search: 0ms, LLM: 2393ms)
- **AERT**: Aeries Technology, Inc. (NASDAQ: AERT) Partners with Skydda.ai to Bring AI-Enabled SOC Operations to... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 2612ms, Search: 0ms, LLM: 2554ms)
- **WINT**: Kraken and Windtree Therapeutics Announce Strategic Partnership for BNB Custody, Trading, and OTC Se... (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 263ms, Search: 0ms, LLM: 1973ms)
- **LASE**: Laser Photonics Subsidiary CMS Laser Wins New Order From Electrical Automation Professionals (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 251ms, Search: 0ms, LLM: 2651ms)
- **EVTL**: Vertical Aerospace Announces Pricing of Underwritten Public Offering (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 86ms, Search: 0ms, LLM: 2700ms)
- **APDN**: Applied DNA Announces New Follow-On LineaDNA Order from Global IVD Manufacturer for Use in Cancer Di... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 269ms, Search: 0ms, LLM: 2222ms)
- **OMER**: Omeros Corporation Announces Pricing of $22 Million Registered Direct Offering (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 89ms, Search: 0ms, LLM: 1911ms)
- **NVVE**: Nuvve Holding Corp. Announces Pricing of Public Offering of Common Stock to Launch HYPE Treasury Str... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 96ms, Search: 0ms, LLM: 2551ms)
- **ABEO**: ZEVASKYN Gene Therapy Now Available at New Qualified Treatment Center in San Francisco Bay Area (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 272ms, Search: 0ms, LLM: 2811ms)
- **KITT**: Nauticus Robotics Signs Master Services Agreement with Advanced Ocean Systems (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 179ms, Search: 0ms, LLM: 2665ms)
- **JSPR**: Jasper Therapeutics Reports Clinical Data Update from Briquilimab Studies in Chronic Spontaneous Urt... (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 281ms, Search: 0ms, LLM: 2233ms)
- **JAMF**: Jamf Announces Strategic Reinvestment Plan (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 1562ms, Search: 0ms, LLM: 1831ms)
- **NAOV**: NanoVibronix Announces Financing of up to $50 Million Private Placement of Preferred Stock (Confidence: 0.88, Actual: False Pump, Similar Examples: 3, Embed: 101ms, Search: 0ms, LLM: 2788ms)
- **ICU**: SeaStar Medical Reports Positive Results for QUELIMMUNE Therapy in Pediatric Acute Kidney Injury (AK... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 201ms, Search: 0ms, LLM: 2444ms)
- **MGRM**: Monogram Technologies Announces the Mandatory Conversion of 8.00% Series D Convertible Cumulative Pr... (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 190ms, Search: 0ms, LLM: 2460ms)
- **XAIR**: Beyond Air Awarded Therapeutic Gases Agreement with Premier, Inc. (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 266ms, Search: 0ms, LLM: 2591ms)
- **CXDO**: Crexendo Announces Completion of Key Oracle Cloud Infrastructure (OCI) Migration Milestones (Confidence: 0.88, Actual: Neutral, Similar Examples: 3, Embed: 182ms, Search: 0ms, LLM: 2193ms)
- **ISPC**: iSpecimen Inc. Announces Pricing of $4 Million Underwritten Offering (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 194ms, Search: 0ms, LLM: 2509ms)
- **HYPR**: Hyperfine Announces the First Commercial Sales of the Next-Generation Swoop® System Powered by Optiv... (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 253ms, Search: 0ms, LLM: 2643ms)
- **GTI**: Graphjet Technology Discloses Filing of Annual Report (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 2179ms, Search: 0ms, LLM: 2035ms)
- **ASTI**: Ascent Solar Technologies to Deliver Thin-Film Solar Technology to a Colorado-based Space Solar Arra... (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 179ms, Search: 0ms, LLM: 2724ms)
- **FLUX**: Flux Power Recognized Among Financial Times' Fastest Growing Companies in the Americas 2025 (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 176ms, Search: 0ms, LLM: 2555ms)
- **OSTX**: OS Therapies Granted End of Phase 2 Meeting by US FDA for OST-HER2 Program in the Prevention or Dela... (Confidence: 0.91, Actual: False Pump, Similar Examples: 3, Embed: 269ms, Search: 0ms, LLM: 3373ms)
- **ICU**: SeaStar Medical Expands QUELIMMUNE Adoption for Critically Ill Pediatric Patients with Acute Kidney ... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 170ms, Search: 0ms, LLM: 2886ms)
- **CLAR**: Clarus Corporation Completes Sale of PIEPS Snow Safety Brand (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 231ms, Search: 0ms, LLM: 2076ms)
- **GTN**: Gray Telemundo Stations to Air Carolina Panthers Preseason Games  in Spanish for the First Time (Confidence: 0.87, Actual: Neutral, Similar Examples: 3, Embed: 174ms, Search: 0ms, LLM: 3034ms)
- **DVLT**: Datavault AI Announces Strategic and Operational Objectives for 3Q 2025 (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 109ms, Search: 0ms, LLM: 1853ms)
- **BMEA**: Biomea Fusion Appoints Julianne Averill to its Board of Directors (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 196ms, Search: 0ms, LLM: 2668ms)
- **SLRX**: Salarius Pharmaceuticals' Seclidemstat Demonstrates Supporting Role in Inhibiting Validated Oncology... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 283ms, Search: 0ms, LLM: 2153ms)
- **AREC**: ReElement Technologies Launches Urban Mining-to-Magnet Tolling Service for 99.5%+ Separated Rare Ear... (Confidence: 0.89, Actual: Neutral, Similar Examples: 3, Embed: 219ms, Search: 0ms, LLM: 2317ms)
- **OKYO**: OKYO Pharma Unveils Strong Phase 2 Clinical Trial Results for Urcosimod to Treat Neuropathic Corneal... (Confidence: 0.89, Actual: False Pump, Similar Examples: 3, Embed: 200ms, Search: 0ms, LLM: 2229ms)
- **MOGO**: Mogo Acquires 9% Stake in Bitcoin & Gold Treasury Company Digital Commodities Capital Corp. (Confidence: 0.90, Actual: False Pump, Similar Examples: 3, Embed: 99ms, Search: 0ms, LLM: 2407ms)

---

## Missed Opportunities Analysis

### TRUE_BULLISH Articles Missed by Traditional Model (23 missed)
- **CAPR**: Capricor Therapeutics Provides Regulatory Update on Deramiocel BLA for Duchenne Muscular Dystrophy (Predicted: SELL, Confidence: high)
- **SMX**: SMX Is Opening the Sustainability Market for GenX and Millennial Investors (Predicted: BUY, Confidence: medium)
- **HOLO**: MicroCloud Hologram Inc. Announces It Has Purchased Up to $200 Million in Bitcoin and Cryptocurrency... (Predicted: BUY, Confidence: medium)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: BUY, Confidence: medium)
- **AEHL**: AEHL Signs $50 Million Strategic Financing Agreement to Launch Bitcoin Acquisition Plan (Predicted: BUY, Confidence: medium)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: medium)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel Announces New LPG Projects for Subsidiary Al Shola Gas (Predicted: BUY, Confidence: medium)
- **HTOO**: Fusion Fuel's BrightHy Solutions Announces Non-Binding Term Sheet for Strategic Partnership with 30 ... (Predicted: BUY, Confidence: medium)
- **ATXG**: Addentax Group Corp. Enters Into US$1.3 Billion Term Sheet for Proposed Acquisition of Up to 12,000 ... (Predicted: BUY, Confidence: medium)
- **SMX**: From Exclusive to Inclusive: SMX's PCT Brings the Value of Sustainable Assets to the New Age Investo... (Predicted: BUY, Confidence: medium)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: SELL, Confidence: medium)
- **CGTX**: Cognition Therapeutics Completes End-of-Phase 2 Meeting with FDA for Zervimesine (CT1812) in Alzheim... (Predicted: BUY, Confidence: medium)
- **CLSD**: Clearside Biomedical Announces Approval of XIPERE Suprachoroidal Treatment for Uveitic Macular Edema... (Predicted: BUY, Confidence: medium)
- **MBIO**: Mustang Bio Granted Orphan Drug Designation by U.S. FDA for MB-101 (IL13Ra2-targeted CAR T-cells) to... (Predicted: BUY, Confidence: medium)
- **DARE**: Positive Interim Phase 3 Results Highlight Potential of Ovaprene, Novel Hormone-Free Contraceptive (Predicted: BUY, Confidence: medium)
- **KAPA**: Kairos Pharma Announces Positive Safety Results from Phase 2 Trial of ENV-105 in Advanced Prostate C... (Predicted: BUY, Confidence: medium)
- **MEIP**: MEI Pharma Announces $100,000,000 Private Placement to Initiate Litecoin Treasury Strategy, Becoming... (Predicted: BUY, Confidence: medium)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: medium)
- **APM**: Aptorum Group Limited and DiamiR Biosciences Enter into Definitive Merger Agreement (Predicted: BUY, Confidence: medium)
- **AIM**: AIM ImmunoTech Reports Positive Mid-year Safety and Efficacy Data from Phase 2 Study Evaluating Ampl... (Predicted: BUY, Confidence: medium)
- **CLDI**: Calidi Biotherapeutics Receives FDA Fast Track Designation for CLD-201 (SuperNova), a First-In-Class... (Predicted: BUY, Confidence: medium)

### TRUE_BULLISH Articles Missed by RAG Model (10 missed)
- **UAVS**: AgEagle Aerial Systems eBee TAC Drone Receives Blue UAS Clearance with Department of Defense (Predicted: HOLD, Confidence: 0.90)
- **PROK**: ProKidney Reports Statistically and Clinically Significant Topline Results for the Phase 2 REGEN-007... (Predicted: HOLD, Confidence: 0.89)
- **QLGN**: Qualigen Granted New Patents Covering 25 Countries (Predicted: HOLD, Confidence: 0.89)
- **LGPS**: LogProstyle Inc. Announces Approval of Cash Dividend at the 2025 Annual General Meeting of Sharehold... (Predicted: HOLD, Confidence: 0.89)
- **PROK**: ProKidney to Participate in the H.C. Wainwright 4th Annual Kidney Virtual Conference (Predicted: HOLD, Confidence: 0.89)
- **EVOK**: Evoke Pharma Receives Notice of Allowance for U.S. Patent Application for GIMOTI Extending Orange Bo... (Predicted: BUY, Confidence: 0.79)
- **HSDT**: Helius Announces Positive Outcome of the Portable Neuromodulation Stimulator PoNS Stroke Registratio... (Predicted: HOLD, Confidence: 0.91)
- **ANEB**: Anebulo Pharmaceuticals Approves Plan to Terminate Registration of Its Common Stock (Predicted: HOLD, Confidence: 0.88)
- **PMN**: ProMIS Neurosciences Granted Fast Track Designation by U.S. FDA for PMN310 in the Treatment of Alzhe... (Predicted: HOLD, Confidence: 0.90)
- **LGPS**: LogProstyle Inc. Announces Approval of Share Repurchase Program by the Board of Directors (Predicted: BUY, Confidence: 0.80)

---

## Success Criteria Assessment

### Target Goals (from README)
- **BUY+High Precision**: Target >80%
  - Traditional: 45.5% ❌
  - RAG: 30.3% ❌

- **TRUE_BULLISH Recall**: Target >90% (BUY any confidence)
  - Traditional: 90.9% ✅
  - RAG: 75.8% ❌

- **BUY+High Recall**: How well each model captures TRUE_BULLISH with high confidence
  - Traditional: 30.3%
  - RAG: 69.7%

### Integration Recommendation
❌ **DO NOT INTEGRATE** - RAG does not meet improvement criteria

**Analysis Time Overhead**: 0.67s per article 🚫


## PnL Analysis Summary

### Overall Trading Performance
- **Total Trades**: 98
- **Total P&L**: $384421.90
- **Total Investment**: $1339165.80
- **Overall Return**: 28.71%

### Model Comparison
| Model | Trades | P&L | Investment | Return |
|-------|--------|-----|------------|--------|
| Traditional | 22 | $117461.20 | $284243.00 | 41.32% |
| RAG | 76 | $266960.70 | $1054922.80 | 25.31% |

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
| **AEHL** | $6550.00 | $20600.00 | 31.80% | 1 |
| **AMOD** | $5280.00 | $10320.00 | 51.16% | 1 |
| **CGTX** | $4465.50 | $5200.00 | 85.88% | 1 |
| **SUGP** | $4451.00 | $4980.00 | 89.38% | 1 |
| **APM** | $3800.00 | $9700.00 | 39.18% | 1 |
| **CLDI** | $3756.00 | $5349.00 | 70.22% | 1 |
| **KIDZ** | $3200.00 | $22880.00 | 13.99% | 1 |
| **ATXG** | $2805.00 | $7160.00 | 39.18% | 1 |
| **CPOP** | $2500.00 | $6410.00 | 39.00% | 1 |
| **TELO** | $2240.00 | $22880.00 | 9.79% | 1 |
| **OKYO** | $1760.00 | $22400.00 | 7.86% | 1 |
| **CLSD** | $1727.00 | $3514.00 | 49.15% | 1 |
| **XAIR** | $1450.00 | $16650.00 | 8.71% | 1 |
| **CELZ** | $1440.00 | $22240.00 | 6.47% | 1 |
| **STXS** | $1440.00 | $18400.00 | 7.83% | 1 |
| **AERT** | $1400.00 | $9000.00 | 15.56% | 1 |
| **APDN** | $1200.00 | $16050.00 | 7.48% | 1 |
| **MOGO** | $1200.00 | $15840.00 | 7.58% | 1 |
| **CING** | $1110.00 | $15090.00 | 7.36% | 1 |
| **OMER** | $1050.00 | $18000.00 | 5.83% | 1 |
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
| **SCOR** | $0.00 | $15930.00 | 0.00% | 1 |
| **PETS** | $0.00 | $18050.00 | 0.00% | 1 |
| **WVVI** | $0.00 | $16140.00 | 0.00% | 1 |
| **FLUX** | $0.00 | $16000.00 | 0.00% | 1 |
| **LASE** | $-100.00 | $15950.00 | -0.63% | 1 |
| **BMEA** | $-240.00 | $14800.00 | -1.62% | 1 |
| **VSTM** | $-250.00 | $23600.00 | -1.06% | 1 |
| **CXDO** | $-270.00 | $19500.00 | -1.38% | 1 |
| **CLAR** | $-450.00 | $18450.00 | -2.44% | 1 |
| **DVLT** | $-670.00 | $8682.00 | -7.72% | 1 |
| **CRGX** | $-1020.00 | $15000.00 | -6.80% | 1 |
| **ISPC** | $-1280.00 | $8480.00 | -15.09% | 1 |
| **CAPR** | $-2140.00 | $16000.00 | -13.38% | 1 |
| **EVTL** | $-2640.00 | $17010.00 | -15.52% | 1 |
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
| **02:00** | $3479.20 | $47400.00 | 7.34% | 3 |
| **03:00** | $18461.10 | $216706.00 | 8.52% | 16 |
| **04:00** | $187440.80 | $533660.80 | 35.12% | 39 |
| **05:00** | $57579.60 | $257156.00 | 22.39% | 18 |

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
| **$0.01-$1.00** | 10,000 | $39685.50 | $157502.00 | 25.20% | 22 |
| **$1.00-$3.00** | 8,000 | $133735.20 | $382720.80 | 34.94% | 25 |
| **$3.00-$5.00** | 5,000 | $51050.00 | $231750.00 | 22.03% | 12 |
| **$5.00-$8.00** | 3,000 | $23850.00 | $233370.00 | 10.22% | 14 |
| **$8.00+** | 2,000 | $18640.00 | $49580.00 | 37.60% | 3 |
