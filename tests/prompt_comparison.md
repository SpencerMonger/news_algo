# Sentiment Analysis Prompt Comparison

## Overview
This document compares the different prompts used across the sentiment analysis systems to understand why performance varies between tests.

---

## 1. **Live Sentiment Service Prompt** (`sentiment_service.py`)

**Location**: `sentiment_service.py` lines 714-748  
**Used By**: Production system, Traditional analysis in RAG comparison test  
**Model**: Claude-3.5-Sonnet (via load balancer)

```
Analyze the following news article about {ticker} and determine if it suggests a BUY, SELL, or HOLD signal based on the sentiment and potential market impact.

Article Content:
{content_to_analyze}

Instructions:
1. Analyze the sentiment (positive, negative, neutral)
2. Consider the potential market impact on stock price
3. Provide a clear recommendation:
   - BUY: For positive sentiment with strong bullish indicators
   - SELL: For negative sentiment with strong bearish indicators  
   - HOLD: For neutral sentiment or unclear market impact
4. Rate confidence as high, medium, or low
5. Give a brief explanation (1-2 sentences)

[USA tickers only:]
Special consideration: If the article discusses Bitcoin, cryptocurrency investments, or crypto-related business activities by the company, these should generally be viewed as high-confidence market movers. Bitcoin/crypto news often has significant immediate market impact on stock prices.

Respond in this exact JSON format:
{
    "ticker": "{ticker}",
    "sentiment": "positive/negative/neutral",
    "recommendation": "BUY/SELL/HOLD",
    "confidence": "high/medium/low",
    "explanation": "Brief explanation of your reasoning"
}

Important: Use exactly "BUY", "SELL", or "HOLD" for recommendation (not "NEUTRAL").
```

**Key Features:**
- Ticker-specific analysis
- Country-aware (Bitcoin consideration for USA tickers)
- Detailed instructions with clear categories
- Structured JSON response
- Scraped full content (up to 6000 chars)

---

## 2. **Sonnet Comparison Test Prompt** (`tests/sonnet_comparison_test.py`)

**Location**: `tests/sonnet_comparison_test.py` lines 212-228  
**Used By**: Sonnet-3.5 vs Sonnet-4 comparison test  
**Models**: Both Claude-3.5-Sonnet and Claude-4

```
Analyze this financial news article and provide sentiment analysis.

ARTICLE CONTENT:
{clean_content[:1500]}

Please analyze the sentiment and provide a trading recommendation. Consider:
- Market impact potential
- Company fundamentals mentioned
- Overall tone and sentiment
- Trading opportunities

Respond with JSON in this exact format:
{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of your analysis"
}
```

**Key Features:**
- Simpler, more direct approach
- Content limited to 1500 chars (vs 6000 in live service)
- No ticker-specific context
- No country-aware features
- Focuses on general sentiment analysis

---

## 3. **RAG-Enhanced Prompt** (`tests/rag_comparison_test.py`)

**Location**: `tests/rag_comparison_test.py` lines 673-688  
**Used By**: RAG vs Traditional comparison test  
**Model**: Claude-3.5-Sonnet (via load balancer)

```
Based on these similar historical examples and their outcomes:

{rag_context}

Now analyze this new article:
{clean_content}

IMPORTANT: Consider the historical patterns above. Look for opportunities that match successful patterns from TRUE_BULLISH examples. Don't be overly conservative - if the article shows strong positive signals similar to past winners, recommend BUY with appropriate confidence.

Key considerations:
- If similar to TRUE_BULLISH examples with high similarity: Consider BUY with high confidence
- If shows positive signals but uncertain: BUY with medium confidence is better than missing opportunities
- Only use HOLD if genuinely neutral or similar to FALSE_PUMP/NEUTRAL examples

Provide sentiment analysis considering the historical patterns shown above.
Respond with JSON: {"action": "BUY/HOLD/SELL", "confidence": "high/medium/low", "reasoning": "explanation based on historical patterns"}
```

**Key Features:**
- Historical context from similar examples
- Outcome-weighted pattern analysis
- Explicit bias toward BUY recommendations
- Anti-conservative instructions
- Pattern-based reasoning

---

## 4. **NEW EXPERIMENTAL PROMPTS** 🆕

### **4A. Breakout vs Pump-Dump Focused Prompt**

**Target**: Distinguish genuine 100-400% breakouts from false pumps  
**Strategy**: Focus on fundamental catalysts vs hype

```
Analyze this financial news article for genuine breakout potential vs pump-and-dump risk.

ARTICLE CONTENT:
{clean_content[:1500]}

CRITICAL DISTINCTION: You must identify news that will cause sustained 100-400% price increases (TRUE BREAKOUTS) vs news that creates temporary pumps followed by quick dumps (FALSE PUMPS).

TRUE BREAKOUT INDICATORS:
- Major business developments (partnerships, acquisitions, regulatory approvals)
- Fundamental game-changers (new products, market expansion, breakthrough technology)
- Concrete financial improvements (revenue growth, profitability, major contracts)
- Regulatory/legal victories with long-term impact

FALSE PUMP INDICATORS:
- Vague announcements without concrete details
- Speculative or "potential" developments
- Social media hype without substance
- Repetitive or recycled news

Respond with JSON:
{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Specific catalyst analysis and breakout vs pump assessment"
}
```

### **4B. Catalyst Strength Prompt**

**Target**: Focus on catalyst quality and sustainability  
**Strategy**: Rank catalyst strength and durability

```
Analyze this financial news for catalyst strength and sustainability.

ARTICLE CONTENT:
{clean_content[:1500]}

CATALYST EVALUATION: Rate the news catalyst's ability to drive sustained price appreciation.

STRONG CATALYSTS (BUY + high confidence):
- FDA approvals, major partnerships, acquisition announcements
- Breakthrough products with clear market demand
- Significant revenue/earnings beats with raised guidance
- Regulatory wins that unlock new markets

MODERATE CATALYSTS (BUY + medium confidence):
- Product launches, smaller partnerships, positive trials
- Good earnings with modest guidance improvements
- Market expansion announcements with clear timelines

WEAK/RISKY CATALYSTS (HOLD):
- Vague "exploring opportunities" statements
- Social media mentions or influencer endorsements
- Speculative research or "potential" developments
- Repeated announcements of old news

Respond with JSON:
{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Catalyst strength assessment and sustainability analysis"
}
```

### **4C. Institutional vs Retail Appeal Prompt**

**Target**: Identify news that attracts institutional vs retail money  
**Strategy**: Focus on professional investor criteria

```
Analyze this financial news from an institutional investor perspective.

ARTICLE CONTENT:
{clean_content[:1500]}

INSTITUTIONAL APPEAL: Evaluate if this news would attract serious institutional money (which drives sustained 100-400% moves) vs retail FOMO (which creates pump-and-dumps).

INSTITUTIONAL ATTRACTORS (BUY + high confidence):
- Concrete business metrics and financial improvements
- Strategic partnerships with established companies
- Regulatory approvals with clear market opportunities
- Management guidance increases with specific numbers

RETAIL FOMO TRIGGERS (HOLD - pump risk):
- Buzzword-heavy announcements (AI, blockchain, quantum)
- Celebrity endorsements or social media viral content
- Vague "revolutionary" claims without specifics
- Penny stock promotion language

EVALUATION CRITERIA:
- Would a pension fund invest based on this news?
- Are there specific, measurable business improvements?
- Is the language professional or promotional?

Respond with JSON:
{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Institutional appeal assessment and pump-dump risk analysis"
}
```

### **4D. Timing and Urgency Prompt**

**Target**: Identify immediate vs delayed impact catalysts  
**Strategy**: Focus on timing of price impact

```
Analyze this financial news for immediate market impact timing.

ARTICLE CONTENT:
{clean_content[:1500]}

TIMING ANALYSIS: Determine if this news will cause immediate explosive price action (hours/days) or delayed appreciation.

IMMEDIATE IMPACT CATALYSTS (BUY + high confidence):
- FDA approvals, merger announcements, major contract wins
- Earnings surprises with immediate market implications
- Breaking regulatory decisions or legal victories
- Emergency use authorizations or critical partnerships

DELAYED IMPACT NEWS (BUY + medium confidence):
- Product development milestones with future potential
- Strategic initiatives with 6-12 month timelines
- Market expansion plans requiring execution time
- Research results requiring further development

LOW IMPACT/SPECULATIVE (HOLD):
- Early-stage research or development updates
- Management commentary without concrete announcements
- Industry trend discussions without company-specific catalysts
- Vague future planning statements

Focus on: Will this move the stock price within 24-48 hours?

Respond with JSON:
{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Immediate impact timing assessment and catalyst urgency analysis"
}
```

### **4E. Concrete vs Speculative Prompt**

**Target**: Separate concrete developments from speculation  
**Strategy**: Demand specificity and measurability

```
Analyze this financial news for concrete vs speculative content.

ARTICLE CONTENT:
{clean_content[:1500]}

CONCRETE VS SPECULATIVE: Only concrete, measurable developments drive sustained 100-400% gains. Speculation creates pump-and-dumps.

CONCRETE DEVELOPMENTS (BUY + high confidence):
- Specific dollar amounts (revenue, contracts, investments)
- Named partners, customers, or acquisition targets
- Exact dates, percentages, or measurable milestones
- Completed transactions or finalized agreements

MODERATELY CONCRETE (BUY + medium confidence):
- Announced partnerships pending final agreements
- Product launches with specific target markets
- Financial guidance with concrete ranges
- Regulatory submissions with clear timelines

SPECULATIVE CONTENT (HOLD):
- "Exploring opportunities" or "considering options"
- "Potential," "possible," or "may" language
- Unnamed partners or "major company" references
- Vague timeline phrases like "in the coming months"

RED FLAGS: Promotional language, excessive superlatives, lack of specific details

Respond with JSON:
{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Concrete vs speculative assessment with specific evidence"
}
```

---

## 5. **System Context Comparison**

| System | System Prompt Addition |
|--------|------------------------|
| **Live Service** | `"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"` |
| **Sonnet Test** | `"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"` |
| **RAG Test** | Same as above (uses live service for traditional analysis) |
| **New Experimental** | Same system prompt for consistency |

---

## 6. **Key Differences Analysis**

### **Content Length**
- **Live Service**: Up to 6000 characters (full scraped content)
- **Sonnet Test**: Up to 1500 characters (truncated)
- **RAG Test**: Full content (uses live service for traditional)
- **New Experimental**: 1500 characters (following successful Sonnet test pattern)

### **Context Awareness**
- **Live Service**: Ticker-specific, country-aware, Bitcoin consideration
- **Sonnet Test**: Generic, no ticker context
- **RAG Test**: Historical pattern context, outcome-weighted
- **New Experimental**: Breakout vs pump-dump focused, no ticker bias

### **Instruction Style**
- **Live Service**: Detailed, structured instructions
- **Sonnet Test**: Simple, direct approach
- **RAG Test**: Pattern-focused, anti-conservative bias
- **New Experimental**: Catalyst-focused, pump-dump aware, specificity-demanding

### **Response Format**
- **Live Service**: `{"ticker", "sentiment", "recommendation", "confidence", "explanation"}`
- **Sonnet Test**: `{"recommendation", "confidence", "reasoning"}`
- **RAG Test**: `{"action", "confidence", "reasoning"}`
- **New Experimental**: `{"recommendation", "confidence", "reasoning"}` (same as Sonnet test)

---

## 7. **NEW EXPERIMENTAL PROMPT STRATEGIES** 🎯

### **4A - Breakout vs Pump-Dump**: Fundamental catalysts vs hype detection
### **4B - Catalyst Strength**: Quality and sustainability ranking
### **4C - Institutional Appeal**: Professional vs retail money attraction
### **4D - Timing Analysis**: Immediate vs delayed impact assessment  
### **4E - Concrete vs Speculative**: Measurability and specificity focus

**Common Design Principles:**
- Keep 1500 character limit (successful from Sonnet test)
- Maintain simple, direct approach
- Focus specifically on 100-400% breakout identification
- Include explicit pump-and-dump risk assessment
- Demand concrete evidence over speculation
- Use same JSON response format for consistency

---

## 8. **Performance Impact Analysis**

### **Why Sonnet Test Shows Higher Performance:**

1. **Simpler Prompt**: Less complex instructions may lead to more consistent responses
2. **Shorter Content**: 1500 char limit vs 6000 may focus on key information
3. **No Ticker Bias**: Generic analysis without company-specific considerations
4. **Direct Format**: Streamlined JSON response structure

### **Why RAG Shows Lower Traditional Performance:**

1. **Same Underlying Service**: RAG test uses the live sentiment service for "traditional" analysis
2. **Full Content Processing**: Processes complete articles (potentially noisier)
3. **Production Complexity**: Includes country lookup, content scraping, caching

### **Why Live Service May Vary:**

1. **Content Source**: Scraped content vs provided content
2. **Context Complexity**: More sophisticated but potentially confusing instructions
3. **Country-Specific Logic**: Additional complexity for Bitcoin/crypto considerations

### **Expected Performance of New Experimental Prompts:**

1. **4A-4E should outperform** current prompts by:
   - Explicitly targeting the 100-400% vs pump-dump distinction
   - Focusing on concrete catalysts over general sentiment
   - Including specific pump-and-dump risk assessment
   - Demanding measurable evidence over speculation

---

## 9. **Recommendations**

### **For Consistent Testing:**
1. Use identical prompts across all test systems
2. Standardize content length limits (1500 chars based on Sonnet success)
3. Remove system-specific features during comparison

### **For Production Optimization:**
1. Test new experimental prompts 4A-4E against current system
2. Consider simplifying the live service prompt
3. Evaluate country-specific logic effectiveness

### **For Future Comparisons:**
1. Create isolated prompt comparison tests with new experimental variants
2. Control for content length and complexity
3. Separate model performance from prompt engineering effects
4. **PRIORITY**: Test experimental prompts against TRUE_BULLISH vs FALSE_PUMP dataset 