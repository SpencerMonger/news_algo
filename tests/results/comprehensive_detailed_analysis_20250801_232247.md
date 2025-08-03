# Comprehensive Sentiment Analysis Comparison - Sonnet Models + Experimental Prompts

## Test Summary
- **Test Date**: 2025-08-01 23:22:47
- **Total Articles Analyzed**: 3
- **TRUE_BULLISH Articles in Test Set**: 1
- **Models/Prompts Tested**: 4A-4E Experimental Prompts

## Overall Performance Metrics

### Experimental Prompts (All using Sonnet-3.5)
#### 4A: Breakout vs Pump-Dump Focused
- **Overall Accuracy**: 33.3%
- **BUY+High Precision**: 0.0% (1 signals)
- **BUY+High Recall**: 0.0%
- **BUY (Any) Recall**: 0.0%

#### 4B: Catalyst Strength
- **Overall Accuracy**: 66.7%
- **BUY+High Precision**: 0.0% (1 signals)
- **BUY+High Recall**: 0.0%
- **BUY (Any) Recall**: 100.0%

#### 4C: Institutional vs Retail Appeal
- **Overall Accuracy**: 33.3%
- **BUY+High Precision**: 0.0% (1 signals)
- **BUY+High Recall**: 0.0%
- **BUY (Any) Recall**: 0.0%

#### 4D: Timing and Urgency
- **Overall Accuracy**: 66.7%
- **BUY+High Precision**: 100.0% (1 signals)
- **BUY+High Recall**: 100.0%
- **BUY (Any) Recall**: 100.0%

#### 4E: Concrete vs Speculative
- **Overall Accuracy**: 66.7%
- **BUY+High Precision**: 0.0% (0 signals)
- **BUY+High Recall**: 0.0%
- **BUY (Any) Recall**: 100.0%

---

## Performance Improvements vs Sonnet-3.5 Baseline

| Model/Prompt | Accuracy | BUY+High Precision | BUY+High Recall | BUY (Any) Recall |
|--------------|----------|-------------------|-----------------|------------------|

---

## Key Findings & Recommendations

### Best Performing Prompts

**Ranked by BUY+High Precision (Most Important for 100-400% Gains):**

1. **4D: Timing & Urgency**: 100.0% precision
2. **4A: Breakout vs Pump-Dump**: 0.0% precision
3. **4B: Catalyst Strength**: 0.0% precision
4. **4C: Institutional Appeal**: 0.0% precision
5. **4E: Concrete vs Speculative**: 0.0% precision

### Test Analysis
- **Experimental Prompts**: All experimental prompts (4A-4E) were designed to distinguish genuine 100-400% breakout opportunities from pump-and-dump scenarios
- **Approach**: Each targets a different aspect of catalyst quality and sustainability
- **Model**: All experimental prompts use Sonnet-3.5 for fair comparison

### Recommendations
1. **Best Overall Performer**: 4D: Timing & Urgency with 100.0% BUY+High precision
2. **Production Implementation**: Consider adopting the top-performing approach for live trading
3. **Further Testing**: Validate top performers on larger datasets and different market conditions
