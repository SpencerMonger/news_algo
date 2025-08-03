# Comprehensive Sentiment Analysis Comparison - Sonnet Models + Experimental Prompts

## Test Summary
- **Test Date**: 2025-08-02 20:21:22
- **Total Articles Analyzed**: 522
- **TRUE_BULLISH Articles in Test Set**: 174
- **Models/Prompts Tested**: 

## Overall Performance Metrics

### Experimental Prompts (All using Sonnet-3.5)
#### 4D: Timing and Urgency
- **Overall Accuracy**: 57.7%
- **BUY+High Precision**: 74.4% (43 signals)
- **BUY+High Recall**: 18.4%
- **BUY (Any) Recall**: 60.3%

---

## Performance Improvements vs Sonnet-3.5 Baseline

| Model/Prompt | Accuracy | BUY+High Precision | BUY+High Recall | BUY (Any) Recall |
|--------------|----------|-------------------|-----------------|------------------|

---

## Key Findings & Recommendations

### Best Performing Prompts

**Ranked by BUY+High Precision (Most Important for 100-400% Gains):**

1. **4D: Timing & Urgency**: 74.4% precision

### Test Analysis
- **Experimental Prompts**: All experimental prompts (4A-4E) were designed to distinguish genuine 100-400% breakout opportunities from pump-and-dump scenarios
- **Approach**: Each targets a different aspect of catalyst quality and sustainability
- **Model**: All experimental prompts use Sonnet-3.5 for fair comparison

### Recommendations
1. **Best Overall Performer**: 4D: Timing & Urgency with 74.4% BUY+High precision
2. **Production Implementation**: Consider adopting the top-performing approach for live trading
3. **Further Testing**: Validate top performers on larger datasets and different market conditions
