# Comprehensive Sentiment Analysis Comparison - Sonnet Models + Experimental Prompts

## Test Summary
- **Test Date**: 2025-08-02 00:21:24
- **Total Articles Analyzed**: 98
- **TRUE_BULLISH Articles in Test Set**: 33
- **Models/Prompts Tested**: Sonnet-3.5, Sonnet-4, 4A-4E Experimental Prompts

## Overall Performance Metrics

### Original Models
#### Sonnet-3.5 Model (Baseline)
- **Overall Accuracy**: 50.0%
- **BUY+High Precision**: 60.0% (10 signals)
- **BUY+High Recall**: 18.2%
- **BUY (Any) Recall**: 90.9%

#### Sonnet-4 Model
- **Overall Accuracy**: 45.9%
- **BUY+High Precision**: 50.0% (18 signals)
- **BUY+High Recall**: 27.3%
- **BUY (Any) Recall**: 69.7%

### Experimental Prompts (All using Sonnet-3.5)
#### 4A: Breakout vs Pump-Dump Focused
- **Overall Accuracy**: 60.2%
- **BUY+High Precision**: 0.0% (2 signals)
- **BUY+High Recall**: 0.0%
- **BUY (Any) Recall**: 9.1%

#### 4B: Catalyst Strength
- **Overall Accuracy**: 51.0%
- **BUY+High Precision**: 50.0% (4 signals)
- **BUY+High Recall**: 6.1%
- **BUY (Any) Recall**: 69.7%

#### 4C: Institutional vs Retail Appeal
- **Overall Accuracy**: 61.2%
- **BUY+High Precision**: 37.5% (8 signals)
- **BUY+High Recall**: 9.1%
- **BUY (Any) Recall**: 39.4%

#### 4D: Timing and Urgency
- **Overall Accuracy**: 54.1%
- **BUY+High Precision**: 69.2% (13 signals)
- **BUY+High Recall**: 27.3%
- **BUY (Any) Recall**: 66.7%

#### 4E: Concrete vs Speculative
- **Overall Accuracy**: 54.1%
- **BUY+High Precision**: 33.3% (6 signals)
- **BUY+High Recall**: 6.1%
- **BUY (Any) Recall**: 45.5%

---

## Performance Improvements vs Sonnet-3.5 Baseline

| Model/Prompt | Accuracy | BUY+High Precision | BUY+High Recall | BUY (Any) Recall |
|--------------|----------|-------------------|-----------------|------------------|
| Sonnet-4 | -4.1% | -10.0% | +9.1% | -21.2% |
| 4A: Breakout vs Pump | +10.2% | -60.0% | -18.2% | -81.8% |
| 4B: Catalyst Strength | +1.0% | -10.0% | -12.1% | -21.2% |
| 4C: Institutional Appeal | +11.2% | -22.5% | -9.1% | -51.5% |
| 4D: Timing & Urgency | +4.1% | +9.2% | +9.1% | -24.2% |
| 4E: Concrete vs Speculative | +4.1% | -26.7% | -12.1% | -45.5% |

---

## Key Findings & Recommendations

### Best Performing Prompts

**Ranked by BUY+High Precision (Most Important for 100-400% Gains):**

1. **4D: Timing & Urgency**: 69.2% precision
2. **Sonnet-3.5 (Baseline)**: 60.0% precision
3. **Sonnet-4**: 50.0% precision
4. **4B: Catalyst Strength**: 50.0% precision
5. **4C: Institutional Appeal**: 37.5% precision

### Test Analysis
- **Experimental Prompts**: All experimental prompts (4A-4E) were designed to distinguish genuine 100-400% breakout opportunities from pump-and-dump scenarios
- **Approach**: Each targets a different aspect of catalyst quality and sustainability
- **Model**: All experimental prompts use Sonnet-3.5 for fair comparison

### Recommendations
1. **Best Overall Performer**: 4D: Timing & Urgency with 69.2% BUY+High precision
2. **Production Implementation**: Consider adopting the top-performing approach for live trading
3. **Further Testing**: Validate top performers on larger datasets and different market conditions
