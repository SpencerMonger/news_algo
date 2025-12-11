#!/usr/bin/env python3
"""
Stock Strength Analyzer using Claude Sonnet 4.5
Analyzes financial data from float_list_detailed_dedup table and generates a strength score (1-10)
indicating likelihood of positive price movement on positive news.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import anthropic

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clickhouse_setup import ClickHouseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockStrengthAnalyzer:
    """Analyzes stock financial data using Claude AI to generate strength scores"""
    
    def __init__(self):
        self.clickhouse_manager = None
        self.anthropic_client = None
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Performance tracking
        self.stats = {
            'stocks_processed': 0,
            'stocks_successful': 0,
            'stocks_failed': 0,
            'api_calls': 0,
            'total_tokens': 0
        }
    
    def initialize(self):
        """Initialize connections to ClickHouse and Anthropic API"""
        logger.info("🚀 Initializing Stock Strength Analyzer...")
        
        # Connect to ClickHouse
        self.clickhouse_manager = ClickHouseManager()
        self.clickhouse_manager.connect()
        logger.info("✅ Connected to ClickHouse")
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        logger.info("✅ Initialized Anthropic API client")
        
        # Create or update table to include strength_score column
        self._setup_strength_score_column()
        
        logger.info("✅ Stock Strength Analyzer initialized successfully")
    
    def _setup_strength_score_column(self):
        """Add strength_score column to float_list_detailed_dedup table if it doesn't exist"""
        try:
            # Check if column exists
            check_query = """
            SELECT count() 
            FROM system.columns 
            WHERE database = 'News' 
            AND table = 'float_list_detailed_dedup' 
            AND name = 'strength_score'
            """
            result = self.clickhouse_manager.client.query(check_query)
            column_exists = result.result_rows[0][0] > 0
            
            if not column_exists:
                logger.info("Adding strength_score column to float_list_detailed_dedup table...")
                alter_query = """
                ALTER TABLE News.float_list_detailed_dedup 
                ADD COLUMN IF NOT EXISTS strength_score Nullable(Float64)
                """
                self.clickhouse_manager.client.command(alter_query)
                logger.info("✅ Added strength_score column")
            else:
                logger.info("✅ strength_score column already exists")
                
            # Also add analysis_timestamp column
            check_timestamp_query = """
            SELECT count() 
            FROM system.columns 
            WHERE database = 'News' 
            AND table = 'float_list_detailed_dedup' 
            AND name = 'analysis_timestamp'
            """
            result = self.clickhouse_manager.client.query(check_timestamp_query)
            timestamp_exists = result.result_rows[0][0] > 0
            
            if not timestamp_exists:
                logger.info("Adding analysis_timestamp column to float_list_detailed_dedup table...")
                alter_query = """
                ALTER TABLE News.float_list_detailed_dedup 
                ADD COLUMN IF NOT EXISTS analysis_timestamp Nullable(DateTime64(3))
                """
                self.clickhouse_manager.client.command(alter_query)
                logger.info("✅ Added analysis_timestamp column")
            else:
                logger.info("✅ analysis_timestamp column already exists")
                
        except Exception as e:
            logger.error(f"Error setting up strength_score column: {e}")
            raise
    
    def get_stocks_to_analyze(self, limit: Optional[int] = None, reanalyze: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve stocks from float_list_detailed_dedup table
        
        Args:
            limit: Optional limit on number of stocks to retrieve
            reanalyze: If True, retrieve all stocks. If False, only retrieve stocks without strength_score
        """
        try:
            # Build query based on reanalyze flag
            if reanalyze:
                query = """
                SELECT * FROM News.float_list_detailed_dedup
                ORDER BY ticker
                """
            else:
                query = """
                SELECT * FROM News.float_list_detailed_dedup
                WHERE strength_score IS NULL
                ORDER BY ticker
                """
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.clickhouse_manager.client.query(query)
            
            # Convert to list of dictionaries
            column_names = result.column_names
            stocks = []
            for row in result.result_rows:
                stock_data = dict(zip(column_names, row))
                stocks.append(stock_data)
            
            logger.info(f"📊 Retrieved {len(stocks)} stocks to analyze")
            return stocks
            
        except Exception as e:
            logger.error(f"Error retrieving stocks: {e}")
            return []
    
    def format_financial_data_for_analysis(self, stock_data: Dict[str, Any]) -> str:
        """
        Format stock financial data into a structured prompt for Claude
        
        Args:
            stock_data: Dictionary containing all stock financial data
            
        Returns:
            Formatted string for LLM analysis
        """
        ticker = stock_data.get('ticker', 'UNKNOWN')
        
        # Helper function to format values with proper handling of None
        def fmt(value, prefix='', suffix='', na_text='N/A'):
            if value is None:
                return na_text
            if isinstance(value, (int, float)):
                if abs(value) >= 1_000_000_000:
                    return f"{prefix}{value / 1_000_000_000:.2f}B{suffix}"
                elif abs(value) >= 1_000_000:
                    return f"{prefix}{value / 1_000_000:.2f}M{suffix}"
                elif abs(value) >= 1_000:
                    return f"{prefix}{value / 1_000:.2f}K{suffix}"
                else:
                    return f"{prefix}{value:.2f}{suffix}"
            return f"{prefix}{value}{suffix}"
        
        # Build comprehensive financial summary
        # Get raw values without converting None to 0
        cash = stock_data.get('cash_and_equivalents')
        debt = stock_data.get('total_debt')
        net_cash = stock_data.get('net_cash')
        shares = stock_data.get('shares_outstanding')
        
        # Calculate derived metrics only if source data exists
        net_cash_per_share = None
        if net_cash is not None and shares is not None and shares > 0:
            net_cash_per_share = net_cash / shares
        
        # Helper for conditional checks
        def check_positive(value):
            """Check if value is positive, handling None properly"""
            if value is None:
                return 'N/A'
            return 'YES ✓' if value > 0 else 'NO'
        
        def check_threshold(value, threshold, operator='>='):
            """Check if value meets threshold, handling None properly"""
            if value is None:
                return 'N/A'
            if operator == '>=':
                return 'YES ✓' if value >= threshold else 'NO'
            elif operator == '>':
                return 'YES ✓' if value > threshold else 'NO'
            return 'N/A'
        
        financial_summary = f"""
STOCK FINANCIAL ANALYSIS REQUEST FOR: {ticker}
{'=' * 80}

CRITICAL: "N/A" means data is MISSING - treat it as unknown, NOT as zero or negative.
Do NOT penalize stocks for missing data. Only evaluate based on available metrics.

KEY INDICATORS (Primary Focus):
1. BALANCE SHEET / CASH POSITION (MOST IMPORTANT):
   - Cash & Equivalents: {fmt(cash, '$')}
   - Total Debt: {fmt(debt, '$')}
   - Net Cash: {fmt(net_cash, '$')}
   - Net Cash per Share: {fmt(net_cash_per_share, '$')}
   - Positive Net Cash Position: {check_positive(net_cash)}

2. ALTMAN Z-SCORE (Financial Stability):
   - Score: {fmt(stock_data.get('altman_z_score'))}
   - Positive (>0) indicates financial stability: {check_threshold(stock_data.get('altman_z_score'), 0, '>')}

3. PIOTROSKI F-SCORE (Quality Indicator):
   - Score: {fmt(stock_data.get('piotroski_f_score'))}
   - Score >= 2 indicates decent quality: {check_threshold(stock_data.get('piotroski_f_score'), 2, '>=')}

PROFITABILITY & MARGINS:
- Profit Margin: {fmt(stock_data.get('profit_margin'), '', '%')}
- Operating Margin: {fmt(stock_data.get('operating_margin'), '', '%')}
- EBITDA Margin: {fmt(stock_data.get('ebitda_margin'), '', '%')}
- Gross Margin: {fmt(stock_data.get('gross_margin'), '', '%')}
- ROE: {fmt(stock_data.get('return_on_equity'), '', '%')}
- ROA: {fmt(stock_data.get('return_on_assets'), '', '%')}

CASH FLOW:
- Operating Cash Flow: {fmt(stock_data.get('operating_cash_flow'), '$')}
- Free Cash Flow: {fmt(stock_data.get('free_cash_flow'), '$')}
- FCF Margin: {fmt(stock_data.get('fcf_margin'), '', '%')}
- FCF per Share: {fmt(stock_data.get('fcf_per_share'), '$')}

ADDITIONAL FINANCIAL POSITION DETAILS:
- Current Ratio: {fmt(stock_data.get('current_ratio'))}
- Quick Ratio: {fmt(stock_data.get('quick_ratio'))}
- Debt/Equity: {fmt(stock_data.get('debt_to_equity'))}
- Interest Coverage: {fmt(stock_data.get('interest_coverage'))}

REVENUE & EARNINGS:
- Revenue: {fmt(stock_data.get('revenue'), '$')}
- Net Income: {fmt(stock_data.get('net_income'), '$')}
- EPS: {fmt(stock_data.get('earnings_per_share'), '$')}
- EBITDA: {fmt(stock_data.get('ebitda'), '$')}

SHARE STATISTICS:
- Shares Outstanding: {fmt(stock_data.get('shares_outstanding'))}
- Float: {fmt(stock_data.get('shares_float'))}
- Shares Change YoY: {fmt(stock_data.get('shares_change_yoy'), '', '%')}
- % Held by Insiders: {fmt(stock_data.get('percent_insiders'), '', '%')}
- % Held by Institutions: {fmt(stock_data.get('percent_institutions'), '', '%')}

VALUATION METRICS (Lower Priority):
- P/E Ratio: {fmt(stock_data.get('pe_ratio'))}
- Forward P/E: {fmt(stock_data.get('forward_pe'))}
- P/B Ratio: {fmt(stock_data.get('pb_ratio'))}
- PEG Ratio: {fmt(stock_data.get('peg_ratio'))}
- EV/EBITDA: {fmt(stock_data.get('ev_to_ebitda'))}
- Market Cap: {fmt(stock_data.get('market_cap'), '$')}
- Enterprise Value: {fmt(stock_data.get('enterprise_value'), '$')}
- Market Cap < Enterprise Value: {'YES ✓' if stock_data.get('market_cap') is not None and stock_data.get('enterprise_value') is not None and stock_data.get('market_cap') < stock_data.get('enterprise_value') else 'N/A'}

SHORT INTEREST:
- Short Interest: {fmt(stock_data.get('short_interest'))}
- Short % of Float: {fmt(stock_data.get('short_percent_float'), '', '%')}
- Short Ratio: {fmt(stock_data.get('short_ratio'))}

STOCK PERFORMANCE:
- 52-Week High: {fmt(stock_data.get('52_week_high'), '$')}
- 52-Week Low: {fmt(stock_data.get('52_week_low'), '$')}
- 52-Week Change: {fmt(stock_data.get('52_week_change'), '', '%')}
- Beta (5Y): {fmt(stock_data.get('beta_5y'))}
- RSI: {fmt(stock_data.get('relative_strength_index'))}
"""
        
        return financial_summary
    
    def analyze_stock_with_claude(self, stock_data: Dict[str, Any]) -> Optional[float]:
        """
        Use Claude Sonnet 4.5 to analyze stock and generate strength score
        
        Args:
            stock_data: Dictionary containing all stock financial data
            
        Returns:
            Strength score (1-10) or None if analysis fails
        """
        ticker = stock_data.get('ticker', 'UNKNOWN')
        
        try:
            # Format financial data
            financial_summary = self.format_financial_data_for_analysis(stock_data)
            
            # Create prompt for Claude
            system_prompt = """You are an expert financial analyst specializing in evaluating stock fundamentals and predicting price movement potential. 

Your task is to analyze the provided financial data and determine a "strength score" (1-10) that indicates how likely the stock price will move POSITIVELY when POSITIVE news is released about the company.

SCORING GUIDELINES:
- Score 1-3: Very weak fundamentals. Even with good news, the market will likely sell off due to poor underlying business quality.
- Score 4-6: Moderate fundamentals. Stock may have mixed reaction to positive news depending on sentiment.
- Score 7-8: Strong fundamentals. Positive news will likely result in positive price movement.
- Score 9-10: Excellent fundamentals. Positive news will very likely result in significant positive price movement.

CRITICAL CONSTRAINT - NEGATIVE ALTMAN Z-SCORE:
- If the Altman Z-Score is NEGATIVE (< 0), the maximum strength score you can assign is 3.5
- A negative Altman Z-Score indicates severe financial distress and high bankruptcy risk
- This constraint ONLY applies when the Altman Z-Score is explicitly negative
- If the Altman Z-Score is N/A (missing data), this constraint does NOT apply - evaluate the stock normally based on other available metrics

KEY EVALUATION CRITERIA (in order of importance):
1. BALANCE SHEET / CASH POSITION (HIGHEST PRIORITY):
   - Net Cash position is critical - positive net cash (cash > debt) is a strong positive indicator
   - Companies with more cash than debt have financial flexibility and resilience
   - Net cash per share shows the actual cash backing per share
   - This should be the PRIMARY factor in your evaluation

2. ALTMAN Z-SCORE (SECOND PRIORITY):
   - Altman Z-Score > 0 indicates financial stability and low bankruptcy risk
   - Higher scores indicate stronger financial health
   - This is a proven predictor of financial distress

3. PROFITABILITY & CASH FLOW GENERATION (THIRD PRIORITY):
   - Profit margins (net, operating, EBITDA margins)
   - Return on Equity (ROE) and Return on Assets (ROA)
   - Free Cash Flow and FCF margin
   - Positive and growing FCF is essential

4. PIOTROSKI F-SCORE:
   - Score >= 2 indicates decent business quality
   - Higher scores indicate better fundamental quality

5. ADDITIONAL FACTORS (Lower weight):
   - Interest coverage and debt ratios
   - Valuation multiples (P/E, P/B, EV/EBITDA)
   - Growth indicators
   - Market Cap vs Enterprise Value (minor indicator only - nice to have but not critical)

CRITICAL RULES FOR HANDLING MISSING DATA:
- "N/A" means the data point is UNAVAILABLE - it does NOT mean zero, negative, or bad
- Missing data is NEUTRAL - do not treat it as positive OR negative
- NEVER assume N/A = 0 (a stock with N/A cash is not the same as a stock with $0 cash)
- Only evaluate based on the metrics that ARE available
- If critical metrics are missing, be more conservative with the score but don't assume the worst
- Weight your analysis toward the available data points

EXAMPLES OF PROPER N/A HANDLING:
- If Net Cash = N/A: Don't assume it's 0 or negative, just note it's unknown
- If Altman Z-Score = N/A: Don't treat as bankruptcy risk, just note it's unavailable
- If a stock has strong margins but N/A for cash position: Score based on margins, note cash is unknown

Return ONLY a single number between 1 and 10 (can include decimals like 7.5). Do not include any explanation or other text."""

            user_prompt = f"""{financial_summary}

Based on the financial data above, provide a strength score (1-10) for {ticker}."""

            # Call Claude API
            logger.info(f"🤖 Analyzing {ticker} with Claude Sonnet 4.5...")
            
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                temperature=0.3,  # Low temperature for consistent scoring
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Extract response
            response_text = message.content[0].text.strip()
            
            # Update stats
            self.stats['api_calls'] += 1
            self.stats['total_tokens'] += message.usage.input_tokens + message.usage.output_tokens
            
            # Parse score from response
            try:
                strength_score = float(response_text)
                
                # Validate score is in range
                if not (1.0 <= strength_score <= 10.0):
                    logger.warning(f"⚠️ Score {strength_score} out of range for {ticker}, clamping to valid range")
                    strength_score = max(1.0, min(10.0, strength_score))
                
                logger.info(f"✅ {ticker} analyzed: Strength Score = {strength_score:.2f}")
                return strength_score
                
            except ValueError:
                logger.error(f"❌ Could not parse score from Claude response for {ticker}: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {ticker} with Claude: {e}")
            return None
    
    def update_strength_score(self, ticker: str, strength_score: float) -> bool:
        """
        Update the strength_score for a ticker in the database
        
        Args:
            ticker: Stock ticker symbol
            strength_score: Calculated strength score (1-10)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Simple ALTER UPDATE for MergeTree table
            update_query = f"""
            ALTER TABLE News.float_list_detailed_dedup 
            UPDATE 
                strength_score = {strength_score},
                analysis_timestamp = now64(3)
            WHERE ticker = '{ticker}'
            """
            
            self.clickhouse_manager.client.command(update_query)
            logger.debug(f"Updated strength_score for {ticker} in database")
            return True
            
        except Exception as e:
            logger.error(f"Error updating strength_score for {ticker}: {e}")
            return False
    
    def analyze_all_stocks(self, limit: Optional[int] = None, reanalyze: bool = False):
        """
        Analyze all stocks in the database and update strength scores
        
        Args:
            limit: Optional limit on number of stocks to analyze
            reanalyze: If True, reanalyze all stocks. If False, only analyze stocks without scores
        """
        try:
            # Get stocks to analyze
            stocks = self.get_stocks_to_analyze(limit=limit, reanalyze=reanalyze)
            
            if not stocks:
                logger.warning("⚠️ No stocks found to analyze")
                return
            
            logger.info(f"📊 Analyzing {len(stocks)} stocks...")
            
            # Analyze each stock
            for idx, stock_data in enumerate(stocks, 1):
                ticker = stock_data.get('ticker', 'UNKNOWN')
                self.stats['stocks_processed'] += 1
                
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing [{idx}/{len(stocks)}]: {ticker}")
                logger.info(f"{'=' * 80}")
                
                # Analyze with Claude
                strength_score = self.analyze_stock_with_claude(stock_data)
                
                if strength_score is not None:
                    # Update database
                    if self.update_strength_score(ticker, strength_score):
                        self.stats['stocks_successful'] += 1
                        logger.info(f"✅ Successfully updated {ticker} with strength score {strength_score:.2f}")
                    else:
                        self.stats['stocks_failed'] += 1
                        logger.error(f"❌ Failed to update database for {ticker}")
                else:
                    self.stats['stocks_failed'] += 1
                    logger.error(f"❌ Failed to analyze {ticker}")
            
            # Print final statistics
            logger.info("\n" + "=" * 80)
            logger.info("📊 ANALYSIS SUMMARY:")
            logger.info(f"   Total Stocks Processed: {self.stats['stocks_processed']}")
            logger.info(f"   ✅ Successful: {self.stats['stocks_successful']}")
            logger.info(f"   ❌ Failed: {self.stats['stocks_failed']}")
            logger.info(f"   API Calls Made: {self.stats['api_calls']}")
            logger.info(f"   Total Tokens Used: {self.stats['total_tokens']:,}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error in analyze_all_stocks: {e}")
            raise
    
    def get_strength_score_statistics(self):
        """Get statistics about strength scores in the database"""
        try:
            query = """
            SELECT 
                count() as total_stocks,
                countIf(strength_score IS NOT NULL) as stocks_with_scores,
                countIf(strength_score IS NULL) as stocks_without_scores,
                round(avg(strength_score), 2) as avg_score,
                round(min(strength_score), 2) as min_score,
                round(max(strength_score), 2) as max_score,
                round(quantile(0.5)(strength_score), 2) as median_score
            FROM News.float_list_detailed_dedup
            """
            
            result = self.clickhouse_manager.client.query(query)
            stats = dict(zip(result.column_names, result.result_rows[0]))
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 DATABASE STRENGTH SCORE STATISTICS:")
            logger.info(f"   Total Stocks: {stats['total_stocks']}")
            logger.info(f"   Stocks with Scores: {stats['stocks_with_scores']}")
            logger.info(f"   Stocks without Scores: {stats['stocks_without_scores']}")
            logger.info(f"   Average Score: {stats['avg_score']}")
            logger.info(f"   Minimum Score: {stats['min_score']}")
            logger.info(f"   Maximum Score: {stats['max_score']}")
            logger.info(f"   Median Score: {stats['median_score']}")
            logger.info("=" * 80)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting strength score statistics: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.clickhouse_manager:
                self.clickhouse_manager.close()
                logger.info("Closed ClickHouse connection")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main function to run the Stock Strength Analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze stocks using Claude AI and generate strength scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all stocks without scores
  python stock_strength_analyzer.py
  
  # Analyze only 5 stocks (for testing)
  python stock_strength_analyzer.py --limit 5
  
  # Reanalyze all stocks (including those with existing scores)
  python stock_strength_analyzer.py --reanalyze
  
  # Get statistics about existing scores
  python stock_strength_analyzer.py --stats-only
        """
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of stocks to analyze (useful for testing)'
    )
    parser.add_argument(
        '--reanalyze',
        action='store_true',
        help='Reanalyze all stocks, including those with existing scores'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only display statistics about existing scores, do not analyze'
    )
    
    args = parser.parse_args()
    
    analyzer = StockStrengthAnalyzer()
    
    try:
        # Initialize analyzer
        analyzer.initialize()
        
        # If stats-only mode, just show statistics
        if args.stats_only:
            analyzer.get_strength_score_statistics()
            return
        
        # Run analysis
        logger.info("🚀 Starting Stock Strength Analysis...")
        analyzer.analyze_all_stocks(limit=args.limit, reanalyze=args.reanalyze)
        
        # Show final statistics
        analyzer.get_strength_score_statistics()
        
    except KeyboardInterrupt:
        logger.info("\n⏸️ Analysis stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()

