#!/usr/bin/env python3
"""
Formula-Based Stock Strength Scorer (No LLM Required)

This script calculates a stock strength score (1-10) using a weighted formula
based on the same criteria used in the Claude-based analyzer.

Usage:
    python test_formula_strength.py AAPL
    python test_formula_strength.py TSLA --verbose
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from clickhouse_setup import ClickHouseManager

load_dotenv()


class FormulaStrengthScorer:
    """Calculate stock strength score using a weighted formula"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.clickhouse_manager = None
        self.score_breakdown = []
    
    def connect(self):
        """Connect to ClickHouse"""
        self.clickhouse_manager = ClickHouseManager()
        self.clickhouse_manager.connect()
    
    def close(self):
        """Close ClickHouse connection"""
        if self.clickhouse_manager:
            self.clickhouse_manager.close()
    
    def get_stock_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Retrieve stock data from float_list_detailed_dedup table"""
        query = f"""
        SELECT * FROM News.float_list_detailed_dedup
        WHERE ticker = '{ticker.upper()}'
        LIMIT 1
        """
        
        result = self.clickhouse_manager.client.query(query)
        
        if not result.result_rows:
            return None
        
        column_names = result.column_names
        stock_data = dict(zip(column_names, result.result_rows[0]))
        return stock_data
    
    def log(self, message: str, points: float = None):
        """Log scoring component if verbose mode"""
        if points is not None:
            self.score_breakdown.append((message, points))
        if self.verbose:
            if points is not None:
                sign = "+" if points >= 0 else ""
                print(f"  {message}: {sign}{points:.2f}")
            else:
                print(message)
    
    def safe_get(self, data: Dict, key: str) -> Optional[float]:
        """Safely get a numeric value from data dict"""
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    def calculate_strength_score(self, data: Dict[str, Any]) -> Tuple[float, list]:
        """
        Calculate strength score using weighted formula
        
        Returns:
            Tuple of (final_score, breakdown_list)
        """
        self.score_breakdown = []
        ticker = data.get('ticker', 'UNKNOWN')
        
        # Start with base score of 5.0 (neutral)
        score = 5.0
        self.log(f"\n{'='*60}")
        self.log(f"STRENGTH SCORE CALCULATION FOR: {ticker}")
        self.log(f"{'='*60}")
        self.log(f"\nBase Score: 5.00")
        
        # =========================================================
        # HARD CAP CHECK: Altman Z-Score < 0 caps score at 3.5
        # =========================================================
        altman_z = self.safe_get(data, 'altman_z_score')
        altman_cap_active = False
        
        if altman_z is not None and altman_z < 0:
            altman_cap_active = True
            self.log(f"\n⚠️  ALTMAN Z-SCORE CAP ACTIVE (Z={altman_z:.2f} < 0)")
            self.log(f"    Maximum possible score capped at 3.5")
        
        # =========================================================
        # PRIORITY 1: Balance Sheet / Cash Position (Weight: ~2.5 points)
        # =========================================================
        self.log(f"\n--- PRIORITY 1: Cash Position (max ±2.5 pts) ---")
        
        net_cash = self.safe_get(data, 'net_cash')
        cash = self.safe_get(data, 'cash_and_equivalents')
        debt = self.safe_get(data, 'total_debt')
        
        if net_cash is not None:
            if net_cash > 0:
                # Positive net cash is a strong signal
                # Scale: small positive = +0.5, large positive = +2.0
                cash_score = min(2.0, 0.5 + (net_cash / 1_000_000_000) * 0.5)
                score += cash_score
                self.log(f"Positive Net Cash (${net_cash/1e6:.1f}M)", cash_score)
            elif net_cash < 0:
                # Negative net cash (more debt than cash)
                # Scale: small negative = -0.5, large negative = -1.5
                cash_penalty = max(-1.5, -0.5 - (abs(net_cash) / 1_000_000_000) * 0.3)
                score += cash_penalty
                self.log(f"Negative Net Cash (${net_cash/1e6:.1f}M)", cash_penalty)
        elif cash is not None and debt is not None:
            # Calculate net cash if not directly available
            calc_net_cash = cash - debt
            if calc_net_cash > 0:
                cash_score = min(1.5, 0.3 + (calc_net_cash / 1_000_000_000) * 0.4)
                score += cash_score
                self.log(f"Calc Positive Net Cash (${calc_net_cash/1e6:.1f}M)", cash_score)
            elif calc_net_cash < 0:
                cash_penalty = max(-1.0, -0.3 - (abs(calc_net_cash) / 1_000_000_000) * 0.2)
                score += cash_penalty
                self.log(f"Calc Negative Net Cash (${calc_net_cash/1e6:.1f}M)", cash_penalty)
        else:
            self.log(f"Net Cash: N/A (no adjustment)", 0)
        
        # =========================================================
        # PRIORITY 2: Altman Z-Score (Weight: ~1.5 points)
        # =========================================================
        self.log(f"\n--- PRIORITY 2: Altman Z-Score (max ±1.5 pts) ---")
        
        if altman_z is not None:
            if altman_z >= 3.0:
                # Safe zone
                z_score = 1.5
                score += z_score
                self.log(f"Altman Z >= 3.0 (Safe Zone: {altman_z:.2f})", z_score)
            elif altman_z >= 1.8:
                # Grey zone but acceptable
                z_score = 0.75
                score += z_score
                self.log(f"Altman Z 1.8-3.0 (Grey Zone: {altman_z:.2f})", z_score)
            elif altman_z >= 0:
                # Risky but not distressed
                z_score = 0.0
                self.log(f"Altman Z 0-1.8 (Risky: {altman_z:.2f})", z_score)
            else:
                # Distress zone
                z_penalty = -1.5
                score += z_penalty
                self.log(f"Altman Z < 0 (Distress: {altman_z:.2f})", z_penalty)
        else:
            self.log(f"Altman Z-Score: N/A (no adjustment)", 0)
        
        # =========================================================
        # PRIORITY 3: Profitability & Cash Flow (Weight: ~1.5 points)
        # =========================================================
        self.log(f"\n--- PRIORITY 3: Profitability & Cash Flow (max ±1.5 pts) ---")
        
        # Free Cash Flow
        fcf = self.safe_get(data, 'free_cash_flow')
        if fcf is not None:
            if fcf > 0:
                fcf_score = min(0.5, 0.2 + (fcf / 1_000_000_000) * 0.1)
                score += fcf_score
                self.log(f"Positive FCF (${fcf/1e6:.1f}M)", fcf_score)
            else:
                fcf_penalty = max(-0.5, -0.2 - (abs(fcf) / 1_000_000_000) * 0.1)
                score += fcf_penalty
                self.log(f"Negative FCF (${fcf/1e6:.1f}M)", fcf_penalty)
        else:
            self.log(f"Free Cash Flow: N/A", 0)
        
        # Profit Margin
        profit_margin = self.safe_get(data, 'profit_margin')
        if profit_margin is not None:
            if profit_margin >= 20:
                pm_score = 0.5
            elif profit_margin >= 10:
                pm_score = 0.3
            elif profit_margin >= 0:
                pm_score = 0.1
            elif profit_margin >= -10:
                pm_score = -0.2
            else:
                pm_score = -0.4
            score += pm_score
            self.log(f"Profit Margin ({profit_margin:.1f}%)", pm_score)
        else:
            self.log(f"Profit Margin: N/A", 0)
        
        # ROE
        roe = self.safe_get(data, 'return_on_equity')
        if roe is not None:
            if roe >= 15:
                roe_score = 0.3
            elif roe >= 5:
                roe_score = 0.15
            elif roe >= 0:
                roe_score = 0.0
            else:
                roe_score = -0.2
            score += roe_score
            self.log(f"ROE ({roe:.1f}%)", roe_score)
        else:
            self.log(f"ROE: N/A", 0)
        
        # =========================================================
        # PRIORITY 4: Piotroski F-Score (Weight: ~0.75 points)
        # =========================================================
        self.log(f"\n--- PRIORITY 4: Piotroski F-Score (max ±0.75 pts) ---")
        
        piotroski = self.safe_get(data, 'piotroski_f_score')
        if piotroski is not None:
            if piotroski >= 7:
                p_score = 0.75
            elif piotroski >= 5:
                p_score = 0.5
            elif piotroski >= 3:
                p_score = 0.25
            elif piotroski >= 2:
                p_score = 0.0
            else:
                p_score = -0.5
            score += p_score
            self.log(f"Piotroski F-Score ({piotroski:.0f})", p_score)
        else:
            self.log(f"Piotroski F-Score: N/A", 0)
        
        # =========================================================
        # PRIORITY 5: Additional Factors (Weight: ~0.75 points)
        # =========================================================
        self.log(f"\n--- PRIORITY 5: Additional Factors (max ±0.75 pts) ---")
        
        # Current Ratio (liquidity)
        current_ratio = self.safe_get(data, 'current_ratio')
        if current_ratio is not None:
            if current_ratio >= 2.0:
                cr_score = 0.25
            elif current_ratio >= 1.5:
                cr_score = 0.15
            elif current_ratio >= 1.0:
                cr_score = 0.0
            else:
                cr_score = -0.25
            score += cr_score
            self.log(f"Current Ratio ({current_ratio:.2f})", cr_score)
        else:
            self.log(f"Current Ratio: N/A", 0)
        
        # Debt to Equity
        debt_equity = self.safe_get(data, 'debt_to_equity')
        if debt_equity is not None:
            if debt_equity <= 0.3:
                de_score = 0.25
            elif debt_equity <= 0.5:
                de_score = 0.15
            elif debt_equity <= 1.0:
                de_score = 0.0
            elif debt_equity <= 2.0:
                de_score = -0.15
            else:
                de_score = -0.25
            score += de_score
            self.log(f"Debt/Equity ({debt_equity:.2f})", de_score)
        else:
            self.log(f"Debt/Equity: N/A", 0)
        
        # Interest Coverage
        interest_cov = self.safe_get(data, 'interest_coverage')
        if interest_cov is not None:
            if interest_cov >= 10:
                ic_score = 0.25
            elif interest_cov >= 5:
                ic_score = 0.15
            elif interest_cov >= 2:
                ic_score = 0.0
            else:
                ic_score = -0.25
            score += ic_score
            self.log(f"Interest Coverage ({interest_cov:.1f}x)", ic_score)
        else:
            self.log(f"Interest Coverage: N/A", 0)
        
        # =========================================================
        # FINAL ADJUSTMENTS
        # =========================================================
        self.log(f"\n--- FINAL ADJUSTMENTS ---")
        
        # Clamp score to 1-10 range
        raw_score = score
        score = max(1.0, min(10.0, score))
        
        if raw_score != score:
            self.log(f"Clamped from {raw_score:.2f} to {score:.2f}")
        
        # Apply Altman Z-Score cap if active
        if altman_cap_active and score > 3.5:
            self.log(f"Altman Z-Score cap applied: {score:.2f} → 3.50")
            score = 3.5
        
        # Round to 2 decimal places
        score = round(score, 2)
        
        self.log(f"\n{'='*60}")
        self.log(f"FINAL STRENGTH SCORE: {score:.2f}")
        self.log(f"{'='*60}\n")
        
        return score, self.score_breakdown
    
    def print_raw_data(self, data: Dict[str, Any]):
        """Print key raw data values"""
        print(f"\n--- KEY FINANCIAL DATA ---")
        
        def fmt(val, prefix='', suffix=''):
            if val is None:
                return 'N/A'
            if isinstance(val, (int, float)):
                if abs(val) >= 1e9:
                    return f"{prefix}{val/1e9:.2f}B{suffix}"
                elif abs(val) >= 1e6:
                    return f"{prefix}{val/1e6:.2f}M{suffix}"
                elif abs(val) >= 1e3:
                    return f"{prefix}{val/1e3:.2f}K{suffix}"
                return f"{prefix}{val:.2f}{suffix}"
            return str(val)
        
        print(f"Cash & Equivalents: {fmt(data.get('cash_and_equivalents'), '$')}")
        print(f"Total Debt: {fmt(data.get('total_debt'), '$')}")
        print(f"Net Cash: {fmt(data.get('net_cash'), '$')}")
        print(f"Altman Z-Score: {fmt(data.get('altman_z_score'))}")
        print(f"Piotroski F-Score: {fmt(data.get('piotroski_f_score'))}")
        print(f"Free Cash Flow: {fmt(data.get('free_cash_flow'), '$')}")
        print(f"Profit Margin: {fmt(data.get('profit_margin'), '', '%')}")
        print(f"ROE: {fmt(data.get('return_on_equity'), '', '%')}")
        print(f"Current Ratio: {fmt(data.get('current_ratio'))}")
        print(f"Debt/Equity: {fmt(data.get('debt_to_equity'))}")
        print(f"Interest Coverage: {fmt(data.get('interest_coverage'))}")
        
        # Show existing LLM score if available
        llm_score = data.get('strength_score')
        if llm_score is not None:
            print(f"\n📊 Existing LLM Score: {llm_score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate stock strength score using weighted formula',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_formula_strength.py AAPL
  python test_formula_strength.py TSLA --verbose
  python test_formula_strength.py MSFT -v
        """
    )
    
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed scoring breakdown')
    
    args = parser.parse_args()
    
    scorer = FormulaStrengthScorer(verbose=args.verbose)
    
    try:
        scorer.connect()
        
        # Get stock data
        data = scorer.get_stock_data(args.ticker)
        
        if data is None:
            print(f"❌ Ticker '{args.ticker.upper()}' not found in database")
            sys.exit(1)
        
        # Print raw data if verbose
        if args.verbose:
            scorer.print_raw_data(data)
        
        # Calculate score
        score, breakdown = scorer.calculate_strength_score(data)
        
        if not args.verbose:
            # Simple output for non-verbose mode
            llm_score = data.get('strength_score')
            print(f"\n{'='*40}")
            print(f"Ticker: {args.ticker.upper()}")
            print(f"Formula Score: {score:.2f}")
            if llm_score is not None:
                print(f"LLM Score: {llm_score:.2f}")
                diff = abs(score - llm_score)
                print(f"Difference: {diff:.2f}")
            print(f"{'='*40}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        scorer.close()


if __name__ == "__main__":
    main()

