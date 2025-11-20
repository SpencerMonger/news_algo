#!/usr/bin/env python3
"""
Example queries for the float_list_detailed table
Demonstrates how to query and analyze the scraped statistics
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clickhouse_setup import ClickHouseManager

def print_separator():
    print("=" * 80)

def example_latest_stats_for_ticker(ch_manager, ticker='AAPL'):
    """Get the latest statistics for a specific ticker"""
    print_separator()
    print(f"Example 1: Latest Statistics for {ticker}")
    print_separator()
    
    query = f"""
    SELECT *
    FROM News.float_list_detailed
    WHERE ticker = '{ticker}'
    ORDER BY scraped_at DESC
    LIMIT 1
    """
    
    result = ch_manager.client.query(query)
    if result.result_rows:
        print(f"Found statistics for {ticker}:")
        for row in result.result_rows:
            print(f"  Scraped At: {row[1]}")
            print(f"  Market Cap: ${row[3]:,.0f}" if row[3] else "  Market Cap: N/A")
            print(f"  P/E Ratio: {row[5]:.2f}" if row[5] else "  P/E Ratio: N/A")
            print(f"  Beta (5Y): {row[11]:.2f}" if row[11] else "  Beta (5Y): N/A")
            print(f"  Dividend Yield: {row[24]:.2f}%" if row[24] else "  Dividend Yield: N/A")
    else:
        print(f"No statistics found for {ticker}")
    print()

def example_high_beta_stocks(ch_manager, min_beta=1.5):
    """Find stocks with high beta (volatility)"""
    print_separator()
    print(f"Example 2: High Beta Stocks (Beta > {min_beta})")
    print_separator()
    
    query = f"""
    SELECT 
        ticker,
        beta_5y,
        market_cap,
        52_week_change,
        scraped_at
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY scraped_at DESC) as rn
        FROM News.float_list_detailed
        WHERE beta_5y > {min_beta}
    ) WHERE rn = 1
    ORDER BY beta_5y DESC
    LIMIT 10
    """
    
    result = ch_manager.client.query(query)
    if result.result_rows:
        print(f"Top 10 High Beta Stocks:")
        print(f"{'Ticker':<10} {'Beta (5Y)':<12} {'Market Cap':<15} {'52W Change':<12}")
        print("-" * 60)
        for row in result.result_rows:
            ticker = row[0]
            beta = f"{row[1]:.2f}" if row[1] else "N/A"
            market_cap = f"${row[2]/1e6:.1f}M" if row[2] else "N/A"
            change = f"{row[3]:.1f}%" if row[3] else "N/A"
            print(f"{ticker:<10} {beta:<12} {market_cap:<15} {change:<12}")
    else:
        print("No high beta stocks found")
    print()

def example_dividend_growth_stocks(ch_manager):
    """Find stocks with strong dividend growth"""
    print_separator()
    print("Example 3: Dividend Growth Stocks")
    print_separator()
    
    query = """
    SELECT 
        ticker,
        dividend_yield,
        dividend_growth_yoy,
        years_dividend_growth,
        payout_ratio
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY scraped_at DESC) as rn
        FROM News.float_list_detailed
        WHERE dividend_yield > 2.0
          AND dividend_growth_yoy > 5.0
          AND years_dividend_growth >= 5
    ) WHERE rn = 1
    ORDER BY dividend_yield DESC
    LIMIT 10
    """
    
    result = ch_manager.client.query(query)
    if result.result_rows:
        print("Top 10 Dividend Growth Stocks:")
        print(f"{'Ticker':<10} {'Yield':<10} {'YoY Growth':<12} {'Years':<8} {'Payout %':<10}")
        print("-" * 60)
        for row in result.result_rows:
            ticker = row[0]
            yield_pct = f"{row[1]:.2f}%" if row[1] else "N/A"
            growth = f"{row[2]:.1f}%" if row[2] else "N/A"
            years = f"{int(row[3])}" if row[3] else "N/A"
            payout = f"{row[4]:.1f}%" if row[4] else "N/A"
            print(f"{ticker:<10} {yield_pct:<10} {growth:<12} {years:<8} {payout:<10}")
    else:
        print("No dividend growth stocks found")
    print()

def example_undervalued_stocks(ch_manager):
    """Find potentially undervalued stocks based on P/E and P/B ratios"""
    print_separator()
    print("Example 4: Potentially Undervalued Stocks")
    print_separator()
    
    query = """
    SELECT 
        ticker,
        pe_ratio,
        pb_ratio,
        market_cap,
        return_on_equity
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY scraped_at DESC) as rn
        FROM News.float_list_detailed
        WHERE pe_ratio > 0 AND pe_ratio < 15
          AND pb_ratio > 0 AND pb_ratio < 2
          AND return_on_equity > 10
          AND market_cap > 100000000
    ) WHERE rn = 1
    ORDER BY pe_ratio ASC
    LIMIT 10
    """
    
    result = ch_manager.client.query(query)
    if result.result_rows:
        print("Top 10 Potentially Undervalued Stocks:")
        print(f"{'Ticker':<10} {'P/E':<10} {'P/B':<10} {'Market Cap':<15} {'ROE':<10}")
        print("-" * 60)
        for row in result.result_rows:
            ticker = row[0]
            pe = f"{row[1]:.2f}" if row[1] else "N/A"
            pb = f"{row[2]:.2f}" if row[2] else "N/A"
            mcap = f"${row[3]/1e6:.1f}M" if row[3] else "N/A"
            roe = f"{row[4]:.1f}%" if row[4] else "N/A"
            print(f"{ticker:<10} {pe:<10} {pb:<10} {mcap:<15} {roe:<10}")
    else:
        print("No undervalued stocks found")
    print()

def example_high_profitability_stocks(ch_manager):
    """Find stocks with high profitability margins"""
    print_separator()
    print("Example 5: High Profitability Stocks")
    print_separator()
    
    query = """
    SELECT 
        ticker,
        profit_margin,
        operating_margin,
        return_on_equity,
        return_on_assets
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY scraped_at DESC) as rn
        FROM News.float_list_detailed
        WHERE profit_margin > 20
          AND operating_margin > 15
    ) WHERE rn = 1
    ORDER BY profit_margin DESC
    LIMIT 10
    """
    
    result = ch_manager.client.query(query)
    if result.result_rows:
        print("Top 10 High Profitability Stocks:")
        print(f"{'Ticker':<10} {'Profit %':<12} {'Operating %':<14} {'ROE %':<10} {'ROA %':<10}")
        print("-" * 60)
        for row in result.result_rows:
            ticker = row[0]
            profit = f"{row[1]:.1f}%" if row[1] else "N/A"
            operating = f"{row[2]:.1f}%" if row[2] else "N/A"
            roe = f"{row[3]:.1f}%" if row[3] else "N/A"
            roa = f"{row[4]:.1f}%" if row[4] else "N/A"
            print(f"{ticker:<10} {profit:<12} {operating:<14} {roe:<10} {roa:<10}")
    else:
        print("No high profitability stocks found")
    print()

def example_statistics_summary(ch_manager):
    """Get summary statistics for the entire database"""
    print_separator()
    print("Example 6: Database Summary Statistics")
    print_separator()
    
    query = """
    SELECT 
        COUNT(DISTINCT ticker) as total_tickers,
        COUNT(*) as total_records,
        MAX(scraped_at) as last_scrape,
        MIN(scraped_at) as first_scrape
    FROM News.float_list_detailed
    """
    
    result = ch_manager.client.query(query)
    if result.result_rows:
        row = result.result_rows[0]
        print(f"Total Unique Tickers: {row[0]}")
        print(f"Total Records: {row[1]}")
        print(f"Latest Scrape: {row[2]}")
        print(f"First Scrape: {row[3]}")
    print()

def main():
    """Run all example queries"""
    print("\n" + "=" * 80)
    print("StockAnalysis.com Statistics - Example Queries")
    print("=" * 80 + "\n")
    
    # Connect to ClickHouse
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    
    try:
        # Run all example queries
        example_statistics_summary(ch_manager)
        example_latest_stats_for_ticker(ch_manager, 'AAPL')
        example_high_beta_stocks(ch_manager)
        example_dividend_growth_stocks(ch_manager)
        example_undervalued_stocks(ch_manager)
        example_high_profitability_stocks(ch_manager)
        
        print_separator()
        print("All example queries completed!")
        print_separator()
        
    except Exception as e:
        print(f"Error running queries: {e}")
    finally:
        ch_manager.close()

if __name__ == "__main__":
    main()

