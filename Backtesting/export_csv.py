#!/usr/bin/env python3
"""
Export CSV for Tradervue
Exports backtest trade results to Tradervue's generic CSV format
Reference: https://www.tradervue.com/help/generic
"""

import os
import sys
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradervueCSVExporter:
    """
    Export backtest trade results to Tradervue generic CSV format
    """
    
    def __init__(self):
        self.ch_manager = None
        
        # Tradervue generic CSV columns (required fields)
        self.csv_columns = [
            'Symbol',           # Ticker symbol
            'Trade Date',       # Date of trade (YYYY-MM-DD)
            'Side',             # 'Buy' or 'Sell'
            'Quantity',         # Number of shares
            'Price',            # Execution price
            'Exec Time',        # Execution time (HH:MM:SS)
            'Notes'             # Optional notes field
        ]
        
        # Optional enhanced columns for more detailed analysis
        self.enhanced_columns = [
            'Symbol',
            'Trade Date',
            'Side',
            'Quantity', 
            'Price',
            'Exec Time',
            'P&L',              # Profit/Loss for the trade
            'Commission',       # Commission (set to 0 for backtesting)
            'Notes'
        ]

    def initialize(self):
        """Initialize the CSV exporter"""
        try:
            # Initialize ClickHouse connection
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            
            logger.info("✅ Tradervue CSV Exporter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing CSV exporter: {e}")
            return False

    def get_trade_data(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get trade data from backtest_trades table"""
        try:
            # Build query
            query = """
            SELECT 
                trade_id,
                ticker,
                published_utc,
                entry_time,
                exit_time,
                entry_price,
                exit_price,
                quantity,
                entry_type,
                exit_type,
                pnl,
                pnl_percent,
                trade_duration_seconds,
                sentiment,
                recommendation,
                confidence,
                explanation
            FROM News.backtest_trades
            ORDER BY entry_time ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.ch_manager.client.query(query)
            
            trades = []
            for row in result.result_rows:
                (trade_id, ticker, published_utc, entry_time, exit_time, 
                 entry_price, exit_price, quantity, entry_type, exit_type,
                 pnl, pnl_percent, trade_duration_seconds, sentiment, 
                 recommendation, confidence, explanation) = row
                
                trades.append({
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'published_utc': published_utc,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'entry_type': entry_type,
                    'exit_type': exit_type,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'trade_duration_seconds': trade_duration_seconds,
                    'sentiment': sentiment,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'explanation': explanation
                })
            
            logger.info(f"📊 Retrieved {len(trades)} trades from database")
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade data: {e}")
            return []

    def format_tradervue_rows(self, trades: List[Dict[str, Any]], enhanced: bool = True) -> List[Dict[str, str]]:
        """Format trade data into Tradervue CSV format"""
        tradervue_rows = []
        
        for trade in trades:
            try:
                ticker = trade['ticker']
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                quantity = trade['quantity']
                pnl = trade['pnl']
                
                # Format dates and times for Tradervue
                entry_date = entry_time.strftime('%Y-%m-%d')
                entry_time_str = entry_time.strftime('%H:%M:%S')
                exit_date = exit_time.strftime('%Y-%m-%d')
                exit_time_str = exit_time.strftime('%H:%M:%S')
                
                # Create notes with trade details - ENHANCED with price trigger info
                price_trigger_info = ""
                if 'price_increase_pct' in trade and 'bars_count' in trade:
                    price_trigger_info = f"Price trigger: {trade['price_increase_pct']:.2f}% move, {trade['bars_count']} bars, "
                
                notes = (f"News Trade - {trade['sentiment'].upper()} sentiment, "
                        f"{trade['confidence']} confidence, "
                        f"{price_trigger_info}"
                        f"Duration: {trade['trade_duration_seconds']}s, "
                        f"Explanation: {trade['explanation'][:100]}")
                
                # Entry leg (BUY)
                entry_row = {
                    'Symbol': ticker,
                    'Trade Date': entry_date,
                    'Side': 'Buy',
                    'Quantity': str(quantity),
                    'Price': f"{entry_price:.4f}",
                    'Exec Time': entry_time_str,
                    'Notes': f"ENTRY - {notes}"
                }
                
                # Add enhanced columns if needed
                if enhanced:
                    entry_row['P&L'] = "0.00"  # No P&L on entry
                    entry_row['Commission'] = "0.00"  # No commission for backtesting
                
                tradervue_rows.append(entry_row)
                
                # Exit leg (SELL)
                exit_row = {
                    'Symbol': ticker,
                    'Trade Date': exit_date,
                    'Side': 'Sell',
                    'Quantity': str(quantity),
                    'Price': f"{exit_price:.4f}",
                    'Exec Time': exit_time_str,
                    'Notes': f"EXIT - {notes}"
                }
                
                # Add enhanced columns if needed
                if enhanced:
                    exit_row['P&L'] = f"{pnl:.2f}"  # P&L on exit
                    exit_row['Commission'] = "0.00"  # No commission for backtesting
                
                tradervue_rows.append(exit_row)
                
            except Exception as e:
                logger.error(f"Error formatting trade {trade.get('trade_id', 'UNKNOWN')}: {e}")
                continue
        
        logger.info(f"📝 Formatted {len(tradervue_rows)} rows for Tradervue export")
        return tradervue_rows

    def export_to_csv(self, filename: str = None, enhanced: bool = True, limit: int = None) -> str:
        """Export trades to Tradervue CSV format"""
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"newshead_backtest_tradervue_{timestamp}.csv"
            
            # Ensure filename ends with .csv
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Create output directory if it doesn't exist
            os.makedirs('exports', exist_ok=True)
            filepath = os.path.join('exports', filename)
            
            # Get trade data
            trades = self.get_trade_data(limit=limit)
            
            if not trades:
                logger.error("No trade data found to export")
                return None
            
            # Format for Tradervue
            tradervue_rows = self.format_tradervue_rows(trades, enhanced=enhanced)
            
            if not tradervue_rows:
                logger.error("No valid rows formatted for export")
                return None
            
            # Determine columns to use
            columns = self.enhanced_columns if enhanced else self.csv_columns
            
            # Write CSV file
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                
                # Write header
                writer.writeheader()
                
                # Write data rows
                for row in tradervue_rows:
                    writer.writerow(row)
            
            # Calculate summary stats
            total_trades = len(trades)
            total_rows = len(tradervue_rows)
            total_pnl = sum(trade['pnl'] for trade in trades)
            profitable_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            logger.info("🎉 CSV EXPORT COMPLETE!")
            logger.info(f"📄 File: {filepath}")
            logger.info(f"📊 Export Summary:")
            logger.info(f"  • Total Trades: {total_trades}")
            logger.info(f"  • CSV Rows: {total_rows}")
            logger.info(f"  • Profitable Trades: {profitable_trades}")
            logger.info(f"  • Win Rate: {win_rate:.1f}%")
            logger.info(f"  • Total P&L: ${total_pnl:.2f}")
            logger.info(f"  • Format: {'Enhanced' if enhanced else 'Basic'}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return None

    def generate_summary_report(self, trades: List[Dict[str, Any]]) -> str:
        """Generate a summary report of the backtest results"""
        try:
            if not trades:
                return "No trades to summarize"
            
            # Calculate summary statistics
            total_trades = len(trades)
            profitable_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            unprofitable_trades = total_trades - profitable_trades
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(trade['pnl'] for trade in trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            profitable_pnl = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            unprofitable_pnl = sum(trade['pnl'] for trade in trades if trade['pnl'] <= 0)
            
            avg_profitable_pnl = profitable_pnl / profitable_trades if profitable_trades > 0 else 0
            avg_unprofitable_pnl = unprofitable_pnl / unprofitable_trades if unprofitable_trades > 0 else 0
            
            # Best and worst trades
            best_trade = max(trades, key=lambda x: x['pnl'])
            worst_trade = min(trades, key=lambda x: x['pnl'])
            
            # Average trade duration
            avg_duration = sum(trade['trade_duration_seconds'] for trade in trades) / total_trades if total_trades > 0 else 0
            
            # Date range
            entry_times = [trade['entry_time'] for trade in trades]
            date_range_start = min(entry_times).strftime('%Y-%m-%d')
            date_range_end = max(entry_times).strftime('%Y-%m-%d')
            
            # Sentiment breakdown
            sentiment_counts = {}
            for trade in trades:
                sentiment = trade['sentiment']
                if sentiment not in sentiment_counts:
                    sentiment_counts[sentiment] = 0
                sentiment_counts[sentiment] += 1
            
            # Generate report
            report = f"""
NEWSHEAD BACKTESTING SUMMARY REPORT
=====================================

OVERVIEW
--------
Total Trades: {total_trades:,}
Date Range: {date_range_start} to {date_range_end}
Average Trade Duration: {avg_duration:.0f} seconds ({avg_duration/60:.1f} minutes)

PERFORMANCE METRICS
------------------
Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trades})
Total P&L: ${total_pnl:.2f}
Average P&L per Trade: ${avg_pnl:.2f}

Profitable Trades: {profitable_trades} (${profitable_pnl:.2f} total, ${avg_profitable_pnl:.2f} avg)
Unprofitable Trades: {unprofitable_trades} (${unprofitable_pnl:.2f} total, ${avg_unprofitable_pnl:.2f} avg)

BEST/WORST TRADES
----------------
Best Trade: {best_trade['ticker']} - ${best_trade['pnl']:.2f} ({best_trade['pnl_percent']:.2f}%)
Worst Trade: {worst_trade['ticker']} - ${worst_trade['pnl']:.2f} ({worst_trade['pnl_percent']:.2f}%)

SENTIMENT BREAKDOWN
------------------"""
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total_trades * 100) if total_trades > 0 else 0
                report += f"\n{sentiment.upper()}: {count} trades ({percentage:.1f}%)"
            
            report += f"""

IMPORT INSTRUCTIONS
------------------
1. Log into your Tradervue account
2. Go to Import > Generic CSV
3. Upload the generated CSV file
4. Map the columns if needed (should auto-detect)
5. Verify the data and complete the import

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return f"Error generating report: {str(e)}"

    def run_export(self, filename: str = None, enhanced: bool = True, limit: int = None, generate_summary: bool = True):
        """Run the complete export process"""
        try:
            logger.info("📤 Starting Tradervue CSV Export...")
            
            # Initialize
            if not self.initialize():
                logger.error("Failed to initialize CSV exporter")
                return False
            
            # Export CSV
            filepath = self.export_to_csv(filename=filename, enhanced=enhanced, limit=limit)
            
            if not filepath:
                logger.error("CSV export failed")
                return False
            
            # Generate summary report if requested
            if generate_summary:
                logger.info("📋 Generating summary report...")
                trades = self.get_trade_data(limit=limit)
                summary = self.generate_summary_report(trades)
                
                # Save summary report
                summary_filename = filepath.replace('.csv', '_summary.txt')
                with open(summary_filename, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                logger.info(f"📋 Summary report saved: {summary_filename}")
                
                # Print summary to console
                print(summary)
            
            logger.info("✅ Export process completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in export process: {e}")
            return False
        finally:
            if self.ch_manager:
                self.ch_manager.close()

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export backtest results to Tradervue CSV format')
    parser.add_argument('--filename', '-f', help='Output filename (default: auto-generated)')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of trades to export')
    parser.add_argument('--basic', action='store_true', help='Use basic CSV format instead of enhanced')
    parser.add_argument('--no-summary', action='store_true', help='Skip generating summary report')
    
    args = parser.parse_args()
    
    # Create exporter and run
    exporter = TradervueCSVExporter()
    
    success = exporter.run_export(
        filename=args.filename,
        enhanced=not args.basic,
        limit=args.limit,
        generate_summary=not args.no_summary
    )
    
    if success:
        print("\n✅ Tradervue CSV export completed successfully!")
        print("📁 Files saved in the 'exports' directory")
        print("🚀 Ready to import into Tradervue!")
    else:
        print("\n❌ Tradervue CSV export failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 