#!/usr/bin/env python3
"""
NewsHead Backtesting System - Main Orchestration Script
Runs the complete backtesting pipeline:
1. Create ClickHouse tables
2. Scrape historical news from Finviz
3. Analyze sentiment with Claude API
4. Simulate trades with Polygon API
5. Export results to Tradervue CSV format
"""

import asyncio
import logging
import argparse
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all backtesting modules
from create_tables import create_backtesting_tables
from finviz_pages import FinvizHistoricalScraper
from sentiment_historical import HistoricalSentimentAnalyzer
from trade_simulation import TradeSimulator
from export_csv import TradervueCSVExporter

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

class BacktestOrchestrator:
    """
    Main orchestrator for the complete backtesting system
    Manages the sequential execution of all backtesting components
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'total_runtime': 0,
            'steps_completed': 0,
            'steps_total': 5,
            'tables_created': False,
            'news_scraped': False,
            'sentiment_analyzed': False,
            'trades_simulated': False,
            'csv_exported': False
        }
        
        # Component instances
        self.finviz_scraper = None
        self.sentiment_analyzer = None
        self.trade_simulator = None
        self.csv_exporter = None

    def log_step_start(self, step_num: int, step_name: str, description: str):
        """Log the start of a backtesting step"""
        logger.info("=" * 80)
        logger.info(f"🚀 STEP {step_num}/{self.stats['steps_total']}: {step_name}")
        logger.info(f"📝 {description}")
        logger.info("=" * 80)

    def log_step_complete(self, step_num: int, step_name: str, success: bool, duration: float):
        """Log the completion of a backtesting step"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.stats['steps_completed'] += 1 if success else 0
        
        logger.info("-" * 80)
        logger.info(f"{status}: STEP {step_num} - {step_name} completed in {duration:.1f}s")
        logger.info("-" * 80)

    async def step_1_create_tables(self) -> bool:
        """STEP 1: Create ClickHouse tables for backtesting"""
        step_start = time.time()
        self.log_step_start(1, "CREATE TABLES", "Setting up ClickHouse database tables for backtesting")
        
        try:
            success = create_backtesting_tables()
            self.stats['tables_created'] = success
            
            step_duration = time.time() - step_start
            self.log_step_complete(1, "CREATE TABLES", success, step_duration)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in Step 1 (Create Tables): {e}")
            step_duration = time.time() - step_start
            self.log_step_complete(1, "CREATE TABLES", False, step_duration)
            return False

    async def step_2_scrape_news(self, ticker_limit: int = None) -> bool:
        """STEP 2: Scrape historical news from Finviz"""
        step_start = time.time()
        limit_desc = f" (limited to {ticker_limit} tickers)" if ticker_limit else ""
        self.log_step_start(2, "SCRAPE NEWS", f"Scraping 6 months of newswire articles from Finviz (5am-9am EST only){limit_desc}")
        
        try:
            self.finviz_scraper = FinvizHistoricalScraper()
            success = await self.finviz_scraper.run_historical_scrape(ticker_limit=ticker_limit)
            self.stats['news_scraped'] = success
            
            step_duration = time.time() - step_start
            self.log_step_complete(2, "SCRAPE NEWS", success, step_duration)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in Step 2 (Scrape News): {e}")
            step_duration = time.time() - step_start
            self.log_step_complete(2, "SCRAPE NEWS", False, step_duration)
            return False

    async def step_3_analyze_sentiment(self) -> bool:
        """STEP 3: Analyze sentiment of scraped articles"""
        step_start = time.time()
        self.log_step_start(3, "ANALYZE SENTIMENT", "Running Claude API sentiment analysis on scraped articles")
        
        try:
            self.sentiment_analyzer = HistoricalSentimentAnalyzer()
            success = await self.sentiment_analyzer.run_historical_sentiment_analysis()
            self.stats['sentiment_analyzed'] = success
            
            step_duration = time.time() - step_start
            self.log_step_complete(3, "ANALYZE SENTIMENT", success, step_duration)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in Step 3 (Analyze Sentiment): {e}")
            step_duration = time.time() - step_start
            self.log_step_complete(3, "ANALYZE SENTIMENT", False, step_duration)
            return False

    async def step_4_simulate_trades(self) -> bool:
        """STEP 4: Simulate trades based on sentiment analysis"""
        step_start = time.time()
        self.log_step_start(4, "SIMULATE TRADES", "Simulating trades using Polygon API (BUY on ask +30s, SELL on bid at 9:28am)")
        
        try:
            self.trade_simulator = TradeSimulator()
            success = await self.trade_simulator.run_trade_simulation()
            self.stats['trades_simulated'] = success
            
            step_duration = time.time() - step_start
            self.log_step_complete(4, "SIMULATE TRADES", success, step_duration)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in Step 4 (Simulate Trades): {e}")
            step_duration = time.time() - step_start
            self.log_step_complete(4, "SIMULATE TRADES", False, step_duration)
            return False

    async def step_5_export_csv(self, filename: str = None) -> bool:
        """STEP 5: Export results to Tradervue CSV format"""
        step_start = time.time()
        self.log_step_start(5, "EXPORT CSV", "Exporting backtest results to Tradervue CSV format")
        
        try:
            self.csv_exporter = TradervueCSVExporter()
            success = self.csv_exporter.run_export(filename=filename)
            self.stats['csv_exported'] = success
            
            step_duration = time.time() - step_start
            self.log_step_complete(5, "EXPORT CSV", success, step_duration)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in Step 5 (Export CSV): {e}")
            step_duration = time.time() - step_start
            self.log_step_complete(5, "EXPORT CSV", False, step_duration)
            return False

    def generate_final_report(self) -> str:
        """Generate a comprehensive final report of the backtesting run"""
        total_runtime = time.time() - self.start_time
        self.stats['total_runtime'] = total_runtime
        
        # Determine overall success
        all_steps_completed = all([
            self.stats['tables_created'],
            self.stats['news_scraped'],
            self.stats['sentiment_analyzed'],
            self.stats['trades_simulated'],
            self.stats['csv_exported']
        ])
        
        overall_status = "✅ SUCCESS" if all_steps_completed else "⚠️ PARTIAL" if self.stats['steps_completed'] > 0 else "❌ FAILED"
        
        report = f"""
{'=' * 80}
🎯 NEWSHEAD BACKTESTING SYSTEM - FINAL REPORT
{'=' * 80}

OVERALL STATUS: {overall_status}
Total Runtime: {total_runtime/60:.1f} minutes ({total_runtime:.1f} seconds)
Steps Completed: {self.stats['steps_completed']}/{self.stats['steps_total']}

STEP-BY-STEP RESULTS:
{'=' * 40}
STEP 1 - Create Tables:     {'✅ SUCCESS' if self.stats['tables_created'] else '❌ FAILED'}
STEP 2 - Scrape News:       {'✅ SUCCESS' if self.stats['news_scraped'] else '❌ FAILED'}
STEP 3 - Analyze Sentiment: {'✅ SUCCESS' if self.stats['sentiment_analyzed'] else '❌ FAILED'}
STEP 4 - Simulate Trades:   {'✅ SUCCESS' if self.stats['trades_simulated'] else '❌ FAILED'}
STEP 5 - Export CSV:        {'✅ SUCCESS' if self.stats['csv_exported'] else '❌ FAILED'}

NEXT STEPS:
{'=' * 40}"""
        
        if all_steps_completed:
            report += """
🎉 Backtesting completed successfully!

1. Check the 'exports' directory for your Tradervue CSV file
2. Import the CSV file into Tradervue for detailed analysis
3. Review the summary report for key performance metrics
4. Consider adjusting parameters for additional backtests

🚀 Ready to analyze your trading strategy performance!
"""
        else:
            report += """
⚠️ Backtesting completed with issues.

1. Check the log file for detailed error messages
2. Ensure all required API keys are configured:
   - ANTHROPIC_API_KEY (for sentiment analysis)
   - POLYGON_API_KEY (for trade simulation)
3. Verify ClickHouse database connectivity
4. Re-run individual steps as needed

Steps that completed successfully can be skipped on re-run.
"""
        
        report += f"""
LOG FILE: {log_filename}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
        
        return report

    async def run_complete_backtest(self, start_step: int = 1, end_step: int = 5, csv_filename: str = None, ticker_limit: int = None) -> bool:
        """Run the complete backtesting pipeline"""
        try:
            logger.info("🎯 STARTING NEWSHEAD BACKTESTING SYSTEM")
            logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🔧 Running steps {start_step} to {end_step}")
            
            success = True
            
            # Step 1: Create tables
            if start_step <= 1 <= end_step:
                if not await self.step_1_create_tables():
                    logger.error("❌ Step 1 failed")
                    success = False
            
            # Step 2: Scrape news
            if start_step <= 2 <= end_step:
                if not await self.step_2_scrape_news(ticker_limit=ticker_limit):
                    logger.error("❌ Step 2 failed")
                    success = False
            
            # Step 3: Analyze sentiment
            if start_step <= 3 <= end_step:
                if not await self.step_3_analyze_sentiment():
                    logger.error("❌ Step 3 failed")
                    success = False
            
            # Step 4: Simulate trades
            if start_step <= 4 <= end_step:
                if not await self.step_4_simulate_trades():
                    logger.error("❌ Step 4 failed")
                    success = False
            
            # Step 5: Export CSV
            if start_step <= 5 <= end_step:
                if not await self.step_5_export_csv(csv_filename):
                    logger.error("❌ Step 5 failed")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Fatal error in backtesting pipeline: {e}")
            return False
        finally:
            # Always generate final report
            final_report = self.generate_final_report()
            logger.info(final_report)
            print(final_report)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='NewsHead Backtesting System - Complete Pipeline')
    
    parser.add_argument('--start-step', type=int, default=1, choices=range(1, 6),
                       help='Starting step (1=tables, 2=scrape, 3=sentiment, 4=trades, 5=export)')
    parser.add_argument('--end-step', type=int, default=5, choices=range(1, 6),
                       help='Ending step (1=tables, 2=scrape, 3=sentiment, 4=trades, 5=export)')
    parser.add_argument('--csv-filename', help='Custom filename for CSV export')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without running')
    parser.add_argument('--limit', type=int, help='Limit number of ticker pages to scrape (e.g., --limit 30)')
    
    args = parser.parse_args()
    
    # Validate step range
    if args.start_step > args.end_step:
        print("❌ Error: start-step cannot be greater than end-step")
        sys.exit(1)
    
    # Validate limit
    if args.limit is not None and args.limit <= 0:
        print("❌ Error: --limit must be a positive integer")
        sys.exit(1)
    
    # Show execution plan
    step_names = {
        1: "Create ClickHouse Tables",
        2: "Scrape Historical News from Finviz",
        3: "Analyze Sentiment with Claude API", 
        4: "Simulate Trades with Polygon API",
        5: "Export to Tradervue CSV"
    }
    
    print("\n🎯 NEWSHEAD BACKTESTING EXECUTION PLAN")
    print("=" * 50)
    
    for step in range(args.start_step, args.end_step + 1):
        print(f"STEP {step}: {step_names[step]}")
    
    print(f"\nCSV Filename: {args.csv_filename or 'Auto-generated'}")
    print(f"Ticker Limit: {args.limit or 'No limit (all tickers)'}")
    print(f"Dry Run: {'Yes' if args.dry_run else 'No'}")
    
    if args.dry_run:
        print("\n✅ Dry run completed - no actual execution performed")
        return
    
    # Confirm execution
    print("\n⚠️  This will run the complete backtesting pipeline.")
    print("Make sure you have the required API keys configured:")
    print("  • ANTHROPIC_API_KEY (for sentiment analysis)")
    print("  • POLYGON_API_KEY (for trade simulation)")
    
    confirm = input("\nProceed with backtesting? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Backtesting cancelled by user")
        return
    
    # Run backtesting
    orchestrator = BacktestOrchestrator()
    
    try:
        success = asyncio.run(orchestrator.run_complete_backtest(
            start_step=args.start_step,
            end_step=args.end_step,
            csv_filename=args.csv_filename,
            ticker_limit=args.limit
        ))
        
        if success:
            print("\n🎉 Backtesting pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Backtesting pipeline completed with errors!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Backtesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 