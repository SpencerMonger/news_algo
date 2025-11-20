#!/usr/bin/env python3
"""
Stock Analysis Pipeline Runner
Orchestrates the complete stock analysis workflow:
1. Scrapes financial data from StockAnalysis.com
2. Analyzes stocks with Claude AI to generate strength scores
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnalysisPipeline:
    """Orchestrates the stock analysis pipeline"""
    
    def __init__(self, script_dir: str):
        self.script_dir = script_dir
        self.scraper_script = os.path.join(script_dir, 'stockanalysis_scraper.py')
        self.analyzer_script = os.path.join(script_dir, 'stock_strength_analyzer.py')
        
        # Validate scripts exist
        if not os.path.exists(self.scraper_script):
            raise FileNotFoundError(f"Scraper script not found: {self.scraper_script}")
        if not os.path.exists(self.analyzer_script):
            raise FileNotFoundError(f"Analyzer script not found: {self.analyzer_script}")
        
        self.start_time = None
        self.scraper_duration = None
        self.analyzer_duration = None
    
    def run_scraper(self, limit: Optional[int] = None) -> bool:
        """
        Run the stock data scraper
        
        Args:
            limit: Optional limit on number of tickers to scrape
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Running Stock Data Scraper")
        logger.info("=" * 80)
        
        try:
            scraper_start = datetime.now()
            
            # Build command
            cmd = [sys.executable, self.scraper_script]
            if limit:
                cmd.extend(['--limit', str(limit)])
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run scraper script
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(self.scraper_script),
                capture_output=False,  # Let output go to console
                text=True
            )
            
            self.scraper_duration = (datetime.now() - scraper_start).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ Scraper completed successfully in {self.scraper_duration:.1f} seconds")
                return True
            else:
                logger.error(f"❌ Scraper failed with exit code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running scraper: {e}")
            return False
    
    def run_analyzer(self, limit: Optional[int] = None, reanalyze: bool = False) -> bool:
        """
        Run the stock strength analyzer
        
        Args:
            limit: Optional limit on number of stocks to analyze
            reanalyze: If True, reanalyze all stocks
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2: Running Stock Strength Analyzer")
        logger.info("=" * 80)
        
        try:
            analyzer_start = datetime.now()
            
            # Build command
            cmd = [sys.executable, self.analyzer_script]
            if limit:
                cmd.extend(['--limit', str(limit)])
            if reanalyze:
                cmd.append('--reanalyze')
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run analyzer script
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(self.analyzer_script),
                capture_output=False,  # Let output go to console
                text=True
            )
            
            self.analyzer_duration = (datetime.now() - analyzer_start).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ Analyzer completed successfully in {self.analyzer_duration:.1f} seconds")
                return True
            else:
                logger.error(f"❌ Analyzer failed with exit code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running analyzer: {e}")
            return False
    
    def run_pipeline(self, scraper_limit: Optional[int] = None, 
                    analyzer_limit: Optional[int] = None,
                    reanalyze: bool = False,
                    skip_scraper: bool = False) -> bool:
        """
        Run the complete analysis pipeline
        
        Args:
            scraper_limit: Optional limit on tickers to scrape
            analyzer_limit: Optional limit on stocks to analyze
            reanalyze: If True, reanalyze all stocks in analyzer
            skip_scraper: If True, skip scraping and only run analyzer
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        self.start_time = datetime.now()
        
        logger.info("")
        logger.info("🚀" * 40)
        logger.info("STOCK ANALYSIS PIPELINE STARTING")
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("🚀" * 40)
        logger.info("")
        
        # Step 1: Run scraper (unless skipped)
        if skip_scraper:
            logger.info("⏭️  Skipping scraper as requested")
            scraper_success = True
        else:
            scraper_success = self.run_scraper(limit=scraper_limit)
            
            if not scraper_success:
                logger.error("❌ Pipeline failed at scraper stage")
                self._print_summary(success=False)
                return False
        
        # Step 2: Run analyzer
        analyzer_success = self.run_analyzer(limit=analyzer_limit, reanalyze=reanalyze)
        
        if not analyzer_success:
            logger.error("❌ Pipeline failed at analyzer stage")
            self._print_summary(success=False)
            return False
        
        # Pipeline completed successfully
        self._print_summary(success=True)
        return True
    
    def _print_summary(self, success: bool):
        """Print pipeline summary"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")
        
        if self.scraper_duration:
            logger.info(f"  - Scraper Duration: {self.scraper_duration:.1f} seconds")
        if self.analyzer_duration:
            logger.info(f"  - Analyzer Duration: {self.analyzer_duration:.1f} seconds")
        
        logger.info("=" * 80)


def main():
    """Main function to run the stock analysis pipeline"""
    parser = argparse.ArgumentParser(
        description='Run the complete stock analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (scrape + analyze all new stocks)
  python run_analysis.py
  
  # Test with limited stocks
  python run_analysis.py --limit 5
  
  # Scrape 10 stocks, analyze all new stocks
  python run_analysis.py --scraper-limit 10
  
  # Skip scraping, only run analyzer
  python run_analysis.py --skip-scraper
  
  # Reanalyze all stocks (including those with existing scores)
  python run_analysis.py --skip-scraper --reanalyze
  
  # Full pipeline with separate limits
  python run_analysis.py --scraper-limit 20 --analyzer-limit 10
        """
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Apply same limit to both scraper and analyzer (shortcut)'
    )
    parser.add_argument(
        '--scraper-limit',
        type=int,
        help='Limit number of tickers to scrape'
    )
    parser.add_argument(
        '--analyzer-limit',
        type=int,
        help='Limit number of stocks to analyze'
    )
    parser.add_argument(
        '--reanalyze',
        action='store_true',
        help='Reanalyze all stocks, including those with existing scores'
    )
    parser.add_argument(
        '--skip-scraper',
        action='store_true',
        help='Skip scraping and only run analyzer'
    )
    
    args = parser.parse_args()
    
    # Handle --limit shortcut
    scraper_limit = args.scraper_limit or args.limit
    analyzer_limit = args.analyzer_limit or args.limit
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        pipeline = StockAnalysisPipeline(script_dir)
        
        success = pipeline.run_pipeline(
            scraper_limit=scraper_limit,
            analyzer_limit=analyzer_limit,
            reanalyze=args.reanalyze,
            skip_scraper=args.skip_scraper
        )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⏸️ Pipeline stopped by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"❌ Unexpected error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

