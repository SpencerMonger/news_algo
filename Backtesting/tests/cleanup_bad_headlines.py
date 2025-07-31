#!/usr/bin/env python3
"""
Cleanup Script for Historical News Database
Removes non-headline entries from the News.historical_news table
"""

import sys
import os
import re
from typing import List, Set, Tuple
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalNewsCleanup:
    """Cleans up bad headlines from historical_news table"""
    
    def __init__(self):
        self.ch_manager = None
        
        # Patterns that indicate non-headlines (navigation/UI elements)
        self.bad_patterns = [
            # Navigation patterns
            r'^\s*open\s+in\s+\w+\s*$',  # "open in yahoo", "open in reuters", etc.
            r'^\s*open\s+in\s+[\w\s]+\s*$',  # "open in market watch", etc.
            
            # Company/institutional names (more specific patterns)
            # Match institutional names that are clearly not headlines
            r'^[A-Z\s]+\s+(LLC|INC|CORP|LTD|LP|LLP)$',  # "RENAISSANCE TECHNOLOGIES LLC"
            r'^[A-Z\s]+\s+(CAPITAL|MANAGEMENT|ADVISORS|SECURITIES|PARTNERS)\s+(LLC|INC|CORP|LTD|LP|LLP)$',  # "CITADEL ADVISORS LLC"
            r'^[A-Z\s]+\s+(CAPITAL|MANAGEMENT|ADVISORS|SECURITIES|PARTNERS)$',  # "MORGAN STANLEY", "CITADEL ADVISORS"
            r'^[A-Z\s]+\s+&\s+[A-Z\s]+(LLC|INC|CORP|LTD|LP|LLP)?/?[A-Z]*$',  # "WELLS FARGO & COMPANY/MN"
            
            # Additional institutional patterns that were missed
            r'^[A-Z\s]+\s+(INVESTMENTS|GROUP|BANK|FINANCIAL|HOLDINGS),?\s+(LLC|INC|CORP|LTD|LP|LLP|AG|SA|PLC)?$',  # "TWO SIGMA INVESTMENTS, LP", "UBS Group AG"
            r'^[A-Z\s]+\s+(AG|SA|PLC|NV|BV)$',  # European company suffixes like "UBS Group AG"
            
            # Timestamp patterns that got scraped as headlines
            r'^\s*[A-Za-z]{3}\s+\d{1,2}\s+\d{1,2}:\d{2}\s*[AP]M\s*$',  # "Dec 19 09:28 PM"
            r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M\s*$',  # "12/19/23 09:28 PM"
            r'^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*$',  # "2023-12-19 21:28:00"
            
            # Very short non-sentence patterns
            r'^\s*[A-Z]{2,10}\s*$',  # Single acronyms/tickers
            r'^\s*\d+\s*$',  # Just numbers
            r'^\s*[^\w\s]*\s*$',  # Just punctuation
            
            # Common UI/navigation elements
            r'^\s*(view|see|more|details?|info|link)\s*$',
            r'^\s*(chart|quote|profile|overview)\s*$',
            r'^\s*(buy|sell|trade)\s*$',
            
            # Empty or very short content
            r'^\s*[\w\s]{1,5}\s*$',  # 5 characters or less
        ]
        
        # Additional specific patterns to check
        self.bad_keywords = {
            'open in yahoo', 'open in reuters', 'open in google', 'open in marketwatch',
            'open in edgar', 'open in sec', 'view chart', 'see profile', 'trade now',
            'buy now', 'sell now', 'get quote', 'view quote', 'more info', 'details',
            'ubs group ag', 'ubs group', 'ubs ag'  # Add specific company names that keep appearing
        }
        
        # Statistics
        self.stats = {
            'total_rows': 0,
            'bad_rows_found': 0,
            'deleted_rows': 0,
            'patterns_matched': {}
        }

    def initialize(self):
        """Initialize database connection"""
        try:
            self.ch_manager = ClickHouseManager()
            self.ch_manager.connect()
            logger.info("✅ Connected to ClickHouse database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False

    def is_bad_headline(self, headline: str) -> Tuple[bool, str]:
        """
        Check if a headline is actually a non-headline (navigation/UI element)
        Returns (is_bad, reason)
        """
        if not headline or not headline.strip():
            return True, "empty_headline"
        
        headline_clean = headline.strip()
        headline_lower = headline_clean.lower()
        
        # Check against bad keywords first (exact matches)
        if headline_lower in self.bad_keywords:
            return True, f"bad_keyword: {headline_lower}"
        
        # Check regex patterns
        for i, pattern in enumerate(self.bad_patterns):
            if re.match(pattern, headline_clean, re.IGNORECASE):
                return True, f"pattern_{i}: {pattern}"
        
        # Additional heuristics for non-headlines
        
        # 1. Headlines should typically have more than 2 words for news articles
        words = headline_clean.split()
        if len(words) <= 2:
            return True, "too_few_words"
        
        # 2. More specific all-caps check - only flag if it looks like a company name
        if headline_clean.isupper() and len(words) > 1:
            # Check if it contains news-like words (if so, it's probably a real headline)
            news_indicators = ['announces', 'reports', 'completes', 'receives', 'launches', 'signs', 'acquires', 
                             'files', 'submits', 'enters', 'agreement', 'partnership', 'deal', 'merger',
                             'earnings', 'revenue', 'results', 'guidance', 'outlook', 'forecast',
                             'appoints', 'names', 'hires', 'promotes', 'executive', 'ceo', 'cfo',
                             'expands', 'opens', 'closes', 'divests', 'invests', 'funding']
            
            has_news_words = any(news_word in headline_lower for news_word in news_indicators)
            
            # Check if it looks like a company/institutional name
            institutional_words = ['llc', 'inc', 'corp', 'ltd', 'capital', 'management', 'advisors', 
                                 'securities', 'partners', 'holdings', 'group', 'fund', 'trust',
                                 'investments', 'bank', 'financial', 'ag', 'sa', 'plc', 'nv', 'bv']
            has_institutional_words = any(inst_word in headline_lower for inst_word in institutional_words)
            
            # Only flag as suspicious if it has institutional words but no news words
            if has_institutional_words and not has_news_words:
                return True, "institutional_name"
        
        # 3. Check for navigation-like patterns that might not be caught by regex
        if any(nav_word in headline_lower for nav_word in ['click here', 'read more', 'full story', 'continue reading']):
            return True, "navigation_text"
        
        # 4. Very repetitive characters (like dashes, dots)
        if len(set(headline_clean)) < 4 and len(headline_clean) > 10:
            return True, "repetitive_characters"
        
        return False, "valid_headline"

    def analyze_headlines(self, dry_run: bool = True) -> List[dict]:
        """Analyze all headlines and identify bad ones"""
        try:
            logger.info("🔍 Analyzing headlines in historical_news table...")
            
            # Get all headlines from the database
            query = """
            SELECT ticker, headline, article_url, published_utc, content_hash
            FROM News.historical_news 
            ORDER BY ticker, published_utc
            """
            
            result = self.ch_manager.client.query(query)
            all_rows = result.result_rows
            self.stats['total_rows'] = len(all_rows)
            
            logger.info(f"📊 Found {len(all_rows)} total rows in historical_news table")
            
            bad_rows = []
            pattern_counts = {}
            
            for row in all_rows:
                ticker, headline, article_url, published_utc, content_hash = row
                
                is_bad, reason = self.is_bad_headline(headline)
                
                if is_bad:
                    bad_rows.append({
                        'ticker': ticker,
                        'headline': headline,
                        'article_url': article_url,
                        'published_utc': published_utc,
                        'content_hash': content_hash,
                        'reason': reason
                    })
                    
                    # Track pattern statistics
                    pattern_counts[reason] = pattern_counts.get(reason, 0) + 1
            
            self.stats['bad_rows_found'] = len(bad_rows)
            self.stats['patterns_matched'] = pattern_counts
            
            # Log findings
            logger.info(f"📊 ANALYSIS RESULTS:")
            logger.info(f"   Total rows: {self.stats['total_rows']}")
            logger.info(f"   Bad headlines found: {self.stats['bad_rows_found']}")
            logger.info(f"   Percentage bad: {(self.stats['bad_rows_found'] / self.stats['total_rows'] * 100):.1f}%")
            
            if pattern_counts:
                logger.info(f"📋 PATTERN BREAKDOWN:")
                for reason, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"   {reason}: {count} rows")
            
            # Show some examples
            if bad_rows and not dry_run:
                logger.info(f"🔍 EXAMPLES OF BAD HEADLINES (first 10):")
                for i, bad_row in enumerate(bad_rows[:10]):
                    logger.info(f"   {i+1}. '{bad_row['headline']}' [{bad_row['reason']}]")
            
            return bad_rows
            
        except Exception as e:
            logger.error(f"❌ Error analyzing headlines: {e}")
            return []

    def delete_bad_headlines(self, bad_rows: List[dict]) -> bool:
        """Delete the identified bad headlines from the database"""
        try:
            if not bad_rows:
                logger.info("✅ No bad headlines to delete")
                return True
            
            logger.info(f"🗑️ Deleting {len(bad_rows)} bad headlines from database...")
            
            # Build delete conditions using content_hash for precise matching
            delete_conditions = []
            for bad_row in bad_rows:
                # Use content_hash for precise matching since it's unique
                condition = f"content_hash = '{bad_row['content_hash']}'"
                delete_conditions.append(condition)
            
            # Execute deletes in batches to avoid query length limits
            batch_size = 100
            total_deleted = 0
            
            for i in range(0, len(delete_conditions), batch_size):
                batch_conditions = delete_conditions[i:i + batch_size]
                where_clause = " OR ".join(batch_conditions)
                
                delete_query = f"""
                DELETE FROM News.historical_news 
                WHERE {where_clause}
                """
                
                self.ch_manager.client.command(delete_query)
                batch_deleted = len(batch_conditions)
                total_deleted += batch_deleted
                
                logger.info(f"   Deleted batch {i//batch_size + 1}: {batch_deleted} rows")
            
            self.stats['deleted_rows'] = total_deleted
            logger.info(f"✅ Successfully deleted {total_deleted} bad headlines")
            
            # Verify deletion
            remaining_query = "SELECT COUNT(*) FROM News.historical_news"
            remaining_count = self.ch_manager.client.query(remaining_query).result_rows[0][0]
            
            logger.info(f"📊 CLEANUP SUMMARY:")
            logger.info(f"   Original rows: {self.stats['total_rows']}")
            logger.info(f"   Deleted rows: {self.stats['deleted_rows']}")
            logger.info(f"   Remaining rows: {remaining_count}")
            logger.info(f"   Expected remaining: {self.stats['total_rows'] - self.stats['deleted_rows']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting bad headlines: {e}")
            return False

    def run_cleanup(self, dry_run: bool = True):
        """Run the complete cleanup process"""
        try:
            logger.info("🚀 Starting Historical News Cleanup...")
            
            if not self.initialize():
                logger.error("❌ Failed to initialize database connection")
                return False
            
            # Analyze headlines
            bad_rows = self.analyze_headlines(dry_run)
            
            if not bad_rows:
                logger.info("✅ No bad headlines found - database is clean!")
                return True
            
            if dry_run:
                logger.info("🔍 DRY RUN MODE - No changes made to database")
                logger.info(f"📋 Would delete {len(bad_rows)} bad headlines")
                
                # Show examples in dry run
                logger.info(f"🔍 EXAMPLES OF BAD HEADLINES TO BE DELETED:")
                for i, bad_row in enumerate(bad_rows[:20]):  # Show more examples in dry run
                    logger.info(f"   {i+1}. '{bad_row['headline']}' - {bad_row['ticker']} [{bad_row['reason']}]")
                
                logger.info("💡 Run with --execute flag to perform actual cleanup")
                return True
            else:
                # Confirm before deletion
                logger.warning(f"⚠️  ABOUT TO DELETE {len(bad_rows)} ROWS FROM DATABASE")
                response = input("Are you sure you want to proceed? (yes/no): ")
                
                if response.lower() != 'yes':
                    logger.info("❌ Cleanup cancelled by user")
                    return False
                
                # Perform deletion
                success = self.delete_bad_headlines(bad_rows)
                
                if success:
                    logger.info("🎉 Cleanup completed successfully!")
                else:
                    logger.error("❌ Cleanup failed")
                
                return success
            
        except Exception as e:
            logger.error(f"❌ Error in cleanup process: {e}")
            return False
        finally:
            if self.ch_manager:
                self.ch_manager.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cleanup bad headlines from historical_news table')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually perform the cleanup (default is dry-run mode)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cleanup = HistoricalNewsCleanup()
    
    if args.execute:
        logger.info("🚨 EXECUTE MODE - Changes will be made to database")
        success = cleanup.run_cleanup(dry_run=False)
    else:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
        success = cleanup.run_cleanup(dry_run=True)
    
    if success:
        print("\n✅ Cleanup process completed successfully!")
    else:
        print("\n❌ Cleanup process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 