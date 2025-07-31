#!/usr/bin/env python3
"""
Test script for Price Movement Analyzer
Allows testing with different parameters and batch sizes
"""

import asyncio
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from price_movement_analyzer import PriceMovementAnalyzer

async def main():
    parser = argparse.ArgumentParser(description='Test Price Movement Analyzer')
    parser.add_argument('--batch-size', type=int, default=50, 
                       help='Number of articles to process in each batch (default: 50)')
    parser.add_argument('--limit', type=int, 
                       help='Limit total number of articles to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without updating the database (show what would be processed)')
    
    args = parser.parse_args()
    
    print(f"🧪 Testing Price Movement Analyzer")
    print(f"   • Batch size: {args.batch_size}")
    if args.limit:
        print(f"   • Article limit: {args.limit}")
    if args.dry_run:
        print(f"   • Dry run: No database updates")
    print()
    
    analyzer = PriceMovementAnalyzer()
    
    if args.dry_run:
        # For dry run, just show what would be processed
        print("🔍 DRY RUN: Showing articles that would be processed...")
        
        if not await analyzer.initialize():
            print("❌ Failed to initialize analyzer")
            return
        
        articles = await analyzer.get_articles_for_analysis(args.limit or 100)
        print(f"📊 Found {len(articles)} articles that would be processed:")
        
        for i, article in enumerate(articles[:10]):  # Show first 10
            print(f"  {i+1}. {article['ticker']} - {article['published_utc']} - {article['headline'][:60]}...")
        
        if len(articles) > 10:
            print(f"  ... and {len(articles) - 10} more articles")
        
        await analyzer.cleanup()
    else:
        # Run the actual analysis
        success = await analyzer.run_analysis(args.batch_size)
        
        if success:
            print("\n✅ Price movement analysis test completed successfully!")
        else:
            print("\n❌ Price movement analysis test failed!")

if __name__ == "__main__":
    asyncio.run(main()) 