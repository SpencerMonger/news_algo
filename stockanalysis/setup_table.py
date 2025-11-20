#!/usr/bin/env python3
"""
Standalone script to drop and recreate the float_list_detailed table
Uses credentials from .env file in the project root
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables from root .env file
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(root_dir, '.env')
load_dotenv(env_path)

def main():
    """Drop and recreate the float_list_detailed table"""
    
    print("=" * 80)
    print("StockAnalysis.com Table Setup")
    print("=" * 80)
    print()
    
    # Connect to ClickHouse
    print("🔌 Connecting to ClickHouse...")
    ch_manager = ClickHouseManager()
    
    try:
        ch_manager.connect()
        print("✅ Connected successfully")
        print()
        
        # Drop the old table if it exists
        print("🗑️  Dropping old float_list_detailed table (if exists)...")
        try:
            ch_manager.client.command("DROP TABLE IF EXISTS News.float_list_detailed")
            print("✅ Old table dropped successfully")
        except Exception as e:
            print(f"⚠️  Note: {e}")
        print()
        
        # Create the new comprehensive table
        print("🔧 Creating new comprehensive float_list_detailed table...")
        ch_manager.create_float_list_detailed_table()
        print("✅ New table created successfully with 115+ fields")
        print()
        
        # Verify the table structure
        print("📋 Verifying table structure...")
        try:
            result = ch_manager.client.query("DESCRIBE TABLE News.float_list_detailed")
            column_count = len(result.result_rows)
            print(f"✅ Table has {column_count} columns")
            print()
            
            # Show first few columns as confirmation
            print("First 10 columns:")
            for i, row in enumerate(result.result_rows[:10]):
                print(f"  {i+1}. {row[0]:<30} {row[1]}")
            print(f"  ... and {column_count - 10} more columns")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not verify table structure: {e}")
            print()
        
        print("=" * 80)
        print("✅ SETUP COMPLETE!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Test with one ticker:")
        print("     python3 stockanalysis/stockanalysis_scraper.py --limit 1")
        print()
        print("  2. Run full scrape:")
        print("     python3 stockanalysis/stockanalysis_scraper.py")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
        
    finally:
        ch_manager.close()
        print("🔌 Connection closed")
        print()


if __name__ == "__main__":
    main()

