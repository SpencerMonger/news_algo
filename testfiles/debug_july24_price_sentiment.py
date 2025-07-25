#!/usr/bin/env python3
"""
Debug Script: July 24th Price Movement vs Sentiment Analysis
Query price_tracking table to find tickers that:
1. Had 5%+ price movement within 40 seconds
2. But failed sentiment requirement (not BUY with high confidence)
"""

import logging
from datetime import datetime
from clickhouse_setup import ClickHouseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class July24PriceDebugger:
    def __init__(self):
        self.ch_manager = ClickHouseManager()
        self.ch_manager.connect()
    
    def find_price_movement_tickers(self):
        """Find all tickers with 5%+ price movement within 40 seconds on July 24th"""
        try:
            # Query for all tickers with significant price movements within 40 seconds
            # FIXED: Exclude quote-only data, require actual trades
            query = """
            SELECT 
                ticker,
                argMax(price, timestamp) as current_price,
                argMin(price, timestamp) as first_price,
                ((argMax(price, timestamp) - argMin(price, timestamp)) / argMin(price, timestamp)) * 100 as change_pct,
                count() as price_count,
                argMin(timestamp, timestamp) as first_timestamp,
                argMax(timestamp, timestamp) as current_timestamp,
                dateDiff('second', argMin(timestamp, timestamp), argMax(timestamp, timestamp)) as seconds_elapsed,
                -- Get sentiment data
                argMax(sentiment, timestamp) as latest_sentiment,
                argMax(recommendation, timestamp) as latest_recommendation,
                argMax(confidence, timestamp) as latest_confidence,
                -- Show source information for debugging
                groupArray(DISTINCT source) as sources,
                sum(volume) as total_volume
            FROM News.price_tracking
            WHERE toDate(timestamp) = '2025-07-24'
            AND ticker != ''
            -- CRITICAL FIX: Only include actual trades, not quotes
            AND (source LIKE '%trade%' OR source = 'polygon' OR source = 'rest_fallback' OR volume > 0)
            GROUP BY ticker
            HAVING change_pct >= 5.0 
            AND price_count >= 2
            AND seconds_elapsed <= 40  -- CRITICAL: Only movements within 40 seconds
            ORDER BY change_pct DESC
            """
            
            logger.info("🔍 Querying price_tracking for July 24th with 40-second filter (TRADES ONLY)...")
            result = self.ch_manager.client.query(query)
            
            print(f"\n{'='*100}")
            print(f"JULY 24TH PRICE MOVEMENT ANALYSIS - TRADES ONLY (NO QUOTES)")
            print(f"{'='*100}")
            
            if not result.result_rows:
                print("❌ NO TICKERS found with 5%+ price movement within 40 seconds (trades only)")
                print("   Let's check what data sources we have...")
                
                # Debug: Check what sources exist
                debug_query = """
                SELECT 
                    source,
                    count() as count,
                    count(DISTINCT ticker) as unique_tickers
                FROM News.price_tracking
                WHERE toDate(timestamp) = '2025-07-24'
                GROUP BY source
                ORDER BY count DESC
                """
                
                debug_result = self.ch_manager.client.query(debug_query)
                print(f"\n📊 DATA SOURCES ON JULY 24TH:")
                for row in debug_result.result_rows:
                    source, count, unique_tickers = row
                    print(f"   • {source}: {count:,} records, {unique_tickers} tickers")
                
                return []
            
            valid_tickers = []
            sentiment_blocked_tickers = []
            
            print(f"✅ Found {len(result.result_rows)} tickers with 5%+ movement within 40 seconds (TRADES ONLY):")
            print(f"{'='*100}")
            
            for i, row in enumerate(result.result_rows, 1):
                ticker, current_price, first_price, change_pct, price_count, first_timestamp, current_timestamp, seconds_elapsed, sentiment, recommendation, confidence, sources, total_volume = row
                
                print(f"\n{i}. {ticker}")
                print(f"   💰 Price: ${first_price:.4f} → ${current_price:.4f} (+{change_pct:.2f}%)")
                print(f"   ⏱️  Time: {seconds_elapsed}s ({price_count} data points)")
                print(f"   📅 Period: {first_timestamp} → {current_timestamp}")
                print(f"   📊 Sources: {sources}, Total Volume: {total_volume}")
                print(f"   🧠 Sentiment: {sentiment}, {recommendation} ({confidence} confidence)")
                
                # Check if this would have been blocked by sentiment filter
                is_blocked = not (recommendation == 'BUY' and confidence == 'high')
                
                if is_blocked:
                    print(f"   ❌ BLOCKED BY SENTIMENT: Not 'BUY' with 'high' confidence")
                    sentiment_blocked_tickers.append({
                        'ticker': ticker,
                        'change_pct': change_pct,
                        'seconds': seconds_elapsed,
                        'sentiment': sentiment,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'sources': sources,
                        'volume': total_volume
                    })
                else:
                    print(f"   ✅ WOULD TRIGGER ALERT: BUY with high confidence")
                    valid_tickers.append(ticker)
            
            # Summary
            print(f"\n{'='*100}")
            print(f"SUMMARY - JULY 24TH PRICE vs SENTIMENT ANALYSIS (TRADES ONLY)")
            print(f"{'='*100}")
            print(f"Total tickers with 5%+ movement in ≤40s: {len(result.result_rows)}")
            print(f"Would trigger alerts (BUY + high confidence): {len(valid_tickers)}")
            print(f"Blocked by sentiment filter: {len(sentiment_blocked_tickers)}")
            
            if valid_tickers:
                print(f"\n✅ TICKERS THAT WOULD TRIGGER ALERTS:")
                for ticker in valid_tickers:
                    print(f"   • {ticker}")
            
            if sentiment_blocked_tickers:
                print(f"\n❌ TICKERS BLOCKED BY SENTIMENT FILTER:")
                for ticker_info in sentiment_blocked_tickers:
                    print(f"   • {ticker_info['ticker']}: +{ticker_info['change_pct']:.2f}% in {ticker_info['seconds']}s")
                    print(f"     Sentiment: {ticker_info['sentiment']}, {ticker_info['recommendation']} ({ticker_info['confidence']})")
                    print(f"     Sources: {ticker_info['sources']}, Volume: {ticker_info['volume']}")
            
            return sentiment_blocked_tickers
            
        except Exception as e:
            logger.error(f"Error querying price movements: {e}")
            return []

    def find_all_significant_movements(self):
        """Find ALL significant price movements regardless of time window to see what we're missing"""
        try:
            query = """
            SELECT 
                ticker,
                argMax(price, timestamp) as current_price,
                argMin(price, timestamp) as first_price,
                ((argMax(price, timestamp) - argMin(price, timestamp)) / argMin(price, timestamp)) * 100 as change_pct,
                count() as price_count,
                argMin(timestamp, timestamp) as first_timestamp,
                argMax(timestamp, timestamp) as current_timestamp,
                dateDiff('second', argMin(timestamp, timestamp), argMax(timestamp, timestamp)) as seconds_elapsed,
                argMax(sentiment, timestamp) as latest_sentiment,
                argMax(recommendation, timestamp) as latest_recommendation,
                argMax(confidence, timestamp) as latest_confidence,
                groupArray(DISTINCT source) as sources,
                sum(volume) as total_volume
            FROM News.price_tracking
            WHERE toDate(timestamp) = '2025-07-24'
            AND ticker != ''
            -- Only include actual trades, not quotes
            AND (source LIKE '%trade%' OR source = 'polygon' OR source = 'rest_fallback' OR volume > 0)
            GROUP BY ticker
            HAVING change_pct >= 5.0 
            AND price_count >= 2
            -- NO TIME RESTRICTION - see all movements
            ORDER BY change_pct DESC
            LIMIT 20
            """
            
            result = self.ch_manager.client.query(query)
            
            print(f"\n{'='*100}")
            print(f"ALL SIGNIFICANT PRICE MOVEMENTS ON JULY 24TH (NO TIME LIMIT)")
            print(f"{'='*100}")
            
            if result.result_rows:
                print(f"Found {len(result.result_rows)} tickers with 5%+ movements (any timeframe):")
                
                within_40s = 0
                for i, row in enumerate(result.result_rows, 1):
                    ticker, current_price, first_price, change_pct, price_count, first_timestamp, current_timestamp, seconds_elapsed, sentiment, recommendation, confidence, sources, total_volume = row
                    
                    time_status = "✅ ≤40s" if seconds_elapsed <= 40 else f"❌ {seconds_elapsed}s"
                    if seconds_elapsed <= 40:
                        within_40s += 1
                    
                    print(f"{i:2d}. {ticker}: +{change_pct:.2f}% in {seconds_elapsed}s {time_status}")
                    print(f"     Price: ${first_price:.4f} → ${current_price:.4f}")
                    print(f"     Sentiment: {sentiment}, {recommendation} ({confidence})")
                    print(f"     Sources: {sources}, Volume: {total_volume}")
                
                print(f"\n📊 SUMMARY:")
                print(f"   Total 5%+ movements: {len(result.result_rows)}")
                print(f"   Within 40 seconds: {within_40s}")
                print(f"   Beyond 40 seconds: {len(result.result_rows) - within_40s}")
            else:
                print("❌ No significant price movements found at all!")
                
        except Exception as e:
            logger.error(f"Error finding all movements: {e}")
    
    def analyze_sentiment_distribution(self):
        """Analyze the distribution of sentiment data on July 24th"""
        try:
            query = """
            SELECT 
                recommendation,
                confidence,
                count() as count,
                count(DISTINCT ticker) as unique_tickers
            FROM News.price_tracking
            WHERE toDate(timestamp) = '2025-07-24'
            AND recommendation != ''
            GROUP BY recommendation, confidence
            ORDER BY count DESC
            """
            
            result = self.ch_manager.client.query(query)
            
            print(f"\n{'='*100}")
            print(f"SENTIMENT DISTRIBUTION ON JULY 24TH")
            print(f"{'='*100}")
            
            if result.result_rows:
                for row in result.result_rows:
                    recommendation, confidence, count, unique_tickers = row
                    print(f"{recommendation} ({confidence}): {count} records, {unique_tickers} unique tickers")
            else:
                print("❌ No sentiment data found in price_tracking for July 24th")
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment distribution: {e}")
    
    def check_data_availability(self):
        """Check what data is available for July 24th"""
        try:
            # Check price_tracking data
            price_query = """
            SELECT 
                count() as total_records,
                count(DISTINCT ticker) as unique_tickers,
                min(timestamp) as earliest,
                max(timestamp) as latest
            FROM News.price_tracking
            WHERE toDate(timestamp) = '2025-07-24'
            """
            
            price_result = self.ch_manager.client.query(price_query)
            
            print(f"\n{'='*100}")
            print(f"DATA AVAILABILITY CHECK - JULY 24TH")
            print(f"{'='*100}")
            
            if price_result.result_rows:
                total_records, unique_tickers, earliest, latest = price_result.result_rows[0]
                print(f"📊 Price Tracking Data:")
                print(f"   Total records: {total_records:,}")
                print(f"   Unique tickers: {unique_tickers}")
                print(f"   Time range: {earliest} → {latest}")
            else:
                print("❌ No price_tracking data found for July 24th")
                return
            
            # Check sentiment data availability
            sentiment_query = """
            SELECT 
                count() as records_with_sentiment,
                count(DISTINCT ticker) as tickers_with_sentiment
            FROM News.price_tracking
            WHERE toDate(timestamp) = '2025-07-24'
            AND recommendation != ''
            AND sentiment != ''
            """
            
            sentiment_result = self.ch_manager.client.query(sentiment_query)
            
            if sentiment_result.result_rows:
                records_with_sentiment, tickers_with_sentiment = sentiment_result.result_rows[0]
                print(f"🧠 Sentiment Data:")
                print(f"   Records with sentiment: {records_with_sentiment:,}")
                print(f"   Tickers with sentiment: {tickers_with_sentiment}")
                
                if records_with_sentiment == 0:
                    print("⚠️  WARNING: No sentiment data found in price_tracking!")
                    print("   This explains why no alerts were triggered.")
                    
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
    
    def close(self):
        """Close database connection"""
        if self.ch_manager:
            self.ch_manager.close()

def main():
    """Main execution"""
    print("🔍 Starting July 24th Price Movement vs Sentiment Analysis")
    print("=" * 60)
    
    debugger = July24PriceDebugger()
    
    try:
        # Check what data is available
        debugger.check_data_availability()
        
        # Analyze sentiment distribution
        debugger.analyze_sentiment_distribution()
        
        # NEW: Find all significant movements first (no time limit)
        debugger.find_all_significant_movements()
        
        # Find actual price movements with sentiment analysis (40s limit)
        blocked_tickers = debugger.find_price_movement_tickers()
        
        print(f"\n{'='*100}")
        print(f"CONCLUSION")
        print(f"{'='*100}")
        
        if blocked_tickers:
            print(f"✅ Found {len(blocked_tickers)} tickers that ACTUALLY passed price movement")
            print(f"   requirements (5%+ in ≤40s) but were blocked by sentiment filter.")
            print(f"\n📋 CORRECTED LIST:")
            for ticker_info in blocked_tickers:
                print(f"   • {ticker_info['ticker']}: +{ticker_info['change_pct']:.2f}% in {ticker_info['seconds']}s")
                print(f"     Volume: {ticker_info['volume']}, Sources: {ticker_info['sources']}")
        else:
            print("❌ NO tickers found that passed price movement requirements within 40 seconds")
            print("   but were blocked by sentiment. This suggests:")
            print("   1. The 40-second filter is working correctly")
            print("   2. Previous debug logs were showing incorrect data")
            print("   3. Most price movements took longer than 40 seconds")
            print("   4. Most movements were quote-only, not actual trades")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        debugger.close()

if __name__ == "__main__":
    main() 