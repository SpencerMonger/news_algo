#!/usr/bin/env python3
"""
Compare sentiment analysis results between database and recent analysis
"""

from clickhouse_setup import ClickHouseManager
from datetime import datetime, timedelta

def get_database_sentiment():
    """Get recent sentiment data from database - individual articles"""
    ch_manager = ClickHouseManager()
    ch_manager.connect()
    
    query = """
    SELECT 
        ticker,
        headline,
        timestamp,
        recommendation,
        confidence,
        sentiment,
        explanation,
        article_url,
        analyzed_at
    FROM News.breaking_news 
    WHERE timestamp >= now() - INTERVAL 24 HOUR
    AND ticker != ''
    AND ticker != 'UNKNOWN'
    ORDER BY timestamp DESC
    """
    
    result = ch_manager.client.query(query)
    
    articles = []
    for row in result.result_rows:
        articles.append({
            'ticker': row[0],
            'headline': row[1],
            'timestamp': row[2],
            'recommendation': row[3],
            'confidence': row[4],
            'sentiment': row[5],
            'explanation': row[6],
            'article_url': row[7],
            'analyzed_at': row[8]
        })
    
    ch_manager.close()
    return articles

def get_recent_analysis_results():
    """Get the recent analysis results - this would need to be populated with actual data"""
    # For now, I'll create a mapping based on your recent analysis output
    # In a real scenario, this would come from the actual analysis run
    
    # From your recent analysis, here are the results by ticker
    ticker_results = {
        'AAPL': {'recommendation': 'SELL', 'confidence': 'high'},
        'AGCO': {'recommendation': 'BUY', 'confidence': 'medium'},
        'AGYS': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'APA': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'BAP': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'BIRD': {'recommendation': 'BUY', 'confidence': 'high'},
        'BN': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'BSY': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'CCI': {'recommendation': 'HOLD', 'confidence': 'high'},
        'CERT': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'CLDI': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'CNC': {'recommendation': 'SELL', 'confidence': 'high'},
        'CNS': {'recommendation': 'HOLD', 'confidence': 'high'},
        'CXT': {'recommendation': 'HOLD', 'confidence': 'high'},
        'DGCMF': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'DV': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'ES': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'FE': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'FNM': {'recommendation': 'HOLD', 'confidence': 'high'},
        'FNWB': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'FONR': {'recommendation': 'BUY', 'confidence': 'high'},
        'FROG': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'FUSE': {'recommendation': 'HOLD', 'confidence': 'high'},
        'GRNT': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'GSBD': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'HANS': {'recommendation': 'SELL', 'confidence': 'high'},
        'INOTF': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'IOSP': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'IOVA': {'recommendation': 'SELL', 'confidence': 'high'},
        'IRBT': {'recommendation': 'SELL', 'confidence': 'high'},
        'ISC': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'KICK': {'recommendation': 'BUY', 'confidence': 'high'},
        'KRY': {'recommendation': 'SELL', 'confidence': 'high'},
        'MATX': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'MDIA': {'recommendation': 'BUY', 'confidence': 'high'},
        'MMC': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'MSFT': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'MSGM': {'recommendation': 'BUY', 'confidence': 'high'},
        'NGG': {'recommendation': 'SELL', 'confidence': 'high'},
        'NTR': {'recommendation': 'HOLD', 'confidence': 'high'},
        'OMC': {'recommendation': 'HOLD', 'confidence': 'high'},
        'ORC': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'RBBN': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'RCUS': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'RMGCF': {'recommendation': 'SELL', 'confidence': 'high'},
        'RYTM': {'recommendation': 'SELL', 'confidence': 'high'},
        'SGA': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'SITE': {'recommendation': 'HOLD', 'confidence': 'high'},
        'SITM': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'SMR': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'SPFI': {'recommendation': 'HOLD', 'confidence': 'high'},
        'SRPT': {'recommendation': 'SELL', 'confidence': 'high'},
        'SSRM': {'recommendation': 'HOLD', 'confidence': 'high'},
        'TASK': {'recommendation': 'SELL', 'confidence': 'high'},
        'TFPM': {'recommendation': 'BUY', 'confidence': 'high'},
        'THG': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'TRNR': {'recommendation': 'BUY', 'confidence': 'high'},
        'TTEK': {'recommendation': 'HOLD', 'confidence': 'high'},
        'UBER': {'recommendation': 'BUY', 'confidence': 'high'},
        'UDR': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'VFC': {'recommendation': 'HOLD', 'confidence': 'high'},
        'VNOM': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'VRDN': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'VRNA': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'WED': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'WTW': {'recommendation': 'HOLD', 'confidence': 'medium'},
        'WYTC': {'recommendation': 'BUY', 'confidence': 'medium'},
        'X': {'recommendation': 'HOLD', 'confidence': 'medium'}
    }
    
    return ticker_results

def compare_articles():
    """Compare database articles with recent analysis results"""
    print("=" * 80)
    print("ARTICLE-BY-ARTICLE SENTIMENT COMPARISON")
    print("=" * 80)
    
    db_articles = get_database_sentiment()
    recent_results = get_recent_analysis_results()
    
    print(f"Database articles: {len(db_articles)}")
    print(f"Recent analysis tickers: {len(recent_results)}")
    print()
    
    # Compare each article
    matches = []
    differences = []
    
    for i, article in enumerate(db_articles, 1):
        ticker = article['ticker']
        db_rec = article['recommendation']
        db_conf = article['confidence']
        headline = article['headline'][:50] + "..." if len(article['headline']) > 50 else article['headline']
        timestamp = article['timestamp']
        
        # Get expected result from recent analysis
        expected = recent_results.get(ticker, {})
        expected_rec = expected.get('recommendation', 'NOT_FOUND')
        expected_conf = expected.get('confidence', 'NOT_FOUND')
        
        if db_rec == expected_rec and db_conf == expected_conf:
            matches.append({
                'ticker': ticker,
                'recommendation': db_rec,
                'confidence': db_conf,
                'headline': headline,
                'timestamp': timestamp
            })
        else:
            differences.append({
                'article_num': i,
                'ticker': ticker,
                'db_recommendation': db_rec,
                'db_confidence': db_conf,
                'expected_recommendation': expected_rec,
                'expected_confidence': expected_conf,
                'headline': headline,
                'timestamp': timestamp
            })
    
    # Print summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"✅ Matches: {len(matches)}")
    print(f"❌ Differences: {len(differences)}")
    print(f"📊 Consistency Rate: {len(matches) / len(db_articles) * 100:.1f}%")
    
    if differences:
        print("\n" + "=" * 80)
        print("DIFFERENCES FOUND")
        print("=" * 80)
        print(f"{'#':<3} {'Ticker':<8} {'DB Rec':<8} {'Expected':<8} {'DB Conf':<8} {'Exp Conf':<8} {'Headline':<40}")
        print("-" * 80)
        
        for diff in differences:
            print(f"{diff['article_num']:<3} {diff['ticker']:<8} {diff['db_recommendation']:<8} {diff['expected_recommendation']:<8} {diff['db_confidence']:<8} {diff['expected_confidence']:<8} {diff['headline']:<40}")
    else:
        print("\n🎉 Perfect Match! All articles match the expected sentiment analysis results.")
    
    # Show detailed breakdown for differences
    if differences:
        print("\n" + "=" * 80)
        print("DETAILED DIFFERENCES")
        print("=" * 80)
        
        for diff in differences:
            print(f"\nArticle #{diff['article_num']}: {diff['ticker']}")
            print(f"  Headline: {diff['headline']}")
            print(f"  Timestamp: {diff['timestamp']}")
            print(f"  Database: {diff['db_recommendation']} ({diff['db_confidence']})")
            print(f"  Expected: {diff['expected_recommendation']} ({diff['expected_confidence']})")
    
    # Show articles that match
    if matches:
        print(f"\n" + "=" * 80)
        print(f"SAMPLE MATCHES ({min(10, len(matches))} of {len(matches)})")
        print("=" * 80)
        
        for match in matches[:10]:
            print(f"{match['ticker']}: {match['recommendation']} ({match['confidence']}) - {match['headline']}")

if __name__ == "__main__":
    compare_articles() 