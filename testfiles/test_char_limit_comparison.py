#!/usr/bin/env python3
"""
Comparison test: 1500 chars vs 6000 chars for sentiment analysis
Tests ALL articles in breaking_news table to see if increased character limit
changes the sentiment recommendations.
"""

import asyncio
import os
import sys
import json
import csv
import aiohttp
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"

# The two limits we're comparing
CHAR_LIMIT_LOW = 1500
CHAR_LIMIT_HIGH = 6000

# Rate limiting
BATCH_SIZE = 5  # Articles per batch
BATCH_DELAY = 2.0  # Seconds between batches


# ============================================================================
# PROMPT CREATION (parameterized char limit)
# ============================================================================

def create_sentiment_prompt(article: dict, char_limit: int) -> str:
    """Create sentiment prompt with specified character limit"""
    headline = article.get('headline', '')
    summary = article.get('summary', '')
    full_content = article.get('full_content', '')
    
    # Use full_content if available, otherwise headline + summary
    if full_content and len(full_content) > 200:
        content_to_analyze = full_content
    else:
        content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
    
    # Apply character limit
    content_to_analyze = content_to_analyze[:char_limit] if content_to_analyze else f"{headline}\n\n{summary}"

    # 4D Timing and Urgency Prompt - EXACT COPY from sentiment_service.py
    prompt = f"""Analyze this financial news for immediate market impact timing.

ARTICLE CONTENT:
{content_to_analyze}

TIMING ANALYSIS: Determine if this news will cause immediate explosive price action (hours/days) or delayed appreciation.

IMMEDIATE IMPACT CATALYSTS (BUY + high confidence):
- FDA approvals, merger/acquisition announcements (being acquired at premium)
- Earnings surprises with immediate market implications
- Breaking regulatory decisions or legal victories
- Emergency use authorizations or critical partnerships

DELAYED IMPACT NEWS (BUY + medium confidence):
- Product development milestones with future potential
- Strategic initiatives with 6-12 month timelines
- Market expansion plans requiring execution time
- Research results requiring further development

IMMEDIATE SELL SIGNALS (SELL + high confidence):
- Going private/delisting plans, self-tender offers with share caps (loss of liquidity for remaining holders)
- Bankruptcy filings, insolvency warnings, going concern notices
- Dilutive offerings, reverse splits to avoid delisting
- Fraud allegations, SEC investigations, regulatory violations
- Failed clinical trials, product recalls, safety issues

LOW IMPACT/SPECULATIVE (HOLD):
- Early-stage research or development updates
- Management commentary without concrete announcements
- Industry trend discussions without company-specific catalysts
- Vague future planning statements

Focus on: Will this move the stock price within 24-48 hours?

Respond with JSON:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Immediate impact timing assessment and catalyst urgency analysis"
}}"""
    return prompt


# ============================================================================
# CLAUDE API CALL
# ============================================================================

async def call_claude_api(prompt: str, session: aiohttp.ClientSession) -> dict:
    """Make a request to Claude API"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {"error": "No ANTHROPIC_API_KEY found"}
    
    system_prefix = "You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n"
    full_prompt = system_prefix + prompt
    
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 300,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": full_prompt}]
    }
    
    headers = {
        'anthropic-version': '2023-06-01',
        'x-api-key': api_key,
        'content-type': 'application/json'
    }
    
    try:
        async with session.post(CLAUDE_ENDPOINT, json=payload, headers=headers) as response:
            response_data = await response.json()
            
            if response.status == 200:
                content = response_data.get("content", [{}])[0].get("text", "")
                usage = response_data.get("usage", {})
                
                # Parse JSON from response
                try:
                    # Clean markdown if present
                    if '```json' in content:
                        start = content.find('```json') + 7
                        end = content.find('```', start)
                        content = content[start:end].strip()
                    elif '```' in content:
                        start = content.find('```') + 3
                        end = content.find('```', start)
                        content = content[start:end].strip()
                    
                    parsed = json.loads(content)
                    return {
                        "status": "success",
                        "recommendation": parsed.get("recommendation", "UNKNOWN"),
                        "confidence": parsed.get("confidence", "unknown"),
                        "reasoning": parsed.get("reasoning", ""),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0)
                    }
                except json.JSONDecodeError:
                    return {"status": "error", "error": "JSON parse failed", "raw": content}
            else:
                return {"status": "error", "http_status": response.status, "response": response_data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# DATABASE
# ============================================================================

def get_all_articles():
    """Fetch all articles from breaking_news table"""
    try:
        from clickhouse_setup import ClickHouseManager
        
        ch_manager = ClickHouseManager()
        ch_manager.connect()
        
        query = """
        SELECT ticker, headline, summary, full_content, article_url, source, published_utc
        FROM News.breaking_news
        ORDER BY published_utc DESC
        """
        
        result = ch_manager.client.query(query)
        ch_manager.close()
        
        articles = []
        for row in result.result_rows:
            articles.append({
                'ticker': row[0],
                'headline': row[1],
                'summary': row[2],
                'full_content': row[3],
                'article_url': row[4],
                'source': row[5],
                'published_at': row[6]
            })
        
        return articles
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return []


# ============================================================================
# MAIN COMPARISON TEST
# ============================================================================

async def run_comparison_test(limit: int = None, output_csv: str = None):
    """
    Run comparison test on all articles
    
    Args:
        limit: Optional limit on number of articles to test
        output_csv: Optional path to save results CSV
    """
    print("=" * 80)
    print("🔬 CHARACTER LIMIT COMPARISON TEST")
    print("=" * 80)
    print(f"Comparing: {CHAR_LIMIT_LOW} chars vs {CHAR_LIMIT_HIGH} chars")
    print(f"Model: {CLAUDE_MODEL}")
    print("=" * 80)
    
    # Fetch articles
    print("\n📥 Fetching articles from breaking_news...")
    articles = get_all_articles()
    
    if not articles:
        print("❌ No articles found")
        return
    
    if limit:
        articles = articles[:limit]
    
    print(f"📊 Testing {len(articles)} articles")
    
    # Results storage
    results = []
    changed_count = 0
    same_count = 0
    error_count = 0
    
    # Token tracking
    total_tokens_low = 0
    total_tokens_high = 0
    
    # Create session
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        # Process in batches
        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} articles)")
            
            for j, article in enumerate(batch):
                ticker = article.get('ticker', 'UNKNOWN')
                headline = article.get('headline', '')[:50]
                
                print(f"  [{i+j+1:3d}/{len(articles)}] {ticker}: {headline}...")
                
                # Create prompts for both limits
                prompt_low = create_sentiment_prompt(article, CHAR_LIMIT_LOW)
                prompt_high = create_sentiment_prompt(article, CHAR_LIMIT_HIGH)
                
                # Call API for both (sequentially to avoid rate limits)
                result_low = await call_claude_api(prompt_low, session)
                await asyncio.sleep(0.5)  # Small delay between calls
                result_high = await call_claude_api(prompt_high, session)
                
                if result_low.get("status") == "success" and result_high.get("status") == "success":
                    rec_low = result_low.get("recommendation", "UNKNOWN")
                    rec_high = result_high.get("recommendation", "UNKNOWN")
                    conf_low = result_low.get("confidence", "unknown")
                    conf_high = result_high.get("confidence", "unknown")
                    
                    # Track tokens
                    total_tokens_low += result_low.get("input_tokens", 0)
                    total_tokens_high += result_high.get("input_tokens", 0)
                    
                    # Check if recommendation or confidence changed
                    rec_changed = rec_low != rec_high
                    conf_changed = conf_low != conf_high
                    
                    if rec_changed or conf_changed:
                        changed_count += 1
                        status = "🔄 CHANGED"
                        print(f"      {status}: {rec_low}/{conf_low} → {rec_high}/{conf_high}")
                    else:
                        same_count += 1
                        status = "✓ Same"
                    
                    results.append({
                        'ticker': ticker,
                        'headline': article.get('headline', ''),
                        'full_content_chars': len(article.get('full_content', '') or ''),
                        'rec_1500': rec_low,
                        'conf_1500': conf_low,
                        'reasoning_1500': result_low.get("reasoning", ""),
                        'tokens_1500': result_low.get("input_tokens", 0),
                        'rec_6000': rec_high,
                        'conf_6000': conf_high,
                        'reasoning_6000': result_high.get("reasoning", ""),
                        'tokens_6000': result_high.get("input_tokens", 0),
                        'rec_changed': rec_changed,
                        'conf_changed': conf_changed,
                        'any_changed': rec_changed or conf_changed
                    })
                else:
                    error_count += 1
                    error_msg = result_low.get("error") or result_high.get("error") or "Unknown error"
                    print(f"      ❌ Error: {error_msg}")
                    results.append({
                        'ticker': ticker,
                        'headline': article.get('headline', ''),
                        'error': error_msg
                    })
            
            # Delay between batches
            if i + BATCH_SIZE < len(articles):
                print(f"  ⏳ Waiting {BATCH_DELAY}s before next batch...")
                await asyncio.sleep(BATCH_DELAY)
    
    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    total_tested = same_count + changed_count
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Articles Tested: {total_tested}")
    print(f"   Errors: {error_count}")
    print(f"   Same Result: {same_count} ({same_count/max(1,total_tested)*100:.1f}%)")
    print(f"   Changed Result: {changed_count} ({changed_count/max(1,total_tested)*100:.1f}%)")
    
    print(f"\n💰 Token Usage:")
    print(f"   Total @ 1500 chars: {total_tokens_low:,} tokens")
    print(f"   Total @ 6000 chars: {total_tokens_high:,} tokens")
    print(f"   Additional tokens: {total_tokens_high - total_tokens_low:,} ({(total_tokens_high/max(1,total_tokens_low)-1)*100:.1f}% increase)")
    print(f"   Avg per article @ 1500: {total_tokens_low/max(1,total_tested):.0f} tokens")
    print(f"   Avg per article @ 6000: {total_tokens_high/max(1,total_tested):.0f} tokens")
    
    # Analyze changes
    if changed_count > 0:
        print(f"\n🔄 Change Analysis:")
        
        # Count recommendation changes
        rec_changes = [r for r in results if r.get('rec_changed')]
        conf_only_changes = [r for r in results if r.get('conf_changed') and not r.get('rec_changed')]
        
        print(f"   Recommendation changes: {len(rec_changes)}")
        print(f"   Confidence-only changes: {len(conf_only_changes)}")
        
        # Show specific changes
        if rec_changes:
            print(f"\n   📋 Recommendation Changes:")
            change_patterns = Counter()
            for r in rec_changes:
                pattern = f"{r['rec_1500']} → {r['rec_6000']}"
                change_patterns[pattern] += 1
            
            for pattern, count in change_patterns.most_common():
                print(f"      {pattern}: {count} articles")
        
        # Show examples of changed articles
        print(f"\n   📝 Example Changed Articles:")
        for r in rec_changes[:5]:
            print(f"      {r['ticker']}: {r['rec_1500']}/{r['conf_1500']} → {r['rec_6000']}/{r['conf_6000']}")
            print(f"         Headline: {r['headline'][:60]}...")
    
    # Recommendation distribution comparison
    print(f"\n📊 Recommendation Distribution:")
    for limit_name, rec_key in [("1500 chars", "rec_1500"), ("6000 chars", "rec_6000")]:
        recs = [r.get(rec_key) for r in results if r.get(rec_key)]
        rec_counts = Counter(recs)
        print(f"   {limit_name}:")
        for rec in ["BUY", "HOLD", "SELL"]:
            count = rec_counts.get(rec, 0)
            pct = count / max(1, len(recs)) * 100
            print(f"      {rec}: {count} ({pct:.1f}%)")
    
    # Save to CSV if requested
    if output_csv:
        csv_path = output_csv if output_csv.endswith('.csv') else f"{output_csv}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\n💾 Results saved to: {csv_path}")
    
    print("\n" + "=" * 80)
    print("✅ COMPARISON TEST COMPLETE")
    print("=" * 80)
    
    return {
        'total_tested': total_tested,
        'same_count': same_count,
        'changed_count': changed_count,
        'error_count': error_count,
        'change_rate': changed_count / max(1, total_tested) * 100,
        'token_increase': (total_tokens_high / max(1, total_tokens_low) - 1) * 100,
        'results': results
    }


# ============================================================================
# CLI
# ============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare 1500 vs 6000 char limit for sentiment analysis')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of articles to test')
    parser.add_argument('--output', '-o', type=str, default='char_limit_comparison', help='Output CSV filename')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to CSV')
    
    args = parser.parse_args()
    
    output_csv = None if args.no_save else args.output
    
    await run_comparison_test(limit=args.limit, output_csv=output_csv)


if __name__ == "__main__":
    asyncio.run(main())

