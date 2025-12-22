#!/usr/bin/env python3
"""
Test file to verify exactly what tokens/text the Claude model receives
during sentiment analysis. Pulls an article from breaking_news table
and replicates the exact sentiment_service.py flow.
"""

import asyncio
import os
import sys
import json
import aiohttp
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"
MAX_CONTENT_CHARS = 1500  # Matches sentiment_service.py limit


# ============================================================================
# ARTICLE SCRAPING - Exact copy from sentiment_service.py
# ============================================================================

async def scrape_article_content_async(url: str, max_chars: int = 6000) -> str:
    """Asynchronous article content scraping - EXACT COPY from sentiment_service.py"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                html_content = await response.text()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to extract article content specifically for Benzinga
        article_content = ""
        
        if 'benzinga.com' in url:
            # Target Benzinga's article paragraph structure
            article_paragraphs = soup.find_all('p')
            content_paragraphs = []
            
            for p in article_paragraphs:
                if p.parent and p.parent.get('class'):
                    parent_classes = p.parent.get('class', [])
                    if any('cAazyy' in str(cls) or 'dIYChw' in str(cls) for cls in parent_classes):
                        text = p.get_text().strip()
                        if len(text) > 20:
                            content_paragraphs.append(text)
            
            article_content = ' '.join(content_paragraphs)
        
        # Fallback to general content extraction if specific method fails
        if not article_content or len(article_content) < 100:
            selectors_to_try = [
                'article',
                '.article-content',
                '.story-body',
                '.post-content',
                '.content',
                '.article-body',
                '[data-module="ArticleBody"]',
                '.article-wrap',
                '.entry-content'
            ]
            
            for selector in selectors_to_try:
                elements = soup.select(selector)
                if elements:
                    article_content = elements[0].get_text()
                    break
            
            if not article_content:
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                article_content = ' '.join(chunk for chunk in chunks if chunk)
        
        # Clean up the content
        lines = (line.strip() for line in article_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_content = ' '.join(chunk for chunk in chunks if chunk)
        
        if len(clean_content) > max_chars:
            clean_content = clean_content[:max_chars]
            
        return clean_content
        
    except Exception as e:
        print(f"❌ Error scraping content from {url}: {e}")
        return ""


# ============================================================================
# PROMPT CREATION - Exact copy from sentiment_service.py
# ============================================================================

async def create_sentiment_prompt(article: dict) -> str:
    """Create sentiment prompt - EXACT COPY from sentiment_service.py"""
    ticker = article.get('ticker', 'UNKNOWN')
    headline = article.get('headline', '')
    summary = article.get('summary', '')
    full_content = article.get('full_content', '')
    article_url = article.get('article_url', '')
    
    # Use existing full_content if available, otherwise scrape from URL
    if full_content and len(full_content) > 200:
        content_to_analyze = full_content
        print(f"📄 Using existing full_content: {len(content_to_analyze)} characters")
    elif article_url:
        print(f"🌐 Scraping full content from URL: {article_url}")
        scraped_content = await scrape_article_content_async(article_url, max_chars=6000)
        if scraped_content:
            content_to_analyze = scraped_content
            print(f"📄 Using scraped content: {len(content_to_analyze)} characters")
        else:
            content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
    else:
        content_to_analyze = full_content if full_content else f"{headline}\n\n{summary}"
    
    # Apply 1500 character limit to match 4D prompt strategy
    content_to_analyze = content_to_analyze[:MAX_CONTENT_CHARS] if content_to_analyze else f"{headline}\n\n{summary}"

    # 4D Timing and Urgency Prompt - EXACT COPY
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
# TOKEN COUNTING
# ============================================================================

def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count using character-based estimation.
    Claude uses ~4 characters per token on average for English text.
    """
    return len(text) // 4


def analyze_prompt_structure(prompt: str) -> dict:
    """Analyze the structure of the prompt"""
    lines = prompt.split('\n')
    
    # Find article content section
    article_start = None
    article_end = None
    for i, line in enumerate(lines):
        if 'ARTICLE CONTENT:' in line:
            article_start = i + 1
        if article_start and 'TIMING ANALYSIS:' in line:
            article_end = i
            break
    
    article_content = '\n'.join(lines[article_start:article_end]) if article_start and article_end else ""
    
    # Calculate sections
    pre_article = '\n'.join(lines[:article_start]) if article_start else ""
    post_article = '\n'.join(lines[article_end:]) if article_end else ""
    
    return {
        'total_chars': len(prompt),
        'total_tokens_approx': count_tokens_approximate(prompt),
        'article_content_chars': len(article_content),
        'article_content_tokens_approx': count_tokens_approximate(article_content),
        'system_prompt_chars': len(pre_article) + len(post_article),
        'system_prompt_tokens_approx': count_tokens_approximate(pre_article + post_article),
        'line_count': len(lines),
        'article_content': article_content.strip()
    }


# ============================================================================
# CLAUDE API CALL
# ============================================================================

async def call_claude_api(prompt: str) -> dict:
    """Make a request to Claude API - matches sentiment_service.py exactly"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {"error": "No ANTHROPIC_API_KEY found in environment"}
    
    # System message prefix - EXACT from sentiment_service.py
    system_prefix = "You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n"
    full_prompt = system_prefix + prompt
    
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 300,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    }
    
    headers = {
        'anthropic-version': '2023-06-01',
        'x-api-key': api_key,
        'content-type': 'application/json'
    }
    
    timeout = aiohttp.ClientTimeout(total=180, connect=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(CLAUDE_ENDPOINT, json=payload, headers=headers) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return {
                        "status": "success",
                        "response": response_data,
                        "usage": response_data.get("usage", {}),
                        "content": response_data.get("content", [{}])[0].get("text", "")
                    }
                else:
                    return {
                        "status": "error",
                        "http_status": response.status,
                        "response": response_data
                    }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# DATABASE QUERY
# ============================================================================

def get_article_from_db(article_id: int = None, ticker: str = None, limit: int = 1):
    """Fetch article(s) from breaking_news table"""
    try:
        from clickhouse_setup import ClickHouseManager
        
        ch_manager = ClickHouseManager()
        ch_manager.connect()
        
        if article_id:
            query = """
            SELECT ticker, headline, summary, full_content, article_url, source, published_utc
            FROM News.breaking_news
            WHERE id = %s
            LIMIT 1
            """
            result = ch_manager.client.query(query, [article_id])
        elif ticker:
            query = """
            SELECT ticker, headline, summary, full_content, article_url, source, published_utc
            FROM News.breaking_news
            WHERE ticker = %s
            ORDER BY published_utc DESC
            LIMIT %s
            """
            result = ch_manager.client.query(query, [ticker, limit])
        else:
            query = """
            SELECT ticker, headline, summary, full_content, article_url, source, published_utc
            FROM News.breaking_news
            ORDER BY published_utc DESC
            LIMIT %s
            """
            result = ch_manager.client.query(query, [limit])
        
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
# MAIN TEST FUNCTION
# ============================================================================

async def test_sentiment_tokens(article: dict = None, article_id: int = None, ticker: str = None, call_api: bool = True):
    """
    Main test function - shows exactly what tokens the model receives
    
    Args:
        article: Pre-defined article dict (optional)
        article_id: Fetch article by ID from database (optional)
        ticker: Fetch latest article for ticker from database (optional)
        call_api: Whether to actually call the Claude API (default True)
    """
    print("=" * 80)
    print("🔍 SENTIMENT TOKEN ANALYSIS TEST")
    print("=" * 80)
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Max Content Characters: {MAX_CONTENT_CHARS}")
    print("=" * 80)
    
    # Get article
    if article:
        test_article = article
    elif article_id or ticker:
        articles = get_article_from_db(article_id=article_id, ticker=ticker)
        if not articles:
            print("❌ No articles found in database")
            return
        test_article = articles[0]
    else:
        # Fetch most recent article
        articles = get_article_from_db(limit=1)
        if not articles:
            print("❌ No articles found in database")
            return
        test_article = articles[0]
    
    # Display article info
    print("\n📰 ARTICLE INFO:")
    print("-" * 40)
    print(f"Ticker: {test_article.get('ticker', 'UNKNOWN')}")
    print(f"Headline: {test_article.get('headline', 'N/A')[:100]}...")
    print(f"URL: {test_article.get('article_url', 'N/A')}")
    print(f"Source: {test_article.get('source', 'N/A')}")
    print(f"Published: {test_article.get('published_at', 'N/A')}")
    print(f"Full Content Length: {len(test_article.get('full_content', '') or '')} chars")
    print(f"Summary Length: {len(test_article.get('summary', '') or '')} chars")
    
    # Create prompt (this includes scraping if needed)
    print("\n🔧 CREATING PROMPT...")
    print("-" * 40)
    prompt = await create_sentiment_prompt(test_article)
    
    # Analyze prompt structure
    analysis = analyze_prompt_structure(prompt)
    
    print("\n📊 TOKEN ANALYSIS:")
    print("-" * 40)
    print(f"Total Characters: {analysis['total_chars']:,}")
    print(f"Total Tokens (approx): {analysis['total_tokens_approx']:,}")
    print(f"Article Content Characters: {analysis['article_content_chars']:,}")
    print(f"Article Content Tokens (approx): {analysis['article_content_tokens_approx']:,}")
    print(f"System Prompt Characters: {analysis['system_prompt_chars']:,}")
    print(f"System Prompt Tokens (approx): {analysis['system_prompt_tokens_approx']:,}")
    print(f"Total Lines: {analysis['line_count']}")
    
    # Show the article content the model sees
    print("\n📄 ARTICLE CONTENT MODEL SEES (truncated to 1500 chars):")
    print("-" * 40)
    print(analysis['article_content'][:2000])
    if len(analysis['article_content']) > 2000:
        print(f"\n... [truncated, showing 2000 of {len(analysis['article_content'])} chars]")
    
    # Show the full prompt
    print("\n📝 FULL PROMPT SENT TO MODEL:")
    print("-" * 40)
    # Add the system prefix that sentiment_service.py adds
    system_prefix = "You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n"
    full_prompt = system_prefix + prompt
    print(full_prompt)
    print("-" * 40)
    print(f"[End of prompt - {len(full_prompt):,} characters, ~{count_tokens_approximate(full_prompt):,} tokens]")
    
    # Call Claude API if requested
    if call_api:
        print("\n🤖 CALLING CLAUDE API...")
        print("-" * 40)
        
        result = await call_claude_api(prompt)
        
        if result.get("status") == "success":
            usage = result.get("usage", {})
            print(f"✅ API Call Successful!")
            print(f"\n📈 ACTUAL TOKEN USAGE (from API):")
            print(f"   Input Tokens: {usage.get('input_tokens', 'N/A')}")
            print(f"   Output Tokens: {usage.get('output_tokens', 'N/A')}")
            
            print(f"\n💬 MODEL RESPONSE:")
            print("-" * 40)
            print(result.get("content", "No content"))
        else:
            print(f"❌ API Call Failed: {result}")
    else:
        print("\n⏭️ Skipping API call (call_api=False)")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    
    return {
        "article": test_article,
        "prompt": prompt,
        "analysis": analysis,
        "full_prompt_length": len(full_prompt)
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test sentiment analysis token usage')
    parser.add_argument('--ticker', '-t', type=str, help='Fetch latest article for this ticker')
    parser.add_argument('--id', type=int, help='Fetch article by ID')
    parser.add_argument('--url', '-u', type=str, help='Test with a specific URL')
    parser.add_argument('--no-api', action='store_true', help='Skip the actual API call')
    parser.add_argument('--headline', type=str, help='Custom headline for URL test')
    
    args = parser.parse_args()
    
    if args.url:
        # Test with custom URL
        article = {
            'ticker': 'TEST',
            'headline': args.headline or 'Test Article',
            'summary': '',
            'full_content': '',
            'article_url': args.url,
            'source': 'manual_test'
        }
        await test_sentiment_tokens(article=article, call_api=not args.no_api)
    elif args.id:
        await test_sentiment_tokens(article_id=args.id, call_api=not args.no_api)
    elif args.ticker:
        await test_sentiment_tokens(ticker=args.ticker, call_api=not args.no_api)
    else:
        # Fetch most recent article
        await test_sentiment_tokens(call_api=not args.no_api)


if __name__ == "__main__":
    asyncio.run(main())

