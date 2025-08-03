#!/usr/bin/env python3
"""
Sonnet-3.5 vs Sonnet-4 + Experimental Prompts Comparison Test

This script compares Claude Sonnet-3.5 vs Sonnet-4 model performance
AND tests new experimental prompts designed to distinguish 100-400% breakouts
from pump-and-dump scenarios.

- Sonnet-3.5: claude-3-5-sonnet-20240620 (current default)
- Sonnet-4: claude-sonnet-4-20250514 (latest model)
- Experimental Prompts: 4A-4E targeting breakout vs pump-dump distinction

Usage:
    python3 tests/sonnet_comparison_test.py --sample-size 50 --test-mode all
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import argparse
from dataclasses import dataclass
import numpy as np
import pytz
import aiohttp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_setup import ClickHouseManager
from dotenv import load_dotenv

# Load environment variables for Polygon API
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Results from sentiment analysis comparison"""
    ticker: str
    headline: str
    # Original models
    sonnet35_sentiment: str
    sonnet35_confidence: str  
    sonnet4_sentiment: str = ""
    sonnet4_confidence: str = "medium"
    # New experimental prompts
    prompt_4a_sentiment: str = ""
    prompt_4a_confidence: str = "medium"
    prompt_4b_sentiment: str = ""
    prompt_4b_confidence: str = "medium"
    prompt_4c_sentiment: str = ""
    prompt_4c_confidence: str = "medium"
    prompt_4d_sentiment: str = ""
    prompt_4d_confidence: str = "medium"
    prompt_4e_sentiment: str = ""
    prompt_4e_confidence: str = "medium"
    # Common fields
    actual_outcome: str = ""
    analysis_time_sonnet35: float = 0.0
    analysis_time_sonnet4: float = 0.0
    analysis_time_4a: float = 0.0
    analysis_time_4b: float = 0.0
    analysis_time_4c: float = 0.0
    analysis_time_4d: float = 0.0
    analysis_time_4e: float = 0.0
    # Add fields for PnL calculation
    published_utc: Optional[datetime] = None
    sonnet35_pnl: Optional[float] = None
    sonnet4_pnl: Optional[float] = None
    sonnet35_entry_price: Optional[float] = None
    sonnet35_exit_price: Optional[float] = None
    sonnet4_entry_price: Optional[float] = None
    sonnet4_exit_price: Optional[float] = None
    # Add detailed PnL tracking fields
    sonnet35_position_size: Optional[int] = None
    sonnet4_position_size: Optional[int] = None
    sonnet35_investment: Optional[float] = None
    sonnet4_investment: Optional[float] = None
    sonnet35_return_pct: Optional[float] = None
    sonnet4_return_pct: Optional[float] = None
    price_bracket: Optional[str] = None
    publication_hour: Optional[int] = None

class SonnetComparisonTester:
    """Compare Sonnet-3.5 vs Sonnet-4 model performance AND experimental prompts using identical test conditions"""
    
    def __init__(self, buy_high_threshold: float = 0.8, buy_medium_threshold: float = 0.5):
        self.ch_manager = ClickHouseManager()
        self.buy_high_threshold = buy_high_threshold
        self.buy_medium_threshold = buy_medium_threshold
        self.confidence_map = {'low': 0.55, 'medium': 0.7, 'high': 0.95}
        
        # PnL calculation setup
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.session = None
        self.est_tz = pytz.timezone('US/Eastern')
        self.default_quantity = 100  # Default shares per trade
        
        # Claude API configuration
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables")
            raise ValueError("ANTHROPIC_API_KEY is required")
        
        # Model specifications
        self.sonnet35_model = "claude-3-5-sonnet-20240620"
        self.sonnet4_model = "claude-sonnet-4-20250514"
        
        # Dynamic position sizing tiers
        self.position_tiers = [
            {'price_min': 0.01, 'price_max': 1.00, 'unit_position_size': 10000, 'max_position_size': 20000},
            {'price_min': 1.00, 'price_max': 3.00, 'unit_position_size': 8000, 'max_position_size': 16000},
            {'price_min': 3.00, 'price_max': 5.00, 'unit_position_size': 5000, 'max_position_size': 10000},
            {'price_min': 5.00, 'price_max': 8.00, 'unit_position_size': 3000, 'max_position_size': 6000},
            {'price_min': 8.00, 'price_max': 999999.99, 'unit_position_size': 2000, 'max_position_size': 4000}
        ]
        
        # Use PROXY_URL if available for Polygon API
        proxy_url = os.getenv('PROXY_URL', '').strip()
        if proxy_url:
            self.polygon_base_url = proxy_url.rstrip('/')
        else:
            self.polygon_base_url = "https://api.polygon.io"
        
    async def initialize(self):
        """Initialize the Sonnet comparison tester"""
        logger.info("🚀 Initializing Enhanced Sonnet + Experimental Prompts Comparison Test Framework...")
        
        # Initialize ClickHouse connection
        self.ch_manager.connect()
        
        # Initialize HTTP session for API calls
        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'anthropic-version': '2023-06-01',
                'x-api-key': self.anthropic_api_key,
                'content-type': 'application/json'
            }
        )
        
        # Initialize Polygon API session for PnL calculations if key available
        if self.polygon_api_key:
            logger.info("✅ Polygon API session initialized for PnL calculations")
        
        logger.info("✅ Enhanced comparison test framework initialized successfully")

    async def make_claude_request(self, prompt: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
        """Make a direct Claude API request with specified model"""
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "max_tokens": 300,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are a financial analyst expert at analyzing news sentiment and its impact on stock prices. Always respond with valid JSON.\n\n{prompt}"
                        }
                    ]
                }
                
                async with self.session.post(
                    "https://api.anthropic.com/v1/messages", 
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if response_data.get("content") and len(response_data["content"]) > 0:
                            content = response_data["content"][0]["text"]
                            
                            # Clean JSON from markdown and fix control characters
                            content = self._clean_json_content(content)
                            
                            try:
                                parsed_result = json.loads(content.strip())
                                return parsed_result
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON parsing failed for {model}: {e}")
                                logger.debug(f"Raw content: {repr(content)}")
                                
                                # Try to fix the JSON and parse again
                                fixed_content = self._fix_malformed_json(content)
                                try:
                                    parsed_result = json.loads(fixed_content)
                                    logger.info(f"✅ JSON fixed and parsed successfully for {model}")
                                    return parsed_result
                                except json.JSONDecodeError as e2:
                                    logger.error(f"❌ JSON still malformed after fixing for {model}: {e2}")
                                    logger.debug(f"Fixed content: {repr(fixed_content)}")
                                    if attempt == max_retries - 1:
                                        return {"error": f"JSON parsing failed: {str(e)}"}
                                
                                if attempt == max_retries - 1:
                                    return {"error": f"JSON parsing failed: {str(e)}"}
                        else:
                            logger.error(f"❌ No content in response from {model}")
                            if attempt == max_retries - 1:
                                return {"error": "No content in response"}
                    
                    elif response.status == 429:
                        wait_time = 2 ** attempt
                        logger.warning(f"⏱️ Rate limited on {model}, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        logger.error(f"❌ API error for {model}: {response.status}")
                        if attempt == max_retries - 1:
                            return {"error": f"API error: {response.status}"}
                        
            except Exception as e:
                logger.error(f"❌ Request failed for {model} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"error": f"Request failed: {str(e)}"}
                await asyncio.sleep(1)
        
        return {"error": "All retries failed"}

    def _clean_json_content(self, content: str) -> str:
        """Clean JSON content from markdown and fix control characters"""
        import re
        import json
        
        # Remove markdown code blocks
        content = content.strip()
        if content.startswith('```json'):
            content = content.split('```json')[1].split('```')[0]
        elif content.startswith('```'):
            content = content.split('```')[1].split('```')[0]
        
        # Remove invalid control characters (keep only valid JSON characters)
        # ASCII control characters (0-31) except tab (9), newline (10), carriage return (13)
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Replace problematic characters that might break JSON
        content = content.replace('\u0000', '')  # null character
        content = content.replace('\u0008', '')  # backspace
        content = content.replace('\u000B', '')  # vertical tab
        content = content.replace('\u000C', '')  # form feed
        content = content.replace('\u000E', '')  # shift out
        content = content.replace('\u000F', '')  # shift in
        
        # Fix common JSON issues
        content = content.replace('\\n', ' ')  # Replace literal \n with space
        content = content.replace('\\r', ' ')  # Replace literal \r with space
        content = content.replace('\\t', ' ')  # Replace literal \t with space
        
        # Clean up multiple spaces
        content = re.sub(r'\s+', ' ', content)
        
        # Handle unterminated strings by finding the last complete JSON object
        content = content.strip()
        
        # Try to find a complete JSON object
        brace_count = 0
        last_complete_pos = -1
        
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_complete_pos = i
        
        # If we found a complete JSON object, use only that part
        if last_complete_pos > 0:
            content = content[:last_complete_pos + 1]
        
        # Additional cleanup for malformed JSON
        # Remove any trailing commas before closing braces
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Fix unescaped quotes in strings
        # This is a simple approach - replace any unescaped quotes inside string values
        try:
            # Test if it's valid JSON first
            json.loads(content)
            return content
        except json.JSONDecodeError:
            # If it fails, try to fix common issues
            # Look for patterns like: "key": "value with "quotes" inside"
            content = re.sub(r'": "([^"]*)"([^"]*)"([^"]*)"', r'": "\1\"\2\"\3"', content)
            
            # Remove any non-printable characters that might remain
            content = ''.join(char if char.isprintable() or char in '\n\r\t' else ' ' for char in content)
            
            # Final cleanup
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content

    def _fix_malformed_json(self, content: str) -> str:
        """Attempt to fix common malformed JSON issues."""
        import re
        import json
        
        # 1. Fix missing commas between JSON key-value pairs
        # Pattern: "key": "value" "nextkey": "nextvalue"
        content = re.sub(r'"\s+"([a-zA-Z_][a-zA-Z0-9_]*)":', r'", "\1":', content)
        
        # 2. Fix missing commas after values before closing braces
        # Pattern: "value" }
        content = re.sub(r'"\s*}', '"}', content)
        
        # 3. Remove trailing commas before closing braces
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # 4. Fix unescaped quotes in strings
        content = re.sub(r'": "([^"]*)"([^"]*)"([^"]*)"', r'": "\1\"\2\"\3"', content)
        
        # 5. Remove any non-printable characters that might remain
        content = ''.join(char if char.isprintable() or char in '\n\r\t' else ' ' for char in content)
        
        # 6. Clean up multiple spaces
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 7. Try to validate and return
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            logger.warning(f"JSON still malformed after fixes: {e}")
            logger.debug(f"Content after fixes: {repr(content)}")
            
            # Last resort: try to construct a basic valid JSON
            if 'recommendation' not in content or 'confidence' not in content:
                return '{"recommendation": "HOLD", "confidence": "medium", "reasoning": "JSON parsing failed"}'
            
            return content

    def create_sentiment_prompt(self, content: str) -> str:
        """Create standardized sentiment analysis prompt (original)"""
        clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
        
        return f"""Analyze this financial news article and provide sentiment analysis.

ARTICLE CONTENT:
{clean_content[:1500]}

Please analyze the sentiment and provide a trading recommendation. Consider:
- Market impact potential
- Company fundamentals mentioned
- Overall tone and sentiment
- Trading opportunities

Respond with JSON in this exact format:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of your analysis"
}}"""

    def create_prompt_4a_breakout_vs_pump(self, content: str) -> str:
        """4A. Breakout vs Pump-Dump Focused Prompt"""
        clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
        
        return f"""Analyze this financial news article for genuine breakout potential vs pump-and-dump risk.

ARTICLE CONTENT:
{clean_content[:1500]}

CRITICAL DISTINCTION: You must identify news that will cause sustained 100-400% price increases (TRUE BREAKOUTS) vs news that creates temporary pumps followed by quick dumps (FALSE PUMPS).

TRUE BREAKOUT INDICATORS:
- Major business developments (partnerships, acquisitions, regulatory approvals)
- Fundamental game-changers (new products, market expansion, breakthrough technology)
- Concrete financial improvements (revenue growth, profitability, major contracts)
- Regulatory/legal victories with long-term impact

FALSE PUMP INDICATORS:
- Vague announcements without concrete details
- Speculative or "potential" developments
- Social media hype without substance
- Repetitive or recycled news

Respond with JSON:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Specific catalyst analysis and breakout vs pump assessment"
}}"""

    def create_prompt_4b_catalyst_strength(self, content: str) -> str:
        """4B. Catalyst Strength Prompt"""
        clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
        
        return f"""Analyze this financial news for catalyst strength and sustainability.

ARTICLE CONTENT:
{clean_content[:1500]}

CATALYST EVALUATION: Rate the news catalyst's ability to drive sustained price appreciation.

STRONG CATALYSTS (BUY + high confidence):
- FDA approvals, major partnerships, acquisition announcements
- Breakthrough products with clear market demand
- Significant revenue/earnings beats with raised guidance
- Regulatory wins that unlock new markets

MODERATE CATALYSTS (BUY + medium confidence):
- Product launches, smaller partnerships, positive trials
- Good earnings with modest guidance improvements
- Market expansion announcements with clear timelines

WEAK/RISKY CATALYSTS (HOLD):
- Vague "exploring opportunities" statements
- Social media mentions or influencer endorsements
- Speculative research or "potential" developments
- Repeated announcements of old news

Respond with JSON:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Catalyst strength assessment and sustainability analysis"
}}"""

    def create_prompt_4c_institutional_appeal(self, content: str) -> str:
        """4C. Institutional vs Retail Appeal Prompt"""
        clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
        
        return f"""Analyze this financial news from an institutional investor perspective.

ARTICLE CONTENT:
{clean_content[:1500]}

INSTITUTIONAL APPEAL: Evaluate if this news would attract serious institutional money (which drives sustained 100-400% moves) vs retail FOMO (which creates pump-and-dumps).

INSTITUTIONAL ATTRACTORS (BUY + high confidence):
- Concrete business metrics and financial improvements
- Strategic partnerships with established companies
- Regulatory approvals with clear market opportunities
- Management guidance increases with specific numbers

RETAIL FOMO TRIGGERS (HOLD - pump risk):
- Buzzword-heavy announcements (AI, blockchain, quantum)
- Celebrity endorsements or social media viral content
- Vague "revolutionary" claims without specifics
- Penny stock promotion language

EVALUATION CRITERIA:
- Would a pension fund invest based on this news?
- Are there specific, measurable business improvements?
- Is the language professional or promotional?

Respond with JSON:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Institutional appeal assessment and pump-dump risk analysis"
}}"""

    def create_prompt_4d_timing_urgency(self, content: str) -> str:
        """4D. Timing and Urgency Prompt"""
        clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
        
        return f"""Analyze this financial news for immediate market impact timing.

ARTICLE CONTENT:
{clean_content[:1500]}

TIMING ANALYSIS: Determine if this news will cause immediate explosive price action (hours/days) or delayed appreciation.

IMMEDIATE IMPACT CATALYSTS (BUY + high confidence):
- FDA approvals, merger announcements, major contract wins
- Earnings surprises with immediate market implications
- Breaking regulatory decisions or legal victories
- Emergency use authorizations or critical partnerships

DELAYED IMPACT NEWS (BUY + medium confidence):
- Product development milestones with future potential
- Strategic initiatives with 6-12 month timelines
- Market expansion plans requiring execution time
- Research results requiring further development

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

    def create_prompt_4e_concrete_vs_speculative(self, content: str) -> str:
        """4E. Concrete vs Speculative Prompt"""
        clean_content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean_content = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in clean_content)
        
        return f"""Analyze this financial news for concrete vs speculative content.

ARTICLE CONTENT:
{clean_content[:1500]}

CONCRETE VS SPECULATIVE: Only concrete, measurable developments drive sustained 100-400% gains. Speculation creates pump-and-dumps.

CONCRETE DEVELOPMENTS (BUY + high confidence):
- Specific dollar amounts (revenue, contracts, investments)
- Named partners, customers, or acquisition targets
- Exact dates, percentages, or measurable milestones
- Completed transactions or finalized agreements

MODERATELY CONCRETE (BUY + medium confidence):
- Announced partnerships pending final agreements
- Product launches with specific target markets
- Financial guidance with concrete ranges
- Regulatory submissions with clear timelines

SPECULATIVE CONTENT (HOLD):
- "Exploring opportunities" or "considering options"
- "Potential," "possible," or "may" language
- Unnamed partners or "major company" references
- Vague timeline phrases like "in the coming months"

RED FLAGS: Promotional language, excessive superlatives, lack of specific details

Respond with JSON:
{{
    "recommendation": "BUY/HOLD/SELL",
    "confidence": "high/medium/low",
    "reasoning": "Concrete vs speculative assessment with specific evidence"
}}"""

    async def analyze_with_prompt(self, content: str, prompt_func, model: str) -> Tuple[str, str, float]:
        """Generic method to analyze with any prompt function"""
        start_time = datetime.now()
        
        try:
            prompt = prompt_func(content)
            result = await self.make_claude_request(prompt, model)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            if result and 'error' not in result:
                recommendation = result.get('recommendation', 'HOLD')
                confidence = result.get('confidence', 'medium')
                return recommendation, confidence, analysis_time
            else:
                logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                return 'HOLD', 'medium', analysis_time
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_time = (datetime.now() - start_time).total_seconds()
            return 'HOLD', 'medium', analysis_time

    async def analyze_sonnet35_sentiment(self, content: str) -> Tuple[str, str, float]:
        """Analyze sentiment using Sonnet-3.5 model"""
        return await self.analyze_with_prompt(content, self.create_sentiment_prompt, self.sonnet35_model)

    async def analyze_sonnet4_sentiment(self, content: str) -> Tuple[str, str, float]:
        """Analyze sentiment using Sonnet-4 model"""
        return await self.analyze_with_prompt(content, self.create_sentiment_prompt, self.sonnet4_model)

    async def analyze_prompt_4a(self, content: str) -> Tuple[str, str, float]:
        """Analyze with 4A: Breakout vs Pump-Dump prompt"""
        return await self.analyze_with_prompt(content, self.create_prompt_4a_breakout_vs_pump, self.sonnet35_model)

    async def analyze_prompt_4b(self, content: str) -> Tuple[str, str, float]:
        """Analyze with 4B: Catalyst Strength prompt"""
        return await self.analyze_with_prompt(content, self.create_prompt_4b_catalyst_strength, self.sonnet35_model)

    async def analyze_prompt_4c(self, content: str) -> Tuple[str, str, float]:
        """Analyze with 4C: Institutional Appeal prompt"""
        return await self.analyze_with_prompt(content, self.create_prompt_4c_institutional_appeal, self.sonnet35_model)

    async def analyze_prompt_4d(self, content: str) -> Tuple[str, str, float]:
        """Analyze with 4D: Timing and Urgency prompt"""
        return await self.analyze_with_prompt(content, self.create_prompt_4d_timing_urgency, self.sonnet35_model)

    async def analyze_prompt_4e(self, content: str) -> Tuple[str, str, float]:
        """Analyze with 4E: Concrete vs Speculative prompt"""
        return await self.analyze_with_prompt(content, self.create_prompt_4e_concrete_vs_speculative, self.sonnet35_model)

    async def get_test_articles(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get balanced articles from the TEST SET for evaluation"""
        try:
            # Get balanced sample from each outcome type
            per_outcome = max(1, limit // 3)  # Divide by 3 outcome types
            
            all_articles = []
            
            for outcome_type in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
                query = """
                SELECT 
                    ticker,
                    headline,
                    full_content,
                    outcome_type,
                    has_30pt_increase,
                    is_false_pump,
                    price_increase_ratio,
                    original_content_hash,
                    published_est
                FROM News.rag_test_set
                WHERE outcome_type = %s AND LENGTH(headline) > 30
                ORDER BY ticker
                LIMIT %s
                """
                
                result = self.ch_manager.client.query(query, parameters=[outcome_type, per_outcome])
                
                for row in result.result_rows:
                    all_articles.append({
                        'ticker': row[0],
                        'headline': row[1],
                        'full_content': row[2],
                        'outcome_type': row[3],
                        'has_30pt_increase': int(row[4]),
                        'is_false_pump': int(row[5]),
                        'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                        'original_content_hash': row[7],
                        'published_est': row[8],  # This is EST timezone, not UTC
                        'content': row[2] or f"{row[0]}: {row[1]}"  # Use full_content if available
                    })
            
            # Shuffle for random order
            import random
            random.shuffle(all_articles)
            
            logger.info(f"📄 Retrieved {len(all_articles)} balanced TEST articles for evaluation")
            
            # Log distribution
            outcome_counts = {}
            for article in all_articles:
                outcome = article['outcome_type']
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            logger.info("📊 Test set distribution:")
            for outcome, count in outcome_counts.items():
                logger.info(f"  • {outcome}: {count} articles")
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error retrieving test articles: {e}")
            return []

    async def get_all_articles(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get articles from ENTIRE dataset (both training and test sets) for prompt evaluation
        
        Since we're doing direct prompt injection (not RAG), there's no data leakage concern.
        The base models haven't seen any of this data before.
        """
        try:
            all_articles = []
            
            # Query both training and test sets
            for table_name in ['rag_training_set', 'rag_test_set']:
                for outcome_type in ['TRUE_BULLISH', 'FALSE_PUMP', 'NEUTRAL']:
                    query = f"""
                    SELECT 
                        ticker,
                        headline,
                        full_content,
                        outcome_type,
                        has_30pt_increase,
                        is_false_pump,
                        price_increase_ratio,
                        original_content_hash,
                        published_est
                    FROM News.{table_name}
                    WHERE outcome_type = %s AND LENGTH(headline) > 30
                    ORDER BY published_est
                    """
                    
                    # Apply limit per outcome type if specified
                    if limit:
                        per_outcome = max(1, limit // 3)  # Divide by 3 outcome types
                        query += f" LIMIT {per_outcome}"
                    
                    result = self.ch_manager.client.query(query, parameters=[outcome_type])
                    
                    for row in result.result_rows:
                        all_articles.append({
                            'ticker': row[0],
                            'headline': row[1],
                            'full_content': row[2],
                            'outcome_type': row[3],
                            'has_30pt_increase': int(row[4]),
                            'is_false_pump': int(row[5]),
                            'price_increase_ratio': float(row[6]) if row[6] else 0.0,
                            'original_content_hash': row[7],
                            'published_est': row[8],
                            'content': row[2] or f"{row[0]}: {row[1]}",  # Use full_content if available
                            'source_table': table_name  # Track which table it came from
                        })
            
            # Shuffle for random order (set seed for reproducibility)
            import random
            random.seed(42)  # Fixed seed for reproducible results
            random.shuffle(all_articles)
            
            logger.info(f"📄 Retrieved {len(all_articles)} articles from ENTIRE dataset for evaluation")
            
            # Log distribution
            outcome_counts = {}
            source_counts = {}
            for article in all_articles:
                outcome = article['outcome_type']
                source = article['source_table']
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            logger.info("📊 Full dataset distribution:")
            for outcome, count in outcome_counts.items():
                logger.info(f"  • {outcome}: {count} articles")
            
            logger.info("📊 By source:")
            for source, count in source_counts.items():
                logger.info(f"  • {source}: {count} articles")
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error retrieving all articles: {e}")
            return []
    
    async def run_comparison_test(self, test_articles: List[Dict[str, Any]], test_mode: str = "all") -> List[SentimentResult]:
        """Run comprehensive comparison test between Sonnet models AND experimental prompts"""
        logger.info(f"🚀 Starting comprehensive comparison test ({test_mode}) with {len(test_articles)} articles...")
        logger.info("📊 Testing: Sonnet-3.5, Sonnet-4, and 5 experimental prompts (4A-4E)")
        
        results = []
        
        for i, article in enumerate(test_articles, 1):
            logger.info(f"📊 Analyzing article {i}/{len(test_articles)}: {article['ticker']} ({article['outcome_type']})")
            
            result = SentimentResult(
                ticker=article['ticker'],
                headline=article['headline'],
                sonnet35_sentiment="",
                sonnet35_confidence="",
                actual_outcome=article['outcome_type'],
                published_utc=article.get('published_est')  # Use published_est from database
            )
            
            # Original Sonnet models
            if test_mode in ["sonnet35", "parallel", "all"]:
                # Sonnet-3.5 analysis
                s35_sentiment, s35_conf, s35_time = await self.analyze_sonnet35_sentiment(article['content'])
                result.sonnet35_sentiment = s35_sentiment
                result.sonnet35_confidence = s35_conf
                result.analysis_time_sonnet35 = s35_time
                
                logger.info(f"  🔍 Sonnet-3.5: {s35_sentiment} ({s35_conf}) in {s35_time:.2f}s")
            
            if test_mode in ["sonnet4", "parallel", "all"]:
                # Sonnet-4 analysis
                s4_sentiment, s4_conf, s4_time = await self.analyze_sonnet4_sentiment(article['content'])
                result.sonnet4_sentiment = s4_sentiment
                result.sonnet4_confidence = s4_conf
                result.analysis_time_sonnet4 = s4_time
                
                logger.info(f"  🧠 Sonnet-4: {s4_sentiment} ({s4_conf}) in {s4_time:.2f}s")
            
            # New experimental prompts (all use Sonnet-3.5 for consistency)
            if test_mode in ["experimental", "all"]:
                # 4A: Breakout vs Pump-Dump
                p4a_sentiment, p4a_conf, p4a_time = await self.analyze_prompt_4a(article['content'])
                result.prompt_4a_sentiment = p4a_sentiment
                result.prompt_4a_confidence = p4a_conf
                result.analysis_time_4a = p4a_time
                
                logger.info(f"  🎯 Prompt 4A (Breakout): {p4a_sentiment} ({p4a_conf}) in {p4a_time:.2f}s")
                
                # 4B: Catalyst Strength
                p4b_sentiment, p4b_conf, p4b_time = await self.analyze_prompt_4b(article['content'])
                result.prompt_4b_sentiment = p4b_sentiment
                result.prompt_4b_confidence = p4b_conf
                result.analysis_time_4b = p4b_time
                
                logger.info(f"  💪 Prompt 4B (Catalyst): {p4b_sentiment} ({p4b_conf}) in {p4b_time:.2f}s")
                
                # 4C: Institutional Appeal
                p4c_sentiment, p4c_conf, p4c_time = await self.analyze_prompt_4c(article['content'])
                result.prompt_4c_sentiment = p4c_sentiment
                result.prompt_4c_confidence = p4c_conf
                result.analysis_time_4c = p4c_time
                
                logger.info(f"  🏛️ Prompt 4C (Institutional): {p4c_sentiment} ({p4c_conf}) in {p4c_time:.2f}s")
                
                # 4D: Timing and Urgency
                p4d_sentiment, p4d_conf, p4d_time = await self.analyze_prompt_4d(article['content'])
                result.prompt_4d_sentiment = p4d_sentiment
                result.prompt_4d_confidence = p4d_conf
                result.analysis_time_4d = p4d_time
                
                logger.info(f"  ⏰ Prompt 4D (Timing): {p4d_sentiment} ({p4d_conf}) in {p4d_time:.2f}s")
                
                # 4E: Concrete vs Speculative
                p4e_sentiment, p4e_conf, p4e_time = await self.analyze_prompt_4e(article['content'])
                result.prompt_4e_sentiment = p4e_sentiment
                result.prompt_4e_confidence = p4e_conf
                result.analysis_time_4e = p4e_time
                
                logger.info(f"  📋 Prompt 4E (Concrete): {p4e_sentiment} ({p4e_conf}) in {p4e_time:.2f}s")
            
            # Run only Prompt 4D (Timing and Urgency)
            elif test_mode == "4d":
                # 4D: Timing and Urgency
                p4d_sentiment, p4d_conf, p4d_time = await self.analyze_prompt_4d(article['content'])
                result.prompt_4d_sentiment = p4d_sentiment
                result.prompt_4d_confidence = p4d_conf
                result.analysis_time_4d = p4d_time
                
                logger.info(f"  ⏰ Prompt 4D (Timing): {p4d_sentiment} ({p4d_conf}) in {p4d_time:.2f}s")
            
            results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Calculate performance metrics for ALL models and prompts with focus on BUY+high precision"""
        metrics = {
            'total_articles': len(results),
            'sonnet35_metrics': {},
            'sonnet4_metrics': {},
            'prompt_4a_metrics': {},
            'prompt_4b_metrics': {},
            'prompt_4c_metrics': {},
            'prompt_4d_metrics': {},
            'prompt_4e_metrics': {},
            'performance_comparison': {},
            'recall_metrics': {}
        }
        
        # Calculate recall metrics first (needed for all analyses)
        true_bullish_articles = [r for r in results if r.actual_outcome == "TRUE_BULLISH"]
        total_true_bullish = len(true_bullish_articles)
        
        metrics['recall_metrics'] = {
            'total_true_bullish_articles': total_true_bullish,
            'sonnet35_buy_high_recall': 0.0,
            'sonnet4_buy_high_recall': 0.0,
            'prompt_4a_buy_high_recall': 0.0,
            'prompt_4b_buy_high_recall': 0.0,
            'prompt_4c_buy_high_recall': 0.0,
            'prompt_4d_buy_high_recall': 0.0,
            'prompt_4e_buy_high_recall': 0.0,
            'sonnet35_buy_any_recall': 0.0,
            'sonnet4_buy_any_recall': 0.0,
            'prompt_4a_buy_any_recall': 0.0,
            'prompt_4b_buy_any_recall': 0.0,
            'prompt_4c_buy_any_recall': 0.0,
            'prompt_4d_buy_any_recall': 0.0,
            'prompt_4e_buy_any_recall': 0.0
        }

        # Helper function to calculate metrics for any prompt/model
        def calculate_prompt_metrics(results, sentiment_field, confidence_field, time_field, recall_key_prefix):
            correct = 0
            buy_high_correct = 0
            buy_high_total = 0
            buy_medium_correct = 0
            buy_medium_total = 0
            buy_high_recall_count = 0
            buy_any_recall_count = 0
            
            for result in results:
                sentiment = getattr(result, sentiment_field, '')
                confidence = getattr(result, confidence_field, 'medium')
                
                if not sentiment:  # Skip if this prompt wasn't run
                    continue
                
                # Check if prediction matches actual outcome
                expected_action = self.outcome_to_expected_action(result.actual_outcome)
                if sentiment == expected_action:
                    correct += 1
                
                # Check BUY+high precision (most critical metric)
                if sentiment == "BUY":
                    if confidence == "high":
                        buy_high_total += 1
                        if result.actual_outcome == "TRUE_BULLISH":
                            buy_high_correct += 1
                    elif confidence in ["medium", "low"]:
                        buy_medium_total += 1
                        if result.actual_outcome == "TRUE_BULLISH":
                            buy_medium_correct += 1
                
                # Calculate recall for TRUE_BULLISH articles
                if result.actual_outcome == "TRUE_BULLISH":
                    if sentiment == "BUY":
                        buy_any_recall_count += 1
                        if confidence == "high":
                            buy_high_recall_count += 1
            
            # Calculate recall percentages
            if total_true_bullish > 0:
                metrics['recall_metrics'][f'{recall_key_prefix}_buy_high_recall'] = buy_high_recall_count / total_true_bullish
                metrics['recall_metrics'][f'{recall_key_prefix}_buy_any_recall'] = buy_any_recall_count / total_true_bullish
            
            # Calculate average analysis time
            times = [getattr(r, time_field, 0) for r in results if getattr(r, sentiment_field, '')]
            avg_time = sum(times) / len(times) if times else 0
            
            return {
                'accuracy': correct / len([r for r in results if getattr(r, sentiment_field, '')]) if any(getattr(r, sentiment_field, '') for r in results) else 0,
                'buy_high_precision': buy_high_correct / max(1, buy_high_total),
                'buy_high_count': buy_high_total,
                'buy_medium_precision': buy_medium_correct / max(1, buy_medium_total),
                'buy_medium_count': buy_medium_total,
                'avg_analysis_time': avg_time,
                'buy_high_recall': metrics['recall_metrics'][f'{recall_key_prefix}_buy_high_recall'],
                'buy_any_recall': metrics['recall_metrics'][f'{recall_key_prefix}_buy_any_recall']
            }

        # Calculate metrics for all prompts/models
        if any(r.sonnet35_sentiment for r in results):
            metrics['sonnet35_metrics'] = calculate_prompt_metrics(results, 'sonnet35_sentiment', 'sonnet35_confidence', 'analysis_time_sonnet35', 'sonnet35')
        
        if any(r.sonnet4_sentiment for r in results):
            metrics['sonnet4_metrics'] = calculate_prompt_metrics(results, 'sonnet4_sentiment', 'sonnet4_confidence', 'analysis_time_sonnet4', 'sonnet4')
        
        if any(r.prompt_4a_sentiment for r in results):
            metrics['prompt_4a_metrics'] = calculate_prompt_metrics(results, 'prompt_4a_sentiment', 'prompt_4a_confidence', 'analysis_time_4a', 'prompt_4a')
        
        if any(r.prompt_4b_sentiment for r in results):
            metrics['prompt_4b_metrics'] = calculate_prompt_metrics(results, 'prompt_4b_sentiment', 'prompt_4b_confidence', 'analysis_time_4b', 'prompt_4b')
        
        if any(r.prompt_4c_sentiment for r in results):
            metrics['prompt_4c_metrics'] = calculate_prompt_metrics(results, 'prompt_4c_sentiment', 'prompt_4c_confidence', 'analysis_time_4c', 'prompt_4c')
        
        if any(r.prompt_4d_sentiment for r in results):
            metrics['prompt_4d_metrics'] = calculate_prompt_metrics(results, 'prompt_4d_sentiment', 'prompt_4d_confidence', 'analysis_time_4d', 'prompt_4d')
        
        if any(r.prompt_4e_sentiment for r in results):
            metrics['prompt_4e_metrics'] = calculate_prompt_metrics(results, 'prompt_4e_sentiment', 'prompt_4e_confidence', 'analysis_time_4e', 'prompt_4e')
        
        # Performance comparison (compare all prompts to Sonnet-3.5 baseline)
        if metrics['sonnet35_metrics']:
            baseline = metrics['sonnet35_metrics']
            metrics['performance_comparison'] = {}
            
            for prompt_name in ['sonnet4', 'prompt_4a', 'prompt_4b', 'prompt_4c', 'prompt_4d', 'prompt_4e']:
                prompt_metrics = metrics.get(f'{prompt_name}_metrics')
                if prompt_metrics:
                    metrics['performance_comparison'][prompt_name] = {
                        'accuracy_improvement': prompt_metrics['accuracy'] - baseline['accuracy'],
                        'buy_high_precision_improvement': prompt_metrics['buy_high_precision'] - baseline['buy_high_precision'],
                        'buy_medium_precision_improvement': prompt_metrics['buy_medium_precision'] - baseline['buy_medium_precision'],
                        'buy_high_recall_improvement': prompt_metrics['buy_high_recall'] - baseline['buy_high_recall'],
                        'buy_any_recall_improvement': prompt_metrics['buy_any_recall'] - baseline['buy_any_recall'],
                        'time_overhead': prompt_metrics['avg_analysis_time'] - baseline['avg_analysis_time']
                    }
        
        return metrics

    def generate_detailed_analysis(self, results: List[SentimentResult], metrics: Dict[str, Any]) -> str:
        """Generate detailed markdown analysis of ALL models and prompts performance"""
        
        md_content = f"""# Comprehensive Sentiment Analysis Comparison - Sonnet Models + Experimental Prompts

## Test Summary
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Articles Analyzed**: {metrics['total_articles']}
- **TRUE_BULLISH Articles in Test Set**: {metrics['recall_metrics']['total_true_bullish_articles']}
- **Models/Prompts Tested**: """
        
        # Only list actually tested models/prompts
        tested_models = []
        if metrics.get('sonnet35_metrics'):
            tested_models.append('Sonnet-3.5')
        if metrics.get('sonnet4_metrics'):
            tested_models.append('Sonnet-4')
        if metrics.get('prompt_4a_metrics'):
            tested_models.append('4A-4E Experimental Prompts')
        
        md_content += ', '.join(tested_models)
        md_content += """

## Overall Performance Metrics

"""
        
        # Original Models - only if they exist
        if metrics.get('sonnet35_metrics') or metrics.get('sonnet4_metrics'):
            md_content += """### Original Models
"""
            
            if metrics.get('sonnet35_metrics'):
                s35 = metrics['sonnet35_metrics']
                md_content += f"""#### Sonnet-3.5 Model (Baseline)
- **Overall Accuracy**: {s35['accuracy']:.1%}
- **BUY+High Precision**: {s35['buy_high_precision']:.1%} ({s35['buy_high_count']} signals)
- **BUY+High Recall**: {s35['buy_high_recall']:.1%}
- **BUY (Any) Recall**: {s35['buy_any_recall']:.1%}

"""
            
            if metrics.get('sonnet4_metrics'):
                s4 = metrics['sonnet4_metrics']
                md_content += f"""#### Sonnet-4 Model
- **Overall Accuracy**: {s4['accuracy']:.1%}
- **BUY+High Precision**: {s4['buy_high_precision']:.1%} ({s4['buy_high_count']} signals)
- **BUY+High Recall**: {s4['buy_high_recall']:.1%}
- **BUY (Any) Recall**: {s4['buy_any_recall']:.1%}

"""
        
        # Experimental Prompts - only if they exist
        experimental_exists = any(metrics.get(f'prompt_4{x}_metrics') for x in ['a', 'b', 'c', 'd', 'e'])
        if experimental_exists:
            md_content += """### Experimental Prompts (All using Sonnet-3.5)
"""
            
            experimental_prompts = [
                ('prompt_4a_metrics', '4A: Breakout vs Pump-Dump Focused'),
                ('prompt_4b_metrics', '4B: Catalyst Strength'),
                ('prompt_4c_metrics', '4C: Institutional vs Retail Appeal'),
                ('prompt_4d_metrics', '4D: Timing and Urgency'),
                ('prompt_4e_metrics', '4E: Concrete vs Speculative')
            ]
            
            for metric_key, display_name in experimental_prompts:
                if metrics.get(metric_key):
                    exp = metrics[metric_key]
                    md_content += f"""#### {display_name}
- **Overall Accuracy**: {exp['accuracy']:.1%}
- **BUY+High Precision**: {exp['buy_high_precision']:.1%} ({exp['buy_high_count']} signals)
- **BUY+High Recall**: {exp['buy_high_recall']:.1%}
- **BUY (Any) Recall**: {exp['buy_any_recall']:.1%}

"""
        
        md_content += """---

## Performance Improvements vs Sonnet-3.5 Baseline

| Model/Prompt | Accuracy | BUY+High Precision | BUY+High Recall | BUY (Any) Recall |
|--------------|----------|-------------------|-----------------|------------------|"""
        
        # Add comparison table - only for models that were actually tested
        if metrics.get('performance_comparison'):
            for prompt_name, comparison in metrics['performance_comparison'].items():
                prompt_display = {
                    'sonnet4': 'Sonnet-4',
                    'prompt_4a': '4A: Breakout vs Pump',
                    'prompt_4b': '4B: Catalyst Strength', 
                    'prompt_4c': '4C: Institutional Appeal',
                    'prompt_4d': '4D: Timing & Urgency',
                    'prompt_4e': '4E: Concrete vs Speculative'
                }.get(prompt_name, prompt_name)
                
                md_content += f"\n| {prompt_display} | {comparison['accuracy_improvement']:+.1%} | {comparison['buy_high_precision_improvement']:+.1%} | {comparison['buy_high_recall_improvement']:+.1%} | {comparison['buy_any_recall_improvement']:+.1%} |"
        
        md_content += f"""

---

## Key Findings & Recommendations

### Best Performing Prompts
"""
        
        # Find best performers - only from actually tested prompts
        all_prompts = ['sonnet35', 'sonnet4', 'prompt_4a', 'prompt_4b', 'prompt_4c', 'prompt_4d', 'prompt_4e']
        buy_high_precision_scores = []
        
        for prompt in all_prompts:
            prompt_metrics = metrics.get(f'{prompt}_metrics')
            if prompt_metrics:
                buy_high_precision_scores.append((prompt, prompt_metrics['buy_high_precision']))
        
        # Sort by BUY+High precision (most important metric)
        buy_high_precision_scores.sort(key=lambda x: x[1], reverse=True)
        
        md_content += f"""
**Ranked by BUY+High Precision (Most Important for 100-400% Gains):**
"""
        
        for i, (prompt, score) in enumerate(buy_high_precision_scores[:5], 1):
            prompt_display = {
                'sonnet35': 'Sonnet-3.5 (Baseline)',
                'sonnet4': 'Sonnet-4',
                'prompt_4a': '4A: Breakout vs Pump-Dump',
                'prompt_4b': '4B: Catalyst Strength', 
                'prompt_4c': '4C: Institutional Appeal',
                'prompt_4d': '4D: Timing & Urgency',
                'prompt_4e': '4E: Concrete vs Speculative'
            }.get(prompt, prompt)
            
            md_content += f"\n{i}. **{prompt_display}**: {score:.1%} precision"
        
        md_content += f"""

### Test Analysis
"""
        
        if experimental_exists:
            md_content += """- **Experimental Prompts**: All experimental prompts (4A-4E) were designed to distinguish genuine 100-400% breakout opportunities from pump-and-dump scenarios
- **Approach**: Each targets a different aspect of catalyst quality and sustainability
- **Model**: All experimental prompts use Sonnet-3.5 for fair comparison

"""
        
        if buy_high_precision_scores:
            best_prompt, best_score = buy_high_precision_scores[0]
            best_display = {
                'sonnet35': 'Sonnet-3.5 (Baseline)',
                'sonnet4': 'Sonnet-4',
                'prompt_4a': '4A: Breakout vs Pump-Dump',
                'prompt_4b': '4B: Catalyst Strength', 
                'prompt_4c': '4C: Institutional Appeal',
                'prompt_4d': '4D: Timing & Urgency',
                'prompt_4e': '4E: Concrete vs Speculative'
            }.get(best_prompt, best_prompt)
            
            md_content += f"""### Recommendations
1. **Best Overall Performer**: {best_display} with {best_score:.1%} BUY+High precision
2. **Production Implementation**: Consider adopting the top-performing approach for live trading
3. **Further Testing**: Validate top performers on larger datasets and different market conditions
"""
        
        return md_content

    def outcome_to_expected_action(self, outcome: str) -> str:
        """Convert outcome type to expected trading action"""
        mapping = {
            'TRUE_BULLISH': 'BUY',
            'FALSE_PUMP': 'HOLD',  # Should avoid BUY for false pumps
            'NEUTRAL': 'HOLD'
        }
        return mapping.get(outcome, 'HOLD')

    async def save_results(self, results: List[SentimentResult], metrics: Dict[str, Any], pnl_results: Optional[Dict[str, Any]] = None):
        """Save comprehensive test results to files"""
        os.makedirs('tests/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare results for JSON serialization
        results_data = {
            'test_timestamp': timestamp,
            'test_type': 'comprehensive_sonnet_experimental_prompts_comparison',
            'data_split': 'proper_train_test_split',
            'test_articles_from': 'News.rag_test_set',
            'models_tested': {
                'sonnet35_model': self.sonnet35_model,
                'sonnet4_model': self.sonnet4_model,
                'experimental_prompts': ['4A_breakout_vs_pump', '4B_catalyst_strength', '4C_institutional_appeal', '4D_timing_urgency', '4E_concrete_vs_speculative']
            },
            'metrics': metrics,
            'pnl_results': pnl_results,
            'detailed_results': []
        }
        
        for result in results:
            results_data['detailed_results'].append({
                'ticker': result.ticker,
                'headline': result.headline,
                'actual_outcome': result.actual_outcome,
                # Original models
                'sonnet35_sentiment': result.sonnet35_sentiment,
                'sonnet35_confidence': result.sonnet35_confidence,
                'sonnet4_sentiment': result.sonnet4_sentiment,
                'sonnet4_confidence': result.sonnet4_confidence,
                # Experimental prompts
                'prompt_4a_sentiment': result.prompt_4a_sentiment,
                'prompt_4a_confidence': result.prompt_4a_confidence,
                'prompt_4b_sentiment': result.prompt_4b_sentiment,
                'prompt_4b_confidence': result.prompt_4b_confidence,
                'prompt_4c_sentiment': result.prompt_4c_sentiment,
                'prompt_4c_confidence': result.prompt_4c_confidence,
                'prompt_4d_sentiment': result.prompt_4d_sentiment,
                'prompt_4d_confidence': result.prompt_4d_confidence,
                'prompt_4e_sentiment': result.prompt_4e_sentiment,
                'prompt_4e_confidence': result.prompt_4e_confidence,
                # Timing data
                'analysis_time_sonnet35': result.analysis_time_sonnet35,
                'analysis_time_sonnet4': result.analysis_time_sonnet4,
                'analysis_time_4a': result.analysis_time_4a,
                'analysis_time_4b': result.analysis_time_4b,
                'analysis_time_4c': result.analysis_time_4c,
                'analysis_time_4d': result.analysis_time_4d,
                'analysis_time_4e': result.analysis_time_4e,
                # Add PnL fields to JSON
                'published_utc': result.published_utc.isoformat() if result.published_utc else None,
                'sonnet35_pnl': result.sonnet35_pnl,
                'sonnet4_pnl': result.sonnet4_pnl,
                'sonnet35_entry_price': result.sonnet35_entry_price,
                'sonnet35_exit_price': result.sonnet35_exit_price,
                'sonnet4_entry_price': result.sonnet4_entry_price,
                'sonnet4_exit_price': result.sonnet4_exit_price,
                # Add detailed PnL tracking fields
                'sonnet35_position_size': result.sonnet35_position_size,
                'sonnet4_position_size': result.sonnet4_position_size,
                'sonnet35_investment': result.sonnet35_investment,
                'sonnet4_investment': result.sonnet4_investment,
                'sonnet35_return_pct': result.sonnet35_return_pct,
                'sonnet4_return_pct': result.sonnet4_return_pct,
                'price_bracket': result.price_bracket,
                'publication_hour': result.publication_hour
            })
        
        # Save JSON results
        with open(f'tests/results/comprehensive_comparison_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Generate and save detailed markdown analysis
        detailed_analysis = self.generate_detailed_analysis(results, metrics)
        
        with open(f'tests/results/comprehensive_detailed_analysis_{timestamp}.md', 'w') as f:
            f.write(detailed_analysis)
        
        logger.info(f"📁 Comprehensive comparison results saved to tests/results/comprehensive_comparison_results_{timestamp}.json")
        logger.info(f"📊 Detailed analysis saved to tests/results/comprehensive_detailed_analysis_{timestamp}.md")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        if self.ch_manager:
            self.ch_manager.close()

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive Sonnet + Experimental Prompts Sentiment Analysis Comparison')
    parser.add_argument('--sample-size', type=int, default=30, help='Number of test articles to analyze')
    parser.add_argument('--test-mode', choices=['sonnet35', 'sonnet4', 'parallel', 'experimental', '4d', 'all'], default='all', 
                        help='Test mode: sonnet35 only, sonnet4 only, parallel (both sonnets), experimental (4A-4E), 4d (only prompt 4D), or all')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save test results')
    parser.add_argument('--buy-high-threshold', type=float, default=0.8, help='Confidence threshold for BUY+high (default: 0.8)')
    parser.add_argument('--buy-medium-threshold', type=float, default=0.5, help='Confidence threshold for BUY+medium (default: 0.5)')
    parser.add_argument('--use-full-dataset', action='store_true', help='Use entire dataset (both training and test sets) instead of just test set')
    
    args = parser.parse_args()
    
    tester = SonnetComparisonTester(buy_high_threshold=args.buy_high_threshold, buy_medium_threshold=args.buy_medium_threshold)
    
    try:
        # Initialize test framework
        await tester.initialize()
        
        # Get articles - either from test set only or full dataset
        if args.use_full_dataset:
            logger.info("🌐 Using FULL DATASET (training + test sets) for evaluation")
            test_articles = await tester.get_all_articles(args.sample_size if args.sample_size != 30 else None)
        else:
            logger.info("📊 Using TEST SET ONLY for evaluation")
            test_articles = await tester.get_test_articles(args.sample_size)
            
        if not test_articles:
            logger.error("No articles found!")
            return
        
        # Run comprehensive comparison test
        results = await tester.run_comparison_test(test_articles, args.test_mode)
        
        # Calculate metrics
        metrics = tester.calculate_metrics(results)
        
        # Log summary for all tested prompts/models
        logger.info("📊 Comprehensive Test Results Summary:")
        
        if metrics.get('sonnet35_metrics'):
            s35 = metrics['sonnet35_metrics']
            logger.info(f"  🔍 Sonnet-3.5 (Baseline): {s35['accuracy']:.1%} accuracy")
            logger.info(f"    • BUY+high: {s35['buy_high_count']} signals, {s35['buy_high_precision']:.1%} precision, {s35['buy_high_recall']:.1%} recall")
            logger.info(f"    • BUY+medium: {s35['buy_medium_count']} signals, {s35['buy_medium_precision']:.1%} precision")
            logger.info(f"    • BUY (any): {s35['buy_any_recall']:.1%} recall of TRUE_BULLISH articles")
        
        if metrics.get('sonnet4_metrics'):
            s4 = metrics['sonnet4_metrics']
            logger.info(f"  🧠 Sonnet-4: {s4['accuracy']:.1%} accuracy")
            logger.info(f"    • BUY+high: {s4['buy_high_count']} signals, {s4['buy_high_precision']:.1%} precision, {s4['buy_high_recall']:.1%} recall")
            logger.info(f"    • BUY+medium: {s4['buy_medium_count']} signals, {s4['buy_medium_precision']:.1%} precision")
            logger.info(f"    • BUY (any): {s4['buy_any_recall']:.1%} recall of TRUE_BULLISH articles")
        
        # Log experimental prompts
        experimental_prompts = [
            ('prompt_4a_metrics', '🎯 4A: Breakout vs Pump-Dump'),
            ('prompt_4b_metrics', '💪 4B: Catalyst Strength'),
            ('prompt_4c_metrics', '🏛️ 4C: Institutional Appeal'),
            ('prompt_4d_metrics', '⏰ 4D: Timing & Urgency'),
            ('prompt_4e_metrics', '📋 4E: Concrete vs Speculative')
        ]
        
        for metric_key, display_name in experimental_prompts:
            if metrics.get(metric_key):
                exp = metrics[metric_key]
                logger.info(f"  {display_name}: {exp['accuracy']:.1%} accuracy")
                logger.info(f"    • BUY+high: {exp['buy_high_count']} signals, {exp['buy_high_precision']:.1%} precision, {exp['buy_high_recall']:.1%} recall")
                logger.info(f"    • BUY+medium: {exp['buy_medium_count']} signals, {exp['buy_medium_precision']:.1%} precision")
                logger.info(f"    • BUY (any): {exp['buy_any_recall']:.1%} recall of TRUE_BULLISH articles")
        
        # Performance comparison summary
        if metrics.get('performance_comparison'):
            logger.info("📈 Performance vs Sonnet-3.5 Baseline:")
            for prompt_name, comparison in metrics['performance_comparison'].items():
                prompt_display = {
                    'sonnet4': 'Sonnet-4',
                    'prompt_4a': '4A: Breakout vs Pump',
                    'prompt_4b': '4B: Catalyst Strength', 
                    'prompt_4c': '4C: Institutional Appeal',
                    'prompt_4d': '4D: Timing & Urgency',
                    'prompt_4e': '4E: Concrete vs Speculative'
                }.get(prompt_name, prompt_name)
                
                logger.info(f"  {prompt_display}: {comparison['accuracy_improvement']:+.1%} accuracy, {comparison['buy_high_precision_improvement']:+.1%} BUY+high precision")
        
        # Find and highlight best performer
        all_prompts = ['sonnet35', 'sonnet4', 'prompt_4a', 'prompt_4b', 'prompt_4c', 'prompt_4d', 'prompt_4e']
        best_performers = []
        
        for prompt in all_prompts:
            prompt_metrics = metrics.get(f'{prompt}_metrics')
            if prompt_metrics:
                best_performers.append((prompt, prompt_metrics['buy_high_precision']))
        
        if best_performers:
            best_performers.sort(key=lambda x: x[1], reverse=True)
            best_prompt, best_score = best_performers[0]
            
            best_display = {
                'sonnet35': 'Sonnet-3.5 (Baseline)',
                'sonnet4': 'Sonnet-4',
                'prompt_4a': '4A: Breakout vs Pump-Dump',
                'prompt_4b': '4B: Catalyst Strength', 
                'prompt_4c': '4C: Institutional Appeal',
                'prompt_4d': '4D: Timing & Urgency',
                'prompt_4e': '4E: Concrete vs Speculative'
            }.get(best_prompt, best_prompt)
            
            logger.info(f"🏆 BEST PERFORMER: {best_display} with {best_score:.1%} BUY+High precision")
        
        if metrics.get('recall_metrics'):
            recall = metrics['recall_metrics']
            logger.info(f"  📋 Recall Analysis: {recall['total_true_bullish_articles']} TRUE_BULLISH articles in test set")
        
        # Save results if requested
        if args.save_results:
            await tester.save_results(results, metrics)
        
        logger.info("✅ Comprehensive model + experimental prompts comparison test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())