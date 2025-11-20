#!/usr/bin/env python3
"""
StockAnalysis.com Statistics Scraper using Crawl4AI
Scrapes detailed stock statistics for tickers from the float_list table
"""

import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlResult
from bs4 import BeautifulSoup

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clickhouse_setup import ClickHouseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnalysisScraper:
    def __init__(self):
        self.clickhouse_manager = None
        self.crawler = None
        self.base_url = "https://stockanalysis.com/stocks/{}/statistics/"
        
        # Performance tracking
        self.stats = {
            'tickers_processed': 0,
            'tickers_successful': 0,
            'tickers_failed': 0,
            'errors': 0
        }

    async def initialize(self):
        """Initialize the scraper with Crawl4AI and ClickHouse connection"""
        logger.info("🚀 Initializing StockAnalysis.com scraper...")
        
        # Connect to ClickHouse
        self.clickhouse_manager = ClickHouseManager()
        self.clickhouse_manager.connect()
        
        # Initialize Crawl4AI AsyncWebCrawler with efficient settings
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.crawler = AsyncWebCrawler(
                    verbose=False,
                    headless=True,
                    browser_type="chromium",
                    max_idle_time=30000,
                    keep_alive=True,
                    max_memory_usage=512,
                    max_concurrent_sessions=2,
                    delay_between_requests=1.0,  # Be respectful to the website
                    extra_args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-software-rasterizer",
                        "--disable-background-timer-throttling",
                        "--disable-backgrounding-occluded-windows",
                        "--disable-renderer-backgrounding",
                        "--disable-features=TranslateUI",
                        "--disable-ipc-flooding-protection",
                        "--memory-pressure-off",
                        "--max_old_space_size=256",
                        "--aggressive-cache-discard",
                        "--disable-extensions",
                        "--disable-plugins",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor",
                        "--disable-background-networking",
                        "--disable-sync",
                        "--disable-default-apps",
                        "--disable-component-update",
                        "--disable-hang-monitor",
                        "--disable-prompt-on-repost",
                        "--disable-client-side-phishing-detection",
                        "--disable-component-extensions-with-background-pages",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-popup-blocking",
                        "--disable-notifications",
                    ]
                )
                await self.crawler.start()
                logger.info(f"✅ Crawl4AI browser started successfully (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.warning(f"❌ Failed to start Crawl4AI browser (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize Crawl4AI after {max_retries} attempts")
                await asyncio.sleep(2)
        
        logger.info("✅ StockAnalysis scraper initialized successfully")

    def _extract_from_json_data(self, data_section: str, stats: Dict[str, Any]):
        """Extract statistics from embedded JSON data structure"""
        # Map of JSON field IDs to our database column names
        field_mapping = {
            # Valuation
            'marketcap': 'market_cap',
            'enterpriseValue': 'enterprise_value',
            # Dates (strings)
            'earningsdate': 'earnings_date',
            'exdivdate': 'ex_dividend_date',
            # Stock Price Statistics
            'beta': 'beta_5y',
            'ch1y': '52_week_change',
            'sma50': '50_day_ma',
            'sma200': '200_day_ma',
            'rsi': 'relative_strength_index',
            'averageVolume': 'average_volume_20d',
            # Share Statistics
            'sharesOutClass': 'current_share_class',
            'sharesout': 'shares_outstanding',
            'sharesgrowthyoy': 'shares_change_yoy',
            'sharesgrowthqoq': 'shares_change_qoq',
            'sharesInsiders': 'percent_insiders',
            'sharesInstitutions': 'percent_institutions',
            'float': 'shares_float',
            # Short Selling
            'shortInterest': 'short_interest',
            'shortPriorMonth': 'short_previous_month',
            'shortShares': 'short_percent_shares_out',
            'shortFloat': 'short_percent_float',
            'shortRatio': 'short_ratio',
            # Valuation Ratios
            'pe': 'pe_ratio',
            'peForward': 'forward_pe',
            'ps': 'ps_ratio',
            'psForward': 'forward_ps',
            'pb': 'pb_ratio',
            'ptbvRatio': 'p_tbv_ratio',
            'pfcf': 'p_fcf_ratio',
            'pocf': 'p_ocf_ratio',
            'pegRatio': 'peg_ratio',
            # Enterprise Valuation
            'evEarnings': 'ev_to_earnings',
            'evSales': 'ev_to_sales',
            'evEbitda': 'ev_to_ebitda',
            'evEbit': 'ev_to_ebit',
            'evFcf': 'ev_to_fcf',
            # Financial Position
            'currentRatio': 'current_ratio',
            'quickRatio': 'quick_ratio',
            'debtEquity': 'debt_to_equity',
            'debtEbitda': 'debt_to_ebitda',
            'debtFcf': 'debt_to_fcf',
            'interestCoverage': 'interest_coverage',
            # Financial Efficiency
            'roe': 'return_on_equity',
            'roa': 'return_on_assets',
            'roic': 'return_on_invested_capital',
            'roce': 'return_on_capital_employed',
            'revPerEmployee': 'revenue_per_employee',
            'profitPerEmployee': 'profits_per_employee',
            'employees': 'employee_count',
            'assetturnover': 'asset_turnover',
            'inventoryturnover': 'inventory_turnover',
            # Taxes
            'taxexp': 'income_tax',
            'taxrate': 'effective_tax_rate',
            # Income Statement
            'revenue': 'revenue',
            'gp': 'gross_profit',
            'opinc': 'operating_income',
            'pretax': 'pretax_income',
            'netinc': 'net_income',
            'ebitda': 'ebitda',
            'ebit': 'ebit',
            'eps': 'earnings_per_share',
            # Balance Sheet
            'totalcash': 'cash_and_equivalents',
            'debt': 'total_debt',
            'netcash': 'net_cash',
            'netcashpershare': 'net_cash_per_share',
            'equity': 'equity_book_value',
            'bvps': 'book_value_per_share',
            'workingcapital': 'working_capital',
            # Cash Flow
            'ncfo': 'operating_cash_flow',
            'capex': 'capital_expenditures',
            'fcf': 'free_cash_flow',
            'fcfps': 'fcf_per_share',
            # Margins
            'grossMargin': 'gross_margin',
            'operatingMargin': 'operating_margin',
            'pretaxMargin': 'pretax_margin',
            'profitMargin': 'profit_margin',
            'ebitdaMargin': 'ebitda_margin',
            'ebitMargin': 'ebit_margin',
            'fcfMargin': 'fcf_margin',
            # Dividends & Yields
            'dps': 'dividend_per_share',
            'dividendYield': 'dividend_yield',
            'dividendGrowth': 'dividend_growth_yoy',
            'dividendGrowthYears': 'years_dividend_growth',
            'payoutRatio': 'payout_ratio',
            'buybackYield': 'buyback_yield',
            'totalReturn': 'shareholder_yield',
            'earningsYield': 'earnings_yield',
            'fcfYield': 'fcf_yield',
            # Stock Splits (strings)
            'lastSplitDate': 'last_split_date',
            'lastSplitType': 'split_type',
            'splitRatio': 'split_ratio',
            # Scores
            'zScore': 'altman_z_score',
            'fScore': 'piotroski_f_score',
        }
        
        # Extract data for each mapped field
        for json_id, db_column in field_mapping.items():
            # Pattern to find: id:"fieldname",title:"...",value:"...",hover:"..."
            pattern = rf'id:"{json_id}".*?value:"([^"]*)".*?hover:"([^"]*)"'
            match = re.search(pattern, data_section, re.DOTALL)
            
            if match:
                value_str = match.group(1) if match.group(1) != 'n/a' else match.group(2)
                
                # String fields (dates, split info) - use empty string for missing values
                if db_column in ['earnings_date', 'ex_dividend_date', 'last_split_date', 'split_type', 'split_ratio']:
                    stats[db_column] = value_str if value_str and value_str != 'n/a' else ''
                else:
                    # Numeric fields - use hover value as it has full precision
                    hover_str = match.group(2)
                    parsed_value = self.parse_float_value(hover_str if hover_str != 'n/a' else value_str)
                    stats[db_column] = parsed_value
                    
                logger.debug(f"Extracted {db_column}: {value_str} -> {stats[db_column]}")

    def parse_float_value(self, value_str: str) -> Optional[float]:
        """Parse string values to float, handling various formats like 35.23M, 1.5B, etc."""
        if not value_str or value_str.lower() in ['n/a', '-', '', 'none']:
            return None
        
        try:
            # Remove common characters
            cleaned = value_str.strip().replace(',', '').replace('$', '').replace('%', '')
            
            # Handle magnitude suffixes (M = million, B = billion, T = trillion)
            multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
            
            for suffix, multiplier in multipliers.items():
                if cleaned.upper().endswith(suffix):
                    number = float(cleaned[:-1])
                    return number * multiplier
            
            # Try direct float conversion
            return float(cleaned)
        except (ValueError, AttributeError):
            logger.debug(f"Could not parse float value: {value_str}")
            return None

    def extract_statistics(self, html_content: str) -> Dict[str, Any]:
        """Extract all statistics from the StockAnalysis.com statistics page"""
        stats = {}
        
        # BETTER APPROACH: Extract structured JSON data embedded in the page
        # StockAnalysis.com embeds all data in a JavaScript object
        try:
            # Find the data object in the script tag
            json_match = re.search(r'data:\s*\[.*?\{type:"data",data:\{valuation:(\{.*?\}),dates:', html_content, re.DOTALL)
            if json_match:
                # Extract the full data structure
                data_start = html_content.find('{type:"data",data:{valuation:')
                if data_start > 0:
                    # Find the end of this data object (before the next one)
                    data_section = html_content[data_start:data_start+50000]  # reasonable chunk
                    
                    # Try to extract the structured data
                    # The data is in format: {valuation:{...}, dates:{...}, shares:{...}, ...}
                    self._extract_from_json_data(data_section, stats)
                    
                    # Also extract 52-week high/low from quote data
                    quote_match = re.search(r'quote:\{[^}]*h52:([\d.]+)[^}]*l52:([\d.]+)', html_content)
                    if quote_match:
                        stats['52_week_high'] = self.parse_float_value(quote_match.group(1))
                        stats['52_week_low'] = self.parse_float_value(quote_match.group(2))
                        logger.debug(f"Extracted 52-week high/low from quote data")
                    
                    if stats:
                        logger.info(f"✅ Successfully extracted {len([v for v in stats.values() if v is not None])} fields from embedded JSON")
                        return stats
        except Exception as e:
            logger.debug(f"Could not extract from JSON: {e}")
        
        # FALLBACK: Parse rendered HTML if JSON extraction fails
        soup = BeautifulSoup(html_content, 'html.parser')
        all_text = soup.get_text()
        logger.info("Using fallback HTML parsing method")
        
        # Comprehensive statistics patterns matching ALL fields from the page
        stat_patterns = {
            # Total Valuation
            'market_cap': r'Market Cap[:\s]*\$?([\d.,]+[BMKT]?)',
            'enterprise_value': r'Enterprise Value[:\s]*\$?([\d.,]+[BMKT]?)',
            
            # Important Dates
            'earnings_date': r'Earnings Date[:\s]*([\w\s,]+\d{4})',
            'ex_dividend_date': r'Ex-Dividend Date[:\s]*([\w\s,/]+|\d+)',
            
            # Stock Price Statistics
            'beta_5y': r'Beta[:\s]*\(5Y\)[:\s]*([\d.]+)',
            '52_week_high': r'52-Week High[:\s]*\$?([\d.,]+)',
            '52_week_low': r'52-Week Low[:\s]*\$?([\d.,]+)',
            '52_week_change': r'52-Week.*?Change[:\s]*([-+]?[\d.]+)%',
            '50_day_ma': r'50-Day Moving Average[:\s]*\$?([\d.,]+)',
            '200_day_ma': r'200-Day Moving Average[:\s]*\$?([\d.,]+)',
            'relative_strength_index': r'Relative Strength Index[:\s]*\(RSI\)[:\s]*([\d.]+)',
            'average_volume_20d': r'Average Volume[:\s]*\(20 Days\)[:\s]*([\d,]+)',
            
            # Share Statistics
            'current_share_class': r'Current Share Class[:\s]*([\d.,]+[BMK]?)',
            'shares_outstanding': r'Shares Outstanding[:\s]*([\d.,]+[BMK]?)',
            'shares_change_yoy': r'Shares Change[:\s]*\(YoY\)[:\s]*\+?([-+]?[\d.]+)%',
            'shares_change_qoq': r'Shares Change[:\s]*\(QoQ\)[:\s]*\+?([-+]?[\d.]+)%',
            'percent_insiders': r'Owned by Insiders[:\s]*\(%\)[:\s]*([\d.]+)%',
            'percent_institutions': r'Owned by Institutions[:\s]*\(%\)[:\s]*([\d.]+)%',
            'shares_float': r'(?:^|\s)Float[:\s]*([\d.,]+[BMK]?)',
            
            # Short Selling Information
            'short_interest': r'Short Interest[:\s]*([\d.,]+[BMK]?)',
            'short_previous_month': r'Short Previous Month[:\s]*([\d.,]+[BMK]?)',
            'short_percent_shares_out': r'Short % of Shares Out[:\s]*([\d.]+)%',
            'short_percent_float': r'Short % of Float[:\s]*([\d.]+)%',
            'short_ratio': r'Short Ratio[:\s]*\(days to cover\)[:\s]*([\d.]+)',
            
            # Valuation Ratios
            'pe_ratio': r'(?:^|\s)PE Ratio[:\s]*([\d.]+)',
            'forward_pe': r'Forward PE[:\s]*([\d.]+)',
            'ps_ratio': r'(?:^|\s)PS Ratio[:\s]*([\d.]+)',
            'forward_ps': r'Forward PS[:\s]*([\d.]+)',
            'pb_ratio': r'(?:^|\s)PB Ratio[:\s]*([\d.]+)',
            'p_tbv_ratio': r'P/TBV Ratio[:\s]*([\d.]+)',
            'p_fcf_ratio': r'P/FCF Ratio[:\s]*([\d.]+)',
            'p_ocf_ratio': r'P/OCF Ratio[:\s]*([\d.]+)',
            'peg_ratio': r'PEG Ratio[:\s]*([\d.]+)',
            
            # Enterprise Valuation
            'ev_to_earnings': r'EV\s*/\s*Earnings[:\s]*([\d.]+)',
            'ev_to_sales': r'EV\s*/\s*Sales[:\s]*([\d.]+)',
            'ev_to_ebitda': r'EV\s*/\s*EBITDA[:\s]*([\d.]+)',
            'ev_to_ebit': r'EV\s*/\s*EBIT[:\s]*([\d.]+)',
            'ev_to_fcf': r'EV\s*/\s*FCF[:\s]*([\d.]+)',
            
            # Financial Position
            'current_ratio': r'Current Ratio[:\s]*([\d.]+)',
            'quick_ratio': r'Quick Ratio[:\s]*([\d.]+)',
            'debt_to_equity': r'Debt\s*/\s*Equity[:\s]*([\d.]+)',
            'debt_to_ebitda': r'Debt\s*/\s*EBITDA[:\s]*([\d.]+)',
            'debt_to_fcf': r'Debt\s*/\s*FCF[:\s]*([\d.]+)',
            'interest_coverage': r'Interest Coverage[:\s]*([-+]?[\d.]+)',
            
            # Financial Efficiency
            'return_on_equity': r'Return on Equity[:\s]*\(ROE\)[:\s]*([-+]?[\d.]+)%',
            'return_on_assets': r'Return on Assets[:\s]*\(ROA\)[:\s]*([-+]?[\d.]+)%',
            'return_on_invested_capital': r'Return on Invested Capital[:\s]*\(ROIC\)[:\s]*([-+]?[\d.]+)%',
            'return_on_capital_employed': r'Return on Capital Employed[:\s]*\(ROCE\)[:\s]*([-+]?[\d.]+)%',
            'revenue_per_employee': r'Revenue Per Employee[:\s]*\$?([\d.,]+[BMK]?)',
            'profits_per_employee': r'Profits Per Employee[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'employee_count': r'Employee Count[:\s]*([\d,]+)',
            'asset_turnover': r'Asset Turnover[:\s]*([\d.]+)',
            'inventory_turnover': r'Inventory Turnover[:\s]*([\d.]+)',
            
            # Taxes
            'income_tax': r'Income Tax[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'effective_tax_rate': r'Effective Tax Rate[:\s]*([-+]?[\d.]+)%',
            
            # Income Statement
            'revenue': r'(?:^|\s)Revenue[:\s]*\$?([\d.,]+[BMK]?)',
            'gross_profit': r'Gross Profit[:\s]*\$?([\d.,]+[BMK]?)',
            'operating_income': r'Operating Income[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'pretax_income': r'Pretax Income[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'net_income': r'Net Income[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'ebitda': r'(?:^|\s)EBITDA[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'ebit': r'(?:^|\s)EBIT[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'earnings_per_share': r'Earnings Per Share[:\s]*\(EPS\)[:\s]*\$?([-+]?[\d.]+)',
            
            # Balance Sheet
            'cash_and_equivalents': r'Cash & Cash Equivalents[:\s]*\$?([\d.,]+[BMK]?)',
            'total_debt': r'Total Debt[:\s]*\$?([\d.,]+[BMK]?)',
            'net_cash': r'Net Cash[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'net_cash_per_share': r'Net Cash Per Share[:\s]*\$?([-+]?[\d.]+)',
            'equity_book_value': r'Equity[:\s]*\(Book Value\)[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'book_value_per_share': r'Book Value Per Share[:\s]*\$?([-+]?[\d.]+)',
            'working_capital': r'Working Capital[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            
            # Cash Flow
            'operating_cash_flow': r'Operating Cash Flow[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'capital_expenditures': r'Capital Expenditures[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'free_cash_flow': r'Free Cash Flow[:\s]*\$?([-+]?[\d.,]+[BMK]?)',
            'fcf_per_share': r'FCF Per Share[:\s]*\$?([-+]?[\d.]+)',
            
            # Margins
            'gross_margin': r'Gross Margin[:\s]*([-+]?[\d.]+)%',
            'operating_margin': r'Operating Margin[:\s]*([-+]?[\d.]+)%',
            'pretax_margin': r'Pretax Margin[:\s]*([-+]?[\d.]+)%',
            'profit_margin': r'Profit Margin[:\s]*([-+]?[\d.]+)%',
            'ebitda_margin': r'EBITDA Margin[:\s]*([-+]?[\d.]+)%',
            'ebit_margin': r'EBIT Margin[:\s]*([-+]?[\d.]+)%',
            'fcf_margin': r'FCF Margin[:\s]*([-+]?[\d.]+)%',
            
            # Dividends & Yields
            'dividend_per_share': r'Dividend Per Share[:\s]*\$?([\d.]+)',
            'dividend_yield': r'Dividend Yield[:\s]*([\d.]+)%',
            'dividend_growth_yoy': r'Dividend Growth[:\s]*\(YoY\)[:\s]*([-+]?[\d.]+)%',
            'years_dividend_growth': r'Years of Dividend Growth[:\s]*([\d]+)',
            'payout_ratio': r'Payout Ratio[:\s]*([\d.]+)%',
            'buyback_yield': r'Buyback Yield[:\s]*([-+]?[\d.]+)%',
            'shareholder_yield': r'Shareholder Yield[:\s]*([-+]?[\d.]+)%',
            'earnings_yield': r'Earnings Yield[:\s]*([-+]?[\d.]+)%',
            'fcf_yield': r'FCF Yield[:\s]*([-+]?[\d.]+)%',
            
            # Stock Splits
            'last_split_date': r'Last Split Date[:\s]*([\w\s,]+\d{4})',
            'split_type': r'Split Type[:\s]*([\w\s]+)',
            'split_ratio': r'Split Ratio[:\s]*([\d:]+)',
            
            # Scores
            'altman_z_score': r'Altman Z-Score[:\s]*([\d.]+)',
            'piotroski_f_score': r'Piotroski F-Score[:\s]*([\d]+)',
        }
        
        # Extract values using regex patterns
        for key, pattern in stat_patterns.items():
            match = re.search(pattern, all_text, re.IGNORECASE | re.DOTALL)
            if match:
                value_str = match.group(1).strip()
                # Special handling for string fields - use empty string for missing values
                if key in ['earnings_date', 'ex_dividend_date', 'last_split_date', 'split_type', 'split_ratio']:
                    stats[key] = value_str if value_str and value_str.lower() != 'n/a' else ''
                else:
                    stats[key] = self.parse_float_value(value_str)
                logger.debug(f"Extracted {key}: {value_str} -> {stats[key]}")
            else:
                # Use empty string for string fields, None for numeric fields
                if key in ['earnings_date', 'ex_dividend_date', 'last_split_date', 'split_type', 'split_ratio']:
                    stats[key] = ''
                else:
                    stats[key] = None
        
        return stats

    async     def scrape_ticker_statistics(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Scrape statistics for a single ticker"""
        url = self.base_url.format(ticker.lower())
        
        try:
            logger.info(f"🔍 Scraping statistics for {ticker} from {url}")
            
            result: CrawlResult = await self.crawler.arun(
                url=url,
                wait_for="css:script",  # Just wait for scripts to load
                delay_before_return_html=0.5,  # Quick load - we're extracting from embedded JSON
                timeout=15
            )
            
            if not result.success or not result.html:
                logger.error(f"❌ Failed to scrape {ticker}: {result.error_message}")
                return None
            
            # Extract statistics from HTML
            stats = self.extract_statistics(result.html)
            
            # Add metadata
            stats['ticker'] = ticker
            stats['scraped_at'] = datetime.now()
            stats['source_url'] = url
            
            # Ensure all fields exist with proper defaults (for database insertion)
            # String fields get empty string, numeric fields get None
            string_fields = ['earnings_date', 'ex_dividend_date', 'last_split_date', 'split_type', 'split_ratio']
            all_fields = [
                'market_cap', 'enterprise_value', 'beta_5y', '52_week_high', '52_week_low', 
                '52_week_change', '50_day_ma', '200_day_ma', 'relative_strength_index', 
                'average_volume_20d', 'current_share_class', 'shares_outstanding', 
                'shares_change_yoy', 'shares_change_qoq', 'percent_insiders', 
                'percent_institutions', 'shares_float', 'short_interest', 'short_previous_month',
                'short_percent_shares_out', 'short_percent_float', 'short_ratio', 'pe_ratio',
                'forward_pe', 'ps_ratio', 'forward_ps', 'pb_ratio', 'p_tbv_ratio', 'p_fcf_ratio',
                'p_ocf_ratio', 'peg_ratio', 'ev_to_earnings', 'ev_to_sales', 'ev_to_ebitda',
                'ev_to_ebit', 'ev_to_fcf', 'current_ratio', 'quick_ratio', 'debt_to_equity',
                'debt_to_ebitda', 'debt_to_fcf', 'interest_coverage', 'return_on_equity',
                'return_on_assets', 'return_on_invested_capital', 'return_on_capital_employed',
                'revenue_per_employee', 'profits_per_employee', 'employee_count', 'asset_turnover',
                'inventory_turnover', 'income_tax', 'effective_tax_rate', 'revenue', 'gross_profit',
                'operating_income', 'pretax_income', 'net_income', 'ebitda', 'ebit',
                'earnings_per_share', 'cash_and_equivalents', 'total_debt', 'net_cash',
                'net_cash_per_share', 'equity_book_value', 'book_value_per_share', 'working_capital',
                'operating_cash_flow', 'capital_expenditures', 'free_cash_flow', 'fcf_per_share',
                'gross_margin', 'operating_margin', 'pretax_margin', 'profit_margin', 'ebitda_margin',
                'ebit_margin', 'fcf_margin', 'dividend_per_share', 'dividend_yield',
                'dividend_growth_yoy', 'years_dividend_growth', 'payout_ratio', 'buyback_yield',
                'shareholder_yield', 'earnings_yield', 'fcf_yield', 'altman_z_score', 'piotroski_f_score'
            ] + string_fields
            
            for field in all_fields:
                if field not in stats:
                    stats[field] = '' if field in string_fields else None
            
            # Count how many fields we successfully extracted
            non_null_fields = sum(1 for k, v in stats.items() if k in all_fields and v is not None and v != '')
            logger.info(f"✅ Extracted {non_null_fields} statistics for {ticker}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error scraping {ticker}: {e}")
            return None

    async def scrape_all_tickers(self, limit: Optional[int] = None):
        """Scrape statistics for all tickers in float_list table"""
        try:
            # Get ticker list from float_list table
            tickers = self.clickhouse_manager.get_active_tickers()
            
            if not tickers:
                logger.error("❌ No tickers found in float_list table")
                return
            
            logger.info(f"📊 Found {len(tickers)} tickers to process")
            
            # Apply limit if specified
            if limit:
                tickers = tickers[:limit]
                logger.info(f"🎯 Processing limited set of {len(tickers)} tickers")
            
            # Process tickers in batches to avoid overwhelming the website
            batch_size = 10
            all_results = []
            
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}")
                
                # Process batch sequentially (be respectful to the website)
                for ticker in batch:
                    self.stats['tickers_processed'] += 1
                    
                    stats = await self.scrape_ticker_statistics(ticker)
                    
                    if stats:
                        all_results.append(stats)
                        self.stats['tickers_successful'] += 1
                        
                        # Insert immediately to avoid losing data
                        try:
                            self.clickhouse_manager.insert_float_list_detailed([stats])
                            logger.info(f"✅ Inserted statistics for {ticker} into float_list_detailed")
                        except Exception as e:
                            logger.error(f"❌ Failed to insert {ticker} into float_list_detailed: {e}")
                            self.stats['errors'] += 1
                        
                        # Also insert into deduplicated table
                        try:
                            self.clickhouse_manager.insert_float_list_detailed_dedup([stats])
                            logger.info(f"✅ Inserted statistics for {ticker} into float_list_detailed_dedup")
                        except Exception as e:
                            logger.error(f"❌ Failed to insert {ticker} into float_list_detailed_dedup: {e}")
                            self.stats['errors'] += 1
                    else:
                        self.stats['tickers_failed'] += 1
                    
                    # Small delay between requests to be respectful
                    await asyncio.sleep(1.5)
                
                # Longer delay between batches
                if i + batch_size < len(tickers):
                    logger.info(f"⏸️ Batch complete, waiting 5 seconds before next batch...")
                    await asyncio.sleep(5)
            
            # Print final statistics
            logger.info("=" * 80)
            logger.info("📊 SCRAPING SUMMARY:")
            logger.info(f"   Total Tickers Processed: {self.stats['tickers_processed']}")
            logger.info(f"   ✅ Successful: {self.stats['tickers_successful']}")
            logger.info(f"   ❌ Failed: {self.stats['tickers_failed']}")
            logger.info(f"   Errors: {self.stats['errors']}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error in scrape_all_tickers: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.crawler:
                await self.crawler.close()
                logger.info("Closed Crawl4AI crawler")
            
            if self.clickhouse_manager:
                self.clickhouse_manager.close()
                logger.info("Closed ClickHouse connection")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main function to run the StockAnalysis scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape stock statistics from StockAnalysis.com')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process (for testing)')
    parser.add_argument('--setup-table', action='store_true', help='Setup the float_list_detailed table')
    args = parser.parse_args()
    
    scraper = StockAnalysisScraper()
    
    try:
        # Initialize scraper
        await scraper.initialize()
        
        # Setup table if requested
        if args.setup_table:
            logger.info("🔧 Setting up float_list_detailed table...")
            scraper.clickhouse_manager.create_float_list_detailed_table()
            logger.info("✅ float_list_detailed table setup complete")
            
            logger.info("🔧 Setting up float_list_detailed_dedup table...")
            scraper.clickhouse_manager.create_float_list_detailed_dedup_table()
            logger.info("✅ float_list_detailed_dedup table setup complete")
            return
        
        # Start scraping
        logger.info("🚀 Starting StockAnalysis.com scraper...")
        await scraper.scrape_all_tickers(limit=args.limit)
        
    except KeyboardInterrupt:
        logger.info("Scraper stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await scraper.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

