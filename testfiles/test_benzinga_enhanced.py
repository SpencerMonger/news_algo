#!/usr/bin/env python3
"""
Enhanced Benzinga Pro Scraper Test
Tests login, newsfeed access, and article extraction with proper JavaScript handling
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenzingaProTester:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.base_url = "https://pro.benzinga.com"
        self.login_url = f"{self.base_url}/login"
        
        # Test URLs for different newsfeeds
        self.test_urls = [
            f"{self.base_url}/dashboard",
            f"{self.base_url}/dashboard/",
            f"{self.base_url}/news",
            f"{self.base_url}/feed",
            f"{self.base_url}/newsfeed",
        ]
        
        # Ticker detection patterns (from existing NewsHead system)
        self.ticker_patterns = [
            r'\b[A-Z]{1,5}\b(?:\s*\([A-Z]+:[A-Z]{1,5}\))?',  # Standard tickers
            r'\$[A-Z]{1,5}\b',  # Dollar prefixed tickers
            r'\b[A-Z]{1,5}:[A-Z]{1,5}\b',  # Exchange:Ticker format
        ]
        
        # Article extraction selectors - expanded for React apps
        self.article_selectors = [
            # Common news article selectors
            'article', '.article', '[data-testid*="article"]', '[data-cy*="article"]',
            '.news-item', '.story', '.headline', '.news-story', '.feed-item',
            '.card', '.news-card', '.story-card', '.content-card',
            
            # Benzinga-specific selectors (guessed from common patterns)
            '.bz-story', '.bz-article', '.bz-news-item', '.story-item',
            '[class*="story"]', '[class*="article"]', '[class*="news"]',
            '[class*="feed"]', '[class*="item"]', '[class*="card"]',
            
            # React component selectors
            '[data-component*="story"]', '[data-component*="article"]',
            '[data-component*="news"]', '[data-component*="feed"]',
            
            # Generic content containers
            '.content', '.main-content', '.feed-content', '.news-content',
            '#content', '#main', '#feed', '#news-feed', '#articles',
        ]

    async def test_connectivity(self) -> Dict:
        """Test basic connectivity to Benzinga Pro"""
        logger.info("Testing basic connectivity...")
        
        browser_config = BrowserConfig(
            headless=True,  # Run headless - no browser window
            browser_type="chromium",
            viewport_width=1920,
            viewport_height=1080,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                result = await crawler.arun(
                    url=self.login_url,
                    config=CrawlerRunConfig(
                        wait_for="css:input[type='email']",
                        delay_before_return_html=2.0,
                        page_timeout=30000,
                        js_code="""
                        // Check for anti-bot indicators
                        const indicators = {
                            captcha: !!document.querySelector('[class*="captcha"], [id*="captcha"], .g-recaptcha, .h-captcha'),
                            cloudflare: !!document.querySelector('.cf-browser-verification, .cf-checking-browser'),
                            blocked: document.title.toLowerCase().includes('blocked') || document.title.toLowerCase().includes('access denied'),
                            login_form: !!document.querySelector('input[type="email"], input[type="password"]'),
                            js_files: document.scripts.length
                        };
                        window.testResults = indicators;
                        """
                    )
                )
                
                # Save login page for debugging
                Path("debug_login_page.html").write_text(result.html)
                
                return {
                    "success": True,
                    "status_code": 200,
                    "title": result.metadata.get("title", ""),
                    "url": result.url,
                    "anti_bot_check": result.js_execution_result,
                    "html_length": len(result.html)
                }
                
            except Exception as e:
                logger.error(f"Connectivity test failed: {e}")
                return {"success": False, "error": str(e)}

    async def login(self, crawler) -> bool:
        """Perform login to Benzinga Pro with enhanced success detection"""
        logger.info("Attempting login...")
        
        try:
            # First, load the login page
            login_result = await crawler.arun(
                url=self.login_url,
                config=CrawlerRunConfig(
                    wait_for="css:input[type='email']",
                    delay_before_return_html=2.0,
                    page_timeout=30000,
                )
            )
            
            logger.info(f"Login page loaded: {login_result.url}")
            Path("simple_login_page.html").write_text(login_result.html)
            
            # Perform login with JavaScript form submission
            login_attempt = await crawler.arun(
                url=self.login_url,
                config=CrawlerRunConfig(
                    wait_for="css:input[type='email']",
                    delay_before_return_html=5.0,  # Wait longer for redirect
                    page_timeout=45000,
                    js_code=f"""
                    async function performLogin() {{
                        console.log('Starting login process...');
                        
                        // Find form elements
                        const emailInput = document.querySelector('input[type="email"]');
                        const passwordInput = document.querySelector('input[type="password"]');
                        const submitButton = document.querySelector('button[type="submit"], input[type="submit"], .login-btn, [class*="submit"], [class*="login"]');
                        
                        if (!emailInput || !passwordInput) {{
                            console.error('Login form elements not found');
                            return {{ success: false, error: 'Form elements not found' }};
                        }}
                        
                        // Fill in credentials
                        emailInput.value = '{self.email}';
                        passwordInput.value = '{self.password}';
                        
                        // Trigger input events (important for React forms)
                        emailInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        emailInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        passwordInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        passwordInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        
                        console.log('Credentials filled');
                        
                        // Wait a moment for React to process
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        
                        // Submit the form
                        if (submitButton) {{
                            console.log('Clicking submit button');
                            submitButton.click();
                        }} else {{
                            console.log('No submit button found, trying form submission');
                            const form = document.querySelector('form');
                            if (form) {{
                                form.submit();
                            }} else {{
                                // Try Enter key on password field
                                passwordInput.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', bubbles: true }}));
                            }}
                        }}
                        
                        // Wait for potential redirect/response
                        await new Promise(resolve => setTimeout(resolve, 3000));
                        
                        // Check for success indicators
                        const currentUrl = window.location.href;
                        const notOnLoginPage = !currentUrl.includes('/login');
                        
                        // Check for error messages (only if visible and has content)
                        const errorContainers = document.querySelectorAll('.error, .auth-error, .login-error, [class*="error"], .alert-danger, .text-danger');
                        let hasVisibleError = false;
                        
                        for (const container of errorContainers) {{
                            const style = window.getComputedStyle(container);
                            const isVisible = style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
                            const hasContent = container.textContent.trim().length > 0;
                            
                            if (isVisible && hasContent) {{
                                console.log('Found visible error:', container.textContent);
                                hasVisibleError = true;
                                break;
                            }}
                        }}
                        
                        const success = notOnLoginPage || !hasVisibleError;
                        
                        console.log('Login check results:', {{
                            currentUrl,
                            notOnLoginPage,
                            hasVisibleError,
                            success
                        }});
                        
                        return {{
                            success,
                            currentUrl,
                            notOnLoginPage,
                            hasVisibleError,
                            errorCount: errorContainers.length
                        }};
                    }}
                    
                    // Execute login
                    performLogin().then(result => {{
                        window.loginResult = result;
                        console.log('Login completed:', result);
                    }}).catch(error => {{
                        console.error('Login error:', error);
                        window.loginResult = {{ success: false, error: error.message }};
                    }});
                    """
                )
            )
            
            # Save post-login page for debugging
            Path("debug_post_login.html").write_text(login_attempt.html)
            
            # Check login result
            login_result = login_attempt.js_execution_result
            if login_result and login_result.get('success'):
                logger.info(f"✅ Login successful! Current URL: {login_result.get('currentUrl')}")
                return True
            else:
                logger.error(f"❌ Login failed: {login_result}")
                return False
                
        except Exception as e:
            logger.error(f"Login process failed: {e}")
            return False

    async def wait_for_content_load(self, crawler, url: str) -> str:
        """Wait for React app to load news content"""
        logger.info(f"Waiting for content to load at {url}...")
        
        try:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    delay_before_return_html=8.0,  # Wait longer for React content
                    page_timeout=45000,
                    js_code="""
                    // Wait for content to load
                    async function waitForContent() {
                        let attempts = 0;
                        const maxAttempts = 20;
                        
                        while (attempts < maxAttempts) {
                            // Check if loading spinner is gone
                            const loadingSpinner = document.querySelector('[alt="loading..."], .loading, .spinner, [class*="loading"], [class*="spinner"]');
                            const hasLoadingSpinner = loadingSpinner && window.getComputedStyle(loadingSpinner).display !== 'none';
                            
                            // Check if we have actual content
                            const contentSelectors = [
                                'article', '.article', '.news-item', '.story', '.feed-item',
                                '.card', '[class*="story"]', '[class*="article"]', '[class*="news"]',
                                '[class*="feed"]', '[class*="item"]', '[data-testid*="article"]'
                            ];
                            
                            let contentFound = false;
                            let contentCount = 0;
                            
                            for (const selector of contentSelectors) {
                                const elements = document.querySelectorAll(selector);
                                if (elements.length > 0) {
                                    contentCount += elements.length;
                                    contentFound = true;
                                }
                            }
                            
                            console.log(`Attempt ${attempts + 1}: Loading spinner: ${hasLoadingSpinner}, Content found: ${contentFound}, Content count: ${contentCount}`);
                            
                            if (!hasLoadingSpinner && contentFound && contentCount > 2) {
                                console.log('Content loaded successfully!');
                                return {
                                    success: true,
                                    contentCount,
                                    loadingComplete: true
                                };
                            }
                            
                            attempts++;
                            await new Promise(resolve => setTimeout(resolve, 1000));
                        }
                        
                        console.log('Content loading timeout reached');
                        return {
                            success: false,
                            contentCount: 0,
                            loadingComplete: false,
                            timeout: true
                        };
                    }
                    
                    waitForContent().then(result => {
                        window.contentLoadResult = result;
                        console.log('Content loading result:', result);
                    });
                    """
                )
            )
            
            load_result = result.js_execution_result
            if load_result and load_result.get('success'):
                logger.info(f"✅ Content loaded! Found {load_result.get('contentCount')} content items")
            else:
                logger.warning(f"⚠️  Content loading may not be complete: {load_result}")
            
            return result.html
            
        except Exception as e:
            logger.error(f"Content loading failed: {e}")
            return ""

    def extract_articles(self, html: str, base_url: str) -> List[Dict]:
        """Extract articles from HTML with multiple selector strategies"""
        if not html:
            return []
        
        logger.info("Extracting articles from HTML...")
        articles = []
        seen_titles = set()
        
        # Import BeautifulSoup here to avoid dependency issues
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("BeautifulSoup not available, using basic regex extraction")
            return self._extract_articles_regex(html)
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try each selector
        for selector in self.article_selectors:
            try:
                elements = soup.select(selector)
                logger.info(f"Selector '{selector}' found {len(elements)} elements")
                
                for element in elements[:20]:  # Limit to prevent overwhelming output
                    article = self._extract_article_data(element, base_url)
                    if article and article.get('title') and article['title'] not in seen_titles:
                        articles.append(article)
                        seen_titles.add(article['title'])
                        
                if len(articles) >= 10:  # Stop when we have enough articles
                    break
                    
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue
        
        logger.info(f"Extracted {len(articles)} unique articles")
        return articles

    def _extract_articles_regex(self, html: str) -> List[Dict]:
        """Fallback regex-based article extraction"""
        articles = []
        
        # Look for common title patterns
        title_patterns = [
            r'<h[1-6][^>]*>([^<]+)</h[1-6]>',
            r'<title>([^<]+)</title>',
            r'title="([^"]+)"',
            r'aria-label="([^"]+)"'
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches[:5]:  # Limit results
                if len(match.strip()) > 10:  # Filter out short/empty titles
                    articles.append({
                        'title': match.strip(),
                        'content': '',
                        'url': '',
                        'tickers': self.extract_tickers(match),
                        'extraction_method': 'regex'
                    })
        
        return articles

    def _extract_article_data(self, element, base_url: str) -> Optional[Dict]:
        """Extract data from a single article element"""
        try:
            # Try to find title
            title_selectors = ['h1', 'h2', 'h3', 'h4', '.title', '.headline', '[class*="title"]', '[class*="headline"]']
            title = ""
            
            for sel in title_selectors:
                title_elem = element.select_one(sel)
                if title_elem and title_elem.get_text(strip=True):
                    title = title_elem.get_text(strip=True)
                    break
            
            if not title:
                # Fallback to element text if no specific title found
                title = element.get_text(strip=True)[:200]  # First 200 chars
            
            if not title or len(title) < 10:
                return None
            
            # Try to find content
            content_selectors = ['.content', '.summary', '.description', 'p', '.text', '[class*="content"]']
            content = ""
            
            for sel in content_selectors:
                content_elem = element.select_one(sel)
                if content_elem and content_elem.get_text(strip=True):
                    content = content_elem.get_text(strip=True)
                    break
            
            # Try to find URL
            url = ""
            link_elem = element.select_one('a[href]')
            if link_elem:
                href = link_elem.get('href')
                if href:
                    url = urljoin(base_url, href)
            
            # Extract tickers from title and content
            full_text = f"{title} {content}"
            tickers = self.extract_tickers(full_text)
            
            return {
                'title': title,
                'content': content[:500],  # Limit content length
                'url': url,
                'tickers': tickers,
                'extraction_method': 'beautifulsoup'
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract article data: {e}")
            return None

    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text using existing NewsHead patterns"""
        if not text:
            return []
        
        tickers = set()
        
        for pattern in self.ticker_patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                # Clean up the ticker
                ticker = re.sub(r'[^\w:]', '', match)
                if ticker and 1 <= len(ticker) <= 6:  # Reasonable ticker length
                    tickers.add(ticker)
        
        return sorted(list(tickers))

    async def discover_news_endpoints(self, crawler) -> Dict:
        """Discover news data endpoints by analyzing the dashboard page"""
        logger.info("Discovering news endpoints from dashboard...")
        
        try:
            # Load dashboard and intercept network requests
            result = await crawler.arun(
                url=f"{self.base_url}/dashboard",
                config=CrawlerRunConfig(
                    delay_before_return_html=15.0,  # Wait longer for React to load
                    page_timeout=60000,  # 60 second timeout
                    js_code="""
                    async function discoverEndpoints() {
                        console.log('Starting enhanced endpoint discovery...');
                        
                        // Store intercepted network requests
                        window.interceptedRequests = [];
                        
                        // Override fetch to intercept API calls
                        const originalFetch = window.fetch;
                        window.fetch = function(...args) {
                            const url = args[0];
                            console.log('Intercepted fetch:', url);
                            window.interceptedRequests.push({
                                url: url,
                                method: args[1]?.method || 'GET',
                                timestamp: Date.now()
                            });
                            return originalFetch.apply(this, args);
                        };
                        
                        // Override XMLHttpRequest
                        const originalXHR = window.XMLHttpRequest;
                        window.XMLHttpRequest = function() {
                            const xhr = new originalXHR();
                            const originalOpen = xhr.open;
                            xhr.open = function(method, url) {
                                console.log('Intercepted XHR:', method, url);
                                window.interceptedRequests.push({
                                    url: url,
                                    method: method,
                                    timestamp: Date.now()
                                });
                                return originalOpen.apply(this, arguments);
                            };
                            return xhr;
                        };
                        
                        // Wait for React app to fully load
                        console.log('Waiting for React app to load...');
                        await new Promise(resolve => setTimeout(resolve, 12000));
                        
                        // Try to find news-related elements
                        const newsSelectors = [
                            '[data-testid*="news"]',
                            '[data-testid*="article"]',
                            '[data-testid*="feed"]',
                            '[class*="news"]',
                            '[class*="article"]',
                            '[class*="feed"]',
                            '[id*="news"]',
                            '[id*="article"]',
                            '[id*="feed"]',
                            'article',
                            '.story',
                            '.headline',
                            '.news-item'
                        ];
                        
                        let foundElements = [];
                        for (const selector of newsSelectors) {
                            const elements = document.querySelectorAll(selector);
                            if (elements.length > 0) {
                                foundElements.push({
                                    selector: selector,
                                    count: elements.length,
                                    sample: elements[0] ? {
                                        tagName: elements[0].tagName,
                                        className: elements[0].className,
                                        id: elements[0].id,
                                        textContent: elements[0].textContent?.substring(0, 100)
                                    } : null
                                });
                            }
                        }
                        
                        // Look for navigation links
                        const navLinks = [];
                        const links = document.querySelectorAll('a[href]');
                        links.forEach(link => {
                            const href = link.getAttribute('href');
                            const text = link.textContent?.toLowerCase();
                            if (href && (
                                href.includes('news') || 
                                href.includes('feed') || 
                                href.includes('article') ||
                                text?.includes('news') ||
                                text?.includes('feed')
                            )) {
                                navLinks.push({
                                    href: href,
                                    text: link.textContent?.trim(),
                                    fullUrl: href.startsWith('http') ? href : window.location.origin + href
                                });
                            }
                        });
                        
                        // Check for React Router or other SPA navigation
                        const reactRoutes = [];
                        if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
                            console.log('React detected on page');
                        }
                        
                        return {
                            success: true,
                            interceptedRequests: window.interceptedRequests || [],
                            foundElements: foundElements,
                            navLinks: navLinks,
                            pageTitle: document.title,
                            currentUrl: window.location.href,
                            reactDetected: !!window.__REACT_DEVTOOLS_GLOBAL_HOOK__,
                            bodyText: document.body.textContent?.substring(0, 500)
                        };
                    }
                    
                    return await discoverEndpoints();
                    """,
                    screenshot=True
                )
            )
            
            if result.success:
                # Save screenshot for debugging
                if result.screenshot:
                    with open("dashboard_screenshot.png", "wb") as f:
                        f.write(result.screenshot)
                    logger.info("Saved dashboard screenshot")
                
                discovery_data = result.extracted_content
                logger.info(f"Discovery completed: {discovery_data}")
                
                return {
                    "success": True,
                    "data": discovery_data,
                    "html_saved": True
                }
            else:
                logger.error(f"Failed to discover endpoints: {result.error_message}")
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            logger.error(f"Error during endpoint discovery: {str(e)}")
            return {"success": False, "error": str(e)}

    def _extract_urls_from_html(self, html: str) -> Dict:
        """Extract potential news URLs from HTML using regex"""
        import re
        
        urls = {
            "all_links": [],
            "news_related": [],
            "api_endpoints": [],
            "dashboard_links": []
        }
        
        # Find all href attributes
        href_pattern = r'href=["\']([^"\']+)["\']'
        all_hrefs = re.findall(href_pattern, html, re.IGNORECASE)
        
        news_keywords = ['news', 'feed', 'story', 'article', 'headline', 'market', 'update', 'ticker']
        api_keywords = ['api', 'json', 'data', 'feed']
        
        for href in all_hrefs:
            urls["all_links"].append(href)
            
            href_lower = href.lower()
            
            # Check for news-related URLs
            if any(keyword in href_lower for keyword in news_keywords):
                urls["news_related"].append(href)
            
            # Check for API endpoints
            if any(keyword in href_lower for keyword in api_keywords):
                urls["api_endpoints"].append(href)
            
            # Check for dashboard-related URLs
            if 'dashboard' in href_lower:
                urls["dashboard_links"].append(href)
        
        # Remove duplicates
        for key in urls:
            urls[key] = list(set(urls[key]))
        
        return urls

    async def test_newsfeed_access(self) -> Dict:
        """Test access to newsfeeds after login"""
        logger.info("Testing newsfeed access...")
        
        browser_config = BrowserConfig(
            headless=True,  # Run headless - no browser window
            browser_type="chromium",
            viewport_width=1920,
            viewport_height=1080,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        results = {
            "login_success": False,
            "dashboard_access": False,
            "screenshot_saved": False,
            "articles": [],
            "total_articles": 0,
        }
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # First login
            login_success = await self.login(crawler)
            results["login_success"] = login_success
            
            if not login_success:
                logger.error("Login failed, cannot test newsfeeds")
                return results
            
            # Navigate to dashboard and take screenshot
            logger.info("📸 Accessing dashboard and taking screenshot...")
            try:
                dashboard_result = await crawler.arun(
                    url=f"{self.base_url}/dashboard",
                    config=CrawlerRunConfig(
                        delay_before_return_html=10.0,  # Wait for page to load
                        page_timeout=60000,
                        screenshot=True,
                        js_code="""
                        (async function() {
                            console.log('Dashboard loaded, checking content...');
                            
                            // Wait a bit more for React to render
                            await new Promise(resolve => setTimeout(resolve, 5000));
                            
                            // Look for any sidebar navigation
                            const sidebarElements = document.querySelectorAll('.sidebar, .nav, .navigation, [role="navigation"]');
                            const allLinks = document.querySelectorAll('a');
                            const allButtons = document.querySelectorAll('button');
                            
                            // Look for newsfeed-related elements
                            const newsfeedElements = [];
                            const searchTerms = ['newsfeed', 'news', 'feed', 'stories', 'articles'];
                            
                            [...allLinks, ...allButtons].forEach(el => {
                                const text = el.textContent?.toLowerCase() || '';
                                const href = el.getAttribute('href') || '';
                                const ariaLabel = el.getAttribute('aria-label')?.toLowerCase() || '';
                                
                                if (searchTerms.some(term => 
                                    text.includes(term) || href.includes(term) || ariaLabel.includes(term)
                                )) {
                                    newsfeedElements.push({
                                        tagName: el.tagName,
                                        text: el.textContent?.trim(),
                                        href: href,
                                        ariaLabel: el.getAttribute('aria-label'),
                                        className: el.className
                                    });
                                }
                            });
                            
                            return {
                                success: true,
                                currentUrl: window.location.href,
                                pageTitle: document.title,
                                sidebarCount: sidebarElements.length,
                                totalLinks: allLinks.length,
                                totalButtons: allButtons.length,
                                newsfeedElements: newsfeedElements,
                                bodyText: document.body.textContent?.substring(0, 500)
                            };
                        })()
                        """
                    )
                )
                
                if dashboard_result.success:
                    results["dashboard_access"] = True
                    
                    # Save screenshot properly
                    if dashboard_result.screenshot:
                        try:
                            screenshot_path = "dashboard_screenshot.png"
                            with open(screenshot_path, "wb") as f:
                                # Handle both bytes and base64 encoded screenshots
                                if isinstance(dashboard_result.screenshot, bytes):
                                    f.write(dashboard_result.screenshot)
                                else:
                                    # If it's base64 encoded, decode it first
                                    import base64
                                    screenshot_data = base64.b64decode(dashboard_result.screenshot)
                                    f.write(screenshot_data)
                            
                            results["screenshot_saved"] = True
                            logger.info(f"✅ Screenshot saved to {screenshot_path}")
                        except Exception as e:
                            logger.error(f"Failed to save screenshot: {e}")
                            results["screenshot_saved"] = False
                    
                    # Save HTML for analysis
                    Path("dashboard_content.html").write_text(dashboard_result.html)
                    
                    # Get dashboard analysis results
                    dashboard_data = dashboard_result.js_execution_result
                    results["dashboard_data"] = dashboard_data
                    
                    logger.info(f"Dashboard analysis: {dashboard_data}")
                    
                    # Look for newsfeed elements found
                    if dashboard_data and dashboard_data.get('newsfeedElements'):
                        logger.info(f"🎯 Found {len(dashboard_data['newsfeedElements'])} potential newsfeed elements:")
                        for i, element in enumerate(dashboard_data['newsfeedElements'][:5], 1):
                            logger.info(f"  {i}. {element.get('tagName')} - '{element.get('text')}' (href: {element.get('href')})")
                    
                    # Now try to navigate to newsfeed
                    logger.info("🔍 Attempting to navigate to newsfeed...")
                    newsfeed_result = await self.navigate_to_newsfeed(crawler)
                    results["newsfeed_navigation"] = newsfeed_result
                    
                    if newsfeed_result.get("success") and newsfeed_result.get("articles_count", 0) > 0:
                        logger.info(f"✅ Successfully found {newsfeed_result['articles_count']} articles!")
                        results["articles"] = newsfeed_result["articles"]
                        results["total_articles"] = newsfeed_result["articles_count"]
                
            except Exception as e:
                logger.error(f"Dashboard access failed: {e}")
                results["dashboard_access"] = False
        
        return results

    async def run_full_test(self) -> Dict:
        """Run the complete test suite"""
        logger.info("🚀 Starting Benzinga Pro scraping test...")
        
        # Test 1: Basic connectivity
        connectivity = await self.test_connectivity()
        logger.info(f"Connectivity test: {'✅ PASSED' if connectivity.get('success') else '❌ FAILED'}")
        
        # Test 2: Login and dashboard access with screenshot
        newsfeed_results = await self.test_newsfeed_access()
        
        # Compile final results
        final_results = {
            "timestamp": "2024-12-19T10:30:00Z",
            "connectivity": connectivity,
            "login_success": newsfeed_results.get("login_success", False),
            "dashboard_access": newsfeed_results.get("dashboard_access", False),
            "screenshot_saved": newsfeed_results.get("screenshot_saved", False),
            "dashboard_data": newsfeed_results.get("dashboard_data", {}),
            "newsfeed_navigation": newsfeed_results.get("newsfeed_navigation", {}),
            "total_articles_found": newsfeed_results.get("total_articles", 0),
            "sample_articles": newsfeed_results.get("articles", [])[:3],  # First 3 articles
            "feasibility_assessment": self._assess_feasibility(connectivity, newsfeed_results)
        }
        
        return final_results

    def _assess_feasibility(self, connectivity: Dict, newsfeed_results: Dict) -> Dict:
        """Assess the feasibility of scraping Benzinga Pro"""
        assessment = {
            "overall_feasible": False,
            "confidence": "low",
            "key_findings": [],
            "challenges": [],
            "recommendations": []
        }
        
        # Check connectivity
        if not connectivity.get("success"):
            assessment["challenges"].append("Basic connectivity failed")
            return assessment
        
        assessment["key_findings"].append("✅ Basic connectivity successful")
        
        # Check anti-bot measures
        anti_bot = connectivity.get("anti_bot_check", {})
        if anti_bot.get("captcha"):
            assessment["challenges"].append("CAPTCHA protection detected")
        if anti_bot.get("cloudflare"):
            assessment["challenges"].append("Cloudflare protection detected")
        if anti_bot.get("blocked"):
            assessment["challenges"].append("Access appears to be blocked")
        
        # Check login success
        if not newsfeed_results.get("login_success"):
            assessment["challenges"].append("Login failed - credentials or anti-bot protection")
            assessment["recommendations"].append("Verify credentials and check for additional anti-bot measures")
            return assessment
        
        assessment["key_findings"].append("✅ Login successful")
        
        # Check article extraction
        total_articles = newsfeed_results.get("total_articles", 0)
        if total_articles == 0:
            assessment["challenges"].append("No articles extracted - content may be heavily JavaScript-dependent")
            assessment["recommendations"].append("Implement more sophisticated JavaScript waiting and content detection")
        elif total_articles < 5:
            assessment["challenges"].append("Limited article extraction - may need better selectors")
            assessment["recommendations"].append("Analyze page structure and improve article selectors")
        else:
            assessment["key_findings"].append(f"✅ Successfully extracted {total_articles} articles")
        
        # Check ticker extraction
        unique_tickers = len(newsfeed_results.get("unique_tickers", []))
        if unique_tickers > 0:
            assessment["key_findings"].append(f"✅ Successfully extracted {unique_tickers} unique tickers")
        
        # Overall assessment
        if (newsfeed_results.get("login_success") and 
            total_articles > 0 and 
            not anti_bot.get("captcha") and 
            not anti_bot.get("blocked")):
            assessment["overall_feasible"] = True
            assessment["confidence"] = "high" if total_articles > 10 else "medium"
            assessment["recommendations"].extend([
                "Implement proper error handling and retry logic",
                "Add monitoring for anti-bot measures",
                "Consider implementing session management for long-running scraping",
                "Test with different user agents and browser configurations"
            ])
        else:
            assessment["recommendations"].extend([
                "Investigate JavaScript loading issues",
                "Consider using more sophisticated browser automation",
                "Implement better content waiting strategies",
                "Test with different timing and waiting approaches"
            ])
        
        return assessment

    async def test_api_endpoints(self, crawler, intercepted_requests) -> Dict:
        """Test intercepted API endpoints to find news data"""
        logger.info(f"Testing {len(intercepted_requests)} intercepted API endpoints...")
        
        api_results = {}
        news_api_endpoints = []
        
        for request in intercepted_requests:
            url = request.get('url', '')
            
            # Filter for potential news API endpoints
            if any(keyword in url.lower() for keyword in ['news', 'feed', 'article', 'story', 'api']):
                logger.info(f"Testing API endpoint: {url}")
                
                try:
                    # Try to access the API endpoint
                    result = await crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(
                            delay_before_return_html=3.0,
                            page_timeout=30000,
                            js_code="""
                            async function testApiEndpoint() {
                                // Check if this is JSON data
                                const bodyText = document.body.textContent;
                                let isJson = false;
                                let jsonData = null;
                                
                                try {
                                    jsonData = JSON.parse(bodyText);
                                    isJson = true;
                                } catch (e) {
                                    isJson = false;
                                }
                                
                                return {
                                    url: window.location.href,
                                    isJson: isJson,
                                    dataLength: bodyText.length,
                                    hasNewsKeywords: /news|article|story|headline|ticker|symbol/i.test(bodyText),
                                    sample: bodyText.substring(0, 500),
                                    jsonData: isJson ? jsonData : null
                                };
                            }
                            
                            return await testApiEndpoint();
                            """
                        )
                    )
                    
                    if result.success:
                        api_data = result.extracted_content
                        api_results[url] = {
                            "success": True,
                            "data": api_data,
                            "method": request.get('method', 'GET')
                        }
                        
                        # If this looks like news data, mark it
                        if api_data and api_data.get('hasNewsKeywords'):
                            news_api_endpoints.append(url)
                            
                    else:
                        api_results[url] = {
                            "success": False,
                            "error": result.error_message,
                            "method": request.get('method', 'GET')
                        }
                        
                except Exception as e:
                    logger.error(f"Error testing API endpoint {url}: {str(e)}")
                    api_results[url] = {
                        "success": False,
                        "error": str(e),
                        "method": request.get('method', 'GET')
                    }
        
        return {
            "total_tested": len(intercepted_requests),
            "api_results": api_results,
            "news_endpoints": news_api_endpoints,
            "successful_calls": len([r for r in api_results.values() if r.get('success')])
        }

    async def navigate_to_newsfeed(self, crawler) -> Dict:
        """Navigate to newsfeed by clicking the sidebar button and extract articles"""
        logger.info("Navigating to newsfeed via sidebar navigation...")
        
        try:
            result = await crawler.arun(
                url=f"{self.base_url}/dashboard",
                config=CrawlerRunConfig(
                    delay_before_return_html=20.0,  # Wait longer for full content load
                    page_timeout=60000,
                    js_code="""
                    // Wait for page to load
                    await new Promise(resolve => setTimeout(resolve, 5000));
                    
                    console.log('Looking for NEWSFEED link...');
                    
                    // Try multiple ways to find the newsfeed link
                    let newsfeedLink = null;
                    
                    // Method 1: Look for text containing "NEWSFEED"
                    const allElements = document.querySelectorAll('*');
                    for (let elem of allElements) {
                        if (elem.textContent && elem.textContent.trim().toUpperCase() === 'NEWSFEED') {
                            newsfeedLink = elem;
                            console.log('Found NEWSFEED by text content');
                            break;
                        }
                    }
                    
                    // Method 2: Look for links or buttons containing "newsfeed"
                    if (!newsfeedLink) {
                        const links = document.querySelectorAll('a, button, div[onclick], span[onclick]');
                        for (let link of links) {
                            if (link.textContent && link.textContent.toLowerCase().includes('newsfeed')) {
                                newsfeedLink = link;
                                console.log('Found newsfeed link by text search');
                                break;
                            }
                        }
                    }
                    
                    let clickResult = 'No newsfeed link found';
                    
                    if (newsfeedLink) {
                        console.log('Found newsfeed element:', newsfeedLink.tagName, newsfeedLink.textContent);
                        
                        // Scroll into view and click
                        newsfeedLink.scrollIntoView();
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        
                        newsfeedLink.click();
                        console.log('Clicked newsfeed link');
                        clickResult = 'Clicked NEWSFEED link successfully';
                        
                        // Wait for content to load
                        await new Promise(resolve => setTimeout(resolve, 10000));
                    }
                    
                    // Return results (this is important for Python to capture)
                    return {
                        found_newsfeed_link: !!newsfeedLink,
                        click_result: clickResult,
                        page_title: document.title,
                        current_url: window.location.href
                    };
                    """
                )
            )
            
            if result.success:
                # Save screenshot for debugging
                if result.screenshot:
                    try:
                        with open("newsfeed_navigation_screenshot.png", "wb") as f:
                            if isinstance(result.screenshot, bytes):
                                f.write(result.screenshot)
                            else:
                                import base64
                                screenshot_data = base64.b64decode(result.screenshot)
                                f.write(screenshot_data)
                        logger.info("Saved newsfeed navigation screenshot")
                    except Exception as e:
                        logger.error(f"Failed to save newsfeed screenshot: {e}")
                
                # Save HTML for analysis
                Path("newsfeed_content.html").write_text(result.html)
                
                navigation_data = result.extracted_content
                logger.info(f"Newsfeed navigation completed: {navigation_data}")
                
                # Extract articles from the loaded content
                articles = self.extract_articles(result.html, self.base_url)
                
                return {
                    "success": True,
                    "navigation_data": navigation_data,
                    "articles": articles,
                    "articles_count": len(articles),
                    "html_length": len(result.html)
                }
            else:
                logger.error(f"Failed to navigate to newsfeed: {result.error_message}")
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            logger.error(f"Error during newsfeed navigation: {str(e)}")
            return {"success": False, "error": str(e)}


async def main():
    """Main test execution"""
    # Get credentials from environment variables
    email = os.getenv('BENZINGA_EMAIL')
    password = os.getenv('BENZINGA_PASSWORD')
    
    if not email or not password:
        print("❌ Please set BENZINGA_EMAIL and BENZINGA_PASSWORD environment variables!")
        print("   Example:")
        print("   export BENZINGA_EMAIL='your_email@domain.com'")
        print("   export BENZINGA_PASSWORD='your_password'")
        print("   python test_benzinga_enhanced.py")
        return
    
    tester = BenzingaProTester(email, password)
    
    try:
        results = await tester.run_full_test()
        
        # Save results to JSON file
        with open("benzinga_test_results.json", "w") as f:
            # Convert sets to lists for JSON serialization
            json_results = json.loads(json.dumps(results, default=list))
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("🧪 BENZINGA PRO SCRAPING TEST RESULTS")
        print("="*60)
        
        print(f"📡 Connectivity: {'✅ SUCCESS' if results['connectivity']['success'] else '❌ FAILED'}")
        print(f"🔐 Login: {'✅ SUCCESS' if results['login_success'] else '❌ FAILED'}")
        print(f"🏠 Dashboard Access: {'✅ SUCCESS' if results['dashboard_access'] else '❌ FAILED'}")
        print(f"📸 Screenshot Saved: {'✅ SUCCESS' if results['screenshot_saved'] else '❌ FAILED'}")
        
        # Show dashboard data
        dashboard_data = results.get('dashboard_data', {})
        if dashboard_data:
            print(f"\n🏠 DASHBOARD ANALYSIS:")
            print(f"   Page Title: {dashboard_data.get('pageTitle', 'N/A')}")
            print(f"   Current URL: {dashboard_data.get('currentUrl', 'N/A')}")
            print(f"   Sidebar Elements: {dashboard_data.get('sidebarCount', 0)}")
            print(f"   Total Links: {dashboard_data.get('totalLinks', 0)}")
            print(f"   Total Buttons: {dashboard_data.get('totalButtons', 0)}")
            
            # Show newsfeed elements found
            newsfeed_elements = dashboard_data.get('newsfeedElements', [])
            if newsfeed_elements:
                print(f"   🎯 Newsfeed Elements Found: {len(newsfeed_elements)}")
                for i, element in enumerate(newsfeed_elements[:5], 1):
                    print(f"      {i}. {element.get('tagName')} - '{element.get('text')}' (href: {element.get('href')})")
            else:
                print(f"   🎯 Newsfeed Elements Found: 0")
        
        # Show newsfeed navigation results
        newsfeed_nav = results.get('newsfeed_navigation', {})
        if newsfeed_nav:
            print(f"\n📱 NEWSFEED NAVIGATION:")
            print(f"   Navigation Success: {'✅' if newsfeed_nav.get('success') else '❌'}")
            if newsfeed_nav.get('navigation_data'):
                nav_data = newsfeed_nav['navigation_data']
                print(f"   Elements Found: {nav_data.get('totalElements', 0)}")
                print(f"   Articles Found: {newsfeed_nav.get('articles_count', 0)}")
                if nav_data.get('clickedElement'):
                    clicked = nav_data['clickedElement']
                    print(f"   Clicked Element: '{clicked.get('text', 'N/A')}'")
        
        # Show sample articles if found
        if results.get('sample_articles'):
            print(f"\n📄 SAMPLE ARTICLES FOUND:")
            for i, article in enumerate(results['sample_articles'], 1):
                print(f"   {i}. {article.get('title', 'No title')[:100]}...")
                if article.get('tickers'):
                    print(f"      Tickers: {', '.join(article['tickers'])}")
        
        # Show sample content from dashboard
        if dashboard_data.get('bodyText'):
            print(f"\n📄 DASHBOARD CONTENT SAMPLE:")
            print(f"   {dashboard_data['bodyText'][:300]}...")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 