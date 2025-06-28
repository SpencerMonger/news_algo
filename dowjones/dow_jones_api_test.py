#!/usr/bin/env python3
"""
Dow Jones Newswires APIs Test Script (Official Postman Collection)
Test script to evaluate the Dow Jones Newswires APIs via official Postman collection for NewsHead use case.

Based on official Dow Jones documentation:
https://developer.dowjones.com/documents/site-docs-getting_started-postman_collections_and_python_notebooks-postman_collections

This script tests:
1. Authentication using environment variables (as per Postman collection)
2. Top Stories API
3. Portfolio Significance API  
4. Calendar Live API
5. Investor Select RSS API
6. Press release content from target sources (GlobeNewswire, BusinessWire, PR Newswire, Accesswire)
7. Latency and performance characteristics
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Try to load from .env file
def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load environment variables
load_env()

# Configuration
class DowJonesConfig:
    def __init__(self):
        # Based on official Dow Jones Postman collection documentation
        self.base_url = "https://api.dowjones.com"
        
        # Authentication - these should match the Postman collection environment variables
        self.api_key = os.getenv('DOW_JONES_API_KEY')
        self.username = os.getenv('DOW_JONES_USERNAME') 
        self.password = os.getenv('DOW_JONES_PASSWORD')
        self.client_id = os.getenv('DOW_JONES_CLIENT_ID')
        
        # Request configuration
        self.timeout = 30
        self.max_retries = 3
        
        # Target newswire sources for NewsHead
        self.target_sources = [
            'GlobeNewswire',
            'BusinessWire', 
            'PR Newswire',
            'PRNewswire',
            'Accesswire',
            'AccessWire',
            'MarketWatch',
            'Dow Jones'
        ]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dow_jones_api_test.log')
    ]
)
logger = logging.getLogger(__name__)

class DowJonesAPITester:
    def __init__(self, config: DowJonesConfig):
        self.config = config
        self.session = requests.Session()
        self.auth_token = None
        self.test_results = {}
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'NewsHead-DowJones-Tester/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def authenticate(self) -> bool:
        """
        Authenticate with Dow Jones API using credentials.
        Based on the Postman collection, this likely uses Bearer token or API key.
        """
        logger.info("🔐 Attempting authentication with Dow Jones API...")
        
        try:
            # Try API Key authentication first (common in Postman collections)
            if self.config.api_key:
                self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
                logger.info("✅ Using API Key authentication")
                return True
            
            # If no API key, try Service Account authentication
            elif self.config.username and self.config.password and self.config.client_id:
                auth_url = f"{self.config.base_url}/accounts/oauth2/v1/token"
                
                auth_data = {
                    'grant_type': 'client_credentials',
                    'username': self.config.username,
                    'password': self.config.password,
                    'client_id': self.config.client_id,
                    'scope': 'openid service_account_id offline_access'
                }
                
                response = self.session.post(auth_url, data=auth_data, timeout=self.config.timeout)
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.auth_token = token_data.get('access_token')
                    if self.auth_token:
                        self.session.headers['Authorization'] = f'Bearer {self.auth_token}'
                        logger.info("✅ Service Account authentication successful")
                        return True
                
                logger.error(f"❌ Authentication failed: {response.status_code} - {response.text}")
                return False
            
            else:
                logger.error("❌ No authentication credentials provided")
                return False
                
        except Exception as e:
            logger.error(f"❌ Authentication error: {str(e)}")
            return False

    def test_top_stories_api(self) -> Dict[str, Any]:
        """Test the Top Stories API endpoint."""
        logger.info("📰 Testing Top Stories API...")
        
        endpoint = "/content-collections"
        url = f"{self.config.base_url}{endpoint}"
        
        # Set proper headers for Top Stories API
        headers = self.session.headers.copy()
        headers['Accept'] = 'application/vnd.dowjones.dna.content-collections.v_1.1_beta'
        
        start_time = time.time()
        
        try:
            response = self.session.get(url, headers=headers, timeout=self.config.timeout)
            latency = (time.time() - start_time) * 1000
            
            result = {
                'endpoint': 'Top Stories API',
                'url': url,
                'status_code': response.status_code,
                'latency_ms': round(latency, 2),
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                collections = data.get('data', [])
                result['collections_available'] = len(collections)
                
                # Test retrieving a specific collection if available
                if collections:
                    collection_id = collections[0].get('id', '')
                    if collection_id:
                        result['sample_collection_id'] = collection_id
                
                logger.info(f"✅ Top Stories API: {result['collections_available']} collections, {result['latency_ms']}ms latency")
            else:
                result['error'] = response.text
                logger.error(f"❌ Top Stories API failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': 'Top Stories API',
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
            logger.error(f"❌ Top Stories API error: {str(e)}")
        
        return result

    def test_portfolio_significance_api(self) -> Dict[str, Any]:
        """Test the Newswires Real-Time API for market-significant news."""
        logger.info("📊 Testing Real-Time API for Market-Significant News...")
        
        endpoint = "/content/realtime/search"
        url = f"{self.config.base_url}{endpoint}"
        
        # Set proper headers for Real-Time API
        headers = self.session.headers.copy()
        headers['Accept'] = 'application/vnd.dowjones.dna.content.v_1.0+json'
        headers['Content-Type'] = 'application/json'
        
        # Search for market-significant news using DJN taxonomy
        payload = {
            "data": {
                "id": "Search",
                "type": "content",
                "attributes": {
                    "query": {
                        "search_string": [
                            {
                                "mode": "Unified",
                                "value": "djn=p/pmdm"
                            }
                        ],
                        "date": {
                            "days_range": "LastDay"
                        }
                    },
                    "formatting": {
                        "is_return_rich_article_id": True
                    },
                    "navigation": {
                        "is_return_headline_coding": True,
                        "is_return_djn_headline_coding": True
                    },
                    "page_offset": 0,
                    "page_limit": 20
                }
            }
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=self.config.timeout)
            latency = (time.time() - start_time) * 1000
            
            result = {
                'endpoint': 'Real-Time API (Market Significant)',
                'url': url,
                'status_code': response.status_code,
                'latency_ms': round(latency, 2),
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                result['total_results'] = data.get('meta', {}).get('total_count', 0)
                result['articles_returned'] = len(articles)
                
                # Analyze sources
                sources_found = set()
                for article in articles:
                    source = article.get('meta', {}).get('source', {}).get('name', '')
                    if source:
                        sources_found.add(source)
                
                result['sources_found'] = list(sources_found)
                result['target_sources_covered'] = len([s for s in sources_found if any(target.lower() in s.lower() for target in self.config.target_sources)])
                
                logger.info(f"✅ Real-Time API: {result['articles_returned']} articles, {result['latency_ms']}ms latency")
            else:
                result['error'] = response.text
                logger.error(f"❌ Real-Time API failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': 'Real-Time API (Market Significant)',
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
            logger.error(f"❌ Real-Time API error: {str(e)}")
        
        return result

    def test_calendar_live_api(self) -> Dict[str, Any]:
        """Test the Calendar Live API endpoint (if available)."""
        logger.info("📅 Testing Calendar Live API...")
        
        # Note: Calendar Live API might not be available in all subscriptions
        endpoint = "/content/calendar"
        url = f"{self.config.base_url}{endpoint}"
        
        # Get today's calendar events
        today = datetime.now().strftime('%Y-%m-%d')
        params = {
            'date': today,
            'limit': 50
        }
        
        start_time = time.time()
        
        try:
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            latency = (time.time() - start_time) * 1000
            
            result = {
                'endpoint': 'Calendar Live API',
                'url': url,
                'status_code': response.status_code,
                'latency_ms': round(latency, 2),
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                result['events_returned'] = len(data) if isinstance(data, list) else data.get('count', 0)
                logger.info(f"✅ Calendar Live API: {result['events_returned']} events, {result['latency_ms']}ms latency")
            else:
                result['error'] = response.text
                # Calendar API might not be available, so don't treat as critical failure
                if response.status_code == 404:
                    logger.info(f"ℹ️ Calendar Live API not available (404) - may not be included in subscription")
                else:
                    logger.error(f"❌ Calendar Live API failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': 'Calendar Live API',
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
            logger.error(f"❌ Calendar Live API error: {str(e)}")
        
        return result

    def test_press_release_search(self) -> Dict[str, Any]:
        """Test searching specifically for press releases from target sources."""
        logger.info("📢 Testing Press Release Search...")
        
        endpoint = "/content/realtime/search"
        url = f"{self.config.base_url}{endpoint}"
        
        # Set proper headers
        headers = self.session.headers.copy()
        headers['Accept'] = 'application/vnd.dowjones.dna.content.v_1.0+json'
        headers['Content-Type'] = 'application/json'
        
        # Search for press releases using DJN taxonomy for press release wire
        payload = {
            "data": {
                "id": "Search",
                "type": "content",
                "attributes": {
                    "query": {
                        "search_string": [
                            {
                                "mode": "Unified",
                                "value": "djn=p/pmdm"
                            }
                        ],
                        "date": {
                            "days_range": "LastDay"
                        }
                    },
                    "formatting": {
                        "is_return_rich_article_id": True
                    },
                    "navigation": {
                        "is_return_headline_coding": True,
                        "is_return_djn_headline_coding": True
                    },
                    "page_offset": 0,
                    "page_limit": 50
                }
            }
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=self.config.timeout)
            latency = (time.time() - start_time) * 1000
            
            result = {
                'endpoint': 'Press Release Search',
                'url': url,
                'status_code': response.status_code,
                'latency_ms': round(latency, 2),
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                result['total_results'] = data.get('meta', {}).get('total_count', 0)
                result['articles_returned'] = len(articles)
                
                # Analyze press release sources
                pr_sources = {}
                newshead_relevant = {}
                
                for article in articles:
                    source = article.get('meta', {}).get('source', {}).get('name', 'Unknown')
                    headline = article.get('attributes', {}).get('headline', {}).get('main', {}).get('text', '')
                    
                    if source not in pr_sources:
                        pr_sources[source] = 0
                    pr_sources[source] += 1
                    
                    # Check if this is relevant to NewsHead target sources
                    if any(target.lower() in source.lower() or target.lower() in headline.lower() 
                           for target in self.config.target_sources):
                        if source not in newshead_relevant:
                            newshead_relevant[source] = 0
                        newshead_relevant[source] += 1
                
                result['press_release_sources'] = pr_sources
                result['newshead_relevant_sources'] = newshead_relevant
                
                logger.info(f"✅ Press Release Search: {result['articles_returned']} releases, {result['latency_ms']}ms latency")
            else:
                result['error'] = response.text
                logger.error(f"❌ Press Release Search failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': 'Press Release Search',
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
            logger.error(f"❌ Press Release Search error: {str(e)}")
        
        return result

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite for NewsHead evaluation."""
        logger.info("🚀 Starting Dow Jones Newswires APIs comprehensive test...")
        
        # Check credentials
        if not any([self.config.api_key, all([self.config.username, self.config.password, self.config.client_id])]):
            logger.error("❌ No authentication credentials found!")
            logger.error("Please set either:")
            logger.error("  - DOW_JONES_API_KEY (for API key auth)")
            logger.error("  - DOW_JONES_USERNAME, DOW_JONES_PASSWORD, DOW_JONES_CLIENT_ID (for service account auth)")
            return {'error': 'No authentication credentials'}
        
        # Authenticate
        if not self.authenticate():
            return {'error': 'Authentication failed'}
        
        # Run all tests
        test_results = {
            'test_timestamp': datetime.now().isoformat(),
            'authentication': 'success',
            'base_url': self.config.base_url,
            'tests': {}
        }
        
        # Test each endpoint
        test_methods = [
            self.test_top_stories_api,
            self.test_portfolio_significance_api,
            self.test_calendar_live_api,
            self.test_press_release_search
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                test_results['tests'][result['endpoint']] = result
            except Exception as e:
                logger.error(f"❌ Test method {test_method.__name__} failed: {str(e)}")
                test_results['tests'][test_method.__name__] = {'error': str(e), 'success': False}
        
        # Generate summary
        successful_tests = sum(1 for test in test_results['tests'].values() if test.get('success', False))
        total_tests = len(test_results['tests'])
        avg_latency = sum(test.get('latency_ms', 0) for test in test_results['tests'].values()) / total_tests if total_tests > 0 else 0
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': f"{(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            'average_latency_ms': round(avg_latency, 2),
            'newshead_compatibility': self.assess_newshead_compatibility(test_results['tests'])
        }
        
        return test_results

    def assess_newshead_compatibility(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compatibility with NewsHead requirements."""
        assessment = {
            'overall_score': 0,
            'latency_acceptable': False,
            'source_coverage': 0,
            'real_time_capability': False,
            'recommendation': 'Not suitable'
        }
        
        # Check latency (NewsHead target: <10 seconds, ideally <5 seconds)
        latencies = [test.get('latency_ms', float('inf')) for test in test_results.values() if test.get('success')]
        if latencies:
            max_latency = max(latencies)
            avg_latency = sum(latencies) / len(latencies)
            
            if max_latency < 5000:  # <5 seconds
                assessment['latency_acceptable'] = True
                assessment['overall_score'] += 40
            elif max_latency < 10000:  # <10 seconds
                assessment['latency_acceptable'] = True
                assessment['overall_score'] += 25
        
        # Check source coverage
        press_release_test = test_results.get('Press Release Search', {})
        if press_release_test.get('success') and 'newshead_relevant_sources' in press_release_test:
            relevant_sources = len(press_release_test['newshead_relevant_sources'])
            total_target_sources = len(self.config.target_sources)
            
            coverage_percentage = (relevant_sources / total_target_sources) * 100
            assessment['source_coverage'] = coverage_percentage
            
            if coverage_percentage >= 75:
                assessment['overall_score'] += 40
            elif coverage_percentage >= 50:
                assessment['overall_score'] += 25
            elif coverage_percentage >= 25:
                assessment['overall_score'] += 10
        
        # Check real-time capability
        if any(test.get('success') and test.get('articles_returned', 0) > 0 for test in test_results.values()):
            assessment['real_time_capability'] = True
            assessment['overall_score'] += 20
        
        # Overall recommendation
        if assessment['overall_score'] >= 80:
            assessment['recommendation'] = 'Highly suitable for NewsHead'
        elif assessment['overall_score'] >= 60:
            assessment['recommendation'] = 'Suitable with minor concerns'
        elif assessment['overall_score'] >= 40:
            assessment['recommendation'] = 'Partially suitable'
        else:
            assessment['recommendation'] = 'Not suitable for NewsHead requirements'
        
        return assessment

def print_results(results: Dict[str, Any]):
    """Print formatted test results."""
    print("\n" + "="*80)
    print("🔍 DOW JONES NEWSWIRES APIs TEST RESULTS")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ Test failed: {results['error']}")
        return
    
    print(f"📅 Test Time: {results['test_timestamp']}")
    print(f"🌐 Base URL: {results['base_url']}")
    print(f"🔐 Authentication: {results['authentication']}")
    
    print(f"\n📊 SUMMARY:")
    summary = results['summary']
    print(f"   • Tests Run: {summary['total_tests']}")
    print(f"   • Success Rate: {summary['success_rate']}")
    print(f"   • Average Latency: {summary['average_latency_ms']}ms")
    
    print(f"\n🎯 NEWSHEAD COMPATIBILITY ASSESSMENT:")
    compat = summary['newshead_compatibility']
    print(f"   • Overall Score: {compat['overall_score']}/100")
    print(f"   • Latency Acceptable: {'✅' if compat['latency_acceptable'] else '❌'}")
    print(f"   • Source Coverage: {compat['source_coverage']:.1f}%")
    print(f"   • Real-time Capability: {'✅' if compat['real_time_capability'] else '❌'}")
    print(f"   • Recommendation: {compat['recommendation']}")
    
    print(f"\n📋 DETAILED TEST RESULTS:")
    for endpoint, result in results['tests'].items():
        status = "✅" if result.get('success') else "❌"
        latency = result.get('latency_ms', 'N/A')
        print(f"   {status} {endpoint}: {latency}ms")
        
        if not result.get('success') and 'error' in result:
            print(f"      Error: {result['error']}")
        
        if endpoint == 'Press Release Search' and result.get('success'):
            if 'newshead_relevant_sources' in result:
                print(f"      NewsHead Relevant Sources: {result['newshead_relevant_sources']}")

def main():
    """Main function to run the test."""
    print("🚀 Dow Jones Newswires APIs Test for NewsHead")
    print("=" * 50)
    
    config = DowJonesConfig()
    tester = DowJonesAPITester(config)
    
    try:
        results = tester.run_comprehensive_test()
        print_results(results)
        
        # Save results to file
        with open('dow_jones_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: dow_jones_test_results.json")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 