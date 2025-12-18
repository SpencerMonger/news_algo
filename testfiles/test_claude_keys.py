#!/usr/bin/env python3
"""
Test all 3 Claude API keys CONCURRENTLY from .env file.
Verifies they work in parallel (the whole point of having 3 keys for load balancing).
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import anthropic
except ImportError:
    print("❌ anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


async def test_api_key_async(key_name: str, api_key: str, request_num: int = 1) -> dict:
    """Test a single API key with a minimal async request."""
    result = {
        "key_name": key_name,
        "request_num": request_num,
        "key_preview": f"{api_key[:12]}...{api_key[-4:]}" if api_key and len(api_key) > 16 else "INVALID",
        "success": False,
        "response_time_ms": 0,
        "error": None,
        "model": None
    }
    
    if not api_key:
        result["error"] = "Key not found in .env"
        return result
    
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        start_time = time.time()
        
        # Minimal request - just say "Hi" and get a short response
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": f"Say 'OK-{request_num}' and nothing else."}]
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result["success"] = True
        result["response_time_ms"] = round(elapsed_ms, 1)
        result["model"] = response.model
        result["response"] = response.content[0].text.strip()
        
    except anthropic.AuthenticationError as e:
        result["error"] = f"Authentication failed: {str(e)}"
    except anthropic.RateLimitError as e:
        result["error"] = f"Rate limited: {str(e)}"
    except anthropic.APIError as e:
        result["error"] = f"API error: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


async def run_concurrent_test():
    """Run all 3 API keys concurrently."""
    print("=" * 60)
    print("🔑 CLAUDE API KEY CONCURRENT VERIFICATION TEST")
    print("=" * 60)
    print()
    
    # Load keys
    keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "ANTHROPIC_API_KEY2": os.getenv("ANTHROPIC_API_KEY2"),
        "ANTHROPIC_API_KEY3": os.getenv("ANTHROPIC_API_KEY3"),
    }
    
    # Check which keys are present
    print("📋 Keys found in .env:")
    for key_name, api_key in keys.items():
        status = "✅" if api_key else "❌"
        preview = f"{api_key[:12]}...{api_key[-4:]}" if api_key and len(api_key) > 16 else "NOT SET"
        print(f"   {status} {key_name}: {preview}")
    print()
    
    # Filter out missing keys
    valid_keys = [(name, key) for name, key in keys.items() if key]
    
    if not valid_keys:
        print("❌ No API keys found in .env file!")
        return 2
    
    # TEST 1: Concurrent single request per key
    print("=" * 60)
    print("🚀 TEST 1: Single request per key (CONCURRENT)")
    print("=" * 60)
    
    start_time = time.time()
    
    tasks = [test_api_key_async(name, key, i+1) for i, (name, key) in enumerate(valid_keys)]
    results = await asyncio.gather(*tasks)
    
    total_time = (time.time() - start_time) * 1000
    
    print()
    for result in results:
        if result["success"]:
            print(f"   ✅ {result['key_name']}: {result['response_time_ms']}ms - \"{result['response']}\"")
        else:
            print(f"   ❌ {result['key_name']}: {result['error']}")
    
    print(f"\n   ⏱️  Total concurrent time: {round(total_time, 1)}ms")
    print(f"   📊 If sequential, would be: ~{round(sum(r['response_time_ms'] for r in results if r['success']), 1)}ms")
    print()
    
    # TEST 2: Multiple concurrent requests (simulating load balancing)
    print("=" * 60)
    print("🚀 TEST 2: 6 requests across all keys (CONCURRENT)")
    print("   Simulating load-balanced sentiment analysis")
    print("=" * 60)
    
    # Create 6 requests distributed across keys (2 per key)
    multi_tasks = []
    for round_num in range(2):
        for i, (name, key) in enumerate(valid_keys):
            request_id = round_num * len(valid_keys) + i + 1
            multi_tasks.append(test_api_key_async(name, key, request_id))
    
    start_time = time.time()
    multi_results = await asyncio.gather(*multi_tasks)
    total_time = (time.time() - start_time) * 1000
    
    print()
    for result in multi_results:
        if result["success"]:
            print(f"   ✅ {result['key_name']} (req #{result['request_num']}): {result['response_time_ms']}ms")
        else:
            print(f"   ❌ {result['key_name']} (req #{result['request_num']}): {result['error']}")
    
    print(f"\n   ⏱️  Total concurrent time: {round(total_time, 1)}ms")
    print(f"   📊 If sequential, would be: ~{round(sum(r['response_time_ms'] for r in multi_results if r['success']), 1)}ms")
    print(f"   🚀 Speedup: {round(sum(r['response_time_ms'] for r in multi_results if r['success']) / total_time, 1)}x")
    print()
    
    # Summary
    print("=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    all_results = results + multi_results
    success_count = sum(1 for r in results if r["success"])
    total_requests = sum(1 for r in all_results if r["success"])
    
    for key_name, _ in keys.items():
        key_results = [r for r in all_results if r["key_name"] == key_name]
        successes = sum(1 for r in key_results if r["success"])
        if key_results:
            status = "✅" if all(r["success"] for r in key_results) else "⚠️"
            print(f"   {status} {key_name}: {successes}/{len(key_results)} requests succeeded")
        else:
            print(f"   ❌ {key_name}: NOT CONFIGURED")
    
    print()
    print(f"   Total: {success_count}/{len(valid_keys)} keys working")
    print(f"   Requests: {total_requests}/{len(all_results)} succeeded")
    print("=" * 60)
    
    if success_count == len(valid_keys):
        print("🎉 All API keys are valid and working concurrently!")
        return 0
    elif success_count > 0:
        print("⚠️  Some keys failed - check the errors above")
        return 1
    else:
        print("❌ All keys failed!")
        return 2


def main():
    return asyncio.run(run_concurrent_test())


if __name__ == "__main__":
    sys.exit(main())
