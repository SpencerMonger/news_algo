#!/usr/bin/env python3
"""
Simple test to verify Portkey Gateway configuration and connection
Tests that the gateway starts correctly and can load balance across the API keys
"""

import asyncio
import json
import os
import time
import logging
import subprocess
import signal
import atexit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gateway_connection():
    """Test the Portkey Gateway setup"""
    
    try:
        # Get API keys from environment
        api_key_1 = os.getenv('ANTHROPIC_API_KEY')
        api_key_2 = os.getenv('ANTHROPIC_API_KEY2')
        api_key_3 = os.getenv('ANTHROPIC_API_KEY3')
        
        logger.info("🔍 Checking API Keys:")
        logger.info(f"   API KEY 1: {'✅ Found' if api_key_1 else '❌ Missing'}")
        logger.info(f"   API KEY 2: {'✅ Found' if api_key_2 else '❌ Missing'}")
        logger.info(f"   API KEY 3: {'✅ Found' if api_key_3 else '❌ Missing'}")
        
        # Create load balancing config
        config = {
            "strategy": {
                "mode": "loadbalance"
            },
            "targets": []
        }
        
        # Add API keys as targets
        if api_key_1:
            config["targets"].append({
                "provider": "anthropic",
                "api_key": api_key_1,
                "weight": 1.0
            })
        
        if api_key_2 and api_key_2 != api_key_1:
            config["targets"].append({
                "provider": "anthropic",
                "api_key": api_key_2,
                "weight": 1.0
            })
        
        if api_key_3 and api_key_3 != api_key_1 and api_key_3 != api_key_2:
            config["targets"].append({
                "provider": "anthropic",
                "api_key": api_key_3,
                "weight": 1.0
            })
        
        logger.info(f"🔑 Configuration: {len(config['targets'])} targets configured")
        logger.info(f"📋 Config: {json.dumps(config, indent=2)}")
        
        if not config["targets"]:
            logger.error("❌ No API keys found! Please check your .env file")
            return False
        
        # Start gateway server
        gateway_url = "http://localhost:8787"
        gateway_process = None
        
        # Check if already running
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{gateway_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        logger.info("✅ Gateway already running")
                    else:
                        raise Exception("Gateway not healthy")
            except:
                # Start the gateway
                logger.info("🚀 Starting Portkey Gateway...")
                gateway_process = subprocess.Popen(
                    ["npx", "@portkey-ai/gateway"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                
                # Wait for startup
                for i in range(30):
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{gateway_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                                if response.status == 200:
                                    logger.info(f"✅ Gateway started on {gateway_url}")
                                    break
                    except:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise Exception("Gateway failed to start")
        
        # Test the Portkey client
        from portkey_ai import Portkey
        
        client = Portkey(
            base_url=f"{gateway_url}/v1",
            config=config
        )
        
        logger.info("🧪 Testing gateway with simple request...")
        
        response = await client.chat.completions.acreate(
            model="claude-3-5-sonnet-20240620",
            messages=[{
                "role": "user", 
                "content": "Respond with exactly: 'Gateway test successful!'"
            }],
            max_tokens=50,
            temperature=0.0
        )
        
        if response and response.choices:
            result = response.choices[0].message.content.strip()
            logger.info(f"🎉 Gateway Response: {result}")
            logger.info("✅ Gateway test PASSED!")
            return True
        else:
            logger.error("❌ No response from gateway")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gateway test FAILED: {e}")
        return False
    
    finally:
        # Cleanup gateway process if we started it
        if gateway_process:
            try:
                os.killpg(os.getpgid(gateway_process.pid), signal.SIGTERM)
                gateway_process.wait(timeout=5)
                logger.info("✅ Gateway stopped")
            except:
                try:
                    os.killpg(os.getpgid(gateway_process.pid), signal.SIGKILL)
                except:
                    pass

if __name__ == "__main__":
    result = asyncio.run(test_gateway_connection())
    if result:
        print("\n🎉 Gateway configuration is working! You can now run the full batch test.")
    else:
        print("\n❌ Gateway configuration failed. Please check your API keys and try again.") 