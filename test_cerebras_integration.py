#!/usr/bin/env python3
"""
Test script to validate Cerebras integration with Biomni framework
"""

import os
import sys
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   You can still use environment variables without .env file support.")

# Add Biomni to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Biomni'))

from biomni.config import BiomniConfig
from biomni.llm import get_llm


def test_cerebras_integration():
    """Test Cerebras as custom provider in Biomni."""
    
    print("🧪 Testing Cerebras integration with Biomni...")
    
    # Configure for Cerebras
    cerebras_config = BiomniConfig(
        llm="zai-glm-4.6",  # Example fast model
        base_url="https://api.cerebras.ai/v1",
        api_key=os.getenv("CEREBRAS_API_KEY"),  # User needs to set this
        source="Custom",
        temperature=0.1
    )
    
    try:
        print(f"📡 Connecting to Cerebras at: {cerebras_config.base_url}")
        print(f"🤖 Using model: {cerebras_config.llm}")
        
        # Get LLM instance
        llm = get_llm(
            model=cerebras_config.llm,
            temperature=cerebras_config.temperature,
            source=cerebras_config.source,
            base_url=cerebras_config.base_url,
            api_key=cerebras_config.api_key,
            config=cerebras_config
        )
        
        print("✅ Successfully created LLM instance")
        
        # Test with a simple biomedical query
        test_query = "What is TP53 and why is it important in cancer research?"
        
        print(f"🔬 Testing query: {test_query}")
        start_time = time.time()
        
        response = llm.invoke(test_query)
        
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"⚡ Response time: {latency:.2f} seconds")
        print(f"📝 Response: {response.content[:200]}...")
        
        return {
            "success": True,
            "latency": latency,
            "response_preview": response.content[:200],
            "model": cerebras_config.llm
        }
        
    except Exception as e:
        print(f"❌ Error testing Cerebras integration: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "model": cerebras_config.llm
        }


def test_openai_comparison():
    """Test OpenAI for comparison."""
    
    print("\n🔄 Testing OpenAI for comparison...")
    
    openai_config = BiomniConfig(
        llm="gpt-4",
        source="OpenAI",
        temperature=0.1
    )
    
    try:
        llm = get_llm(config=openai_config)
        print("✅ Successfully created OpenAI LLM instance")
        
        test_query = "What is TP53 and why is it important in cancer research?"
        print(f"🔬 Testing query: {test_query}")
        
        start_time = time.time()
        response = llm.invoke(test_query)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"⚡ Response time: {latency:.2f} seconds")
        print(f"📝 Response: {response.content[:200]}...")
        
        return {
            "success": True,
            "latency": latency,
            "response_preview": response.content[:200],
            "model": openai_config.llm
        }
        
    except Exception as e:
        print(f"❌ Error testing OpenAI: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "model": openai_config.llm
        }


def main():
    """Run integration tests."""
    
    print("🚀 Biomni-Cerebras Integration Test")
    print("=" * 50)
    
    # Check for required API keys
    if not os.getenv("CEREBRAS_API_KEY"):
        print("⚠️  CEREBRAS_API_KEY not found in environment variables.")
        print("Please set it with: export CEREBRAS_API_KEY=your_key_here")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping OpenAI comparison.")
    
    # Test Cerebras
    cerebras_result = test_cerebras_integration()
    
    # Test OpenAI if available
    openai_result = None
    if os.getenv("OPENAI_API_KEY"):
        openai_result = test_openai_comparison()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    if cerebras_result["success"]:
        print(f"✅ Cerebras ({cerebras_result['model']}): {cerebras_result['latency']:.2f}s")
    else:
        print(f"❌ Cerebras: {cerebras_result['error']}")
    
    if openai_result:
        if openai_result["success"]:
            print(f"✅ OpenAI ({openai_result['model']}): {openai_result['latency']:.2f}s")
        else:
            print(f"❌ OpenAI: {openai_result['error']}")
    
    if cerebras_result["success"] and openai_result and openai_result["success"]:
        speedup = openai_result["latency"] / cerebras_result["latency"]
        print(f"🚀 Speedup: {speedup:.2f}x faster with Cerebras")


if __name__ == "__main__":
    main()
