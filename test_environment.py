#!/usr/bin/env python3
"""
Environment Verification Script for Multi-Agent Demo
Tests all components before running the full demo
"""

import os
import sys
import time
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   You can still use environment variables without .env file support.")

def test_environment():
    """Verify all components are properly configured."""
    
    print("🔧 Multi-Agent Demo Environment Verification")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Python version
    print("\n1. Testing Python version...")
    if sys.version_info >= (3, 8):
        print("   ✅ Python version OK")
    else:
        print("   ❌ Python 3.8+ required")
        all_tests_passed = False
    
    # Test 2: Required packages
    print("\n2. Testing required packages...")
    required_packages = [
        'biomni', 'langchain', 'pandas', 'requests', 
        'matplotlib', 'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} installed")
        except ImportError:
            print(f"   ❌ {package} missing - run: pip install {package}")
            all_tests_passed = False
    
    # Test 3: Environment variables
    print("\n3. Testing API keys...")
    
    if os.getenv("CEREBRAS_API_KEY"):
        print("   ✅ CEREBRAS_API_KEY set")
    else:
        print("   ❌ CEREBRAS_API_KEY not set")
        print("      Run: export CEREBRAS_API_KEY=your_key_here")
        all_tests_passed = False
    
    if os.getenv("OPENAI_API_KEY"):
        print("   ✅ OPENAI_API_KEY set")
    else:
        print("   ❌ OPENAI_API_KEY not set")
        print("      Run: export OPENAI_API_KEY=your_key_here")
        all_tests_passed = False
    
    # Test 4: Biomni framework
    print("\n4. Testing Biomni framework...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Biomni'))
        from biomni.config import BiomniConfig
        from biomni.llm import get_llm
        print("   ✅ Biomni framework accessible")
    except ImportError as e:
        print(f"   ❌ Biomni framework error: {e}")
        all_tests_passed = False
    
    # Test 5: Cerebras connectivity (quick test)
    print("\n5. Testing Cerebras connectivity...")
    if os.getenv("CEREBRAS_API_KEY"):
        try:
            from biomni.config import BiomniConfig
            from biomni.llm import get_llm
            
            config = BiomniConfig(
                llm="zai-glm-4.6",
                source="Custom",
                base_url="https://api.cerebras.ai/v1",
                api_key=os.getenv("CEREBRAS_API_KEY"),
                temperature=0.1
            )
            
            llm = get_llm(config=config)
            
            # Quick test call
            start_time = time.time()
            response = llm.invoke("Say 'Cerebras test OK'")
            test_time = time.time() - start_time
            
            if "OK" in response.content:
                print(f"   ✅ Cerebras API working ({test_time:.2f}s)")
            else:
                print("   ❌ Cerebras API unexpected response")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ Cerebras API error: {e}")
            all_tests_passed = False
    else:
        print("   ⚠️  Skipping Cerebras test (no API key)")
    
    # Test 6: OpenAI connectivity (quick test)
    print("\n6. Testing OpenAI connectivity...")
    if os.getenv("OPENAI_API_KEY"):
        try:
            from biomni.llm import get_llm
            
            llm = get_llm(model="gpt-5-mini", source="OpenAI", temperature=0.1)
            
            # Quick test call
            start_time = time.time()
            response = llm.invoke("Say 'OpenAI test OK'")
            test_time = time.time() - start_time
            
            if "OK" in response.content:
                print(f"   ✅ OpenAI API working ({test_time:.2f}s)")
            else:
                print("   ❌ OpenAI API unexpected response")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ OpenAI API error: {e}")
            all_tests_passed = False
    else:
        print("   ⚠️  Skipping OpenAI test (no API key)")
    
    # Test 7: cBioPortal connectivity
    print("\n7. Testing cBioPortal connectivity...")
    try:
        import requests
        
        response = requests.get("https://www.cbioportal.org/api/studies", timeout=10)
        if response.status_code == 200:
            studies = response.json()
            print(f"   ✅ cBioPortal API accessible ({len(studies)} studies)")
        else:
            print(f"   ❌ cBioPortal API error: {response.status_code}")
            all_tests_passed = False
            
    except Exception as e:
        print(f"   ❌ cBioPortal connectivity error: {e}")
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED - Environment ready for demo!")
        print("\nNext steps:")
        print("   python final_demo.py    # Run full demo")
        print("   python enhanced_demo.py # Run comparative demo")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before running demo")
        print("\nTroubleshooting:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Set API keys: export CEREBRAS_API_KEY=...")
        print("   3. Check network connectivity")
    
    return all_tests_passed


if __name__ == "__main__":
    test_environment()
