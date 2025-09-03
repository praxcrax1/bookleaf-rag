"""
Test script for the new LangGraph React agent implementation.
This script tests both backward compatibility and new features.
"""
import sys
import os
import asyncio
import logging

# Add the server directory to the path
sys.path.append('/home/abhay/Desktop/Repos/bookleaf-rag/server')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backward_compatibility():
    """Test that the original interface still works"""
    print("\n" + "="*50)
    print("TESTING BACKWARD COMPATIBILITY")
    print("="*50)
    
    try:
        from agent import create_agent
        
        # Test with original interface
        user_id = "test_user_123"
        query = "What are the pricing plans?"
        
        print(f"Creating agent for user: {user_id}")
        print(f"Query: {query}")
        
        # Create agent (should use LangChain by default)
        agent = create_agent(user_id=user_id, query=query)
        print(f"✅ Agent created successfully: {type(agent)}")
        
        # Test invoke method
        response = agent.invoke({"input": query})
        print(f"✅ Agent response received")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        
        if isinstance(response, dict) and "output" in response:
            print(f"Output preview: {response['output'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_langgraph_agent():
    """Test the new LangGraph agent"""
    print("\n" + "="*50)
    print("TESTING LANGGRAPH AGENT")
    print("="*50)
    
    try:
        from agent import create_agent
        
        # Test with LangGraph backend
        user_id = "test_user_456"
        query = "How do I upload a document?"
        
        print(f"Creating LangGraph agent for user: {user_id}")
        print(f"Query: {query}")
        
        # Create agent with LangGraph backend
        agent = create_agent(user_id=user_id, query=query, use_langgraph=True)
        print(f"✅ LangGraph agent created successfully: {type(agent)}")
        
        # Test invoke method
        response = agent.invoke({"input": query})
        print(f"✅ LangGraph agent response received")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        
        if isinstance(response, dict) and "output" in response:
            print(f"Output preview: {response['output'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LangGraph agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_agent():
    """Test the hybrid agent selection"""
    print("\n" + "="*50)
    print("TESTING HYBRID AGENT")
    print("="*50)
    
    try:
        from agent import create_hybrid_agent
        
        # Test FAQ query (should use LangGraph)
        user_id = "test_user_789"
        faq_query = "How to create a new book?"
        
        print(f"Testing FAQ query: {faq_query}")
        agent_faq = create_hybrid_agent(user_id=user_id, query=faq_query)
        print(f"✅ FAQ agent created: {type(agent_faq)}")
        
        # Test personal query (should use LangChain)
        personal_query = "Show me my books"
        print(f"Testing personal query: {personal_query}")
        agent_personal = create_hybrid_agent(user_id=user_id, query=personal_query)
        print(f"✅ Personal agent created: {type(agent_personal)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variable():
    """Test environment variable configuration"""
    print("\n" + "="*50)
    print("TESTING ENVIRONMENT VARIABLE CONFIGURATION")
    print("="*50)
    
    try:
        # Test with AGENT_BACKEND=langgraph
        os.environ["AGENT_BACKEND"] = "langgraph"
        
        from agent import create_agent
        
        user_id = "test_user_env"
        query = "Test environment variable"
        
        print(f"AGENT_BACKEND set to: {os.environ.get('AGENT_BACKEND')}")
        agent = create_agent(user_id=user_id, query=query)
        print(f"✅ Agent created with env var: {type(agent)}")
        
        # Reset environment variable
        os.environ["AGENT_BACKEND"] = "langchain"
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Starting LangGraph React Agent Tests")
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("LangGraph Agent", test_langgraph_agent),
        ("Hybrid Agent", test_hybrid_agent),
        ("Environment Variable", test_environment_variable)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! LangGraph React agent is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
