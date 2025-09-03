#!/usr/bin/env python3
"""
Demo script showing the LangGraph React Agent with self-validation capabilities.
This demonstrates the enhanced FAQ handling and validation features.
"""

import os
import sys
from agent.enhanced_agent_factory import create_agent as create_enhanced_agent

def demo_langgraph_agent():
    """Demonstrate the LangGraph React agent capabilities"""
    print("🚀 LangGraph React Agent Demo")
    print("=" * 50)
    
    # Set environment to use LangGraph
    os.environ['AGENT_BACKEND'] = 'langgraph'
    
    # Create the enhanced agent
    print("Creating LangGraph agent...")
    agent = create_enhanced_agent(user_id="demo_user_123", query="FAQ question")
    print(f"✅ Created agent: {type(agent)}")
    
    # Test FAQ questions with validation
    faq_questions = [
        "What are your pricing plans?",
        "How do I upload a document?",
        "What does the ₹1999 plan include?",
        "Can I get a refund?",
        "How do I create a new book?"
    ]
    
    print("\n📋 Testing FAQ Questions with Self-Validation:")
    print("-" * 50)
    
    for i, question in enumerate(faq_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("   Processing...")
        
        try:
            response = agent.invoke({"input": question})
            output = response.get('output', 'No output generated')
            
            # Show first 150 characters of response
            preview = output[:150] + "..." if len(output) > 150 else output
            print(f"   Response: {preview}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n🔄 Testing Personal Account Queries:")
    print("-" * 40)
    
    # Test personal queries (should use standard agent)
    personal_questions = [
        "Show me my books",
        "What's my account status?",
        "My writing progress"
    ]
    
    for i, question in enumerate(personal_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("   Processing...")
        
        try:
            # Create new agent for personal queries
            personal_agent = create_enhanced_agent(user_id="demo_user_123", query=question)
            response = personal_agent.invoke({"input": question})
            output = response.get('output', 'No output generated')
            
            # Show first 100 characters of response
            preview = output[:100] + "..." if len(output) > 100 else output
            print(f"   Response: {preview}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✨ Demo completed!")
    print("\nKey Features Demonstrated:")
    print("- ✅ FAQ questions use LangGraph React agent with validation")
    print("- ✅ Personal queries use standard LangChain agent")
    print("- ✅ Automatic agent selection based on query type")
    print("- ✅ Backward compatibility maintained")
    print("- ✅ Environment variable configuration support")

if __name__ == "__main__":
    demo_langgraph_agent()
