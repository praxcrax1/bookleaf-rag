# LangGraph React Agent Implementation

This implementation adds a new LangGraph-based React agent with self-validation capabilities while maintaining full backward compatibility with the existing LangChain agent system.

## 🚀 Features

### LangGraph React Agent
- **Self-Validation**: FAQ responses are validated and may trigger additional searches for accuracy
- **Enhanced Reasoning**: Uses React (Reason + Act) pattern for better decision making
- **Conversation Memory**: Persistent chat history using MongoDB checkpointer
- **Tool Integration**: Same tools as original system (`faq_and_support`, `user_book_lookup`)

### Backward Compatibility
- **Drop-in Replacement**: Existing code continues to work without changes
- **Same Interface**: Maintains identical input/output format as LangChain agents
- **Gradual Migration**: Can be enabled per-query or via environment variable

## 📁 File Structure

```
server/agent/
├── __init__.py                 # Updated exports
├── agent_factory.py            # Original LangChain agent (unchanged)
├── enhanced_agent_factory.py   # Smart agent selection logic
├── langgraph_agent.py          # New LangGraph React agent
├── langgraph_wrapper.py        # Compatibility wrapper
└── tools.py                    # Shared tools
```

## 🔧 Usage

### 1. Basic Usage (Automatic Selection)
```python
from agent.enhanced_agent_factory import create_enhanced_agent

# Automatically selects best agent based on query type
agent = create_enhanced_agent(user_id="user123", query="What are your pricing plans?")
response = agent.invoke({"input": "What are your pricing plans?"})
```

### 2. Force LangGraph Agent
```python
import os
os.environ['AGENT_BACKEND'] = 'langgraph'

agent = create_enhanced_agent(user_id="user123")
response = agent.invoke({"input": "How do I upload a document?"})
```

### 3. Force LangChain Agent (Default)
```python
import os
os.environ['AGENT_BACKEND'] = 'langchain'

agent = create_enhanced_agent(user_id="user123")
response = agent.invoke({"input": "Show me my books"})
```

### 4. Original Agent (Still Available)
```python
from agent.agent_factory import create_agent

# Original implementation, unchanged
agent = create_agent(user_id="user123")
response = agent.invoke({"input": "Any question"})
```

## 🧠 Intelligent Agent Selection

The enhanced factory automatically chooses the best agent:

### LangGraph React Agent (with validation)
- FAQ questions: "How to...", "What is...", "Can I..."
- Technical support: "I can't...", "Not working..."
- Policy questions: "Refund policy", "Terms..."
- General inquiries about features, pricing, etc.

### LangChain Agent (faster, direct)
- Personal queries: "My books", "My account"
- User-specific data: "Show me...", "My progress"
- Account management: "My subscription"

## ⚙️ Configuration

### Environment Variables
```bash
# Use LangGraph for all queries
export AGENT_BACKEND=langgraph

# Use LangChain for all queries (default)
export AGENT_BACKEND=langchain

# Let system auto-select (recommended)
unset AGENT_BACKEND
```

### MongoDB Integration
The LangGraph agent uses MongoDB for conversation persistence:
- Collection: `chat_histories`
- Session ID: Based on `user_id`
- Automatic conversation continuity

## 🛠️ Tools Available

Both agents have access to the same tools:

### 1. FAQ and Support Tool
```python
def faq_and_support(query: str) -> str:
    """Search FAQ database and support documentation"""
```

### 2. User Book Lookup Tool
```python
def user_book_lookup(user_id: str) -> str:
    """Look up user's books and account information"""
```

## 📊 Performance Comparison

| Feature | LangChain Agent | LangGraph Agent |
|---------|----------------|-----------------|
| Response Speed | Fast | Moderate |
| Validation | None | Self-validation |
| Reasoning | Basic | Enhanced (React) |
| Memory | Buffer | Persistent (MongoDB) |
| Tool Usage | Direct | Validated |
| Best For | Personal queries | FAQ/Support |

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd server
source venv/bin/activate
python test_langgraph_agent.py
```

Run the demo:
```bash
python demo_langgraph_agent.py
```

## 🔄 Migration Path

### Phase 1: Parallel Operation (Current)
- Both systems running side-by-side
- Automatic selection based on query type
- Zero breaking changes

### Phase 2: Gradual Rollout
- Use environment variable to control usage
- Monitor performance and accuracy
- Gather user feedback

### Phase 3: Full Migration
- Default to LangGraph for all FAQ queries
- Keep LangChain for personal queries
- Remove old agent if desired

## 🚨 Error Handling

The system includes comprehensive error handling:

1. **Graceful Fallback**: If LangGraph fails, falls back to LangChain
2. **Error Logging**: All errors logged with context
3. **User-Friendly Messages**: Clear error messages for users
4. **Retry Logic**: Automatic retry for transient failures

## 📝 Response Format

Both agents return identical response format:
```python
{
    "input": "User's question",
    "output": "Agent's response", 
    "intermediate_steps": []  # Tool calls and reasoning
}
```

## 🔍 Debugging

Enable verbose logging:
```python
import logging
logging.getLogger('agent').setLevel(logging.DEBUG)
```

Check which agent was used:
```python
response = agent.invoke({"input": "question"})
print(f"Agent type: {type(agent)}")
```

## 📚 Dependencies

Required packages (already in requirements.txt):
- `langgraph`: For React agent implementation
- `langchain-google-genai`: For LLM integration
- `langchain-mongodb`: For chat history persistence

## 🤝 Contributing

When adding new features:
1. Update both agent implementations if needed
2. Add tests to `test_langgraph_agent.py`
3. Update documentation
4. Ensure backward compatibility

## 📖 References

- [LangGraph Agentic RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- [LangGraph Agents Documentation](https://langchain-ai.github.io/langgraph/agents/run_agents/)
- [LangGraph Tools Reference](https://langchain-ai.github.io/langgraph/concepts/tools/)
