
import sys
import os
import json
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from src.agent.intent_analyzer import analyze_intent
from src.agent.core import QwenAgent

def test_analyze_intent():
    print("Testing analyze_intent...")
    mock_llm = MagicMock()
    
    # Mock response for analyze_intent
    mock_json = json.dumps({
        "goal": "Find bitcoin price",
        "language": "Vietnamese"
    })
    
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": mock_json}}]
    }
    
    result = analyze_intent(mock_llm, "Giá Bitcoin là bao nhiêu?")
    print(f"Result: {result}")
    
    assert result['goal'] == "Find bitcoin price"
    assert result['language'] == "Vietnamese"
    print("✅ analyze_intent Passed")

def test_agent_integration():
    print("\nTesting QwenAgent integration...")
    mock_llm = MagicMock()
    mock_rag = MagicMock()
    
    agent = QwenAgent(mock_llm, mock_rag)
    
    # We need to mock the sequence of LLM calls:
    # 1. analyze_intent
    # 2. analyze_request (planner)
    # 3. generate (step 1) -> Tool Call
    # 4. generate (step 2) -> Final Answer
    
    intent_resp = json.dumps({"goal": "Test Goal", "language": "Test Lang"})
    planner_resp = "PLAN: Just answer."
    step1_resp = "I will answer."
    
    # Mocking create_chat_completion side effects
    # Note: analyze_intent calls create_chat_completion
    # analyze_request calls create_chat_completion
    # generate calls create_chat_completion
    
    mock_llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": intent_resp}}]}, # Intent
        {"choices": [{"message": {"content": planner_resp}}]}, # Planner
        {"choices": [{"message": {"content": step1_resp}}]}   # Generate
    ]
    
    try:
        agent.run("Hello")
        print("✅ QwenAgent.run executed without crashing")
        
        # Verify analyze_intent was called (it's the first call)
        args, _ = mock_llm.create_chat_completion.call_args_list[0]
        # The prompt should be the intent prompt
        assert "Intent Classifier" in args[0][0]['content']
        print("✅ Intent prompt verification successful")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    test_analyze_intent()
    test_agent_integration()
