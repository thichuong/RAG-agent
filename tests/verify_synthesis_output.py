import unittest
from unittest.mock import MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.agent.nodes.synthesis import SynthesisNode

class TestSynthesisNode(unittest.TestCase):
    def test_output_format(self):
        # Mock LLM
        mock_llm = MagicMock()
        expected_content = "This is the final answer."
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": expected_content
                    }
                }
            ]
        }
        mock_llm.create_chat_completion.return_value = mock_response

        # Initialize Node
        node = SynthesisNode(mock_llm)

        # Mock State
        state = {
            "messages": [{"role": "user", "content": "Hello"}],
            "intent": {"goal": "Answer", "language": "English"},
            "logs": []
        }

        # Run Node
        result = node(state)

        # Assertions
        self.assertEqual(result["final_answer"], expected_content)
        self.assertIn(f"🤖 **Final Answer**: {expected_content}", result["logs"][1])

if __name__ == "__main__":
    unittest.main()
