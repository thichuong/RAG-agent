from .analyze_intent import AnalyzeIntentNode
from .planning import PlanningNode
from .generate import GenerateNode
from .execute_tools import ExecuteToolsNode
from .synthesis import SynthesisNode

class GraphNodes:
    def __init__(self, llm, rag):
        self.analyze_intent_node = AnalyzeIntentNode(llm)
        self.planning_node = PlanningNode(llm)
        self.generate_node = GenerateNode(llm)
        self.execute_tools_node = ExecuteToolsNode(llm, rag)
        self.synthesis_node = SynthesisNode(llm)
