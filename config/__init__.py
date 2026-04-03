"""
LangGraph-based PyKGML configuration agent.

Usage (direct graph invoke):
    from config_LangGraph import create_config_graph
    graph = create_config_graph(llm)
    state = {"messages": [], "user_input": "I want to create a model structure"}
    result = graph.invoke(state)

Usage (server-friendly one-turn runner; use this in agent_server_app or similar):
    from config_LangGraph import create_config_graph, run_one_turn
    graph = create_config_graph(llm)
    result = run_one_turn(graph, user_message, previous_state)
    # result["output"], result["state"], result["complete"], result["generated_code"]
"""

from .state import ConfigAgentState
from .graph import create_config_graph
from .templates import get_model_structure_template, get_loss_function_template, get_required_fields
from .runner import run_one_turn

__all__ = [
    "ConfigAgentState",
    "create_config_graph",
    "run_one_turn",
    "get_model_structure_template",
    "get_loss_function_template",
    "get_required_fields",
]
