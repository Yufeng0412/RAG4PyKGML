"""
LangGraph-based PyKGML configuration agent.

Usage:
    from config_LangGraph import create_config_graph

    graph = create_config_graph(llm)
    state = {"messages": [], "user_input": "I want to create a model structure"}
    result = graph.invoke(state)
    # result["output"] is the assistant reply; persist result as state for next turn.
"""

from .state import ConfigAgentState
from .graph import create_config_graph
from .templates import get_model_structure_template, get_loss_function_template, get_required_fields

__all__ = [
    "ConfigAgentState",
    "create_config_graph",
    "get_model_structure_template",
    "get_loss_function_template",
    "get_required_fields",
]
