"""
One-turn runner for the LangGraph config agent.

This module is not used inside the config_LangGraph package; it is intended for
the server (e.g. backend_server.py) or any script that wants to run one
user turn and get back a server-friendly result: output, state, complete, generated_code.

Usage:
    from config_LangGraph import create_config_graph, run_one_turn
    graph = create_config_graph(llm)
    result = run_one_turn(graph, user_message, previous_state)
    # result["output"], result["state"], result["complete"], result["generated_code"]
"""

from typing import Any, Dict


def run_one_turn(
    graph,
    user_input: str,
    state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run one turn of the config agent.

    Args:
        graph: Compiled graph from create_config_graph(llm)
        user_input: Latest user message
        state: Previous state (messages, config, script_type, etc.) or None for first turn

    Returns:
        {
            "output": str,           # Assistant message to show
            "state": dict,           # Full state to persist for next turn
            "complete": bool,        # True when config is done and code was generated
            "generated_code": str|None,
        }
    """
    state = state or {}
    # Ensure we pass user_input; graph expects it for this turn
    invoke_state = {
        "messages": state.get("messages", []),
        "user_input": user_input,
        "script_type": state.get("script_type"),
        "config": state.get("config"),
        "next_field": state.get("next_field"),
        "current_field": state.get("current_field"),
        "complete": state.get("complete", False),
        "needs_confirmation": state.get("needs_confirmation", False),
        "layers_phase": state.get("layers_phase"),
        "current_layer_name": state.get("current_layer_name"),
        "forward_phase": state.get("forward_phase"),
        "forward_steps": state.get("forward_steps", []),
        "loss_phase": state.get("loss_phase"),
        "loss_term_index": state.get("loss_term_index"),
        "forward_valid": state.get("forward_valid"),
    }
    result = graph.invoke(invoke_state)
    output = result.get("output", "")
    # Persist full graph state for next turn (omit user_input; it's per-turn)
    next_state = {k: v for k, v in result.items() if k != "user_input"}
    if "generated_code" in result:
        next_state["generated_code"] = result["generated_code"]
    return {
        "output": output,
        "state": next_state,
        "complete": result.get("complete", False),
        "generated_code": result.get("generated_code"),
    }
