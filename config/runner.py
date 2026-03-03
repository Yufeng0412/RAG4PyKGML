"""
One-turn runner for the LangGraph config agent.
Use from a server or script: pass current state + user message, get back output and updated state.
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
