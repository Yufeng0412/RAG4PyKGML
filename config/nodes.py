"""LangGraph nodes for the config agent."""

import json
from typing import Any, Dict

from .state import ConfigAgentState
from .templates import (
    get_model_structure_template,
    get_loss_function_template,
    get_next_missing_field,
)
from .prompts import get_question_for_field, CONFIRM_MESSAGE, CONFIRM_ASK
from .extractor import extract_value_for_field, parse_dict_from_text


def receive(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Normalize user input and append to messages. llm not used here."""
    messages = list(state.get("messages") or [])
    user_input = (state.get("user_input") or "").strip()
    if isinstance(user_input, list) and len(user_input) > 0:
        last = user_input[-1]
        user_input = last.get("content", str(last)) if isinstance(last, dict) else str(last)
    user_input = str(user_input).strip()
    if user_input:
        messages.append({"role": "user", "content": user_input})
    return {"messages": messages, "user_input": user_input}


def select_script_type(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Detect script type from user message and initialize config."""
    user_input = (state.get("user_input") or "").strip().lower()
    script_type = None
    if "model" in user_input and "structure" in user_input:
        script_type = "model_structure"
    elif "loss" in user_input and "function" in user_input:
        script_type = "loss_function"
    if not script_type:
        return {
            "output": "Please choose: **I want to create a model structure** or **I want to create a loss function**.",
        }
    config = get_model_structure_template() if script_type == "model_structure" else get_loss_function_template()
    next_field = get_next_missing_field(script_type, config)
    return {
        "script_type": script_type,
        "config": config,
        "next_field": next_field,
        "current_field": next_field,
        "complete": False,
        "needs_confirmation": False,
        "output": get_question_for_field(script_type, next_field) if next_field else CONFIRM_MESSAGE,
    }


def extract(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Extract value for current_field from user message and update config."""
    user_input = (state.get("user_input") or "").strip()
    script_type = state.get("script_type")
    config = dict(state.get("config") or {})
    current_field = state.get("current_field")
    if not current_field or not script_type:
        return {"current_field": None}
    value, ok = extract_value_for_field(user_input, current_field, script_type or "", llm)
    if not ok or value is None:
        return {
            "output": f"I couldn't parse **{current_field}**. Please provide a valid value (e.g. a Python dict).",
            "current_field": current_field,
        }
    if current_field in ("init_params", "parameters", "variables", "layers", "forward", "loss_formula"):
        if isinstance(value, dict):
            config[current_field] = value
        else:
            return {"output": f"**{current_field}** must be a dictionary.", "current_field": current_field}
    else:
        config[current_field] = value
    next_field = get_next_missing_field(script_type, config)
    complete = next_field is None
    out_msg = CONFIRM_MESSAGE if complete else get_question_for_field(script_type, next_field)
    return {
        "config": config,
        "current_field": next_field if not complete else None,
        "next_field": next_field,
        "complete": complete,
        "needs_confirmation": complete,
        "output": out_msg,
    }


def decide_next(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Compute next missing field or mark complete. Used after extract or when no current_field."""
    script_type = state.get("script_type")
    config = state.get("config") or {}
    next_field = get_next_missing_field(script_type, config)
    complete = next_field is None
    if complete:
        return {
            "next_field": None,
            "current_field": None,
            "complete": True,
            "needs_confirmation": True,
            "output": CONFIRM_MESSAGE,
        }
    return {
        "next_field": next_field,
        "current_field": next_field,
        "complete": False,
        "needs_confirmation": False,
        "output": get_question_for_field(script_type, next_field),
    }


def ask_question(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Produce the question for next_field and set current_field so next turn we extract."""
    script_type = state.get("script_type")
    next_field = state.get("next_field")
    if not next_field:
        out = CONFIRM_MESSAGE
        return {"output": out, "current_field": None, "needs_confirmation": True, "complete": True}
    out = get_question_for_field(script_type, next_field)
    return {"output": out, "current_field": next_field}


def confirm(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """If user confirmed, go to generate_code; else ask for confirmation."""
    user_input = (state.get("user_input") or "").strip().lower()
    confirmed = user_input in ("yes", "confirm", "y", "ok", "generate")
    if confirmed:
        return {"output": "[Generating code...]"}
    return {"output": CONFIRM_ASK}


def generate_code(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Turn config into PyKGML config code string."""
    script_type = state.get("script_type")
    config = state.get("config") or {}
    if script_type == "model_structure":
        code = "archt_config = " + json.dumps(config, indent=2)
    elif script_type == "loss_function":
        code = "lossfn_config = " + json.dumps(config, indent=2)
    else:
        code = json.dumps(config, indent=2)
    return {
        "output": "Here is your configuration:\n\n```python\n" + code + "\n```",
        "generated_code": code,
        "complete": True,
        "needs_confirmation": False,
    }


def route_after_receive(state: ConfigAgentState) -> str:
    """Route from receive: select_script_type | confirm | extract | decide_next."""
    script_type = state.get("script_type")
    current_field = state.get("current_field")
    user_input = (state.get("user_input") or "").strip()
    if not script_type:
        return "select_script_type"
    if state.get("complete") and state.get("needs_confirmation") and user_input:
        return "confirm"
    if current_field and user_input:
        return "extract"
    return "decide_next"


def route_after_select_script_type(state: ConfigAgentState) -> str:
    """After select_script_type we already set first question; go to END via ask_question."""
    return "ask_question"


def route_after_extract(state: ConfigAgentState) -> str:
    """After extract: if complete -> confirm, else END (output already set)."""
    if state.get("complete"):
        return "confirm"
    return "__end__"


def route_after_decide_next(state: ConfigAgentState) -> str:
    """After decide_next: ask_question or confirm."""
    if state.get("complete") and state.get("needs_confirmation"):
        return "confirm"
    return "ask_question"


def route_after_confirm(state: ConfigAgentState) -> str:
    """After confirm: generate_code if user said yes, else END (we already set output)."""
    user_input = (state.get("user_input") or "").strip().lower()
    if user_input in ("yes", "confirm", "y", "ok", "generate"):
        return "generate_code"
    return "__end__"
