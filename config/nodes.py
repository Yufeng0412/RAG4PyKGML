"""LangGraph nodes for the config agent."""

import copy
import json
import re
from pprint import pformat
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
    next_field = get_next_missing_field(script_type, config, state={})
    return {
        "script_type": script_type,
        "config": config,
        "next_field": next_field,
        "current_field": next_field,
        "complete": False,
        "needs_confirmation": False,
        "output": get_question_for_field(script_type, next_field) if next_field else CONFIRM_MESSAGE,
        "layers_phase": None,
        "current_layer_name": None,
        "forward_phase": None,
        "forward_steps": [],
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
        hint = "a number" if current_field and "init_params." in current_field else "a valid value (e.g. a Python dict)"
        return {
            "output": f"I couldn't parse **{current_field}**. Please provide {hint}.",
            "current_field": current_field,
        }
    # Step-by-step init_params: dotted field (e.g. init_params.input_dim) or form-submit dict
    if current_field and current_field.startswith("init_params."):
        if "init_params" not in config or not isinstance(config["init_params"], dict):
            config["init_params"] = {}
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("input_dim", "hidden_dim", "num_layers", "output_dim", "dropout") and v is not None:
                    if k == "dropout":
                        config["init_params"][k] = float(v) if isinstance(v, (int, float)) else float(str(v).strip())
                    else:
                        config["init_params"][k] = int(v) if isinstance(v, (int, float)) else int(str(v).strip())
        else:
            sub = current_field.split(".", 1)[1]
            config["init_params"][sub] = value
        next_field = get_next_missing_field(script_type, config, state)
        complete = next_field is None
        if complete:
            out_msg = get_confirm_message_with_preview(config, script_type, CONFIRM_MESSAGE)
        else:
            out_msg = get_question_for_field(script_type, next_field)
        return {
            "config": config,
            "current_field": next_field if not complete else None,
            "next_field": next_field,
            "complete": complete,
            "needs_confirmation": complete,
            "output": out_msg,
        }
    # Stage 2: layers.name -> store name, ask for spec
    if current_field == "layers.name" and script_type == "model_structure":
        name = (value if isinstance(value, str) else str(value)).strip()
        if name in (config.get("layers") or {}):
            return {
                "output": f"Layer name **{name}** already exists. Please choose a unique name.",
                "current_field": current_field,
            }
        return {
            "config": config,
            "current_field": "layers.spec",
            "next_field": "layers.spec",
            "layers_phase": "spec",
            "current_layer_name": name,
            "output": get_question_for_field(script_type, "layers.spec"),
        }
    # Stage 2: layers.spec -> add layer to config, ask continue or complete
    if current_field == "layers.spec" and script_type == "model_structure":
        if "layers" not in config or not isinstance(config["layers"], dict):
            config["layers"] = {}
        spec = tuple(value) if isinstance(value, list) else value
        config["layers"][state.get("current_layer_name") or "layer"] = spec
        return {
            "config": config,
            "current_field": "layers.continue",
            "next_field": "layers.continue",
            "layers_phase": "continue",
            "current_layer_name": None,
            "output": get_question_for_field(script_type, "layers.continue"),
        }
    # Stage 2: layers.continue -> next layer name or go to forward
    if current_field == "layers.continue" and script_type == "model_structure":
        if value == "complete":
            return {
                "current_field": "forward",
                "next_field": "forward",
                "layers_phase": None,
                "output": get_question_for_field(script_type, "forward"),
            }
        return {
            "current_field": "layers.name",
            "next_field": "layers.name",
            "layers_phase": "name",
            "output": get_question_for_field(script_type, "layers.name"),
        }
    # Stage 3: forward.step -> append step, ask continue or complete (layer_name must match defined layers)
    if current_field == "forward.step" and script_type == "model_structure":
        layer_name = (value.get("layer_name") or "").strip()
        defined = list((config.get("layers") or {}).keys())
        if layer_name and defined and layer_name not in defined:
            return {
                "output": f"**layer_name** must match a defined layer. Defined: {', '.join(defined)}. You gave: {layer_name!r}.",
                "current_field": current_field,
            }
        forward_steps = list(state.get("forward_steps") or [])
        forward_steps.append(value)
        return {
            "forward_steps": forward_steps,
            "current_field": "forward.continue",
            "next_field": "forward.continue",
            "forward_phase": "continue",
            "output": get_question_for_field(script_type, "forward.continue"),
        }
    # Stage 3: forward.continue -> next step or build forward and complete
    if current_field == "forward.continue" and script_type == "model_structure":
        if value == "complete":
            forward_steps = list(state.get("forward_steps") or [])
            config["forward"] = _build_forward_from_steps(forward_steps)
            next_field = get_next_missing_field(script_type, config, {"forward_phase": None})
            return {
                "config": config,
                "forward_phase": None,
                "forward_steps": [],
                "current_field": None,
                "next_field": next_field,
                "complete": next_field is None,
                "needs_confirmation": next_field is None,
                "output": get_confirm_message_with_preview(config, script_type, CONFIRM_MESSAGE)
                if next_field is None
                else get_question_for_field(script_type, next_field),
            }
        return {
            "current_field": "forward.step",
            "next_field": "forward.step",
            "forward_phase": "step",
            "output": get_question_for_field(script_type, "forward.step"),
        }
    if current_field in ("init_params", "parameters", "variables", "layers", "forward", "loss_formula"):
        if isinstance(value, dict):
            if current_field == "layers":
                init_param_keys = {"input_dim", "hidden_dim", "num_layers", "output_dim", "dropout"}
                if set(value.keys()) <= init_param_keys or not any(isinstance(v, (list, tuple)) for v in value.values()):
                    return {
                        "output": "**layers** must map layer names to tuples, e.g. gru_basic: ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout'). Not init_params keys.",
                        "current_field": current_field,
                    }
            config[current_field] = value
        else:
            return {"output": f"**{current_field}** must be a dictionary.", "current_field": current_field}
    else:
        config[current_field] = value
    next_field = get_next_missing_field(script_type, config, state)
    complete = next_field is None
    if complete:
        out_msg = get_confirm_message_with_preview(config, script_type, CONFIRM_MESSAGE)
    else:
        out_msg = get_question_for_field(script_type, next_field)
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
    next_field = get_next_missing_field(script_type, config, state)
    complete = next_field is None
    if complete:
        output = get_confirm_message_with_preview(config, script_type, CONFIRM_MESSAGE)
        return {
            "next_field": None,
            "current_field": None,
            "complete": True,
            "needs_confirmation": True,
            "output": output,
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
    config = state.get("config") or {}
    if not next_field:
        out = get_confirm_message_with_preview(config, script_type, CONFIRM_MESSAGE)
        return {"output": out, "current_field": None, "needs_confirmation": True, "complete": True}
    out = get_question_for_field(script_type, next_field)
    return {"output": out, "current_field": next_field}


def confirm(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """If user confirmed, route to validate_forward; else ask for confirmation with preview."""
    user_input = (state.get("user_input") or "").strip().lower()
    confirmed = user_input in ("yes", "confirm", "y", "ok", "generate")
    if confirmed:
        return {"output": "[Checking dimensions...]"}
    config = state.get("config") or {}
    script_type = state.get("script_type") or "model_structure"
    output = get_confirm_message_with_preview(config, script_type, CONFIRM_ASK)
    return {"output": output}


def validate_forward(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """
    After user said 'yes', validate input-output dimensions in forward calls.
    If valid -> go to generate_code; if invalid -> return error message and suggestions.
    """
    config = state.get("config") or {}
    script_type = state.get("script_type") or "model_structure"
    if script_type != "model_structure":
        return {"forward_valid": True, "output": "[Generating code...]"}
    is_valid, message = _validate_forward_dimensions(config)
    if is_valid:
        return {"forward_valid": True, "output": "[Generating code...]"}
    return {"forward_valid": False, "output": message}


def _parse_forward_expressions(forward_dict: Dict[str, Any]) -> list:
    """Parse config['forward'] dict into list of (forward_key, layer_name, inputs_list). forward_key may be 'out_1, hidden1' or 'pred'."""
    if not isinstance(forward_dict, dict):
        return []
    steps = []
    for forward_key, expression in forward_dict.items():
        if not isinstance(expression, str) or "(" not in expression:
            continue
        layer_name = expression.split("(")[0].strip()
        rest = expression[expression.index("(") + 1 : expression.rfind(")")]
        inputs_list = [s.strip() for s in re.split(r"\s*&\s*", rest) if s.strip()]
        steps.append((forward_key, layer_name, inputs_list))
    return steps


def _is_rnn_layer(layer_spec: Any) -> bool:
    """True if layer type is RNN/GRU/LSTM (produces output and hidden state)."""
    if not layer_spec or not isinstance(layer_spec, (list, tuple)) or len(layer_spec) < 1:
        return False
    t = (layer_spec[0] or "").lower()
    return t in ("gru", "rnn", "lstm")


def _get_layer_io_dims(
    layer_name: str,
    layer_spec: tuple,
    init_params: Dict[str, Any],
) -> tuple:
    """
    Return (expected_input_dim, output_dim) for a layer from its spec and init_params.
    GRU/RNN: first data param = input dim, second = output (hidden) dim.
    Linear: first = in, second = out.
    """
    if not layer_spec or not isinstance(layer_spec, (list, tuple)):
        return None, None
    init_params = init_params or {}
    layer_type = (layer_spec[0] or "").lower()
    if len(layer_spec) < 3:
        return None, None
    in_param = layer_spec[1]
    out_param = layer_spec[2]
    try:
        in_dim = init_params.get(in_param)
        out_dim = init_params.get(out_param)
        if in_dim is None and isinstance(in_param, str) and in_param.isdigit():
            in_dim = int(in_param)
        if out_dim is None and isinstance(out_param, str) and str(out_param).isdigit():
            out_dim = int(out_param)
        if in_dim is not None and out_dim is not None:
            return int(in_dim), int(out_dim)
    except (TypeError, ValueError):
        pass
    return None, None


def _validate_forward_dimensions(config: Dict[str, Any]) -> tuple:
    """
    Check that (1) input-output dimensions in forward calls are consistent, and
    (2) layer output format matches layer type (RNN -> two outputs 'out, hidden', linear/dropout -> single output).
    Returns (is_valid: bool, message: str).
    """
    init_params = config.get("init_params") or {}
    layers = config.get("layers") or {}
    forward_dict = config.get("forward") or {}
    if not isinstance(forward_dict, dict) or not forward_dict:
        return True, ""
    steps = _parse_forward_expressions(forward_dict)
    if not steps:
        return True, ""
    var_dim = {"x": init_params.get("input_dim")}
    try:
        var_dim["x"] = int(var_dim["x"]) if var_dim["x"] is not None else None
    except (TypeError, ValueError):
        var_dim["x"] = None
    errors = []
    for forward_key, layer_name, inputs_list in steps:
        layer_spec = layers.get(layer_name) if isinstance(layers.get(layer_name), (list, tuple)) else None
        if not layer_spec:
            continue
        is_rnn = _is_rnn_layer(layer_spec)
        has_two_outputs = "," in forward_key
        if is_rnn and not has_two_outputs:
            errors.append(
                f"**{layer_name}** is an RNN/GRU/LSTM layer and produces both output and hidden state. "
                f"Use format `'output_name, hidden_name': '{layer_name}(...)'` (e.g. `'out_1, hidden1': '{layer_name}(x)'`)."
            )
        if not is_rnn and has_two_outputs:
            errors.append(
                f"**{layer_name}** is a linear/dropout layer and has only one output. "
                f"Use a single key (e.g. `'pred': '{layer_name}(...)`) instead of two."
            )
        exp_in_dim, out_dim = _get_layer_io_dims(layer_name, layer_spec, init_params)
        if exp_in_dim is None or out_dim is None:
            continue
        actual_in_dim = 0
        for v in inputs_list:
            d = var_dim.get(v)
            if d is None:
                errors.append(f"Unknown input variable **{v}** in step {forward_key} = {layer_name}(...).")
                break
            actual_in_dim += d
        else:
            if actual_in_dim != exp_in_dim:
                in_desc = " & ".join(inputs_list)
                errors.append(
                    f"**{layer_name}** expects input dimension **{exp_in_dim}** (from its layer spec), "
                    f"but the input ({in_desc}) has total dimension **{actual_in_dim}** "
                    f"({' + '.join(f'{v}={var_dim[v]}' for v in inputs_list)})."
                )
        for var in [s.strip() for s in forward_key.split(",") if s.strip()]:
            var_dim[var] = out_dim
    if not errors:
        return True, ""
    suggestion = (
        "**Suggested fix:** Adjust the layer definition so its input dimension matches the concatenated input. "
        "For a call like `gru2(x & output1)`, the layer's expected input dimension should equal "
        "the sum of the dimensions of each input (e.g. input_dim + hidden_dim from the previous layer). "
        "You can add an entry in **init_params** for the combined dimension (e.g. `concat_dim: input_dim + hidden_dim`) "
        "and reference it in the layer spec, or define a layer that accepts the correct total input size."
    )
    message = "**Dimension mismatch in forward calls:**\n\n" + "\n\n".join(errors) + "\n\n" + suggestion
    return False, message


def _build_forward_from_steps(steps: list) -> Dict[str, Any]:
    """
    Build config['forward'] as a readable dict. RNN layers (gru, rnn, lstm) produce output and hidden state,
    so use key 'output_name, hidden_name': 'layer(inputs)'. Linear/dropout use single key 'output_name': 'layer(inputs)'.
    """
    if not steps:
        return {}
    forward_dict = {}
    for step in steps:
        layer_name = step.get("layer_name", "")
        inputs = (step.get("inputs") or "x").strip()
        output_name = (step.get("output_name") or "").strip()
        hidden_name = (step.get("hidden_name") or "").strip()
        for prev in steps:
            if prev is step:
                break
            prev_hidden = (prev.get("hidden_name") or "").strip()
            prev_output = (prev.get("output_name") or "").strip()
            if prev_hidden and prev_output and prev_hidden != prev_output:
                inputs = re.sub(r"\b" + re.escape(prev_hidden) + r"\b", prev_output, inputs)
        inputs_display = re.sub(r"\s*\+\s*", " & ", inputs)
        expression = f"{layer_name}({inputs_display})"
        if hidden_name and hidden_name != output_name:
            forward_dict[f"{output_name}, {hidden_name}"] = expression
        else:
            forward_dict[output_name] = expression
    return forward_dict


def _config_for_python_display(config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy config and convert layers values from list to tuple so output uses tuples."""
    out = copy.deepcopy(config)
    layers = out.get("layers")
    if isinstance(layers, dict):
        for k, v in list(layers.items()):
            if isinstance(v, list):
                out["layers"][k] = tuple(v)
    return out


# Key order for generated script (matches question order / example script)
_MODEL_TOP_ORDER = ["class_name", "base_class", "init_params", "layers", "forward"]
_INIT_PARAMS_ORDER = ["input_dim", "hidden_dim", "num_layers", "output_dim", "dropout"]
_LOSSFN_TOP_ORDER = ["parameters", "variables", "loss_formula"]


def _format_value_ordered(v: Any, key_order: list = None, indent: int = 2) -> str:
    """Format value for Python display; dicts use key_order when provided; lists/tuples as tuple literal."""
    if isinstance(v, dict):
        order = key_order or list(v.keys())
        parts = []
        for k in order:
            if k not in v:
                continue
            vv = v[k]
            if k == "init_params" and isinstance(vv, dict):
                sub = _format_value_ordered(vv, _INIT_PARAMS_ORDER, indent + 2)
            else:
                sub = _format_value_ordered(vv, None, indent + 2)
            parts.append(" " * indent + repr(k) + ": " + sub)
        return "{\n" + ",\n".join(parts) + "\n" + " " * (indent - 2) + "}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(repr(x) for x in v) + ")"
    return repr(v)


def format_config_ordered(config: Dict[str, Any], script_type: str, var_name: str = "") -> str:
    """Format config with keys in the desired order (class_name, base_class, init_params, layers, forward)."""
    config_display = _config_for_python_display(config)
    top_order = _MODEL_TOP_ORDER if script_type == "model_structure" else _LOSSFN_TOP_ORDER
    body = _format_value_ordered(config_display, top_order, indent=2)
    if var_name:
        return var_name + " = " + body
    return body


def get_confirm_message_with_preview(config: Dict[str, Any], script_type: str, intro: str) -> str:
    """Build confirmation message with a preview of the config (ordered keys)."""
    var = "archt_config" if script_type == "model_structure" else "lossfn_config"
    preview = format_config_ordered(config, script_type, var)
    return intro + "\n\n**Preview:**\n```python\n" + preview + "\n```"


def generate_code(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """Turn config into PyKGML config code string. Keys in order: class_name, base_class, init_params, layers, forward."""
    script_type = state.get("script_type")
    config = state.get("config") or {}
    var = "archt_config" if script_type == "model_structure" else "lossfn_config"
    code = format_config_ordered(config, script_type, var)
    return {
        "output": "Here is your configuration:\n\n```python\n" + code + "\n```",
        "generated_code": code,
        "complete": True,
        "needs_confirmation": False,
    }


def start_layers(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """User said 'Start Building Layers'; transition to asking for first layer name."""
    return {
        "layers_phase": "name",
        "current_field": "layers.name",
        "next_field": "layers.name",
        "output": get_question_for_field(state.get("script_type") or "model_structure", "layers.name"),
    }


def start_forward(state: ConfigAgentState, *, llm: Any = None) -> Dict[str, Any]:
    """User said 'Start Building Forward Function'; transition to asking for first step."""
    return {
        "forward_phase": "step",
        "current_field": "forward.step",
        "next_field": "forward.step",
        "output": get_question_for_field(state.get("script_type") or "model_structure", "forward.step"),
    }


def route_after_receive(state: ConfigAgentState) -> str:
    """Route from receive: select_script_type | confirm | extract | start_layers | start_forward | decide_next."""
    script_type = state.get("script_type")
    current_field = state.get("current_field")
    user_input = (state.get("user_input") or "").strip().lower()
    # Allow starting over: "create a (new) model structure" or "create a (new) loss function" always re-initialize
    if user_input and "model" in user_input and "structure" in user_input:
        return "select_script_type"
    if user_input and "loss" in user_input and "function" in user_input:
        return "select_script_type"
    if not script_type:
        return "select_script_type"
    if state.get("complete") and state.get("needs_confirmation") and user_input:
        return "confirm"
    if current_field == "layers" and user_input and "start" in user_input and "layer" in user_input:
        return "start_layers"
    if current_field == "forward" and user_input and "start" in user_input and ("forward" in user_input or "building" in user_input):
        return "start_forward"
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
    """After confirm: validate_forward if user said yes, else END."""
    user_input = (state.get("user_input") or "").strip().lower()
    if user_input in ("yes", "confirm", "y", "ok", "generate"):
        return "validate_forward"
    return "__end__"


def route_after_validate(state: ConfigAgentState) -> str:
    """After validate_forward: generate_code if dimensions OK, else END with error message."""
    if state.get("forward_valid"):
        return "generate_code"
    return "__end__"
