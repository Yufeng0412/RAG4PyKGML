"""Extract structured values from free-form user text."""

import re
from typing import Any, Dict, Optional


def parse_dict_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a single Python dict from text (bracket matching + eval)."""
    text = (text or "").strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return eval(text[start : i + 1])
                except Exception:
                    pass
                break
    return None


def parse_key_value_pairs(text: str) -> Optional[Dict[str, Any]]:
    """Parse key=value pairs (e.g. input_dim=19, hidden_dim=128) into a dict. Numbers become int/float."""
    text = (text or "").strip()
    result = {}
    for m in re.finditer(r"(\w+)\s*=\s*([\w.]+)", text):
        k, v = m.group(1), m.group(2)
        try:
            result[k] = int(v) if v.isdigit() else float(v)
        except ValueError:
            result[k] = v
    return result if result else None


def parse_forward_step_pairs(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse forward step key=value where values may contain +, *, etc. (e.g. inputs=x+hidden1).
    Splits on comma, then each part is key=value with value = rest of string after first '='.
    """
    text = (text or "").strip()
    result = {}
    for part in re.split(r"\s*,\s*", text):
        part = part.strip()
        eq = part.find("=")
        if eq <= 0:
            continue
        k = part[:eq].strip()
        v = part[eq + 1 :].strip()
        if not k or v is None:
            continue
        if k in ("layer_name", "inputs", "output_name", "hidden_name"):
            result[k] = v
        else:
            try:
                result[k] = int(v) if v.isdigit() else float(v)
            except ValueError:
                result[k] = v
    return result if result else None


# Keys that belong to init_params, not layers (reject if layers dict has only these)
_INIT_PARAM_KEYS = {"input_dim", "hidden_dim", "num_layers", "output_dim", "dropout"}


def _looks_like_valid_layers(d: Dict[str, Any]) -> bool:
    """True if d looks like layer_name -> tuple/list (not init_params keys)."""
    if not d or not isinstance(d, dict):
        return False
    keys = set(d.keys())
    if keys <= _INIT_PARAM_KEYS:
        return False
    for v in d.values():
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            return True
    return False


def parse_layers_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse layers dict from text. Accepts:
    - Full dict: {"gru_basic": ("gru", "input_dim", "hidden_dim", "num_layers", "dropout")}
    - Key-value without outer braces: gru_basic: ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout')
    Returns None if result looks like init_params (wrong keys) or invalid.
    """
    text = (text or "").strip()
    parsed = parse_dict_from_text(text)
    if parsed is not None and isinstance(parsed, dict) and _looks_like_valid_layers(parsed):
        return parsed
    if parsed is not None and isinstance(parsed, dict) and not _looks_like_valid_layers(parsed):
        return None
    # Try wrapping key: value in braces; quote bare keys for valid Python
    if ":" in text and "(" in text:
        try:
            wrapped = "{" + re.sub(r"(\w+)\s*:\s*", r'"\1": ', text) + "}"
            parsed = eval(wrapped)
            if isinstance(parsed, dict) and _looks_like_valid_layers(parsed):
                return parsed
        except Exception:
            pass
    return None


def _parse_single_number(text: str) -> Optional[Any]:
    """Parse a single int or float from text (e.g. '19' or '0.2')."""
    text = (text or "").strip().lower()
    if text == "skip":
        return None
    for part in re.split(r"[\s,]+", text):
        part = part.strip()
        if not part:
            continue
        try:
            if "." in part:
                return float(part)
            return int(part)
        except ValueError:
            continue
    return None


def extract_value_for_field(
    user_input: str,
    field_name: str,
    script_type: str,
    llm=None,
) -> tuple[Any, bool]:
    """
    Extract a value for the given field from user input.
    For init_params.input_dim etc. extracts a single number. For init_params (whole) uses dict or key=value.
    Returns (value, success).
    """
    # Step-by-step init_params: single subfield (e.g. init_params.input_dim)
    if script_type == "model_structure" and field_name and field_name.startswith("init_params."):
        sub = field_name.split(".", 1)[1]
        # Allow form submit: "input_dim=19, hidden_dim=128, ..." to fill multiple at once
        key_val = parse_key_value_pairs(user_input)
        if key_val and any(k in key_val for k in _INIT_PARAM_KEYS):
            return key_val, True
        num = _parse_single_number(user_input)
        if sub == "dropout" and (user_input or "").strip().lower() == "skip":
            return 0.0, True
        if num is not None:
            if sub == "dropout" and isinstance(num, int) and 0 <= num <= 1:
                return float(num), True
            if sub in ("input_dim", "hidden_dim", "num_layers", "output_dim") and isinstance(num, int) and num > 0:
                return num, True
            if sub == "dropout" and isinstance(num, (int, float)) and 0 <= num <= 1:
                return float(num), True
        return None, False

    parsed = parse_dict_from_text(user_input)
    if parsed is not None and isinstance(parsed, dict):
        if field_name in ("init_params", "parameters", "variables", "layers", "forward", "loss_formula"):
            return parsed, True
        if field_name in parsed:
            return parsed[field_name], True
        return parsed, True

    # For init_params/parameters (whole dict), accept key=value style
    if field_name in ("init_params", "parameters"):
        key_val = parse_key_value_pairs(user_input)
        if key_val and len(key_val) >= 1:
            return key_val, True

    # For layers, accept "layer_name: ('gru', 'input_dim', ...)" or full dict; reject init_params-like keys
    if field_name == "layers":
        layers_val = parse_layers_from_text(user_input)
        if layers_val is not None:
            return layers_val, True

    # Stage 2: layers.name — single layer name (identifier)
    if field_name == "layers.name":
        name = (user_input or "").strip()
        if name and all(c.isalnum() or c in "_" for c in name):
            return name, True
        return None, False

    # Stage 2: layers.spec — tuple like ('gru', 'input_dim', 'hidden_dim', ...)
    if field_name == "layers.spec":
        text = (user_input or "").strip()
        # Try parsing as Python tuple directly
        start = text.find("(")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        try:
                            t = eval(text[start : i + 1])
                            if isinstance(t, (list, tuple)) and len(t) >= 1:
                                return tuple(t), True
                        except Exception:
                            pass
                        break
        # Try as single key: value (e.g. gru_basic: ('gru', ...))
        spec = parse_layers_from_text(f"dummy: {text}")
        if spec and "dummy" in spec:
            v = spec["dummy"]
            return (tuple(v) if isinstance(v, list) else v), True
        return None, False

    # Stage 2: layers.continue — "complete" or "continue adding more layers"
    if field_name == "layers.continue":
        low = (user_input or "").strip().lower()
        if "complete" in low or low in ("done", "finish"):
            return "complete", True
        if "continue" in low or "more layer" in low or "add" in low:
            return "continue", True
        return None, False

    # Stage 3: forward.step — layer_name, inputs, output_name, hidden_name (values may contain +, e.g. inputs=x+hidden1)
    if field_name == "forward.step":
        kv = parse_forward_step_pairs(user_input)
        if not kv:
            kv = parse_key_value_pairs(user_input)
        if not kv:
            return None, False
        step = {}
        for k in ("layer_name", "inputs", "output_name", "hidden_name"):
            if k in kv:
                step[k] = kv[k] if isinstance(kv[k], str) else str(kv[k])
        if "layer_name" in step and "inputs" in step and "output_name" in step:
            if "hidden_name" not in step:
                step["hidden_name"] = step.get("output_name", "")
            return step, True
        return None, False

    # Stage 3: forward.continue — "complete" or "continue adding"
    if field_name == "forward.continue":
        low = (user_input or "").strip().lower()
        if "complete" in low or low in ("done", "finish"):
            return "complete", True
        if "continue" in low or "more" in low or "add" in low:
            return "continue", True
        return None, False

    if llm is not None:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You extract structured data from user messages. Reply with ONLY a valid Python dict, no other text."),
                ("human", "User said: {user_input}\n\nExtract the value for field '{field_name}' (script type: {script_type}). "
                         "Return ONLY one Python dict, e.g. {{'input_dim': 19, 'hidden_dim': 128}} for init_params."),
            ])
            out = llm.invoke(prompt.format_messages(user_input=user_input, field_name=field_name, script_type=script_type))
            raw = out.content if hasattr(out, "content") else str(out)
            parsed = parse_dict_from_text(raw)
            if parsed is not None and isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

    return None, False
