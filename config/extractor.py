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


def extract_value_for_field(
    user_input: str,
    field_name: str,
    script_type: str,
    llm=None,
) -> tuple[Any, bool]:
    """
    Extract a value for the given field from user input.
    Uses simple parse first; if llm is provided and parse fails, can use LLM to extract.
    Returns (value, success).
    """
    parsed = parse_dict_from_text(user_input)
    if parsed is not None and isinstance(parsed, dict):
        if field_name in ("init_params", "parameters", "variables", "layers", "forward", "loss_formula"):
            return parsed, True
        if field_name in parsed:
            return parsed[field_name], True
        return parsed, True

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


def parse_init_params_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to get init_params dict from text (e.g. input_dim=19, hidden_dim=128)."""
    parsed = parse_dict_from_text(text)
    if isinstance(parsed, dict):
        return parsed
    # Try key=value pairs
    result = {}
    for m in re.finditer(r"(\w+)\s*=\s*([\w.]+)", text):
        k, v = m.group(1), m.group(2)
        try:
            result[k] = int(v) if v.isdigit() else float(v)
        except ValueError:
            result[k] = v
    return result if result else None
