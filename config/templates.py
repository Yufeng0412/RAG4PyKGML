"""Configuration templates and required fields for PyKGML config agent."""

from typing import Dict, List, Any


def get_model_structure_template() -> Dict[str, Any]:
    """Empty template for model structure (archt_config)."""
    return {
        "class_name": "my_KGML",
        "base_class": "TimeSeriesModel",
        "init_params": {},
        "layers": {},
        "forward": {},
    }


def get_loss_function_template() -> Dict[str, Any]:
    """Empty template for loss function (lossfn_config)."""
    return {
        "parameters": {},
        "variables": {},
        "loss_formula": {},
    }


def get_required_fields(script_type: str) -> Dict[str, List[str]]:
    """Required fields per script type. Order of top_level defines ask order."""
    if script_type == "model_structure":
        return {
            "top_level": ["init_params", "layers", "forward"],
            "init_params": ["input_dim", "hidden_dim", "num_layers", "output_dim"],
            "layers": [],
            "forward": [],
        }
    if script_type == "loss_function":
        return {
            "top_level": ["parameters", "variables", "loss_formula"],
            "parameters": [],
            "variables": [],
            "loss_formula": ["loss"],
        }
    return {}


def get_next_missing_field(script_type: str, config: Dict[str, Any]) -> str | None:
    """
    Return the next field name to ask for (from top_level order), or None if complete.
    Does not drill into nested keys; we treat init_params/layers/forward as single steps.
    """
    required = get_required_fields(script_type)
    if not required:
        return None
    top = required.get("top_level", [])
    for field in top:
        val = config.get(field)
        if field == "init_params" and script_type == "model_structure":
            sub = required.get("init_params", [])
            if not isinstance(val, dict):
                return field
            for k in sub:
                if k not in val or val[k] is None:
                    return field
        elif field == "layers":
            if not isinstance(val, dict) or len(val) == 0:
                return field
        elif field == "forward":
            if not isinstance(val, dict) or len(val) == 0:
                return field
        elif field == "parameters":
            if not isinstance(val, dict):
                return field
        elif field == "variables":
            if not isinstance(val, dict) or len(val) == 0:
                return field
        elif field == "loss_formula":
            if not isinstance(val, dict) or "loss" not in val:
                return field
    return None


def is_config_complete(script_type: str, config: Dict[str, Any]) -> bool:
    """True if all required fields are filled."""
    return get_next_missing_field(script_type, config) is None
