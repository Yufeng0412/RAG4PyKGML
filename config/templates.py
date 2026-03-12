"""Configuration templates and required fields for PyKGML config agent."""

from typing import Dict, List, Any, Optional

# Step-by-step: ask for each init_param separately (order preserved)
INIT_PARAM_SUBFIELDS = ["input_dim", "hidden_dim", "num_layers", "output_dim", "dropout"]


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
            "init_params": INIT_PARAM_SUBFIELDS,
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


def _layers_valid(val: Any) -> bool:
    """True if config['layers'] is a valid non-empty layers dict (layer_name -> tuple)."""
    if not isinstance(val, dict) or len(val) == 0:
        return False
    init_param_keys = {"input_dim", "hidden_dim", "num_layers", "output_dim", "dropout"}
    if set(val.keys()) <= init_param_keys:
        return False
    return any(isinstance(v, (list, tuple)) for v in val.values())


def get_next_missing_field(
    script_type: str,
    config: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> str | None:
    """
    Return the next field name to ask for. For init_params returns subfields one by one.
    For layers/forward uses state (layers_phase, forward_phase) for step-by-step flow.
    """
    state = state or {}
    required = get_required_fields(script_type)
    if not required:
        return None
    top = required.get("top_level", [])
    for field in top:
        val = config.get(field)
        if field == "init_params" and script_type == "model_structure":
            if not isinstance(val, dict):
                return "init_params.input_dim"
            for k in INIT_PARAM_SUBFIELDS:
                if k not in val or val[k] is None:
                    return f"init_params.{k}"
        elif field == "layers" and script_type == "model_structure":
            if _layers_valid(val):
                continue
            layers_phase = state.get("layers_phase")
            if layers_phase == "name":
                return "layers.name"
            if layers_phase == "spec":
                return "layers.spec"
            if layers_phase == "continue":
                return "layers.continue"
            return "layers"
        elif field == "forward" and script_type == "model_structure":
            if isinstance(val, dict) and len(val) > 0:
                continue
            forward_phase = state.get("forward_phase")
            if forward_phase == "step":
                return "forward.step"
            if forward_phase == "continue":
                return "forward.continue"
            return "forward"
        elif field == "layers":
            if not _layers_valid(val):
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


def is_config_complete(
    script_type: str, config: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> bool:
    """True if all required fields are filled."""
    return get_next_missing_field(script_type, config, state) is None
