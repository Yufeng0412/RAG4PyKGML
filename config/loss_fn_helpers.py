"""Helpers to build parameters and variables from user-defined loss-function variable entries."""

from typing import Any, Dict, Tuple


def _normalize_loss_vtype(vtype: str) -> str:
    """
    Map user/UI type strings to input | prediction | ground_truth.
    Accepts e.g. 'input (x)', 'Input', 'ground truth (y_true)'.
    """
    s = (vtype or "").strip().lower()
    if not s:
        return ""
    base = s.split("(", 1)[0].strip()
    if "ground" in base and "truth" in base:
        return "ground_truth"
    first = base.split()[0] if base else ""
    if first in ("input", "x", "batch_x", "inputs"):
        return "input"
    if base.startswith("input(") or base.startswith("input "):
        return "input"
    if first in ("prediction", "pred", "y_pred", "output"):
        return "prediction"
    if first in ("ground_truth", "y_true", "true", "target"):
        return "ground_truth"
    return base


def apply_loss_variable(
    config: Dict[str, Any],
    name: str,
    vtype: str,
    index: int,
    reverse_norm: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Update parameters and variables dicts for one variable definition.

    vtype: "input" | "prediction" | "ground_truth" (normalized from user)
    """
    params = dict(config.get("parameters") or {})
    variables = dict(config.get("variables") or {})
    name = (name or "").strip()
    if not name:
        return params, variables
    vtype = _normalize_loss_vtype(str(vtype or ""))
    idx_key = f"{name}_idx"
    params[idx_key] = index

    if vtype == "input":
        params.setdefault("x_scaler", "x_scaler")
        if reverse_norm:
            variables[f"{name}_reverse"] = (
                f"Z_norm_reverse(batch_x[:, :, {idx_key}], x_scaler[{idx_key}])"
            )
        else:
            # Raw input slice (normalized batch); needed when reverse is off — still use in loss.
            variables[f"{name}_input"] = f"batch_x[:, :, {idx_key}]"
    elif vtype == "prediction":
        params.setdefault("y_scaler", "y_scaler")
        variables[f"{name}_pred"] = f"y_pred[:, :, {idx_key}]"
        if reverse_norm:
            variables[f"{name}_pred_reverse"] = (
                f"Z_norm_reverse(y_pred[:, :, {idx_key}], y_scaler[{idx_key}])"
            )
    elif vtype == "ground_truth":
        params.setdefault("y_scaler", "y_scaler")
        variables[f"{name}_true"] = f"y_true[:, :, {idx_key}]"
        if reverse_norm:
            variables[f"{name}_true_reverse"] = (
                f"Z_norm_reverse(y_true[:, :, {idx_key}], y_scaler[{idx_key}])"
            )

    return params, variables


def merge_loss_final(loss_formula: Dict[str, Any]) -> Dict[str, Any]:
    """Set 'loss' key from numbered terms unless already explicitly set."""
    lf = dict(loss_formula or {})
    if isinstance(lf.get("loss"), str) and lf.get("loss").strip():
        return lf
    terms = [k for k in lf if k.startswith("loss") and k != "loss" and k[4:].isdigit()]
    terms.sort(key=lambda k: int(k[4:]))
    if terms:
        lf["loss"] = " + ".join(terms)
    return lf
