"""State schema for the LangGraph config agent."""

from typing import TypedDict, Any, Optional, List


class ConfigAgentState(TypedDict, total=False):
    """State for the configuration generation graph."""

    messages: List[dict]
    script_type: Optional[str]
    config: dict
    next_field: Optional[str]
    current_field: Optional[str]
    complete: bool
    needs_confirmation: bool
    output: str
    generated_code: Optional[str]
    user_input: str  # latest user message for this turn
    # Stage 2: layers (step-by-step)
    layers_phase: Optional[str]  # "intro" | "name" | "spec" | "continue"
    current_layer_name: Optional[str]  # name of layer being added
    # Stage 3: forward (step-by-step)
    forward_phase: Optional[str]  # "intro" | "step" | "continue"
    forward_steps: List[dict]  # list of {layer_name, inputs, output_name, hidden_name}
    forward_valid: Optional[bool]  # set by validate_forward: True -> generate_code, False -> show error
    loss_phase: Optional[str]  # loss function step-by-step flow
    loss_term_index: Optional[int]  # next loss term number (loss1, loss2, …)