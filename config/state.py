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
