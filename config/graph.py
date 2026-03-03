"""Build and compile the LangGraph config agent."""

from typing import Any

from langgraph.graph import StateGraph, START, END

from .state import ConfigAgentState
from . import nodes


def create_config_graph(llm: Any):
    """
    Create the compiled config agent graph.
    Each invocation: pass state + user_input; graph runs until it produces an output for the user.
    """
    builder = StateGraph(ConfigAgentState)

    # Nodes (bind llm)
    builder.add_node("receive", lambda s: nodes.receive(s, llm=llm))
    builder.add_node("select_script_type", lambda s: nodes.select_script_type(s, llm=llm))
    builder.add_node("extract", lambda s: nodes.extract(s, llm=llm))
    builder.add_node("decide_next", lambda s: nodes.decide_next(s, llm=llm))
    builder.add_node("ask_question", lambda s: nodes.ask_question(s, llm=llm))
    builder.add_node("confirm", lambda s: nodes.confirm(s, llm=llm))
    builder.add_node("generate_code", lambda s: nodes.generate_code(s, llm=llm))

    # Entry
    builder.add_edge(START, "receive")

    # Receive -> route to select_script_type | confirm | extract | decide_next
    builder.add_conditional_edges(
        "receive",
        nodes.route_after_receive,
        {
            "select_script_type": "select_script_type",
            "confirm": "confirm",
            "extract": "extract",
            "decide_next": "decide_next",
        },
    )

    # select_script_type already set output and current_field -> end
    builder.add_edge("select_script_type", END)

    # extract -> confirm or END (output already set for next question)
    builder.add_conditional_edges(
        "extract",
        nodes.route_after_extract,
        {"confirm": "confirm", "__end__": END},
    )

    # decide_next -> ask_question or confirm
    builder.add_conditional_edges(
        "decide_next",
        nodes.route_after_decide_next,
        {"confirm": "confirm", "ask_question": "ask_question"},
    )

    builder.add_edge("ask_question", END)

    # confirm -> generate_code or END
    builder.add_conditional_edges(
        "confirm",
        nodes.route_after_confirm,
        {"generate_code": "generate_code", "__end__": END},
    )

    builder.add_edge("generate_code", END)

    return builder.compile()
