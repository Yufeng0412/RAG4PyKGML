# Config LangGraph — PyKGML configuration agent

LangGraph-based, stateful config agent that:

1. Maintains a partially-filled configuration dictionary
2. Decides what question to ask next
3. Extracts structured values from free-form user text
4. Decides when all required fields are complete
5. Loops until complete and user confirms
6. Then generates configuration code

## Plan

See [PLAN.md](PLAN.md) for the design, state schema, nodes, and edges.

## Usage

```python
from config_LangGraph import create_config_graph
from config_LangGraph.runner import run_one_turn

# Build graph (pass your LLM)
graph = create_config_graph(llm)

# First turn
result = run_one_turn(graph, "I want to create a model structure", state=None)
print(result["output"])   # First question (e.g. init_params)
state = result["state"]

# Next turns: pass previous state and new user message
result = run_one_turn(graph, "input_dim=19, hidden_dim=128, num_layers=2, output_dim=3", state=state)
state = result["state"]
# ... until result["complete"] and result.get("generated_code")
```

## Server integration

Use `run_one_turn(graph, user_input, state)` in your FastAPI/LangServe endpoint: accept `{"input": message, "state": state}`, return `{"output", "state", "complete", "generated_code"}` so the frontend can persist state between turns.

## Dependencies

- `langgraph`
- `langchain_core` (for prompts if using LLM in extractor)

Install: `pip install langgraph langchain_core`
