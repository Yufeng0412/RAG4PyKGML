# LangGraph Config Agent — Plan

## Goal

A **stateful, goal-driven agent** that guides users to fill PyKGML configuration (model structure or loss function) via conversation, then generates code only after all required fields are complete and confirmed.

## Design Principles

1. **Single source of truth**: One graph state holds the partially-filled config and conversation context.
2. **Explicit control flow**: Nodes and conditional edges decide what happens next (ask question vs extract vs generate code).
3. **Human-in-the-loop**: Each invocation processes one user message and returns one assistant response; the frontend sends back the updated state on the next turn.
4. **Code only when done**: Code generation runs only after all required fields are filled and the user confirms.

## State (TypedDict)

- **messages**: List of `{role, content}` (conversation history).
- **script_type**: `"model_structure"` | `"loss_function"` | `None`.
- **config**: Partially or fully filled configuration dict (schema depends on script_type).
- **next_field**: The next required field to ask about (e.g. `"init_params"`, `"layers"`, `"forward"`).
- **current_field**: Field we last asked for (so we know what to extract from the user’s reply).
- **complete**: Whether all required fields are filled.
- **needs_confirmation**: Whether we are waiting for user to confirm before generating code.
- **output**: Latest assistant message to show the user.
- **generated_code**: Final generated code string when done.

## Flow (One Turn = One User Message)

1. **Receive input**: Normalize user message and append to `messages`.
2. **Route**:
   - No `script_type` → **select_script_type** (from user message) → init config → set first `next_field`.
   - Has `script_type` and we have **current_field** (we asked for it last) → **extract** (parse user reply, update config, clear current_field).
   - Has `script_type`, no current_field → **decide_next** (run completion check, set next_field or complete/needs_confirmation).
3. **select_script_type**: Detect "model structure" vs "loss function", initialize config template, set `next_field` to first required field.
4. **extract**: From last user message, extract value for `current_field` (LLM + parsing); validate; update `config`; then go to **decide_next**.
5. **decide_next**: Compute missing required fields; if any, set `next_field`; if none, set `complete=True` and `needs_confirmation=True`.
6. **ask_question**: Generate a clear question for `next_field`, set `current_field = next_field`, set `output` to that question.
7. **confirm**: If user message is confirmation (e.g. "yes", "confirm") and `complete` → **generate_code**; else ask "Confirm to generate code?"
8. **generate_code**: Turn `config` into PyKGML config code (or JSON); set `output` and `generated_code`; set done.

## Required Fields (from templates)

- **model_structure**: `init_params` (with input_dim, hidden_dim, num_layers, output_dim), `layers` (≥1), `forward` (≥1). `class_name` and `base_class` can be pre-filled.
- **loss_function**: `parameters`, `variables`, `loss_formula` (must include `loss` key).

## Nodes

| Node               | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `receive`          | Normalize input, append user message to state.                         |
| `route`            | Conditional: select_script_type / extract / decide_next.                |
| `select_script_type` | Set script_type + init config + first next_field.                     |
| `extract`          | Parse last user message for current_field; validate; update config.    |
| `decide_next`      | Check completion; set next_field or complete + needs_confirmation.     |
| `ask_question`     | Build question for next_field; set current_field; set output.           |
| `confirm`          | If user confirmed → generate_code; else ask for confirmation.          |
| `generate_code`    | Produce code from config; set output and generated_code.                |

## Edges

- `START → receive`
- `receive → route` (conditional: select_script_type | extract | decide_next)
- `select_script_type → decide_next`
- `extract → decide_next`
- `decide_next` (conditional: ask_question | confirm | generate_code)
- `ask_question → END`
- `confirm → END` or `confirm → generate_code` (conditional)
- `generate_code → END`

## Files

- **state.py**: TypedDict state schema.
- **templates.py**: Config templates and required-field lists (can mirror pykgml_config_agent).
- **nodes.py**: All node functions (receive, select_script_type, extract, decide_next, ask_question, confirm, generate_code).
- **graph.py**: StateGraph construction, add_node, add_conditional_edges, compile.
- **extractor.py**: Helper to extract structured value from free-form text (LLM + parse).
- **__init__.py**: Export `create_config_graph()` or `compile_config_graph(llm)`.

## Integration

- **Server**: New endpoint (e.g. in agent_server_app or a small LangGraph server) that receives `{ "input": user_message, "state": graph_state }`, invokes the compiled graph with that state, returns `{ "output", "state", "complete", "generated_code" }`.
- **Frontend**: Same as current config bot: send last user message + previous state; display `output`; persist returned `state` for next turn.

## Out of Scope (for this folder)

- RAG / retriever (tutorial-only context or fixed prompts).
- Changes to pykgml_config_agent (this is a separate, LangGraph-based implementation).
