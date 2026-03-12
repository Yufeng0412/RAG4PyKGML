"""Prompt snippets for question generation and code generation."""

# Stage 2: Layers (step-by-step)
LAYERS_INTRO = (
    "You will now define the neural network layers.\n"
    "You may create one or more layers.\n"
    "Layers may accept parameter names set in the init_params or values provided by the user.\n\n"
    "Reply with **Start Building Layers** or click the button to begin."
)
LAYERS_NAME = "What is the **name** for this layer? (e.g. gru_basic, fc)"
LAYERS_SPEC = (
    "What is the **layer spec**? Provide the type and param names as a tuple, "
    "e.g. `('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout')` for a GRU layer, "
    "or ('linear',  'hidden_dim',  1) for a linear layer with 1 output dimension."
)
LAYERS_CONTINUE = (
    "Layer added. Choose:\n"
    "**Continue Adding More Layers** — add another layer\n"
    "**Complete** — finish and go to the forward function"
)

# Stage 3: Forward (step-by-step)
FORWARD_INTRO = (
    "The forward function defines how data flows through the layers to produce an output.\n"
    "All defined layers should be used in the forward pass; unused layers are considered redundant.\n\n"
    "Reply with **Start Building Forward Function** or click the button to begin."
)
FORWARD_STEP = (
    "**Forward step**: provide **layer_name** (must match a defined layer), **inputs** (e.g. `x` or `x & out_1` for concat), "
    "**output_name**, and **hidden_name** for GRU/RNN/LSTM layers (they produce output and hidden state; the script will show `'out_1, hidden1': 'gru1(x)'`). "
    "For linear/dropout layers only **output_name** is needed.\n"
    "Example (RNN): layer_name=gru1, inputs=x, output_name=out_1, hidden_name=hidden1. Example (linear): layer_name=fc1, inputs=out_2, output_name=pred"
)
FORWARD_CONTINUE = (
    "Step added. Choose:\n"
    "**Continue Adding Layer Calls** — add another step\n"
    "**Complete** — finish the forward definition"
)

# Step-by-step init_params: one question per subfield
INIT_PARAM_QUESTIONS = {
    "input_dim": "What is **input_dim**? (integer, number of input features)",
    "hidden_dim": "What is **hidden_dim**? (integer, number of nodes in the hidden layers)",
    "num_layers": "What is **num_layers**? (integer, number of hidden layers)",
    "output_dim": "What is **output_dim**? (integer, number of output targets)",
    "dropout": "What is **dropout**? (float 0–1, e.g. 0.2; or skip by replying 'skip')",
}

QUESTION_FOR_FIELD = {
    "model_structure": {
        "init_params": "Please provide **init_params** (step by step we'll ask for each). First: **input_dim**? (integer)",
        "layers": "Please provide the **layers** dict: layer names mapping to tuples like ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout'). Example: gru_basic: ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout')",
        "forward": "Please provide the **forward** dict: output names and expressions using layer names and 'x', with '&' for concat. Example: 'output1, hidden1': 'gru1(x)', 'output2': 'fc1(output1)'",
    },
    "loss_function": {
        "parameters": "Please provide the **parameters** dict (e.g. Ra_idx, Rh_idx, NEE_idx, GPP_idx, tol_MB, x_scaler, y_scaler).",
        "variables": "Please provide the **variables** dict: name -> expression strings like 'y_pred[:, :, Ra_idx]'.",
        "loss_formula": "Please provide the **loss_formula** dict with a 'loss' key (e.g. loss formula string or expression).",
    },
}

CONFIRM_MESSAGE = "All required fields are filled. Review the preview below, then reply **yes** to generate the final code."
CONFIRM_ASK = "Review the preview below. Reply **yes** to generate the configuration code, or describe any changes you'd like (e.g. key order, values)."


def get_question_for_field(script_type: str, field_name: str) -> str:
    if script_type == "model_structure" and field_name and field_name.startswith("init_params."):
        sub = field_name.split(".", 1)[1]
        return INIT_PARAM_QUESTIONS.get(sub, f"What is **{sub}**?")
    if script_type == "model_structure" and field_name == "layers":
        return LAYERS_INTRO
    if script_type == "model_structure" and field_name == "layers.name":
        return LAYERS_NAME
    if script_type == "model_structure" and field_name == "layers.spec":
        return LAYERS_SPEC
    if script_type == "model_structure" and field_name == "layers.continue":
        return LAYERS_CONTINUE
    if script_type == "model_structure" and field_name == "forward":
        return FORWARD_INTRO
    if script_type == "model_structure" and field_name == "forward.step":
        return FORWARD_STEP
    if script_type == "model_structure" and field_name == "forward.continue":
        return FORWARD_CONTINUE
    return QUESTION_FOR_FIELD.get(script_type, {}).get(field_name, f"Please provide the value for **{field_name}**.")
