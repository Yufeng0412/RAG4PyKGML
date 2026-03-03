"""Prompt snippets for question generation and code generation."""

QUESTION_FOR_FIELD = {
    "model_structure": {
        "init_params": "Please provide **init_params**: `input_dim`, `hidden_dim`, `num_layers`, `output_dim`, and optionally `dropout`. Example: input_dim=19, hidden_dim=128, num_layers=2, output_dim=3, dropout=0.2",
        "layers": "Please provide the **layers** dict: layer names mapping to tuples like ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout'). Example: gru_basic: ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout')",
        "forward": "Please provide the **forward** dict: output names and expressions using layer names and 'x', with '&' for concat. Example: 'output': 'fc(gru_basic(x))'",
    },
    "loss_function": {
        "parameters": "Please provide the **parameters** dict (e.g. Ra_idx, Rh_idx, NEE_idx, GPP_idx, tol_MB, x_scaler, y_scaler).",
        "variables": "Please provide the **variables** dict: name -> expression strings like 'y_pred[:, :, Ra_idx]'.",
        "loss_formula": "Please provide the **loss_formula** dict with a 'loss' key (e.g. loss formula string or expression).",
    },
}

CONFIRM_MESSAGE = "All required fields are filled. Reply **yes** or **confirm** to generate the configuration code."
CONFIRM_ASK = "Is this correct? Reply **yes** to generate the configuration code."


def get_question_for_field(script_type: str, field_name: str) -> str:
    return QUESTION_FOR_FIELD.get(script_type, {}).get(field_name, f"Please provide the value for **{field_name}**.")
