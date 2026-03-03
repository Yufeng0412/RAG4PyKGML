# PyKGML Configuration Script Generator Agent

This module provides an interactive agent that helps users create PyKGML configuration scripts through a conversational interface.

## Overview

The agent guides users through creating two types of configuration scripts:
1. **Model Structure Configuration** - For designing neural network architectures
2. **Loss Function Configuration** - For customizing loss functions

## Architecture

The agent follows a step-by-step question-answer loop:

1. **Script Type Selection** - Determines if user wants model structure or loss function
2. **Status Check** - Validates required fields and identifies missing information
3. **Question Generation** - Uses RAG to retrieve relevant documentation and generates contextual questions
4. **Information Extraction** - Parses user responses to extract structured values
5. **Script Update** - Updates configuration dictionary with extracted values
6. **Loop** - Returns to status check until configuration is complete

## Usage

### Server Integration

The agent is integrated into `server_app.py` with a `/config_generator` endpoint:

```python
from pykgml_config_agent.agent_chain import create_simple_config_chain

# Create chain
config_agent_chain = create_simple_config_chain(llm, retriever)

# Add to FastAPI app
add_routes(app, config_generator_chain, path="/config_generator")
```

### Frontend Integration

The frontend (`frontend/frontend_block.py`) includes a "Config Generator" mode that:
- Maintains state across conversation turns
- Displays generated questions
- Shows final configuration when complete

### Example Flow

1. User: "I want to create a model structure"
2. Agent: "Let's start! What would you like to name your model class?"
3. User: "MyGRUModel"
4. Agent: "Great! What will be your input dimension?"
5. User: "19"
6. [Continues until all fields are filled]
7. Agent: "Configuration complete! Here's your script: [shows config]"

## Components

### State Management (`state.py`)
- `ConfigGenerationState` - Pydantic model tracking script generation progress

### Templates (`templates.py`)
- `get_model_structure_template()` - Empty model structure template
- `get_loss_function_template()` - Empty loss function template

### Status Checker (`status_checker.py`)
- `check_completion_status()` - Validates configuration completeness

### Question Generator (`question_generator.py`)
- Uses RAG to retrieve relevant documentation
- Generates contextual questions for missing fields

### Information Extractor (`extractor.py`)
- Parses natural language responses
- Extracts structured values using LLM

### Script Updater (`script_updater.py`)
- Validates extracted values
- Updates configuration dictionaries safely

### Agent Chain (`agent_chain.py`)
- Orchestrates the question-answer-update loop
- `create_simple_config_chain()` - Main chain for integration

## Configuration Script Formats

### Model Structure Template

```python
{
    "class_name": "MyModel",
    "base_class": "TimeSeriesModel",
    "init_params": {
        "input_dim": 19,
        "hidden_dim": 128,
        "num_layers": 2,
        "output_dim": 3
    },
    "layers": {
        "gru_basic": ("gru", "input_dim", "hidden_dim", "num_layers", "dropout")
    },
    "forward": {
        "out_basic, hidden": "gru_basic(x)",
        "output": "fc(out_basic)"
    }
}
```

### Loss Function Template

```python
{
    "parameters": {
        "Ra_idx": 0,
        "Rh_idx": 1
    },
    "variables": {
        "Ra_pred": "y_pred[:, :, Ra_idx]",
        "Ra_true": "y_true[:, :, Ra_idx]"
    },
    "loss_formula": {
        "loss": "mean((Ra_pred - Ra_true)**2)"
    }
}
```

## Requirements

- LangChain
- Pydantic
- FastAPI
- Gradio (for frontend)
- Existing RAG infrastructure (FAISS vector store, retriever, LLM)
