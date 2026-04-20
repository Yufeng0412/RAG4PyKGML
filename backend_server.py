# https://python.langchain.com/docs/langserve#server
# Server app with PyKGML Configuration Script Generator Agent integration
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

## May be useful later
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Config generator: LangGraph-based stateful config agent
from config_LangGraph import create_config_graph, run_one_turn
import logging

# set API key from environment (container/runtime friendly)
import os
if "NVIDIA_API_KEY" not in os.environ and os.getenv("NVIDIA_NIM_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = os.environ["NVIDIA_NIM_API_KEY"]


## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

runtime_init_error: Optional[Exception] = None

def extract_text(x):
    if isinstance(x, list) and len(x) > 0:
        return x[-1].get('text', str(x[-1]))
    return str(x)

def _raise_runtime_error(_: Any):
    raise RuntimeError(
        "Backend runtime failed to initialize. "
        "Check NVIDIA_NIM_API_KEY/NVIDIA_API_KEY and startup logs."
    ) from runtime_init_error

# For the RAG chain
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document chatbot. Help the user as they ask questions about PyKGML and its related documents."
               " Use the provided context to answer. Only cite sources that are used. Make your response conversational."),
    ("human", "Context: {context}\n\nQuestion: {input}")
])

try:
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
    # Allow runtime override and keep a broadly available default.
    chat_model = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-8b-instruct")
    instruct_llm = ChatNVIDIA(model=chat_model)

    basic_chat = (
        RunnableLambda(extract_text)
        | instruct_llm
        | StrOutputParser()
    )

    base_path = Path(__file__).resolve().parent
    docstore = FAISS.load_local(
        str(base_path / "pykgml_vector_db"),
        embedder,
        allow_dangerous_deserialization=True,
    )
    retriever = docstore.as_retriever(search_type="similarity", k=3)
except Exception as exc:
    runtime_init_error = exc
    logger.exception("Backend initialization failed")
    basic_chat = RunnableLambda(_raise_runtime_error)
    retriever = RunnableLambda(_raise_runtime_error)


def docs2str(docs, title="Document"):
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

def assert_docs(d):
    if isinstance(d, list) and len(d) and isinstance(d[0], (Document, dict)):
        return d
    print(f"Warning: Retriever outputs should be a list of documents, but instead got {str(d)[:100]}...")
    return []

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about PyKGML and its related documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

def output_puller(inputs):
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

if runtime_init_error is None:
    long_reorder = RunnableLambda(LongContextReorder().transform_documents)
    context_getter = itemgetter('input') | retriever | assert_docs | long_reorder | docs2str
    retrieval_chain = RunnableLambda(lambda x: {"input": extract_text(x)}) | RunnableAssign({'context' : context_getter})
    generator_chain = chat_prompt | instruct_llm | StrOutputParser()
    # Config generator: compiled LangGraph (stateful config agent)
    config_agent_graph = create_config_graph(instruct_llm)
else:
    retrieval_chain = RunnableLambda(_raise_runtime_error)
    generator_chain = RunnableLambda(_raise_runtime_error)
    config_agent_graph = None

# Explicit input schema so LangServe does not strip "state" when validating the request body
class ConfigGeneratorInput(BaseModel):
    """Request body for /config_generator. input can be str or list of blocks (Gradio 6 / LangServe)."""
    input: Union[str, List[Dict[str, Any]]] = Field(default="", description="Latest user message (string or content blocks)")
    state: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Config agent state from previous turn")


def config_generator_wrapper(input_data):
    """Normalize API input and delegate to LangGraph config agent via run_one_turn."""
    if config_agent_graph is None:
        _raise_runtime_error(input_data)
    if isinstance(input_data, dict):
        user_input = input_data.get("input", input_data.get("user_input", ""))
        state = input_data.get("state") or {}
    elif hasattr(input_data, "input") and hasattr(input_data, "state"):
        user_input = getattr(input_data, "input", "") or ""
        state = getattr(input_data, "state", None) or {}
    else:
        user_input = str(input_data)
        state = {}
    # Gradio/LangServe may send input as list of blocks [{"text": "...", "type": "text"}]; extract plain text
    user_input = extract_text(user_input) if isinstance(user_input, list) else user_input
    user_input = str(user_input).strip()
    result = run_one_turn(config_agent_graph, user_input, state)
    # Frontend expects top-level "config" when complete
    out = {"output": result["output"], "state": result["state"], "complete": result["complete"]}
    if result.get("generated_code") is not None:
        out["generated_code"] = result["generated_code"]
    if result.get("state") and "config" in result["state"]:
        out["config"] = result["state"]["config"]
    return out


config_generator_chain = RunnableLambda(config_generator_wrapper).with_types(
    input_type=ConfigGeneratorInput,
)


app = FastAPI(
  title="LangChain Server with Config Generator",
  version="1.0",
  description="API server with LangChain/LangGraph: RAG + PyKGML Configuration Script Generator Agent",
)

add_routes(app, basic_chat, path="/basic_chat")
add_routes(app, generator_chain, path="/generator")
add_routes(app, retrieval_chain, path="/retriever")
add_routes(app, config_generator_chain, path="/config_generator")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "9012"))
    uvicorn.run(app, host="0.0.0.0", port=port)
