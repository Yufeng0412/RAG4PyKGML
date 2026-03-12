
## NOTE: THIS SERVER IS RUNNING PERPETUALLY FOR THIS COURSE.
## DO NOT CHANGE CODE HERE; INSTEAD, INTERFACE WITH IT VIA USER INTERFACE
## AND BY DEPLOYING ON PORT :9012

import os
import random

from copy import deepcopy
from datetime import datetime
from fastapi import FastAPI

from operator import itemgetter

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.runnables import RunnableLambda
from langserve import RemoteRunnable
import gradio as gr

import logging
import traceback

def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#####################################################################
## Chain Dictionary

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        if isinstance(doc, dict):
            out_str += doc.get('page_content', doc) + "\n"
        else: 
            out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str


def output_puller(inputs):
    """If you want to support streaming, implement final step as a generator extractor."""
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

## Necessary Endpoints
chains_dict = {
    'basic' : RemoteRunnable("http://localhost:9012/basic_chat/"),
    'retriever' : RemoteRunnable("http://localhost:9012/retriever/"),
    'generator' : RemoteRunnable("http://localhost:9012/generator/"),
    'config_generator' : RemoteRunnable("http://localhost:9012/config_generator/"),
}

# basic_chain = (RunnableLambda(lambda x: x[-1]) | chains_dict['basic'])
basic_chain = chains_dict['basic']


## Retrieval-Augmented Generation Chain

def assert_docs(d):
    if isinstance(d, list) and len(d) and isinstance(d[0], (Document, dict)):
        return d
    gr.Warning(f"Retriever outputs should be a list of documents, but instead got {str(d)[:100]}...")
    return []



retrieval_chain = chains_dict['retriever']
generator_chain = chains_dict['generator']
output_chain = RunnableAssign({"output" : generator_chain}) | output_puller
rag_chain = retrieval_chain | output_chain

#####################################################################
## ChatBot utilities

def add_message(msg_stream, history):
    buffer = ""

    try:
        for chunk in msg_stream:
            if isinstance(chunk, str):
                buffer += chunk
            elif isinstance(chunk, dict):
                if "output" in chunk and isinstance(chunk["output"], str):
                    buffer += chunk["output"]
                elif "text" in chunk and isinstance(chunk["text"], str):
                    buffer += chunk["text"]

            # If assistant message doesn't exist yet, create it
            if not history or history[-1]["role"] != "assistant":
                history.append({"role": "assistant", "content": buffer})
            else:
                history[-1]["content"] = buffer

            yield history, buffer, True

    except Exception as e:
        history.append(
            {"role": "assistant", "content": f"⚠️ Error: {e}"}
        )
        yield history, buffer, True


# def add_text(history, text):
#     history = history + [(text, None)]
#     return history, gr.Textbox(value="", interactive=False)

# in gradio >=4.0, each message must be one of {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}
# In Gradio 6, content can be a list of blocks: [{"text": "...", "type": "text"}, ...]
def content_to_text(content):
    """Extract plain text from message content (string or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", block.get("content", str(block))))
            else:
                parts.append(str(block))
        return " ".join(parts).strip()
    return str(content).strip()


def add_text(history, text):
    history = history + [{"role": "user", "content": text}]
    return history, gr.Textbox(value="", interactive=False)


# Add this helper function
def debug_stream_call(chain, input_data, chain_name="chain"):
    """Debug streaming calls to understand the issue"""
    print(f"\n=== DEBUG {chain_name} ===")
    print(f"Input type: {type(input_data)}")
    print(f"Input: {input_data}")
    
    try:
        # Try streaming first
        print("Attempting to stream...")
        for i, chunk in enumerate(chain.stream(input_data)):
            print(f"Chunk {i}: type={type(chunk)}, content={chunk}")
            if i > 5:  # Limit debug output
                print("... (truncated)")
                break
        print("Stream completed")
    except Exception as e:
        print(f"Stream error: {e}")
        # Try invoke as fallback
        try:
            print("\nTrying invoke instead...")
            result = chain.invoke(input_data)
            print(f"Invoke result: {result}")
        except Exception as e2:
            print(f"Invoke error: {e2}")
    print("=== END DEBUG ===\n")

    
def rag_bot(history):
    """Bot for RAG chain - handles general questions"""
    if not history or "content" not in history[-1]:
        if history:
            history[-1]["content"] = history[-1].get("content", "")
        else:
            return

    user_msg = content_to_text(history[-1]["content"])
    msg_stream = rag_chain.stream(user_msg)
    for history, buffer, is_error in add_message(msg_stream, history):
        yield history


def _show_init_params_form(state):
    """True when agent is asking for init_params (step-by-step)."""
    if not isinstance(state, dict):
        return False
    cur = state.get("current_field") or ""
    nxt = state.get("next_field") or ""
    return cur.startswith("init_params.") or nxt.startswith("init_params.")


def _visibility_from_state(state):
    """From config state, return (show_init, show_layers_start, show_layers_continue, show_forward_start, show_forward_continue)."""
    if not isinstance(state, dict):
        return False, False, False, False, False
    cur = (state.get("current_field") or "").strip()
    nxt = (state.get("next_field") or "").strip()
    show_init = _show_init_params_form(state)
    show_layers_start = (cur == "layers" or nxt == "layers") and not (cur.startswith("layers.") or nxt.startswith("layers."))
    show_layers_continue = cur == "layers.continue" or nxt == "layers.continue"
    show_forward_start = (cur == "forward" or nxt == "forward") and not (cur.startswith("forward.") or nxt.startswith("forward."))
    show_forward_continue = cur == "forward.continue" or nxt == "forward.continue"
    return show_init, show_layers_start, show_layers_continue, show_forward_start, show_forward_continue


def _update_stage_visibility(state):
    """Return gr.update(visible=...) for init_params row, init inputs row, layers start, layers continue, forward start, forward continue."""
    a, b, c, d, e = _visibility_from_state(state)
    return (
        gr.update(visible=a), gr.update(visible=a),
        gr.update(visible=b), gr.update(visible=c),
        gr.update(visible=d), gr.update(visible=e),
    )


def config_bot(history, config_state):
    """Bot for Config Generator. config_state is passed in/out so it persists across requests (gr.State)."""
    if not history or "content" not in history[-1]:
        if history:
            history[-1]["content"] = history[-1].get("content", "")
        else:
            return history, config_state or {}

    user_msg = content_to_text(history[-1].get("content"))
    config_state = config_state if isinstance(config_state, dict) else {}

    input_data = {"input": user_msg, "state": config_state}

    try:
        result = chains_dict["config_generator"].invoke(input_data)
        if not isinstance(result, dict):
            output = str(result)
            if not history or history[-1]["role"] != "assistant":
                history.append({"role": "assistant", "content": output})
            else:
                history[-1]["content"] = output
            if "state" in result:
                config_state = dict(result["state"])
            return history, config_state

        if "state" in result:
            config_state = dict(result["state"])

        output = result.get("output", "")
        if not history or history[-1]["role"] != "assistant":
            history.append({"role": "assistant", "content": output})
        else:
            history[-1]["content"] = output

        return history, config_state

    except Exception as e:
        logger.error(f"Config generator error: {get_traceback(e)}")
        err = f"⚠️ Error: {str(e)}"
        if not history or history[-1]["role"] != "assistant":
            history.append({"role": "assistant", "content": err})
        else:
            history[-1]["content"] = err
        return history, config_state


def submit_init_params_form(history, config_state, input_dim, hidden_dim, num_layers, output_dim, dropout):
    """Build init_params from form fields, add as user message, call config bot, return (history, state)."""
    config_state = config_state if isinstance(config_state, dict) else {}
    parts = []
    if input_dim is not None and str(input_dim).strip() != "":
        parts.append(f"input_dim={input_dim}")
    if hidden_dim is not None and str(hidden_dim).strip() != "":
        parts.append(f"hidden_dim={hidden_dim}")
    if num_layers is not None and str(num_layers).strip() != "":
        parts.append(f"num_layers={num_layers}")
    if output_dim is not None and str(output_dim).strip() != "":
        parts.append(f"output_dim={output_dim}")
    if dropout is not None and str(dropout).strip() != "":
        parts.append(f"dropout={dropout}")
    if not parts:
        return history, config_state
    user_msg = ", ".join(parts)
    history = history + [{"role": "user", "content": user_msg}]
    input_data = {"input": user_msg, "state": config_state}
    try:
        result = chains_dict["config_generator"].invoke(input_data)
        if not isinstance(result, dict):
            output = str(result)
            history.append({"role": "assistant", "content": output})
            config_state = result.get("state", config_state)
            return history, dict(config_state) if isinstance(config_state, dict) else config_state
        if "state" in result:
            config_state = dict(result["state"])
        output = result.get("output", "")
        history.append({"role": "assistant", "content": output})
        return history, config_state
    except Exception as e:
        logger.error(f"Config generator error: {get_traceback(e)}")
        history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
        return history, config_state


#####################################################################
## Document/Assessment Utilities


def get_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    content = document[0].page_content
    content = content.replace("{", "[").replace("}", "]")
    if "References" in content:
        content = content[:content.index("References")]
    document[0].page_content = content
    return text_splitter.split_documents(document)


def get_day_difference(date_str):
    given_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    current_date = datetime.now().date()
    difference = current_date - given_date
    return difference.days


def get_fresh_chunks(chunks):
    return [
        chunk for chunk in chunks 
            if get_day_difference(chunk.metadata.get("Published", "2000-01-01")) < 90
    ]


def format_chunk(doc):
    prep_str = lambda x: x.replace('{', '<').replace('}', '>')
    return (
        f"Paper: {prep_str(doc.metadata.get('Title', 'unknown'))}"
        f"\n\nSummary: {prep_str(doc.metadata.get('Summary', 'unknown'))}"
        f"\n\nPage Body: {prep_str(doc.page_content)}"
    )


def get_synth_prompt(docs):
    doc1, doc2 = random.sample(docs, 2)
    sys_msg = (
        "Use the documents provided by the user to generate an interesting question-answer pair."
        " Try to use both documents if possible, and rely more on the document bodies than the summary. Be specific!"
        " Use the format:\nQuestion: (good question, 1-3 sentences, detailed)\n\nAnswer: (answer derived from the documents)"
        " DO NOT SAY: \"Here is an interesting question pair\" or similar. FOLLOW FORMAT!"
    )
    usr_msg = f"Document1: {format_chunk(doc1)}\n\nDocument2: {format_chunk(doc2)}"
    return ChatPromptTemplate.from_messages([('system', sys_msg), ('user', usr_msg)])


def get_eval_prompt():
    eval_instruction = (
        "Evaluate the following Question-Answer pair for human preference and consistency."
        "\nAssume the first answer is a ground truth answer and has to be correct."
        "\nAssume the second answer may or may not be true."
        "\n[1] The first answer is extremely preferable, or the second answer heavily deviates."
        "\n[2] The second answer does not contradict the first and significantly improves upon it."
        "\n\nOutput Format:"
        "\nJustification\n[2] if 2 is strongly preferred, [1] otherwise"
        "\n\nQuestion-Answer Pair:"
        "\n{input}\n\n"
        "[/INST]</s><s>[INST]Justification: "
    )
    return {"input" : lambda x:x} | ChatPromptTemplate.from_messages([('system', eval_instruction), ('user', '{input}')])



#####################################################################
## GRADIO EVENT LOOP

# https://github.com/gradio.app/gradio/issues/4001
CSS ="""
.contain { display: flex; flex-direction: column; height:80vh;}
#component-0 { height: 100%; }
.chatbot { flex-grow: 1; overflow: auto;}
"""
THEME = gr.themes.Default(primary_hue="green")

def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# PyKGML Assistant - Dual Chat Interface")
        # gr.Markdown("### Ask general questions on the left, generate configuration scripts on the right")
        
        with gr.Row():
            # Left column: RAG Chatbot
            with gr.Column(scale=1):
                gr.Markdown("## 📚 RAG Chatbot")
                gr.Markdown("Ask general questions on the left.")
                gr.Markdown("For example: how to use the model structure configuration in PyKGML?")
                rag_chatbot = gr.Chatbot(
                    value=[],
                    elem_id="rag_chatbot",
                    label="RAG Chatbot",
                    avatar_images=(None, (os.path.join(os.path.dirname(__file__), "parrot.png"))),
                    height=500,
                )
                rag_txt = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question about PyKGML...",
                    container=False,
                )
                
                # RAG chatbot event handlers
                rag_txt_msg = (
                    rag_txt.submit(
                        fn=add_text,
                        inputs=[rag_chatbot, rag_txt],
                        outputs=[rag_chatbot, rag_txt],
                        queue=False
                    )
                    .then(rag_bot, [rag_chatbot], [rag_chatbot])
                    .then(lambda: gr.Textbox(interactive=True), None, [rag_txt], queue=False)
                )
            
            # Right column: Config Generator Chatbot
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Config Generator")
                gr.Markdown("Generate PyKGML model structure or loss function configurations")
                with gr.Row():
                    btn_model = gr.Button("Create a new model structure", variant="secondary")
                    btn_loss = gr.Button("Create a new loss function", variant="secondary")
                config_chatbot = gr.Chatbot(
                    value=[],
                    elem_id="config_chatbot",
                    label="Config Generator",
                    avatar_images=(None, (os.path.join(os.path.dirname(__file__), "parrot.png"))),
                    height=380,
                )
                # Step-by-step init_params form (visible when agent asks for init_params)
                with gr.Row(visible=False) as init_params_row:
                    gr.Markdown("**Initial parameters** (fill any subset and submit):")
                with gr.Row(visible=False) as init_params_inputs_row:
                    input_dim_in = gr.Number(label="input_dim", value=None, precision=0, min_width=80)
                    hidden_dim_in = gr.Number(label="hidden_dim", value=None, precision=0, min_width=80)
                    num_layers_in = gr.Number(label="num_layers", value=None, precision=0, min_width=80)
                    output_dim_in = gr.Number(label="output_dim", value=None, precision=0, min_width=80)
                    dropout_in = gr.Number(label="dropout", value=None, precision=2, min_width=80)
                    btn_submit_init = gr.Button("Submit init params", variant="primary")
                # Stage 2: Layers — Start / Continue / Complete
                with gr.Row(visible=False) as layers_start_row:
                    btn_start_layers = gr.Button("Start Building Layers", variant="primary")
                with gr.Row(visible=False) as layers_continue_row:
                    btn_continue_layers = gr.Button("Continue Adding More Layers", variant="secondary")
                    btn_complete_layers = gr.Button("Complete", variant="primary")
                # Stage 3: Forward — Start / Continue / Complete
                with gr.Row(visible=False) as forward_start_row:
                    btn_start_forward = gr.Button("Start Building Forward Function", variant="primary")
                with gr.Row(visible=False) as forward_continue_row:
                    btn_continue_forward = gr.Button("Continue Adding Layer Calls", variant="secondary")
                    btn_complete_forward = gr.Button("Complete", variant="primary")
                config_state = gr.State(value={})  # Persist config agent state across requests
                config_txt = gr.Textbox(
                    show_label=False,
                    placeholder="Or type your choice / answer the bot's questions here...",
                    container=False,
                )

                stage_rows = [init_params_row, init_params_inputs_row, layers_start_row, layers_continue_row, forward_start_row, forward_continue_row]

                btn_model.click(
                    fn=lambda h: (h + [{"role": "user", "content": "I want to create a model structure"}], ""),
                    inputs=[config_chatbot],
                    outputs=[config_chatbot, config_txt],
                    queue=False,
                ).then(
                    config_bot, [config_chatbot, config_state], [config_chatbot, config_state]
                ).then(
                    _update_stage_visibility,
                    [config_state],
                    stage_rows,
                )

                btn_loss.click(
                    fn=lambda h: (h + [{"role": "user", "content": "I want to create a loss function"}], ""),
                    inputs=[config_chatbot],
                    outputs=[config_chatbot, config_txt],
                    queue=False,
                ).then(config_bot, [config_chatbot, config_state], [config_chatbot, config_state]).then(
                    _update_stage_visibility,
                    [config_state],
                    stage_rows,
                )

                # Config generator chatbot event handlers (text submit)
                config_txt_msg = (
                    config_txt.submit(
                        fn=add_text,
                        inputs=[config_chatbot, config_txt],
                        outputs=[config_chatbot, config_txt],
                        queue=False
                    )
                    .then(config_bot, [config_chatbot, config_state], [config_chatbot, config_state])
                    .then(_update_stage_visibility, [config_state], stage_rows)
                    .then(lambda: gr.Textbox(interactive=True), None, [config_txt], queue=False)
                )

                # Submit init params form
                btn_submit_init.click(
                    fn=submit_init_params_form,
                    inputs=[
                        config_chatbot, config_state,
                        input_dim_in, hidden_dim_in, num_layers_in, output_dim_in, dropout_in,
                    ],
                    outputs=[config_chatbot, config_state],
                ).then(_update_stage_visibility, [config_state], stage_rows)

                # Stage 2 & 3: Layers / Forward button clicks — add user message and run config_bot
                for btn, msg in [
                    (btn_start_layers, "Start Building Layers"),
                    (btn_continue_layers, "Continue Adding More Layers"),
                    (btn_complete_layers, "Complete"),
                    (btn_start_forward, "Start Building Forward Function"),
                    (btn_continue_forward, "Continue Adding Layer Calls"),
                    (btn_complete_forward, "Complete"),
                ]:
                    btn.click(
                        fn=lambda h, s, m=msg: (h + [{"role": "user", "content": m}], s),
                        inputs=[config_chatbot, config_state],
                        outputs=[config_chatbot, config_state],
                        queue=False,
                    ).then(config_bot, [config_chatbot, config_state], [config_chatbot, config_state]).then(
                        _update_stage_visibility, [config_state], stage_rows
                    )

    return demo

#####################################################################
## Final App Deployment

if __name__ == "__main__":
    import uvicorn
    demo = get_demo()
    demo.queue()

    logger.warning("Starting FastAPI app")
    app = FastAPI()

    app = gr.mount_gradio_app(app, demo, '/')

    @app.route("/health")
    async def health():
        return {"success": True}, 200
    
    uvicorn.run(app, host="0.0.0.0", port=9012, reload=True)
