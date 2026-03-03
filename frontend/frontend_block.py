
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
}

# basic_chain = (RunnableLambda(lambda x: x[-1]) | chains_dict['basic'])
basic_chain = chains_dict['basic']


## Retrieval-Augmented Generation Chain

def assert_docs(d):
    if isinstance(d, list) and len(d) and isinstance(d[0], (Document, dict)):
        return d
    gr.Warning(f"Retriever outputs should be a list of documents, but instead got {str(d)[:100]}...")
    return []


# retrieval_chain = (
#     {'input' : (lambda x: x)}
#     | RunnableAssign(
#         {'context' : itemgetter('input') 
#         | chains_dict['retriever'] 
#         | assert_docs
#         | LongContextReorder().transform_documents
#         | docs2str
#     })
# )

retrieval_chain = chains_dict['retriever']
generator_chain = chains_dict['generator']
output_chain = RunnableAssign({"output" : generator_chain}) | output_puller
rag_chain = retrieval_chain | output_chain

#####################################################################
## ChatBot utilities

# def add_message(message, history, role=0, preface=""):
#     if not history or history[-1][role] is not None:
#         history += [[None, None]]
#     history[-1][role] = preface
#     buffer = ""
#     try:
#         for chunk in message:
#             token = getattr(chunk, 'content', chunk)
#             buffer += token
#             history[-1][role] += token
#             yield history, buffer, False 
#     except Exception as e:
#         logger.error(f"Gradio Stream failed:\n{get_traceback(e)}")
#         history[-1][role] += f"...\nGradio Stream failed: {e}"
#         yield history, buffer, True

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

    
def bot(history, chain_key):
    chain = {'Basic' : basic_chain, 'RAG' : rag_chain}.get(chain_key)

    if not history or "content" not in history[-1]:
        print("⚠️ WARNING: 'content' key missing, initializing")
        history[-1]["content"] = ""

    user_msg = history[-1]["content"]
    msg_stream = chain.stream(user_msg)
    for history, buffer, is_error in add_message(msg_stream, history):
        yield history


# def bot(history, chain_btn):
#     chain = {
#         "Basic": chains_dict["basic"],
#         "RAG": rag_chain,
#     }[chain_btn]

#     user_msg = history[-1]["content"]

#     # Create assistant message ONCE
#     history.append({"role": "assistant", "content": ""})

#     try:
#         for chunk in chain.stream(user_msg):
#             # LangServe streams strings
#             history[-1]["content"] += chunk
#             yield history

#     except Exception as e:
#         history[-1]["content"] = f"⚠️ Error: {e}"
#         yield history



# # this is for debugging bot error
# def bot(history, chain_key):
#     try:
#         chain = {
#             "Basic": basic_chain,
#             "RAG": rag_chain,
#         }[chain_key]

#         user_msg = history[-1][0]
#         # print(user_msg)
#         msg_stream = chain.stream(user_msg)

#         for history, buffer, is_error in add_message(
#             msg_stream, history, role=1
#         ):
#             yield history

#     except Exception as e:
#         import traceback
#         tb = traceback.format_exc()
#         print("🔥 BOT ERROR 🔥")
#         print(tb)

#         history[-1][1] = f"ERROR:\n{tb}"
#         yield history



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


## Document names, and the overall chunk list
class Globals:
    doc_names = set()
    doc_chunks = []



#####################################################################
## GRADIO EVENT LOOP

# https://github.com/gradio-app/gradio/issues/4001
CSS ="""
.contain { display: flex; flex-direction: column; height:80vh;}
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
THEME = gr.themes.Default(primary_hue="green")

def get_demo():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            value=[],
            elem_id="chatbot",
            avatar_images=(None, (os.path.join(os.path.dirname(__file__), "parrot.png"))),
        )

        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )

            chain_btn  = gr.Radio(["Basic", "RAG"], value="Basic", label="Main Route")
            # test_btn   = gr.Button("🎓\nEvaluate")

        # Reference: https://www.gradio.app/guides/blocks-and-event-listeners

        # This listener is triggered when the user presses the Enter key while the Textbox is focused.
        txt_msg = (
            # first update the chatbot with the user message immediately. Also, disable the textbox
            txt.submit(              ## On textbox submit (or enter)...
                fn=add_text,            ## Run the add_text function...
                inputs=[chatbot, txt],  ## Pass in the values of chatbot and txt...
                outputs=[chatbot, txt], ## Assign the results to the values of chatbot and txt...
                queue=False             ## And don't use the function as a generator (so no streaming)!
            )
            # then update the chatbot with the bot response (same variable logic)
            .then(bot, [chatbot, chain_btn], [chatbot])
            ## Then, unblock the textbox by assigning an active status to it
            .then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
        )


    return demo

#####################################################################
## Final App Deployment
import uvicorn
if __name__ == "__main__":

    demo = get_demo()
    demo.queue()

    logger.warning("Starting FastAPI app")
    app = FastAPI()

    app = gr.mount_gradio_app(app, demo, '/')

    @app.route("/health")
    async def health():
        return {"success": True}, 200
    
    uvicorn.run(app, host="0.0.0.0", port=9012, reload=True)
