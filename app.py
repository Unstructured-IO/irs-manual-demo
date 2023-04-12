import os
from typing import Optional, Tuple
import gradio as gr
from threading import Lock
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")


def get_chain_and_vectorstore():
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    return chain, vectorstore


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]], chain, vectorstore=None
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        if not chain or not vectorstore:
            chain, vectorstore = get_chain_and_vectorstore()
        try:
            history = history or []
            docs = vectorstore.similarity_search(inp, k=2)
            output = chain.run(input_documents=docs, question=inp, chat_history=history)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chat = ChatWrapper()


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Chat-IRS-Manuals</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about the IRS Manuals",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What is the definition of a taxpayer?",
            "What kinds of factors affect how much I owe in taxes?",
            "What if I don't pay my taxes?",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        """<center>
            Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a>
            and <a href='https://github.com/unstructured-io/unstructured'>Unstructured.IO</a>
        </center>"""
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])

block.launch(debug=True)
