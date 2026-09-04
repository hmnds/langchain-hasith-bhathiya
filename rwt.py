# rwt.py - Retrieval With Tools: building up from a RAG chain to an agent.
#
# Four ideas, in order:
#   1. embed documents into a vector store, retrieve the relevant ones
#   2. wire that retrieval into a prompt with LCEL (the "dict at the front" pattern)
#   3. expose a plain Python function to the model as a @tool
#   4. hand the model that tool and let IT decide when to call it (the agent)
#
# The gap between 2 and 4 is the real lesson: a chain always runs every step in the
# same order. An agent chooses what to run, and can choose to run nothing.

import os
from langchain_core.documents import Document             # the {page_content, metadata} unit
from langchain_huggingface import HuggingFaceEmbeddings   # local embedding model, no API cost
from langchain_chroma import Chroma                       # vector store (in-memory here)
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # "hand the input along untouched"
from langchain_core.tools import tool                     # decorator: function -> model-callable tool
from langchain.agents import create_agent                 # prebuilt reason-and-act loop
from dotenv import load_dotenv
import json

# Reads .env into os.environ. Must run BEFORE any os.environ.get() below, or the
# variables will still look unset.
load_dotenv()

# Identity-linked API keys aren't bound to a workspace, so requests must name one.
workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
if not workspace_id:
    raise SystemExit(
        "ANTHROPIC_WORKSPACE_ID is not set. Find the wrkspc_... id in the Anthropic "
        "Console under Settings > Workspaces (it's in the workspace's URL)."
    )

# temperature=0 keeps both the wording AND the tool-calling decisions repeatable,
# which matters when you demo the same script twice.
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0 , default_headers={"anthropic-workspace-id": workspace_id})

# --- Step 1-2: point Chroma at docs, build the retrieval chain ---
print("=== Step 1-2: retrieval ===")

# Our knowledge base. A real project would load PDFs or web pages with a document
# loader and chop them up with a text splitter; four hand-written lines keep the
# demo readable and the retrieval easy to reason about.
docs = [
    Document(page_content="Standard items can be returned within 30 days of delivery for a full refund."),
    Document(page_content="Sale and clearance items are final sale and cannot be returned or exchanged, except for manufacturing defects."),
    Document(page_content="To start a return, log into your account, go to Order History, and select Start a Return."),
    Document(page_content="Shipping costs are non-refundable unless the return is due to our error."),
]
# Turns text into vectors so we can search by meaning rather than by keyword. Runs
# locally - the first run downloads ~90MB of weights, after that it is cached.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Embeds all four documents and indexes them. Nothing is written to disk here.
store = Chroma.from_documents(docs, embedding=embeddings)
# .as_retriever() hands back a runnable: question string in, list of Documents out.
# k=2 means "give me the 2 closest matches", so we don't pad the prompt with noise.
retriever = store.as_retriever(search_kwargs={"k": 2})          # <- "two lines," step 1

def format_docs(docs):
    # Retrieved Document objects -> one plain string the prompt can interpolate.
    return "\n".join(d.page_content for d in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    # "Answer only from the context provided" is the grounding instruction. It is what
    # makes this retrieval-augmented, rather than the model answering from memory.
    ("system", "Answer only from the context provided.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
# The dict at the front runs both branches on the SAME input string, side by side:
#   "What's the return window?" -> retriever -> format_docs -> fills {context}
#   "What's the return window?" -> passthrough             -> fills {question}
# LangChain then feeds the resulting dict into the prompt. RunnablePassthrough just
# means "don't transform this" - without it we'd have no way to reuse the raw question.
rag_chain = (                                                     # <- the dict-at-front chain, step 2
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)
print(rag_chain.invoke("What's the return window?"))
print()

# --- Step 3: the tool ---
print("=== Step 3: @tool ===")
# Stands in for a real ticketing system API call.
MOCK_TICKETS = {
    "TCK-1042": {"status": "awaiting parts", "opened": "2026-08-28"},
    "TCK-2050": {"status": "resolved", "opened": "2026-08-30"},
}

@tool
def ticket_lookup(ticket_id: str) -> str:
    """Look up the current status of a support ticket by its ID, e.g. TCK-1042."""
    # That docstring is not just documentation. @tool ships it to the model as the
    # tool's description, and turns the type hints into its input schema. The model
    # reads both to decide whether to call this and what to pass it. A vague docstring
    # is the usual reason a tool never gets called.
    ticket = MOCK_TICKETS.get(ticket_id)
    if not ticket:
        # Return the failure as a string instead of raising. The model can read this
        # and explain the problem, whereas an exception would kill the agent loop.
        return f"No ticket {ticket_id} found."
    return json.dumps(ticket)

print("ticket_lookup ready.\n")

# --- Step 4-5: the agent ---
print("=== Step 4-5: create_agent ===")
# The agent loop: send the question plus the tool definitions -> if the model asks for
# a tool, run it and feed the result back -> repeat until it replies in plain text.
agent = create_agent(model=llm, tools=[ticket_lookup])
# Safety net: give up after 10 model/tool steps, so a confused loop can't run forever
# (or run up a bill).
CONFIG = {"recursion_limit": 10}

def print_trace(step):
    # Narrates each step, so viewers see the decisions and not just the final answer.
    if "model" in step:
        for msg in step["model"]["messages"]:
            if msg.tool_calls:
                # The model decided it needs a tool and picked the arguments itself -
                # note that we never parsed the ticket ID out of the question.
                for call in msg.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    print(f"  -> calling {call['name']}({args})")
            else:
                # No tool call, so the model went straight to an answer.
                print(f"  -> answered directly: {msg.content}")
    elif "tools" in step:
        # Our Python function actually ran; this is what was handed back to the model.
        for msg in step["tools"]["messages"]:
            print(f"  <- {msg.name} returned: {msg.content}")

def ask(question):
    print(f"Q: {question}")
    # stream_mode="updates" emits one event per completed node, which is what lets us
    # narrate the loop as it runs. A plain .invoke() returns only the final answer.
    for step in agent.stream({"messages": [{"role": "user", "content": question}]},
                              stream_mode="updates", config=CONFIG):
        print_trace(step)
    print("---")

# Two questions that make the point. Same agent, same tool, opposite behaviour:
ask("What's the status of TCK-1042?")  # fits the tool -> model calls it, then answers
ask("What are your opening hours?")    # nothing fits -> model skips the tool entirely
