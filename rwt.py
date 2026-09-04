import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import json

load_dotenv()

workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
if not workspace_id:
    raise SystemExit(
        "ANTHROPIC_WORKSPACE_ID is not set. Find the wrkspc_... id in the Anthropic "
        "Console under Settings > Workspaces (it's in the workspace's URL)."
    )

llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0 , default_headers={"anthropic-workspace-id": workspace_id})

# --- Step 1-2: point Chroma at docs, build the retrieval chain ---
print("=== Step 1-2: retrieval ===")

docs = [
    Document(page_content="Standard items can be returned within 30 days of delivery for a full refund."),
    Document(page_content="Sale and clearance items are final sale and cannot be returned or exchanged, except for manufacturing defects."),
    Document(page_content="To start a return, log into your account, go to Order History, and select Start a Return."),
    Document(page_content="Shipping costs are non-refundable unless the return is due to our error."),
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma.from_documents(docs, embedding=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 2})          # <- "two lines," step 1

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the context provided.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
rag_chain = (                                                     # <- the dict-at-front chain, step 2
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)
print(rag_chain.invoke("What's the return window?"))
print()

# --- Step 3: the tool ---
print("=== Step 3: @tool ===")
MOCK_TICKETS = {
    "TCK-1042": {"status": "awaiting parts", "opened": "2026-08-28"},
    "TCK-2050": {"status": "resolved", "opened": "2026-08-30"},
}

@tool
def ticket_lookup(ticket_id: str) -> str:
    """Look up the current status of a support ticket by its ID, e.g. TCK-1042."""
    ticket = MOCK_TICKETS.get(ticket_id)
    if not ticket:
        return f"No ticket {ticket_id} found."
    return json.dumps(ticket)

print("ticket_lookup ready.\n")

# --- Step 4-5: the agent ---
print("=== Step 4-5: create_agent ===")
agent = create_agent(model=llm, tools=[ticket_lookup])
CONFIG = {"recursion_limit": 10}

def print_trace(step):
    if "model" in step:
        for msg in step["model"]["messages"]:
            if msg.tool_calls:
                for call in msg.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    print(f"  -> calling {call['name']}({args})")
            else:
                print(f"  -> answered directly: {msg.content}")
    elif "tools" in step:
        for msg in step["tools"]["messages"]:
            print(f"  <- {msg.name} returned: {msg.content}")

def ask(question):
    print(f"Q: {question}")
    for step in agent.stream({"messages": [{"role": "user", "content": question}]},
                              stream_mode="updates", config=CONFIG):
        print_trace(step)
    print("---")

ask("What's the status of TCK-1042?")
ask("What are your opening hours?")