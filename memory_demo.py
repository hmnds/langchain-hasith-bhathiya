# memory_demo.py - giving a LangChain chain a memory.
#
# The question this demo answers: how does the bot know what "And for sale items?"
# means? On its own, a chain is stateless - every .invoke() is a fresh call with no
# idea a previous one happened. RunnableWithMessageHistory fixes that by replaying
# past turns back into the prompt.

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic                       # Claude chat model wrapper
from langchain_core.chat_history import InMemoryChatMessageHistory  # keeps turns in a Python list
from langchain_core.output_parsers import StrOutputParser           # AIMessage object -> plain str
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory  # the memory wrapper

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

# Policy text - the only facts this assistant is allowed to answer from.
# Hard-coded so the demo stays about memory. (In rwt.py this comes from a vector
# store instead, retrieved per question.)
CONTEXT = """\
- Standard items can be returned within 30 days of delivery for a full refund.
- Sale items can be returned within 14 days, for store credit only.
- All returns must be unused and in the original packaging.
"""

# A three-slot prompt. The middle slot is what makes memory possible.
prompt = ChatPromptTemplate.from_messages([
    (
        # 1. System: the role, plus the grounding rule. "Say you don't know" is what
        #    stops the model inventing a policy that isn't in CONTEXT.
        "system",
        "You are the Wireapps support assistant. Answer only from the context "
        "below. If the answer isn't in it, say you don't know.\n\nContext:\n{context}",
    ),
    # 2. History: a placeholder for a LIST of past messages, not a single string.
    #    RunnableWithMessageHistory fills this on every call. Delete this one line
    #    and the second question below stops making sense.
    MessagesPlaceholder("history"),
    # 3. The new question for this turn.
    ("human", "{question}"),
]).partial(context=CONTEXT)  # pre-fill {context} once so callers only supply {question}

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,  # deterministic - the same question gives the same answer
    default_headers={"anthropic-workspace-id": workspace_id},
)

# LCEL: the | operator wires runnables left to right.
# inputs dict -> filled prompt -> model -> plain string
chain = prompt | llm | StrOutputParser()

# One history per session id, so separate conversations don't bleed into each other.
# "InMemory" means exactly that - this dies with the process. Production would swap
# in a Redis- or Postgres-backed history here; nothing else would change.
sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    # LangChain calls this on every invoke. setdefault = return the existing history,
    # or create an empty one the first time we see this session id.
    return sessions.setdefault(session_id, InMemoryChatMessageHistory())


# Wraps the stateless chain to make it stateful. Each invoke now:
#   1. calls get_history(session_id) to load past turns
#   2. injects them into the "history" placeholder
#   3. runs the chain
#   4. appends this question and its answer back into that history
chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",   # which input key holds the new user message
    history_messages_key="history",  # which prompt placeholder receives the past turns
)

# Picks which conversation we're in. Change "demo" to any other string and the bot
# starts from a blank slate - sessions are isolated.
SESSION = {"configurable": {"session_id": "demo"}}


def reply(question):
    return chat.invoke({"question": question}, config=SESSION)


print(reply("What's the return window?"))
# The payoff. "And for sale items?" has no subject and no verb - it's meaningless in
# isolation. It resolves only because turn 1 was replayed in via MessagesPlaceholder.
print(reply("And for sale items?"))
