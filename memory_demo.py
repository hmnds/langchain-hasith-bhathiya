import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# Identity-linked API keys aren't bound to a workspace, so requests must name one.
workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
if not workspace_id:
    raise SystemExit(
        "ANTHROPIC_WORKSPACE_ID is not set. Find the wrkspc_... id in the Anthropic "
        "Console under Settings > Workspaces (it's in the workspace's URL)."
    )

# Policy text
CONTEXT = """\
- Standard items can be returned within 30 days of delivery for a full refund.
- Sale items can be returned within 14 days, for store credit only.
- All returns must be unused and in the original packaging.
"""

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the Wireapps support assistant. Answer only from the context "
        "below. If the answer isn't in it, say you don't know.\n\nContext:\n{context}",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
]).partial(context=CONTEXT)

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    default_headers={"anthropic-workspace-id": workspace_id},
)

chain = prompt | llm | StrOutputParser()

# One history per session id, so separate conversations don't bleed into each other.
sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return sessions.setdefault(session_id, InMemoryChatMessageHistory())


chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

SESSION = {"configurable": {"session_id": "demo"}}


def reply(question):
    return chat.invoke({"question": question}, config=SESSION)


print(reply("What's the return window?"))
print(reply("And for sale items?"))
