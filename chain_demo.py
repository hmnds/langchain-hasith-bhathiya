import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
WS = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support assistant. Answer in one short sentence."),
    ("human", "{question}"),
])
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0, default_headers=WS)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "How long do refunds take?"}))
