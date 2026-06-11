from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip(),
    temperature=0.7,
    streaming=True,
)

# Define Tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for calculator/math operations."""
    try:
        # Safe evaluation of basic math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        # Use a safe dict for evaluation
        return str(eval(expression, {"__builtins__": None}))
    except Exception as e:
        return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information, news, or facts."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Search error: {e}"

@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    """Search the uploaded documents/PDFs for relevant context. Use this whenever the user asks questions about their uploaded files or documents."""
    try:
        configurable = config.get("configurable", {})
        retriever = configurable.get("retriever", None)
        if retriever is not None:
            retriever_runnable = retriever.with_config({"run_name": "Document Retriever"})
            docs = retriever_runnable.invoke(query)
            if not docs:
                return "No matching context found in the uploaded documents."
            context = "\n---\n".join([d.page_content for d in docs])
            return f"Relevant document passages found:\n{context}"
    except Exception as e:
        return f"Error retrieving document context: {e}"
    return "No documents have been uploaded yet. Please ask the user to upload a document first."

tools = [calculate, web_search, search_documents]
llm_with_tools = llm.bind_tools(tools)
 
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    message = state["messages"]
    response = llm_with_tools.invoke(message)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Setup persistent SQLite checkpointer in the unified database
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "chat_sessions.db")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)
graph.add_edge("tools", "chat_node")

chatbot_graph = graph.compile(checkpointer=checkpointer)


