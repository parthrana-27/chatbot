import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, messages_to_dict, messages_from_dict
import json
import sqlite3
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda

# Ensure backend config and graph are in path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.graph import chatbot_graph
import backend.config as cfg

# SQLite Database Configurations
DB_FILE = "chat_sessions.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            messages TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def load_sessions():
    init_db()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, title, messages FROM chat_sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        sessions = {}
        for session_id, title, messages_str in rows:
            try:
                messages = json.loads(messages_str)
            except Exception:
                messages = []
            sessions[session_id] = {
                "title": title,
                "messages": messages
            }
        return sessions
    except Exception:
        return {}

def save_session(session_id: str, title: str, messages: list):
    init_db()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        messages_json = json.dumps(messages_to_dict(messages), ensure_ascii=False)
        cursor.execute("""
            INSERT INTO chat_sessions (session_id, title, messages)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                title=excluded.title,
                messages=excluded.messages
        """, (session_id, title, messages_json))
        conn.commit()
        conn.close()
    except Exception:
        pass

def delete_all_sessions():
    init_db()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_sessions")
        conn.commit()
        conn.close()
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Set up page configurations
st.set_page_config(
    page_title="Gemma 4 Chatbot - LangGraph AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100..800&family=Outfit:wght@100..900&display=swap');
    
    /* Main body background gradient */
    .stApp {
        background: radial-gradient(circle at top right, #17103a 0%, #090815 60%, #030207 100%) !important;
        color: #e2e8f0 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Title styling */
    .glow-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.15);
        margin-bottom: 0.1rem;
    }
    
    .glow-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(9, 8, 22, 0.95) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #a78bfa !important;
    }

    /* Custom styles for chat messages */
    div[data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(139, 92, 246, 0.15) !important;
    }

    /* User messages specifically - violet gradient background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] img[alt="user"]) {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.05) !important;
    }

    /* Input area customization */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
        background-color: rgba(15, 14, 30, 0.8) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.05) !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #e2e8f0 !important;
    }

    /* Expander / Thinking block formatting */
    .stExpander {
        border: 1px solid rgba(59, 130, 246, 0.25) !important;
        background-color: rgba(15, 23, 42, 0.5) !important;
        border-radius: 10px !important;
        margin-bottom: 1rem;
    }

    .stExpander div[data-testid="stExpanderHeader"] {
        background-color: transparent !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
    }

    /* Thinking code block styles */
    .thinking-box {
        font-family: 'JetBrains Mono', monospace;
        color: #93c5fd;
        border-left: 3px solid #3b82f6;
        padding-left: 1rem;
        font-size: 0.9rem;
        background-color: rgba(30, 41, 59, 0.3);
        padding: 0.8rem;
        border-radius: 6px;
        white-space: pre-wrap;
    }

    /* Suggested prompt cards */
    .suggestion-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .suggestion-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .suggestion-card:hover {
        background: rgba(139, 92, 246, 0.05);
        border-color: rgba(139, 92, 246, 0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to parse response content and extract thinking tokens
def parse_response(text: str):
    thinking = ""
    response = text
    
    if "<think>" in text:
        parts = text.split("<think>", 1)
        before_think = parts[0]
        after_think = parts[1]
        
        if "</think>" in after_think:
            think_parts = after_think.split("</think>", 1)
            thinking = think_parts[0]
            response = before_think + think_parts[1]
        else:
            thinking = after_think
            response = before_think
            
    return thinking.strip(), response.strip()

from langchain_core.callbacks import BaseCallbackHandler

class StreamlitStreamingHandler(BaseCallbackHandler):
    def __init__(self, response_placeholder):
        self.response_placeholder = response_placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        thinking, response = parse_response(self.text)
        
        # Display the streaming text formatted
        if thinking and response:
            self.response_placeholder.markdown(
                f"**🧠 Reasoning:**\n```\n{thinking}\n```\n\n{response} ▌"
            )
        elif thinking:
            self.response_placeholder.markdown(
                f"**🧠 Reasoning:**\n```\n{thinking} ▌\n```"
            )
        elif response:
            self.response_placeholder.markdown(response + " ▌")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_selector = st.session_state.session_id

if "hf_token_validated" not in st.session_state:
    st.session_state.hf_token_validated = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""

# Token retrieval (Load from .env file or environment variable)
st.session_state.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

# Hidden configurations (hardcoded parameters)
selected_model = "google/gemma-4-E2B-it"
selected_temp = 0.7

# Session state controls
st.sidebar.markdown("### 🛠️ Chat Management")

if st.sidebar.button("➕ Start New Chat", use_container_width=True):
    new_id = str(uuid.uuid4())
    st.session_state.session_id = new_id
    st.session_state.session_selector = new_id
    st.rerun()

# Load saved sessions
sessions = load_sessions()
if sessions:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Past Conversations")
    
    session_options = list(sessions.keys())
    session_labels = [sessions[sid]["title"] for sid in session_options]
    
    try:
        current_index = session_options.index(st.session_state.session_id)
    except ValueError:
        # Prepend the current active session
        session_options.insert(0, st.session_state.session_id)
        session_labels.insert(0, "🆕 Current Active Chat")
        current_index = 0
        
    def on_session_change():
        st.session_state.session_id = st.session_state.session_selector

    selected_sid = st.sidebar.selectbox(
        "Select conversation:",
        options=session_options,
        format_func=lambda x: session_labels[session_options.index(x)],
        index=current_index,
        key="session_selector",
        on_change=on_session_change,
        label_visibility="collapsed"
    )

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Saved Chats", use_container_width=True):
    delete_all_sessions()
    new_id = str(uuid.uuid4())
    st.session_state.session_id = new_id
    st.session_state.session_selector = new_id
    st.success("All chats cleared!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Document Q&A (RAG)")

# If document is loaded, display status and removal option
if st.session_state.retriever is not None:
    st.sidebar.markdown(f"""
    <div style='padding: 1rem; background: rgba(16, 185, 129, 0.05); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2); font-size: 0.85rem; margin-bottom: 0.5rem;'>
        <strong style='color: #10b981;'>🟢 Document Indexed</strong><br>
        <span style='color: #94a3b8; font-size: 0.8rem; word-break: break-all;'>File: {st.session_state.uploaded_filename}</span>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("🗑️ Remove Document", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.uploaded_filename = ""
        st.success("Document removed!")
        st.rerun()
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload a file to ask questions about its content.",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        with st.sidebar.spinner("Parsing and indexing document..."):
            try:
                # Read file content
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
                if filename.endswith(".txt"):
                    text = file_bytes.decode("utf-8")
                elif filename.endswith(".pdf"):
                    # Open PDF from bytes stream using PyMuPDF (fitz)
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    text = ""
                    for page in doc:
                        text += page.get_text()
                
                if not text.strip():
                    st.sidebar.error("The uploaded file is empty or could not be read.")
                else:
                    # Define the ingestion pipeline function
                    def ingest_document_flow(text_content):
                        # Safely handle cases where LangChain wraps input in a list
                        if isinstance(text_content, list):
                            text_content = text_content[0] if text_content else ""
                        
                        # Chunker runnable
                        chunker = RunnableLambda(
                            lambda t: RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([t])
                        ).with_config({"run_name": "Text Chunker"})
                        
                        # Indexer runnable wrapping index creation
                        def index_docs(splits_list):
                            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            vectorstore = FAISS.from_documents(splits_list, embeddings)
                            # Wrap in dictionary to prevent LangChain from automatically invoking the retriever
                            return {"retriever": vectorstore.as_retriever(search_kwargs={"k": 3})}
                        
                        indexer = RunnableLambda(index_docs).with_config({"run_name": "FAISS Vector Indexer"})
                        
                        # Run steps sequentially inside the trace span
                        splits = chunker.invoke(text_content)
                        return indexer.invoke(splits)

                    # Wrap the pipeline in a parent runnable
                    ingestion_chain = RunnableLambda(ingest_document_flow).with_config({"run_name": "Document Ingestion Flow"})
                    
                    # Run the ingestion chain to obtain the retriever (and unpack the dict)
                    res_dict = ingestion_chain.invoke(text)
                    st.session_state.retriever = res_dict["retriever"]
                    st.session_state.uploaded_filename = filename
                    st.sidebar.success("🟢 Document indexed successfully!")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Indexing error: {e}")

st.sidebar.markdown("---")

# LangSmith Observability Indicator
if cfg.LANGCHAIN_TRACING_V2 and cfg.LANGCHAIN_API_KEY:
    st.sidebar.markdown(f"""
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(16, 185, 129, 0.05); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2); font-size: 0.85rem;'>
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;'>
            <strong style='color: #10b981;'>🕵️‍♂️ LangSmith Tracing</strong>
            <span style='background-color: #10b981; color: #090815; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold; font-size: 0.75rem;'>ACTIVE</span>
        </div>
        <div style='color: #94a3b8; margin-bottom: 0.5rem;'>
            Project: <code style='color: #a78bfa; background: transparent; padding: 0;'>{cfg.LANGCHAIN_PROJECT}</code>
        </div>
        <a href="https://smith.langchain.com/" target="_blank" style='color: #3b82f6; text-decoration: none; font-weight: 500;'>Go to Console ↗</a>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(255, 255, 255, 0.02); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.85rem;'>
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;'>
            <strong style='color: #64748b;'>🕵️‍♂️ LangSmith Tracing</strong>
            <span style='background-color: #334155; color: #94a3b8; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold; font-size: 0.75rem;'>INACTIVE</span>
        </div>
        <div style='color: #64748b;'>
            Enable tracing by setting <code>LANGCHAIN_TRACING_V2=true</code> in your environment or <code>.env</code> file.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; color: #64748b;'>
    <strong>Model Specs: Gemma 4 E2B</strong><br>
    • Activates 2B params dynamically<br>
    • Multimodal & Audio native<br>
    • 128K context length<br>
    • Runs on HF Serverless Inference API
</div>
""", unsafe_allow_html=True)

# Main UI setup
st.markdown("<h1 class='glow-title'>🔮 Gemma 4 Chat Companion</h1>", unsafe_allow_html=True)
st.markdown("<p class='glow-subtitle'>Stateful AI Chatbot orchestrated by LangGraph & Hugging Face</p>", unsafe_allow_html=True)

# Fetch conversation history from LangGraph checkpointer
config = {"configurable": {"thread_id": st.session_state.session_id}}
try:
    state = chatbot_graph.get_state(config)
    messages = state.values.get("messages", []) if state.values else []
except Exception:
    messages = []

# Display previous messages
for msg in messages:
    if isinstance(msg, SystemMessage):
        continue  # skip system prompt rendering
    
    if isinstance(msg, ToolMessage):
        st.markdown(f"🔧 **Tool Output (`{msg.name}`):**\n```\n{msg.content}\n```")
        continue

    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.content)
        else:
            # Render tool calls if any exist on the assistant message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    st.caption(f"🤖 *Decided to use tool: `{tc['name']}` with arguments `{tc['args']}`*")
            
            # Parse and render assistant thinking and markdown response
            thinking, response = parse_response(msg.content)
            if thinking:
                with st.expander("🧠 Reasoned Thought Process", expanded=False):
                    st.markdown(f"<div class='thinking-box'>{thinking}</div>", unsafe_allow_html=True)
            if response:
                st.markdown(response)

# Handle suggestions if chat is empty
if len([m for m in messages if not isinstance(m, SystemMessage)]) == 0:
    st.markdown("### Suggested Prompts to Start:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Explain the Gemma 4 E2B parameter size approach", use_container_width=True):
            st.session_state.pending_prompt = "Explain the Gemma 4 E2B parameter size approach"
            st.rerun()
        if st.button("📝 Draft a science fiction synopsis about floating cities", use_container_width=True):
            st.session_state.pending_prompt = "Draft a science fiction synopsis about floating cities"
            st.rerun()
    with col2:
        if st.button("💻 Write a Python function for binary search with comments", use_container_width=True):
            st.session_state.pending_prompt = "Write a Python function for binary search with comments"
            st.rerun()
        if st.button("🤖 Show a basic LangGraph workflow structure in python", use_container_width=True):
            st.session_state.pending_prompt = "Show a basic LangGraph workflow structure in python"
            st.rerun()

# Check for pending prompt triggers from cards
prompt_input = ""
if "pending_prompt" in st.session_state:
    prompt_input = st.session_state.pending_prompt
    del st.session_state.pending_prompt

# Main Chat Input
user_input = st.chat_input("Message Gemma 4...")

# Override chat input if button was clicked
if prompt_input:
    user_input = prompt_input

# Process new user message
if user_input:
    # Check for token availability
    if not st.session_state.hf_token:
        st.error("❌ A Hugging Face API token is required. Please configure HUGGINGFACEHUB_API_TOKEN in your .env file.")
    else:
        # Render user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Gemma 4 is thinking..."):
                try:
                    # Setup LangGraph configurations
                    run_config = {
                        "configurable": {
                            "thread_id": st.session_state.session_id,
                            "retriever": st.session_state.retriever,
                        }
                    }
                    # Prepare graph input
                    graph_input = {"messages": [HumanMessage(content=user_input)]}
                    
                    thinking_placeholder = st.empty()
                    stream_state = {"full_text": ""}
                    
                    # Define generator for st.write_stream
                    def response_generator():
                        last_response_len = 0
                        for chunk, metadata in chatbot_graph.stream(
                            graph_input,
                            config=run_config,
                            stream_mode='messages'
                        ):
                            # Detect when agent outputs tool call chunks
                            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                                for tc in chunk.tool_calls:
                                    if tc.get("name"):
                                        st.caption(f"🤖 *Calling tool `{tc['name']}` with args: `{tc['args']}`...*")
                            
                            # Detect tool messages from the tools node
                            if metadata.get("langgraph_node") == "tools":
                                if chunk.content:
                                    st.markdown(f"🔧 **Tool Output (`{chunk.name}`):**\n```\n{chunk.content}\n```")
                            
                            if chunk.content and metadata.get("langgraph_node") == "chat_node":
                                stream_state["full_text"] += chunk.content
                                thinking, response = parse_response(stream_state["full_text"])
                                
                                # Display thinking in real-time
                                if thinking:
                                    with thinking_placeholder.container():
                                        st.markdown(f"**🧠 Reasoning:**\n```\n{thinking}\n```")
                                
                                # Yield new parts of response
                                if response:
                                    new_chunk = response[last_response_len:]
                                    if new_chunk:
                                        yield new_chunk
                                        last_response_len = len(response)
                    
                    # Stream response using st.write_stream
                    st.write_stream(response_generator())
                    
                    # Render final formatted expander for thinking
                    thinking_placeholder.empty()
                    thinking, response = parse_response(stream_state["full_text"])
                    if thinking:
                        with thinking_placeholder.container():
                            with st.expander("🧠 Reasoned Thought Process", expanded=False):
                                st.markdown(f"<div class='thinking-box'>{thinking}</div>", unsafe_allow_html=True)
                                
                    # Save the conversation history to JSON
                    try:
                        new_state = chatbot_graph.get_state(config)
                        all_messages = new_state.values.get("messages", []) if new_state.values else []
                        
                        # Retrieve or generate title
                        sessions_dict = load_sessions()
                        if st.session_state.session_id in sessions_dict:
                            title = sessions_dict[st.session_state.session_id].get("title", user_input[:30] + "...")
                        else:
                            title = user_input[:30] + "..."
                            
                        if all_messages:
                            save_session(st.session_state.session_id, title, all_messages)
                    except Exception:
                        pass
                        
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"Invocation error: {e}")
                    st.info("💡 Tip: Make sure your Hugging Face API token in your .env file is correct and has access to the requested model.")
