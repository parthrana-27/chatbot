# 🔮 Gemma 4 Chat Companion

A production-grade, stateful AI chatbot built with **Streamlit**, **LangGraph**, and the **Hugging Face Inference API**. Designed with a multi-tool agentic loop, native SQLite-backed conversation persistence, real-time streaming with reasoning token rendering, and document Q&A via Retrieval-Augmented Generation (RAG).

Live Demo: **[https://parthrana-27-chatbot.streamlit.app](https://parthrana-27-chatbot.streamlit.app)**
GitHub: **[https://github.com/parthrana-27/chatbot](https://github.com/parthrana-27/chatbot)**

---

## 📋 Table of Contents

1. [Features](#-features)
2. [System Design](#-system-design)
3. [Project Structure](#-project-structure)
4. [Component Deep Dive](#-component-deep-dive)
5. [Data Flow](#-data-flow)
6. [Key Design Decisions](#-key-design-decisions)
7. [Quick Start](#-quick-start)
8. [Deployment](#-deployment-streamlit-community-cloud)
9. [Tech Stack](#-tech-stack)
10. [Roadmap](#-roadmap)

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Persistent Chat History** | Full conversation state saved in SQLite via LangGraph's native `SqliteSaver` — survives app restarts |
| 📂 **Resume Past Sessions** | Sidebar dropdown to switch between and continue any past conversation by its `thread_id` |
| 🧠 **Reasoning Token Rendering** | Streams and displays `<think>...</think>` tokens from the model in collapsible expanders |
| ⚡ **Real-time Streaming** | Token-by-token output using `chatbot_graph.stream(stream_mode='messages')` and `st.write_stream` |
| 📄 **Document Q&A (RAG)** | Upload PDFs/TXTs → chunk → embed → FAISS index → agent can search via `search_documents` tool |
| 🔍 **Web Search Tool** | DuckDuckGo integration for real-time, factual information retrieval |
| 🧮 **Calculator Tool** | Safe arithmetic expression evaluator with character allowlist (no `eval` injection) |
| 🕵️ **LangSmith Observability** | Optional full LLM trace logging with configurable project name |
| ☁️ **Streamlit Cloud Ready** | Token resolution via `st.secrets` for cloud; falls back to `.env` for local dev |

---

## 🏗️ System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                             │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (port 8000)
┌────────────────────────────▼────────────────────────────────────┐
│                   Streamlit Frontend (app.py)                   │
│                                                                 │
│  ┌──────────────────┐     ┌────────────────────────────────┐   │
│  │   Sidebar Panel  │     │         Main Chat Panel        │   │
│  │                  │     │                                │   │
│  │  • New Chat      │     │  • Message history display     │   │
│  │  • Past Sessions │     │  • Suggested prompts           │   │
│  │  • Clear All     │     │  • Real-time streaming output  │   │
│  │  • RAG Uploader  │     │  • Thinking token expanders    │   │
│  │  • LangSmith     │     │  • Tool call captions          │   │
│  │    status        │     │                                │   │
│  └──────────────────┘     └────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Python function call
┌────────────────────────────▼────────────────────────────────────┐
│                  LangGraph Agent (backend/graph.py)             │
│                                                                 │
│   START → [chat_node] → tools_condition ──┬──→ [tools] ──┐     │
│                                           │               └──→  │
│                                           └──→ END              │
│                                                                 │
│   LLM: Qwen/Qwen2.5-72B (HF Inference API, OpenAI-compat)     │
│   Tools: calculate | web_search | search_documents             │
└────────────────────────────┬────────────────────────────────────┘
                             │ read/write
┌────────────────────────────▼────────────────────────────────────┐
│                     SQLite (chat_sessions.db)                   │
│                                                                 │
│  ┌──────────────────────────┐  ┌──────────────────────────┐    │
│  │    chat_sessions table   │  │  LangGraph checkpoint     │    │
│  │  (custom session index)  │  │  tables (SqliteSaver)     │    │
│  │                          │  │                           │    │
│  │  session_id (PK)         │  │  checkpoints              │    │
│  │  title                   │  │  checkpoint_blobs         │    │
│  │  messages (JSON)         │  │  checkpoint_writes        │    │
│  │  created_at              │  │  (keyed by thread_id)     │    │
│  └──────────────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph Agent Flow

```
User Input (HumanMessage)
        │
        ▼
  ┌─────────────┐     Invokes LLM with all messages + bound tools
  │  chat_node  │─────────────────────────────────────────────────────►
  └──────┬──────┘                                           Qwen2.5-72B
         │                                                  (HF API)
         ▼                                                       │
  tools_condition ◄───────────────────────────────────────────────
         │
   ┌─────┴─────┐
   │           │
   ▼           ▼
[tool_calls] [no tool_calls]
   │               │
   ▼               ▼
┌──────┐         END
│tools │  (ToolNode executes the requested tool)
│ node │
└──┬───┘
   │  Tool result (ToolMessage)
   └──────────────────► back to chat_node (loop continues)
```

### RAG Pipeline (Document Q&A)

```
User uploads PDF/TXT
        │
        ▼
   PyMuPDF / UTF-8 decode
        │
        ▼
   RecursiveCharacterTextSplitter
   (chunk_size=1000, overlap=200)
        │
        ▼
   HuggingFaceEmbeddings
   (all-MiniLM-L6-v2, 384-dim)
        │
        ▼
   FAISS.from_documents()
   → vectorstore.as_retriever(k=3)
        │
        ▼
   Stored in st.session_state.retriever
   Passed via LangGraph config["configurable"]["retriever"]
        │
        ▼
   search_documents tool reads retriever from RunnableConfig
   → returns top-3 relevant chunks to the LLM
```

---

## 📁 Project Structure

```
chatbot/
│
├── app.py                      # Streamlit app entry point & UI logic
│   ├── DB functions            #   init_db, load_sessions, save_session, delete_all_sessions
│   ├── Session state init      #   session_id, session_selector, retriever, hf_token
│   ├── Sidebar panel           #   New Chat, Past Conversations, Clear, RAG uploader, LangSmith
│   ├── Message rendering       #   HumanMessage, AIMessage, ToolMessage display + thinking expander
│   ├── Suggested prompts       #   Quick-start cards for empty sessions
│   ├── RAG ingestion pipeline  #   chunker → embedder → FAISS indexer as RunnableLambda chain
│   └── Chat processing         #   stream loop, tool call captions, session save, st.rerun()
│
├── backend/
│   ├── graph.py                # LangGraph StateGraph definition
│   │   ├── Token resolution    #   st.secrets → os.getenv fallback
│   │   ├── LLM setup           #   ChatOpenAI pointing to HF Inference API router
│   │   ├── Tool definitions    #   @tool decorators: calculate, web_search, search_documents
│   │   ├── ChatState           #   TypedDict with Annotated[list[BaseMessage], add_messages]
│   │   ├── chat_node           #   LLM invocation node
│   │   ├── tool_node           #   ToolNode(tools) — prebuilt executor
│   │   ├── SqliteSaver setup   #   conn + checkpointer.setup() + graph.compile(checkpointer=...)
│   │   └── chatbot_graph       #   Exported compiled graph used by app.py
│   │
│   └── config.py               # Centralized config & token resolution
│       ├── HUGGINGFACEHUB_API_TOKEN
│       ├── DEFAULT_MODEL
│       ├── PORT / HOST
│       ├── LangSmith settings
│       └── get_hf_token()      #   Utility function for override → secrets → env chain
│
├── run.py                      # Programmatic Streamlit launcher (sys.argv override)
│
├── chat_sessions.db            # Unified SQLite database (gitignored)
├── requirements.txt            # All Python dependencies
├── .env                        # Local secrets (gitignored)
├── .gitignore
└── .devcontainer/
    └── devcontainer.json       # Dev Container for Streamlit Cloud / GitHub Codespaces
```

---

## 🔬 Component Deep Dive

### `app.py` — Frontend & Orchestration

- **Session Management**: Each conversation is identified by a UUID (`session_id`) stored in `st.session_state`. This same UUID is used as the `thread_id` in LangGraph's config, connecting the UI session to the correct checkpoint chain.
- **Dual Storage Strategy**: When a message is submitted, the app saves both to the LangGraph checkpoint (automatic, via `chatbot_graph.stream()`) and to the custom `chat_sessions` table (for sidebar title/index display).
- **`on_change` Pattern**: The Past Conversations selectbox uses Streamlit's `on_change=on_session_change` callback. This ensures `session_id` is only updated when the user *manually* changes the dropdown — not during reruns triggered by chat submission — preventing accidental session switching.
- **Streaming with Thinking Tokens**: The `response_generator()` function yields response tokens incrementally while simultaneously updating a `thinking_placeholder` for `<think>` content. After streaming completes, the raw streaming placeholder is replaced with a proper collapsible expander.

### `backend/graph.py` — LangGraph Agent

- **State Schema**: Uses `TypedDict` with `Annotated[list[BaseMessage], add_messages]` — the `add_messages` reducer ensures new messages are *appended* to history rather than overwriting it.
- **LLM Binding**: `llm.bind_tools(tools)` attaches tool schemas to the LLM, enabling the model to emit structured `tool_calls` in its responses which `ToolNode` then executes.
- **`search_documents` Tool**: Receives the FAISS retriever through `RunnableConfig` (passed as `config["configurable"]["retriever"]`), keeping the retriever stateless at the graph level while allowing per-invocation injection from the frontend.
- **Checkpointer**: `SqliteSaver(conn)` with `check_same_thread=False` supports Streamlit's multi-threaded execution model. `checkpointer.setup()` is called once at module load to create the required schema tables.

### `backend/config.py` — Configuration Layer

- Centralizes all environment variable resolution with a three-layer priority: UI override → `st.secrets` → `.env`/OS environment.
- `get_hf_token()` is a reusable utility that allows token injection from any caller, useful for future multi-user or API-key-per-request scenarios.

### SQLite Database Schema

```sql
-- Custom session index (managed by app.py)
CREATE TABLE chat_sessions (
    session_id TEXT PRIMARY KEY,
    title      TEXT,
    messages   TEXT,              -- JSON-serialized via messages_to_dict()
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LangGraph checkpoint tables (managed by SqliteSaver.setup())
-- checkpoints          — full graph state snapshots
-- checkpoint_blobs     — large blob storage for message content
-- checkpoint_writes    — pending writes between checkpoints
-- (all keyed by thread_id = session_id)
```

---

## 🔄 Data Flow

### New Message Flow (Step by Step)

```
1. User types in st.chat_input("Message Gemma 4...")
2. app.py constructs HumanMessage(content=user_input)
3. Calls chatbot_graph.stream({"messages": [HumanMessage]}, config=run_config, stream_mode='messages')
4. LangGraph loads previous state from SQLite (via thread_id / session_id)
5. chat_node invokes llm_with_tools.invoke(all_messages)
   → If LLM returns tool_calls: routes to tools node
   → tools node executes tool, appends ToolMessage, returns to chat_node
   → chat_node invokes LLM again with tool results in context
   → If LLM returns plain AIMessage: routes to END
6. Each content chunk is yielded through response_generator() → st.write_stream()
7. LangGraph auto-saves checkpoint to SQLite after each node execution
8. After stream completes: chatbot_graph.get_state(config) retrieves final state
9. save_session() updates the chat_sessions table with new title + messages
10. st.rerun() triggers full page refresh to render the completed message
```

### Session Restore Flow

```
1. User selects past session from sidebar dropdown
2. on_session_change() sets st.session_state.session_id = selected_sid
3. st.rerun() fires
4. app.py calls chatbot_graph.get_state({"configurable": {"thread_id": session_id}})
5. SqliteSaver loads checkpoint from SQLite by thread_id
6. state.values["messages"] is returned — full history restored
7. Message rendering loop displays all historical messages
```

---

## 🤔 Key Design Decisions

### Why LangGraph over a simple LLM call loop?

LangGraph provides a **stateful, cyclical computation graph** that natively handles:
- Tool call → result → re-invoke loops without manual orchestration
- First-class checkpoint/persistence via swappable `checkpointer` backends
- Structured node/edge routing (e.g., `tools_condition` prebuilt)
- Future extensibility (multi-agent subgraphs, human-in-the-loop, etc.)

### Why SQLite over MemorySaver?

`MemorySaver` stores state **in-process RAM only** — all history is lost on Streamlit rerun or app restart. `SqliteSaver` writes to disk after every graph step, making conversation history durable across:
- Browser refreshes
- App restarts
- Cloud container cold starts (within the same deployment instance)

### Why a single `chat_sessions.db` for both tables?

Keeping LangGraph checkpoints and the session metadata index in the same file eliminates the need to synchronize two separate database connections and simplifies backup, export, and cleanup operations (one file to manage).

### Why `on_change` for the session selector?

Streamlit re-executes the entire script on every interaction. Without `on_change`, assigning `session_id = selected_sid` directly in the script body would re-trigger on every chat input submission rerun, potentially switching the active session mid-conversation. The `on_change` callback fires *only* when the user explicitly changes the dropdown value.

### Why HuggingFace Inference API with OpenAI-compatible interface?

Using `ChatOpenAI` pointed at `https://router.huggingface.co/v1` allows the project to use any HF-hosted model without custom wrappers, while retaining full LangChain tool-binding and streaming compatibility. Switching models requires only changing the `model=` string.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Hugging Face account with an API token

### 1. Clone

```bash
git clone https://github.com/parthrana-27/chatbot.git
cd chatbot
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_your_key_here
LANGCHAIN_PROJECT=chatbot-observability
```

### 4. Run

```bash
python run.py
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## ☁️ Deployment (Streamlit Community Cloud)

Live at: **[https://parthrana-27-chatbot.streamlit.app](https://parthrana-27-chatbot.streamlit.app)**

### Steps

1. Push to [GitHub](https://github.com/parthrana-27/chatbot).
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app** → connect repository.
3. Set **Main file path**: `app.py`
4. Under **Advanced → Secrets**, add:

```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "lsv2_your_key_here"
LANGCHAIN_PROJECT = "chatbot-observability"
```

> ⚠️ **Cloud Storage Note**: On Streamlit Community Cloud, the `chat_sessions.db` file lives in the ephemeral container filesystem and resets on each redeploy. For persistent cloud storage, integrate an external DB (Supabase, PlanetScale, or Neon) and swap `SqliteSaver` for `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`.

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Frontend** | Streamlit | Rapid Python-native UI with built-in session state and streaming |
| **Agent Orchestration** | LangGraph `StateGraph` | Cyclical graph with native checkpointing and tool routing |
| **LLM** | Qwen/Qwen2.5-72B (HF Inference API) | Free-tier access, OpenAI-compatible, strong reasoning |
| **Checkpointing** | `SqliteSaver` (langgraph-checkpoint-sqlite) | Persistent, zero-dependency local persistence |
| **Session DB** | SQLite | Lightweight, serverless, single-file storage |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Fast, lightweight, high-quality sentence embeddings |
| **Vector Store** | FAISS (`faiss-cpu`) | In-memory ANN search, no server required |
| **PDF Parsing** | PyMuPDF (`fitz`) | Fast, accurate text extraction from PDFs |
| **Web Search** | DuckDuckGo | No API key required, privacy-respecting |
| **Observability** | LangSmith | Full LLM call tracing, latency, token usage |

---

## 📦 Requirements

```
langchain>=0.3.0
langgraph>=0.1.0
langgraph-checkpoint-sqlite>=2.0.0
langsmith>=0.1.0
langchain-openai>=0.1.0
langchain-huggingface>=0.1.0
langchain-community>=0.3.0
streamlit>=1.30.0
python-dotenv>=1.0.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
PyMuPDF>=1.22.0
duckduckgo-search>=6.0.0
pydantic>=2.0.0
typing-extensions>=4.13.0
```

---

## 🗺️ Roadmap

- [ ] Persistent cloud database (swap SQLite → PostgreSQL via `AsyncPostgresSaver`)
- [ ] Multi-user session isolation with authentication
- [ ] Image/multimodal input support (native Gemma 4 capability)
- [ ] Conversation export (JSON / Markdown download)
- [ ] Custom system prompt editor in the sidebar UI
- [ ] Streaming tool output display (real-time tool result injection)
