# 🔮 Gemma 4 Chat Companion

A **production-grade, stateful AI chatbot** built with Streamlit, LangGraph, and the Hugging Face Inference API. Features a multi-tool agentic loop, native SQLite-backed conversation persistence, real-time token streaming with reasoning block rendering, and document Q&A via Retrieval-Augmented Generation (RAG).

**Live Demo:** [https://parthrana-27-chatbot.streamlit.app](https://parthrana-27-chatbot.streamlit.app)  
**GitHub:** [https://github.com/parthrana-27/chatbot](https://github.com/parthrana-27/chatbot)

---

## 📋 Table of Contents

1. [Features](#-features)
2. [System Design](#-system-design)
3. [Project Structure](#-project-structure)
4. [Component Deep Dive](#-component-deep-dive)
5. [Data Flow](#-data-flow)
6. [Key Design Decisions & Trade-offs](#-key-design-decisions--trade-offs)
7. [Interview Q&A](#-interview-qa)
8. [Quick Start](#-quick-start)
9. [Deployment](#-deployment-streamlit-community-cloud)
10. [Tech Stack](#-tech-stack)
11. [Roadmap](#-roadmap)

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Persistent Chat History** | Full conversation state saved to SQLite via LangGraph's native `SqliteSaver` — survives app restarts |
| 📂 **Resume Past Sessions** | Sidebar dropdown to switch between and continue any past conversation by UUID |
| 🧠 **Reasoning Token Rendering** | Streams and parses `<think>...</think>` tokens in real-time; renders in collapsible expanders |
| ⚡ **Real-time Token Streaming** | Per-token output using `stream_mode='messages'` + `st.write_stream()` |
| 📄 **Document Q&A (RAG)** | Upload PDFs/TXTs → chunk → embed → FAISS index → agent searches via `search_documents` tool |
| 🔍 **Web Search Tool** | DuckDuckGo integration for real-time factual information retrieval |
| 🧮 **Calculator Tool** | Safe arithmetic evaluator with character allowlist (no arbitrary code execution) |
| 🕵️ **LangSmith Observability** | Optional full LLM trace logging — latency, token counts, tool calls, all tracked per run |
| ☁️ **Streamlit Cloud Ready** | Token resolution via `st.secrets` for cloud; graceful fallback to `.env` for local dev |

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
User uploads PDF or TXT file
        │
        ├──[PDF]── PyMuPDF (fitz.open stream) → page.get_text()
        └──[TXT]── file_bytes.decode("utf-8")
        │
        ▼
   RecursiveCharacterTextSplitter
   chunk_size=1000, chunk_overlap=200
   → List[Document]
        │
        ▼
   HuggingFaceEmbeddings("all-MiniLM-L6-v2")
   384-dimensional dense vectors, runs locally
        │
        ▼
   FAISS.from_documents(splits, embeddings)
   → vectorstore.as_retriever(search_kwargs={"k": 3})
        │
        ▼
   Stored in st.session_state.retriever
   Passed at query time via:
   config["configurable"]["retriever"] → RunnableConfig
        │
        ▼
   search_documents tool reads retriever from RunnableConfig
   → runs retriever.invoke(query) → top-3 chunks
   → returns formatted context string to the LLM
```

> **Why inject via RunnableConfig?** Keeps the LangGraph state schema lean (no binary blob in `ChatState`), allows per-invocation retriever injection without graph recompilation, and cleanly separates ephemeral session data from persistent conversation history.

---

### Streaming Architecture

```
chatbot_graph.stream(input, config, stream_mode='messages')
        │
        ├── yields (chunk, metadata) for every LLM token / tool output
        │
        ▼
response_generator()  [Python generator function]
        │
        ├── metadata["langgraph_node"] == "chat_node"
        │       → accumulate to stream_state["full_text"]
        │       → parse_response() splits <think>...</think>
        │       → yield new response tokens only (delta)
        │       → update thinking_placeholder in real-time
        │
        └── metadata["langgraph_node"] == "tools"
                → display tool output as inline markdown block
        │
        ▼
st.write_stream(response_generator())
        │
        ▼
After stream ends:
  • thinking_placeholder replaced with collapsible st.expander
  • chatbot_graph.get_state() fetches final state
  • save_session() persists to chat_sessions table
  • st.rerun() refreshes the full UI
```

---

## 📁 Project Structure

```
chatbot/
│
├── app.py                          # Streamlit app — all UI + orchestration logic
│   │
│   ├── [DB Layer]
│   │   ├── init_db()               # Creates chat_sessions table if missing
│   │   ├── load_sessions()         # Loads all sessions ordered by created_at DESC
│   │   ├── save_session()          # Upserts session (ON CONFLICT DO UPDATE)
│   │   └── delete_all_sessions()  # Deletes all rows + removes DB file
│   │
│   ├── [Streaming]
│   │   ├── parse_response(text)    # Splits <think>...</think> from response text
│   │   └── response_generator()   # Generator: routes chunks by langgraph_node
│   │
│   ├── [Session State]
│   │   ├── session_id             # Active UUID = LangGraph thread_id
│   │   ├── session_selector       # Bound to sidebar selectbox widget
│   │   ├── retriever              # FAISS retriever (None if no doc uploaded)
│   │   └── hf_token               # Resolved HF token for validation gate
│   │
│   ├── [Sidebar]
│   │   ├── ➕ Start New Chat       # Generates new uuid4, calls st.rerun()
│   │   ├── 📂 Past Conversations  # on_change selectbox — avoids rerun side effects
│   │   ├── 🗑️ Clear All Chats    # Deletes sessions + resets state
│   │   └── 📄 RAG Uploader        # PDF/TXT → chunker → FAISS → session_state
│   │
│   └── [Chat Processing]
│       ├── Token validation gate  # Blocks if hf_token is empty
│       ├── st.write_stream()      # Progressive rendering via generator
│       ├── Tool call captions     # Inline display of tool name + args
│       └── post-stream save       # get_state → save_session → st.rerun()
│
├── backend/
│   │
│   ├── graph.py                    # LangGraph StateGraph + tools + checkpointer
│   │   ├── Token resolution        # st.secrets → os.getenv (raises if missing)
│   │   ├── ChatOpenAI LLM          # HF router, OpenAI-compat, streaming=True
│   │   ├── @tool: calculate        # Safe eval, allowlist chars only
│   │   ├── @tool: web_search       # DuckDuckGoSearchRun wrapper
│   │   ├── @tool: search_documents # Reads retriever from RunnableConfig
│   │   ├── ChatState               # TypedDict + Annotated add_messages reducer
│   │   ├── chat_node               # llm_with_tools.invoke(messages)
│   │   ├── tool_node               # ToolNode(tools) — prebuilt executor
│   │   ├── SqliteSaver setup       # Unified DB, check_same_thread=False
│   │   └── chatbot_graph           # Compiled graph exported to app.py
│   │
│   └── config.py                   # Centralized configuration
│       ├── HUGGINGFACEHUB_API_TOKEN  # st.secrets → os.getenv
│       ├── DEFAULT_MODEL           # google/gemma-4-E2B-it (display label)
│       ├── PORT / HOST             # Server bind config
│       ├── LangSmith settings      # TRACING_V2, API_KEY, PROJECT, ENDPOINT
│       └── get_hf_token(override)  # Override → secrets → env chain
│
├── run.py                          # Programmatic Streamlit launcher
│                                   # sys.argv = ["streamlit", "run", "app.py",
│                                   #             "--server.port", "8000"]
│
├── chat_sessions.db                # Unified SQLite DB (gitignored)
│   ├── chat_sessions               # Custom session metadata index
│   ├── checkpoints                 # LangGraph state snapshots
│   ├── checkpoint_blobs            # Large message blob storage
│   └── checkpoint_writes           # Pending write buffer
│
├── requirements.txt                # All Python dependencies (19 packages)
├── .env                            # Local secrets (gitignored)
├── .gitignore
└── .devcontainer/
    └── devcontainer.json           # GitHub Codespaces / Dev Container config
```

---

## 🔬 Component Deep Dive

### `app.py` — Frontend & Orchestration

**Session Identity:**  
Each conversation is a UUID (`session_id`) stored in `st.session_state`. This UUID is used as the `thread_id` in LangGraph's config dict, linking the Streamlit UI session directly to the correct SQLite checkpoint chain. The same UUID is the primary key in the `chat_sessions` table.

**Dual Storage Strategy:**  
After each message, the app writes to two places:
1. **LangGraph checkpoint** — automatic, via `chatbot_graph.stream()`. Stores the full `ChatState` (all `BaseMessage` objects) as binary blobs in SQLite.
2. **`chat_sessions` table** — manual, via `save_session()`. Stores JSON-serialized messages (via `messages_to_dict()`) plus a human-readable title for the sidebar.

This dual-write exists because LangGraph's checkpoint tables are not designed for easy UI enumeration — they're keyed by internal checkpoint IDs, not human-readable titles.

**`on_change` Callback Pattern:**
```python
def on_session_change():
    st.session_state.session_id = st.session_state.session_selector

selected_sid = st.sidebar.selectbox(..., key="session_selector", on_change=on_session_change)
```
Streamlit re-executes the entire script on every widget interaction. Without `on_change`, assigning `session_id = selected_sid` in the script body would re-trigger on every chat submit rerun, silently switching sessions mid-conversation. The callback fires **only** when the user explicitly changes the dropdown.

**Streaming with Think-Token Parsing:**
```python
def parse_response(text: str) -> tuple[str, str]:
    if "<think>" in text:
        # splits thinking content from response content
        ...
    return thinking.strip(), response.strip()
```
During streaming, `response_generator()` accumulates the full text and calls `parse_response()` on every chunk to progressively extract the thinking region and yield only the response portion to `st.write_stream()`. After the stream ends, the raw placeholder is swapped for a collapsible `st.expander`.

---

### `backend/graph.py` — LangGraph Agent

**State Schema:**
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```
The `add_messages` reducer means new messages are **appended** to the list rather than overwriting it — this is what gives the graph its conversation memory across multiple calls within the same thread.

**Tool Binding:**
```python
llm_with_tools = llm.bind_tools(tools)
```
`bind_tools()` attaches JSON schemas of all tools to the LLM's system prompt. When the model decides to call a tool, it returns an `AIMessage` with a `tool_calls` field (structured JSON) instead of a text response. `ToolNode` reads these, dispatches to the correct Python function, and returns a `ToolMessage`.

**`search_documents` — Stateless Retriever Injection:**
```python
@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    retriever = config.get("configurable", {}).get("retriever", None)
    ...
```
The retriever is not stored in `ChatState` (which would require serializing the FAISS index to SQLite — impractical). Instead, it's injected per-invocation via `RunnableConfig`. The frontend passes it as:
```python
run_config = {"configurable": {"thread_id": session_id, "retriever": st.session_state.retriever}}
```

**Checkpointer Setup:**
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "chat_sessions.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()   # Creates checkpoint schema tables (idempotent)
chatbot_graph = graph.compile(checkpointer=checkpointer)
```
`check_same_thread=False` is required because Streamlit may execute callbacks from different OS threads. `setup()` is safe to call on every app load — it uses `CREATE TABLE IF NOT EXISTS` internally.

---

### `backend/config.py` — Configuration Layer

- Centralizes all environment variable resolution with a three-layer priority: UI override → `st.secrets` → `.env`/OS environment.
- `get_hf_token()` is a reusable utility that allows token injection from any caller, useful for future multi-user or API-key-per-request scenarios.
- LangSmith tracing is enabled by setting `LANGCHAIN_TRACING_V2=true` — the sidebar shows an active/inactive badge with a direct link to the LangSmith console.

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

### New Message (Step by Step)

```
1.  User types in st.chat_input() → user_input string
2.  app.py wraps as HumanMessage(content=user_input)
3.  Calls chatbot_graph.stream(
          {"messages": [HumanMessage]},
          config={"configurable": {"thread_id": session_id, "retriever": retriever}},
          stream_mode='messages'
      )
4.  LangGraph loads previous ChatState from SQLite checkpoint (by thread_id)
5.  chat_node: llm_with_tools.invoke(all_messages)
        → If AIMessage has tool_calls:
              routes to tools node
              ToolNode dispatches to calculate / web_search / search_documents
              ToolMessage appended to state, returns to chat_node
              LLM re-invoked with tool result in context
        → If plain AIMessage (no tool_calls):
              routes to END
6.  Each content chunk yielded → response_generator() filters by langgraph_node
7.  st.write_stream() renders tokens progressively in browser
8.  LangGraph auto-saves checkpoint after each node completes
9.  Stream ends → chatbot_graph.get_state(config) fetches final state
10. save_session() upserts to chat_sessions table (title + serialized messages)
11. st.rerun() triggers full page re-render with completed message in history
```

### Session Restore Flow

```
1.  User selects past session from sidebar selectbox
2.  on_session_change() fires → st.session_state.session_id = selected_sid
3.  st.rerun() executes
4.  app.py: chatbot_graph.get_state({"configurable": {"thread_id": session_id}})
5.  SqliteSaver loads latest checkpoint from SQLite (by thread_id)
6.  state.values["messages"] → full BaseMessage list restored
7.  Message rendering loop: displays HumanMessage / AIMessage / ToolMessage
        • Skips SystemMessage (internal prompt, not shown)
        • ToolMessage rendered as code block
        • AIMessage: parse_response() renders think block + markdown
```

---

## 🤔 Key Design Decisions & Trade-offs

### 1. LangGraph over a Raw LLM Loop

**Chosen:** LangGraph `StateGraph`  
**Alternatives considered:** Manual `while True` loop, `AgentExecutor` (LangChain legacy)

LangGraph provides:
- **Cyclical graph execution** — tool call → result → re-invoke loops handled automatically via edge routing
- **First-class persistence** — swappable `checkpointer` backends (Memory → SQLite → Postgres) with zero application code change
- **Structured routing** — `tools_condition` prebuilt handles `tool_calls` vs no `tool_calls` branching
- **Future-proof** — native support for multi-agent subgraphs, human-in-the-loop interrupts, and streaming

**Trade-off:** LangGraph adds import complexity and a learning curve vs. a simple loop.

---

### 2. SqliteSaver over MemorySaver

**Chosen:** `SqliteSaver` (`langgraph-checkpoint-sqlite`)  
**Alternative:** `MemorySaver` (in-process RAM)

`MemorySaver` stores state in a Python dict — lost on every Streamlit rerun, app restart, or container cold start. `SqliteSaver` writes to disk after every node execution:
- Survives browser refreshes and app restarts
- Survives Streamlit's aggressive script-rerun model
- Zero external dependencies (no Postgres server needed)

**Trade-off:** On Streamlit Community Cloud, the filesystem is ephemeral per-deployment — the DB resets on each redeploy. For true cloud persistence, swap `SqliteSaver` → `AsyncPostgresSaver` (from `langgraph-checkpoint-postgres`).

---

### 3. Single `chat_sessions.db` for Both Tables

**Chosen:** One unified SQLite file for custom metadata + LangGraph checkpoint tables  
**Alternative:** Two separate DB files

Benefits:
- Single file to backup, export, and clean up
- One SQLite connection pool to manage
- Atomic consistency — session metadata and checkpoint always in sync

**Trade-off:** LangGraph's checkpoint tables are opaque (keyed by internal checkpoint IDs) — the custom `chat_sessions` table exists purely for UI enumeration with human-readable titles.

---

### 4. `on_change` Callback for Session Selector

**Chosen:** `on_change=on_session_change` callback  
**Alternative:** Reading `session_selector` directly from `st.session_state` in script body

Streamlit re-runs the entire script on every widget interaction (chat submit, button click, etc.). Reading the selectbox value directly in the script body would reassign `session_id` on *every* rerun — including chat submits — silently switching sessions. The `on_change` callback fires only when the user explicitly changes the selectbox value.

---

### 5. HuggingFace Inference API with OpenAI-Compatible Client

**Chosen:** `ChatOpenAI` pointed at `https://router.huggingface.co/v1`  
**Alternative:** `HuggingFaceEndpoint` from `langchain-huggingface`

Using the OpenAI-compatible endpoint means:
- Full LangChain `bind_tools()` and structured output support (designed for OpenAI schema)
- `streaming=True` works natively with LangGraph's `stream_mode='messages'`
- Switching models requires changing only the `model=` string
- No custom wrapper code

---

### 6. FAISS In-Memory Vector Store for RAG

**Chosen:** `faiss-cpu` (in-process)  
**Alternative:** Chroma, Pinecone, Weaviate (persistent vector DBs)

FAISS requires no server, no external dependencies, and no API keys. For a single-user chatbot with per-session document uploads, in-memory is perfectly sufficient. The retriever lives in `st.session_state` (not serialized to disk) — it's rebuilt fresh on each upload, which is appropriate since documents are session-scoped.

**Trade-off:** FAISS index is lost when the session ends or the app restarts. For a multi-user, multi-document knowledge base, a persistent vector DB would be required.

---

## 💼 Interview Q&A

### "Walk me through the system architecture."

> "The app has three layers. The **Streamlit frontend** (`app.py`) handles the UI — session management, RAG document upload, and streaming the response. The **LangGraph agent** (`backend/graph.py`) is the brain — it's a stateful graph with two nodes: a chat node that calls the LLM, and a tools node that executes external tools like web search or document retrieval. The **SQLite database** (`chat_sessions.db`) is the persistence layer — it holds both the LangGraph checkpoints (full conversation state) and a custom session index table for the sidebar UI. The frontend calls the agent's `stream()` method, which handles the LLM + tool loop and streams tokens back, and the checkpointer automatically saves state to SQLite after every node."

---

### "How does conversation persistence work?"

> "LangGraph's `SqliteSaver` checkpointer writes the full `ChatState` — which is a list of `BaseMessage` objects — to SQLite after every node execution in the graph. Each conversation is identified by a `thread_id` (a UUID). When the user switches to a past conversation, the app calls `chatbot_graph.get_state()` with that thread's ID, and the checkpointer loads the latest checkpoint from SQLite and returns the full message history. This means every conversation survives app restarts, browser refreshes, and Streamlit reruns."

---

### "How does the RAG pipeline work?"

> "When a user uploads a PDF, PyMuPDF extracts the text. We then chunk it using `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), embed the chunks using HuggingFace's `all-MiniLM-L6-v2` model (384-dim vectors), and index them in FAISS. The retriever is stored in `st.session_state`. When the agent needs to answer a question about the document, it calls the `search_documents` tool, which receives the retriever via `RunnableConfig` (not from `ChatState`) and runs a similarity search, returning the top 3 most relevant chunks as context to the LLM."

---

### "Why do you inject the retriever via `RunnableConfig` instead of storing it in `ChatState`?"

> "Storing a FAISS retriever in `ChatState` would mean trying to serialize a Python object containing a NumPy index to SQLite — that's not feasible. `ChatState` only holds serializable `BaseMessage` objects. By injecting the retriever via `RunnableConfig` at invocation time, I keep the graph state schema clean and serializable, while still giving the tool access to the retriever. It's a clean separation: persistent conversation history in `ChatState`, ephemeral session data in `RunnableConfig`."

---

### "How does real-time streaming work?"

> "I use LangGraph's `stream(stream_mode='messages')` which yields `(chunk, metadata)` tuples for every token the LLM produces and every tool execution. The `metadata['langgraph_node']` field tells me whether the chunk came from the `chat_node` (LLM response) or the `tools` node (tool output). I yield only the `chat_node` tokens through a Python generator that `st.write_stream()` consumes. Simultaneously, if the token stream contains `<think>` tags, I parse them in real-time and display a reasoning placeholder. When the stream ends, I replace that placeholder with a proper collapsible expander."

---

### "How do you prevent the session from switching accidentally on every chat message?"

> "Streamlit re-executes the entire script on every user interaction. If I read the selectbox value directly in the script body, it would reset `session_id` on every chat submit rerun. I solve this with Streamlit's `on_change` callback — the function only fires when the user explicitly changes the dropdown, not on other reruns. So session switching is intentional and explicit, not a side effect of the reactive re-run model."

---

### "What are the limitations of this design? How would you scale it?"

> "A few key limitations:
> 1. **Ephemeral cloud storage** — SQLite on Streamlit Cloud resets on redeploy. Fix: swap `SqliteSaver` for `AsyncPostgresSaver` (Neon, Supabase).
> 2. **In-memory FAISS** — the vector index is lost when the session ends. Fix: use a persistent vector DB like Chroma or Weaviate with a stable namespace per session.
> 3. **Single-user design** — no authentication, all sessions visible to any user. Fix: add auth (e.g., `st-login`) and scope session queries by user ID.
> 4. **No horizontal scaling** — Streamlit is a single-process server. Fix: deploy with FastAPI + a React frontend, move LangGraph to a background worker (Celery/Redis)."

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
| **Frontend** | Streamlit | Python-native UI with built-in session state, widgets, and streaming |
| **Agent Orchestration** | LangGraph `StateGraph` | Cyclical graph with native checkpointing and conditional tool routing |
| **LLM** | Qwen/Qwen2.5-72B via HF Inference API | Free-tier, OpenAI-compatible, strong tool-calling and reasoning |
| **Checkpointing** | `SqliteSaver` (`langgraph-checkpoint-sqlite`) | Zero-dependency persistent state, survives app restarts |
| **Session DB** | SQLite | Lightweight, serverless, single-file — perfect for single-user apps |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Fast, 384-dim, high-quality sentence embeddings, runs locally |
| **Vector Store** | FAISS (`faiss-cpu`) | In-process ANN search, no server required, sufficient for session-scoped RAG |
| **PDF Parsing** | PyMuPDF (`fitz`) | Fast, accurate text extraction, handles complex PDFs |
| **Web Search** | DuckDuckGo | No API key required, privacy-respecting, real-time results |
| **Observability** | LangSmith | Full LLM call tracing — latency, token counts, tool calls, per-run |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langchain` | ≥0.3.0 | Core orchestration, message types, callbacks |
| `langgraph` | ≥0.1.0 | Stateful agent graph |
| `langgraph-checkpoint-sqlite` | ≥2.0.0 | Native SQLite checkpointer |
| `langchain-openai` | ≥0.1.0 | OpenAI-compat LLM client (points to HF router) |
| `langchain-huggingface` | ≥0.1.0 | HuggingFace embeddings |
| `langchain-community` | ≥0.3.0 | DuckDuckGo search tool |
| `langsmith` | ≥0.1.0 | Tracing & observability |
| `streamlit` | ≥1.30.0 | Frontend framework |
| `python-dotenv` | ≥1.0.0 | `.env` file loading |
| `faiss-cpu` | ≥1.7.4 | Vector similarity search |
| `sentence-transformers` | ≥2.2.2 | `all-MiniLM-L6-v2` embedding model |
| `PyMuPDF` | ≥1.22.0 | PDF text extraction |
| `duckduckgo-search` | ≥6.0.0 | DuckDuckGo search backend |
| `pydantic` | ≥2.0.0 | Data validation |
| `typing-extensions` | ≥4.13.0 | Backported type hints |

---

## 🗺️ Roadmap

- [ ] Persistent cloud database — swap SQLite → PostgreSQL via `AsyncPostgresSaver`
- [ ] Multi-user session isolation with authentication (`st-login` or OAuth)
- [ ] Image/multimodal input support (native Gemma 4 capability)
- [ ] Conversation export — JSON / Markdown download button
- [ ] Custom system prompt editor in the sidebar UI
- [ ] Support multiple simultaneous documents (multi-namespace FAISS or Chroma)
- [ ] Streaming tool output display — real-time tool result injection mid-stream
