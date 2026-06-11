# Gemma 4 Chat Companion

A chatbot companion built with Streamlit, LangGraph, and Hugging Face.

## Features

* Chat Interface: Dark-themed conversational interface.
* Reasoned Thought Process: Displays the thinking steps of the model.
* Saved Chats: Saves conversation history using a local SQLite database.
* Document Q&A: Allows uploading PDF or TXT files to search for relevant information.
* Tools: Integrates DuckDuckGo web search and a calculator.
* Observability: LangSmith support to trace LLM calls.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your Hugging Face API token:
```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

3. Run the application:
```bash
python run.py
```
