# Intelligent Travel Planning AI Agent

[![Deployment](https://img.shields.io/badge/Deployed%20on-Google%20Cloud-blue?logo=googlecloud)](https://travel-agent-349302450067.asia-east1.run.app/)

This project is an advanced AI agent built to fulfill the "Intelligent Travel Planning AI Agent" assessment. It provides a conversational interface for planning personalized 2-day trips to any city in the world, deployed as a robust API and web UI on Google Cloud Run.

## 🚀 Demo & Testing

The agent is fully deployed and operational. You can interact with it in two ways:

* **Chat UI:** [**https://travel-agent-349302450067.asia-east1.run.app/**](https://travel-agent-349302450067.asia-east1.run.app/)
* **API Docs:** [**https://travel-agent-349302450067.asia-east1.run.app/docs**](https://travel-agent-349302450067.asia-east1.run.app/docs)

### How to Test (Recommended Flow)

1.  **Open the [Chat UI](https://travel-agent-349302450067.asia-east1.run.app/)**.
2.  **Test Long-Term Memory:**
    * **Send:** `Hi, please remember I am vegetarian and I love museums.`
    * The agent will use the `save_preference` tool to store this.
3.  **Test Planning & Tool Use:**
    * **Send:** `OK, now please plan a 3-day trip to Singapore for me.`
    * Observe as the agent:
        * Recalls your "vegetarian" and "museums" preferences.
        * Calls the `get_weather` tool for Paris.
        * Calls the `web_search` tool for museums.
        * Synthesizes this information into a personalized plan.
4.  **Test Proactive Suggestions & Multi-hop:**
    * The agent will proactively ask if you need help with flights or hotels.
    * **Send:** `Yes, find me some good hotels.`
    * The agent will call the `find_hotels` tool, which internally triggers another `web_search` for `best rated hotels in Paris`.

## ✅ System Structure

### 1. Tool Use & Decision-Making
* **3+ Tools Integrated:** The agent has access to **6** distinct tools defined in `tools.py`.
* **Dynamic Decision-Making:** The agent (Gemini 2.5 Flash) dynamically decides which tool to call.

### 2. Memory Management
* **Short-Term Memory:** Implemented via `ChatSession` objects in `main.py`.
* **Long-Term Memory (Vector Database):** Implemented using a **FAISS** database (`user_prefs.faiss`) and managed by `save_preference` / `load_preferences` tools.

### 3. Planning & Reasoning
* **Planning Mechanism (Custom Logic):** The core planner is the Gemini 2.5 Flash model, and the executor is the custom `while True:` tool-calling loop in `main.py`.
* **Multi-hop RAG:** The agent combines knowledge from `search_knowledge` (local RAG) and `web_search` (real-time facts) to generate synthesized plans.

### 4. Conversational Interface
* **Chat API (FastAPI Preferred):** The entire application is a **FastAPI** service.
* **Simple Web UI:** A clean HTML/JS UI is served from the root (`/`) route.
* **Clarify Ambiguous Inputs:** Handled natively by the model's reasoning when required tool arguments (e.g., `city`) are missing.
* **Maintain Context:** Guaranteed by the `ChatSession` (short-term memory).

### 5. Deployment
* The application is fully containerized (`Dockerfile`) and deployed on **Google Cloud Run**.

## 📁 Project File Structure

This project uses a modular structure to separate concerns:

```

travel-agent/
│
├── .venv/                  \# Python virtual environment (ignored)
│
├── data/                   \# Stores the persistent vector databases
│   ├── knowledge\_base.faiss  \# RAG DB for generic travel strategies
│   └── user\_prefs.faiss      \# Long-Term Memory DB for user preferences
│
├── main.py                 \# FastAPI server, API endpoints, and core agent loop
│
├── tools.py                \# All agent tools (Weather, Search, RAG, Memory)
│
├── build\_rag.py            \# One-time script to build the .faiss databases
│
├── requirements.txt        \# Python package dependencies
│
├── .env                    \# Local file for API keys (ignored)
│
├── .gitignore              \# Files to ignore for Git
│
└── Dockerfile              \# Container definition for GCloud deployment

```

## 🏗️ Project Architecture

This project is built as a self-contained FastAPI server, deployed on Google Cloud Run. All logic, tools, and data are packaged within a single Docker container.

```

[ User (Browser) ]
|
|  (1. HTTP Request to / or /chat)
v
\+-------------------------------------------+
| [ FastAPI Server (main.py) on Cloud Run ] |
|      |                                    |
|      +---- [ / (serves index.html UI) ]   |
|      |                                    |
|      +---- [ /docs (serves API Docs) ]    |
|      |                                    |
|      v                                    |
|  [ /chat (API Endpoint) ]                 |
|      |                                    |
|      v                                    |
|  [ Custom Logic (main.py) ]---------------+
|  (Tool-Calling Loop `while True:`)        |
|      |                                    |
| (Manages Short-Term Memory/ChatSessions)  |
|      |                                    |
|      v (2. send\_message(tools=...) )      |
|  [ LLM (Gemini 2.5 Flash) ] \<-------------+
|    (Planner)                              |  (5. send\_message(result=...) )
|      |                                    |
|      | (3. Returns FunctionCall)          |
|      v                                    |
|  [ Tool Registry (main.py) ]              |
|    (Executor)                             |
|      |                                    |
|      | (4. Calls specific tool)           |
|      v                                    |
|  [ Toolbox (tools.py) ] ------------------+
|      |  
|      +--\> [ get\_weather() ] ----\> (OpenWeather API)
|      |  
|      +--\> [ web\_search() ] -----\> (Google Search API)
|      |  
|      +--\> [ find\_hotels() ] ----|  
|      |      (uses web\_search)   |  
|      |                          v  
|      +--\> [ find\_flights() ] ---|  
|      |      (uses web\_search)   |  
|      |                          v  
|      +--\> [ search\_knowledge() ] -\> (FAISS RAG DB)
|      |  
|      +--\> [ load/save\_preference() ] -\> (FAISS Mem DB)
|  
\+-------------------------------------------+

````

## 🏛️ Architecture & Core Logic

This project is a FastAPI server, where all logic is self-contained.

1.  **`main.py` (The Server & Brain)**
    * Initializes the FastAPI app and the `genai.GenerativeModel` (using `gemini-2.5-flash`).
    * **`/` Endpoint:** Serves the `index.html` file as the main chat UI.
    * **`/chat` Endpoint:** This is the core of the agent. It:
        1.  Calls `get_chat_session` to retrieve or create a session.
        2.  Injects System Rules and Long-Term Memory into the `history`.
        3.  Calls `chat.send_message(message, tools=...)` **once** to send the user message and register the tools.
        4.  Enters the **Tool-Calling Loop** (`while True:`).
        5.  Checks the model's response. If it's a `FunctionCall`:
            * It looks up the function in the `tool_registry`.
            * It **injects** the `user_id` into memory tools.
            * It executes the tool (`tools.py`).
            * It sends the `{"function_response": ...}` **back** to the model (this time *without* the `tools` argument, to prevent the `multiple values` error).
        6.  If the response is text, the loop breaks and the text is returned to the user.

2.  **`tools.py` (The Tools)**
    * This file loads all API keys and (most importantly) the FAISS databases and embedding model into memory **on startup**. This ensures low-latency tool calls.
    * It defines all 6 Python functions, each with a clear docstring that the Gemini model reads to understand its capabilities.
    * The `find_hotels` and `find_flights` functions demonstrate an advanced "expert tool" pattern by re-using the `web_search` tool internally.

3.  **`data/` (The Knowledge)**
    * `knowledge_base.faiss`: The RAG database for generic planning tips.
    * `user_prefs.faiss`: The Long-Term Memory database for user-specific facts.
    * *These files are generated by running `build_rag.py` locally.*

## 🛠️ How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [your-repo-url]
    cd travel-agent
    ```

2.  **Install Dependencies**
    * Ensure you have Python 3.11+ installed.
    * (Optional but recommended: `python -m venv .venv` and activate it)
    * ```bash
        pip install -r requirements.txt
        ```

3.  **Set Up Environment (.env)**
    * Create a file named `.env` in the root of the project.
    * Add the following keys:
    ```
    GEMINI_API_KEY="AIzaSy..."
    OPENWEATHER_API_KEY="..."
    CUSTOM_SEARCH_API_KEY="AIzaSy..."
    CUSTOM_SEARCH_CX="..."
    ```

4.  **Build Vector Databases**
    * This only needs to be run once.
    * ```bash
        python build_rag.py
        ```
    * This will create the `data/knowledge_base.faiss` and `data/user_prefs.faiss` folders.

5.  **Run the Server**
    * ```bash
        python main.py
        ```
    * The server will start on `http://127.0.0.1:8080`.
    * You can test using the Chat UI (`http://127.0.0.1:8080/`) or the API docs (`http://127.0.0.1:8080/docs`).
