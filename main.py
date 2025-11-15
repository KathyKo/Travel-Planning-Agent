import os
import json
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

import google.generativeai as genai

# Import our custom tools
import tools 

# --- 1. Load Environment Variables & Configure API ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found in .env file. "
        "Please get your key from https://aistudio.google.com/app/apikey"
    )

genai.configure(api_key=GEMINI_API_KEY)


# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="Travel Planning AI Agent",
    description="AI agent for planning personalized trips."
)


# --- 3. Initialize Model and Tools ---

# This is a Python dictionary that maps the tool *name* (str)
# to the actual *function* (callable)
tool_registry = {
    tool.__name__: tool for tool in tools.available_tools
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash"
)


# --- 4. Manage Memory (Short-term & Long-term) ---

chat_sessions = {}

def get_chat_session(user_id: str) -> genai.ChatSession:
    """
    Retrieves or creates a chat session for a user.
    This is where Long-term Memory is loaded.
    """
    if user_id not in chat_sessions:
        print(f"[Memory] Creating new session for user: {user_id}")
        
        # Step 1. ALWAYS add the System Instruction first
        history = [
            {
                "role": "user", 
                "parts": [
                    "SYSTEM_RULE: You are a helpful and expert travel planning agent. "
                    "Your goal is to help the user plan a trip to any city. "
                    "First, understand their preferences (like 'vegetarian' or 'museums'). "
                    "Then, use your tools (web_search, get_weather) to build a plan. "
                    "After you have presented the plan, ALWAYS proactively ask the user if they would also like help finding hotels OR flights for their trip."
                    "CRITICAL RULE: You must *never* show your internal reasoning, thoughts, or the specific tools you are calling (e.g., 'Tool Call: web_search...'). "
                    "You must synthesize the information from your tools and present only the final, helpful answer directly to the user."
                ]
            },
            {
                "role": "model", 
                "parts": [
                    "Understood. I am a helpful travel agent. I will follow your rules."
                ]
            }
        ]
        
        # Step 2. NOW, try to load and *append* long-term memory
        try:
            prefs_json = tools.load_preferences(user_id=user_id)
            prefs = json.loads(prefs_json) # This is safe (tools.py guarantees JSON)
            
            if isinstance(prefs, list) and len(prefs) > 0:
                print(f"[Memory] Found {len(prefs)} preferences for user {user_id}.")
                # Append the preferences to the history
                history.append(
                    {"role": "user", "parts": [
                        f"Please remember these are my long-term preferences: {'; '.join(prefs)}"
                    ]}
                )
                history.append(
                    {"role": "model", "parts": [
                        "Understood. I have loaded your preferences."
                    ]}
                )
            else:
                 print(f"[Memory] No preferences found for user {user_id}.")
        except Exception as e:
            print(f"[Memory] Error loading preferences: {e}. Starting session without them.")
        
        # Step 3. Start the chat session
        chat_sessions[user_id] = model.start_chat(history=history)

    return chat_sessions[user_id]


# --- 5. Define API Endpoints ---

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    The main chat endpoint for the agent.
    """
    print(f"\n[Request] User: {request.user_id}, Message: {request.message}")
    
    chat = get_chat_session(request.user_id)
    
    try:
        response = chat.send_message(request.message)
    except Exception as e:
        print(f"[ERROR] chat.send_message failed: {e}")
        # Return the error to the user
        return {"user_id": request.user_id, "response": f"Sorry, I encountered an error: {e}"}

    
    # --- This is the CORE "Custom Logic" (Tool-Calling Loop) ---
    
    while True:
        if not response.candidates or not response.candidates[0].content.parts:
            print("[Response] No valid response from model.")
            break
            
        part = response.candidates[0].content.parts[0]

        if not part.function_call:
            break
        
        fc = part.function_call
        tool_name = fc.name
        tool_args = {key: value for key, value in fc.args.items()}
        
        print(f"[Tool Loop] Gemini wants to call: {tool_name}({tool_args})")
        
        if tool_name not in tool_registry:
            print(f"[Tool Loop] Error: Unknown tool '{tool_name}'")
            # --- FIX: Do NOT pass 'tools' here ---
            response = chat.send_message(
                {"function_response": {
                    "name": tool_name,
                    "response": {"error": f"Unknown tool: {tool_name}"}
                }}
            )
            continue 

        function_to_call = tool_registry[tool_name]
        
        if tool_name in ["save_preference", "load_preferences"]:
            tool_args["user_id"] = request.user_id
        
        try:
            tool_result = function_to_call(**tool_args)
            
            print(f"[Tool Loop] Tool '{tool_name}' returned: {tool_result}")
            
            # --- FIX: Do NOT pass 'tools' here ---
            response = chat.send_message(
                 {"function_response": {
                    "name": tool_name,
                    "response": {"result": tool_result}
                }}
            )
            
        except Exception as e:
            print(f"[Tool Loop] Error executing tool '{tool_name}': {e}")
            # --- FIX: Do NOT pass 'tools' here ---
            response = chat.send_message(
                 {"function_response": {
                    "name": tool_name,
                    "response": {"error": f"Error executing tool: {e}"}
                }}
            )

    if response.candidates and response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
        final_response = response.candidates[0].content.parts[0].text
        print(f"[Response] Agent: {final_response[:100]}...")
        return {"user_id": request.user_id, "response": final_response}
    else:
        print("[Response] Error: Agent did not return a final text response.")
        return {"user_id": request.user_id, "response": "Sorry, I encountered an error."}


@app.get("/")
async def read_root():
    """
    Serves the main HTML chat interface.
    """
    return FileResponse('index.html')


# --- 6. Run the Server ---
if __name__ == "__main__":
    print("Starting Travel Planning Agent server...")
    # Read the PORT from environment, default to 8080 if not set
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="127.0.0.1", port=port)
