import os
import requests
import json
from dotenv import load_dotenv

# --- LangChain/FAISS Imports for RAG & Memory ---
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document                
from langchain_huggingface import HuggingFaceEmbeddings      

# === 1. LOAD API KEYS & GLOBAL RESOURCES ===
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
CUSTOM_SEARCH_CX = os.getenv("CUSTOM_SEARCH_CX")

if not all([OPENWEATHER_API_KEY, CUSTOM_SEARCH_API_KEY, CUSTOM_SEARCH_CX]):
    raise EnvironmentError(
        "API Keys not fully configured in .env file. "
        "Please check OPENWEATHER_API_KEY, CUSTOM_SEARCH_API_KEY, and CUSTOM_SEARCH_CX."
    )

KNOWLEDGE_BASE_PATH = "data/knowledge_base.faiss"
USER_PREFS_PATH = "data/user_prefs.faiss"

print("Tools: Loading local embedding model (all-MiniLM-L6-v2)...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Tools: Local embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    raise

try:
    print(f"Tools: Loading Knowledge Base from {KNOWLEDGE_BASE_PATH}...")
    knowledge_db = FAISS.load_local(KNOWLEDGE_BASE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Tools: Knowledge Base loaded.")
    
    print(f"Tools: Loading User Preferences DB from {USER_PREFS_PATH}...")
    prefs_db = FAISS.load_local(USER_PREFS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Tools: User Preferences DB loaded.")
except Exception as e:
    print(f"Error loading FAISS databases: {e}")
    print("Did you run 'python build_rag.py' first?")
    raise


# === 2. DEFINE ALL AGENT TOOLS ===

def get_weather(city: str) -> str:
    """
    Gets the 2-day weather forecast for a specific city using the 
    5-day/3-hour forecast API.
    """
    print(f"[Tool Call] get_weather(city='{city}')")
    
    api_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "cnt": 16 
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        forecast_data = response.json()
        
        list_data = forecast_data.get("list", [])
        
        if not list_data:
            return f"Error: No forecast data found for city '{city}'."

        day1_forecast = list_data[7] 
        day2_forecast = list_data[15]

        simplified_forecasts = [
            {
                "day": "Tomorrow",
                "summary": day1_forecast.get("weather", [{}])[0].get("description"),
                "temp": day1_forecast.get("main", {}).get("temp"),
            },
            {
                "day": "The Day After Tomorrow",
                "summary": day2_forecast.get("weather", [{}])[0].get("description"),
                "temp": day2_forecast.get("main", {}).get("temp"),
            }
        ]
            
        return json.dumps(simplified_forecasts)

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 401:
            return "Error: Unauthorized. Please check your OPENWEATHER_API_KEY in the .env file."
        return f"Error connecting to Forecast API: {http_err}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def web_search(query: str) -> str:
    """
    Searches the web for real-time information, news, or specific facts 
    about a city, location, or topic.
    """
    print(f"[Tool Call] web_search(query='{query}')")
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": CUSTOM_SEARCH_API_KEY,
        "cx": CUSTOM_SEARCH_CX,
        "q": query,
        "num": 3 
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        search_data = response.json()
        
        results = search_data.get("items", [])
        
        if not results:
            return f"No web search results found for query: '{query}'"
            
        simplified_results = []
        for item in results:
            simplified_results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "source": item.get("link")
            })
            
        return json.dumps(simplified_results)

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Google Search API: {e}"


def find_hotels(city: str, preferences: str = "best rated") -> str:
    """
    Finds real-time hotel information for a given city by using a 
    specialized web search.
    You can optionally specify preferences like 'budget', 'luxury', or 'near city center'.
    """
    print(f"[Tool Call] find_hotels(city='{city}', preferences='{preferences}')")
    
    # 1. Create a specialized, high-quality search query
    query = f"{preferences} hotels in {city}"
    
    # 2. Call our *other* tool (web_search) internally.
    # This is just a regular Python function call, NOT an LLM tool call.
    print(f"[Tool Internal] Calling web_search with query: '{query}'")
    
    try:
        # We re-use the web_search logic
        search_results_json = web_search(query)
        
        # 3. Return the results from web_search
        # The format is already a JSON string of snippets, which is perfect.
        return search_results_json

    except Exception as e:
        # Return a JSON error
        return json.dumps({"error": f"Failed to search for hotels: {e}"})


def search_knowledge(query: str) -> str:
    """
    Searches the local knowledge base (RAG) for *generic* travel planning 
    strategies, templates, and advice.
    Use this to find out *how* to plan a trip, not *what* to see.
    """
    print(f"[Tool Call] search_knowledge(query='{query}')")
    
    try:
        docs = knowledge_db.similarity_search(query, k=2)
        
        if not docs:
            return "No relevant planning advice found in the knowledge base."
            
        simplified_docs = [
            {"content": doc.page_content, "source": doc.metadata.get("source")}
            for doc in docs
        ]
        return json.dumps(simplified_docs)
        
    except Exception as e:
        return f"Error searching knowledge base: {e}"

def load_preferences(user_id: str = None) -> str:
    """
    Loads ALL saved long-term preferences for a specific user_id.
    The user_id is handled automatically by the system. Do NOT ask the user for it.
    """
    print(f"[Tool Call] load_preferences(user_id='{user_id}')")
    
    if not user_id:
        return json.dumps({"error": "System error: user_id was not provided to tool."})
    
    try:
        docstore = prefs_db.docstore._dict
        
        if not docstore:
            return json.dumps([]) # Return empty JSON list

        user_prefs = []
        for doc_id, doc in docstore.items():
            if doc.metadata.get("user_id") == user_id:
                user_prefs.append(doc.page_content)
                
        if not user_prefs:
            return json.dumps([]) # Return empty JSON list
            
        return json.dumps(user_prefs)

    except Exception as e:
        print(f"[Tool Error] Error loading preferences: {e}")
        return json.dumps({"error": str(e)})

def save_preference(preference: str, user_id: str = None) -> str:
    """
    Saves a user's specific preference (e.g., 'I am vegetarian').
    The user_id is handled automatically by the system.
    You ONLY need to provide the 'preference' argument.
    """
    print(f"[Tool Call] save_preference(user_id='{user_id}', preference='{preference}')")
    
    if not user_id:
        return json.dumps({"error": "System error: user_id was not provided to tool."})
    
    try:
        new_doc = Document(
            page_content=preference,
            metadata={"user_id": user_id, "type": "preference"}
        )
        
        prefs_db.add_documents([new_doc])
        prefs_db.save_local(USER_PREFS_PATH)
        
        return f"Successfully saved preference: '{preference}'"
        
    except Exception as e:
        return f"Error saving preference: {e}"

# In tools.py ... (paste this before the available_tools list)

def find_flights(origin_city: str, destination_city: str, travel_date: str) -> str:
    """
    Finds potential flight options using a specialized web search.
    'travel_date' should be in YYYY-MM-DD format.
    This tool performs a web search, it does not book the flight.
    """
    print(f"[Tool Call] find_flights(from: '{origin_city}', to: '{destination_city}', on: '{travel_date}')")
    
    # 1. Create a specialized, high-quality search query
    query = (
        f"flights from {origin_city} to {destination_city} "
        f"on {travel_date}"
    )
    
    # 2. Call our *other* tool (web_search) internally.
    print(f"[Tool Internal] Calling web_search with query: '{query}'")
    
    try:
        # We re-use the web_search logic we already built
        search_results_json = web_search(query)
        
        # 3. Return the results from web_search
        return search_results_json

    except Exception as e:
        return json.dumps({"error": f"Failed to search for flights: {e}"})
    
# --- 3. CREATE THE FINAL LIST OF TOOLS ---
available_tools = [
    get_weather,
    web_search,
    find_hotels,
    search_knowledge,
    save_preference,
    load_preferences,
    find_flights
]