import os
import faiss
from langchain_core.documents import Document                
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# --- 2. Define File Paths ---
KNOWLEDGE_BASE_PATH = "data/knowledge_base.faiss"
USER_PREFS_PATH = "data/user_prefs.faiss"

# --- 3. Define our *GENERIC* Knowledge Data ---
generic_travel_data = [
    {
        "content": """
        How to plan a personalized 2-day trip:
        1. Identify the user's main interests (e.g., food, anime, history, nature).
        2. Use web_search to find the Top 3-5 specific locations in the
           target city that match those interests.
        3. Group locations geographically. Dedicate Day 1 to one area
           (e.g., downtown, north side) and Day 2 to another to minimize travel time.
        4. Use get_weather to check the forecast. If Day 1 is rainy,
           plan indoor activities (museums, indoor markets) for that day.
        5. Use find_hotels to suggest accommodation near one of the key areas.
        """,
        "source": "general_planning_guide.txt"
    },
    {
        "content": """
        Travel Tips for Anime Lovers:
        When a user asks for an 'anime' trip to a new city:
        1. Use web_search for terms like "anime district [City Name]", 
           "manga cafe [City Name]", or "video game museum [City Name]".
        2. If the city is outside Japan, famous spots might be specific stores
           like "Manga Story" in Paris or "Kinokuniya" in various cities.
        3. Check for any special events or anime conventions happening 
           during the user's travel dates using web_search.
        """,
        "source": "anime_travel_tips.txt"
    },
    {
        "content": """
        Travel Tips for Foodies:
        When a user asks for a 'food' trip:
        1. Use web_search for "traditional food market [City Name]" or
           "famous local dish [City Name]".
        2. Suggest a mix of experiences: one high-end restaurant,
           one famous street food stall, and one local market.
        3. Use web_search for "best restaurants near [Landmark Name]" to
           combine sightseeing with good food.
        """,
        "source": "foodie_travel_tips.txt"
    }
]

def create_vector_store(documents, embeddings_model, file_path):
    """Creates and saves a FAISS vector store from text documents."""
    
    print(f"Starting vector store creation for {file_path}...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        all_chunks.extend(
            [{"text": chunk, "source": doc["source"]} for chunk in chunks]
        )
    
    if not all_chunks:
        print("No documents to process. Exiting.")
        return

    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")

    # Convert our chunks into LangChain 'Document' objects
    lc_documents = []
    for chunk in all_chunks:
        lc_documents.append(
            Document(page_content=chunk["text"], metadata={"source": chunk["source"]})
        )

    print("Creating FAISS index from documents (this may take a moment)...")
    
    vector_store = FAISS.from_documents(lc_documents, embeddings_model)
    
    # Save the FAISS index locally
    vector_store.save_local(file_path)
    
    print(f"Successfully created and saved vector store at: {file_path}")


def main():
    os.makedirs("data", exist_ok=True)
    
    # --- Initialize the *Local* Embeddings Model ---
    # 2. 替換為 HuggingFace (本地) 模型
    print("Loading local embedding model (all-MiniLM-L6-v2)...")
    print("This will download the model the first time you run it.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Local model loaded.")

    # --- 1. Create Knowledge Base (RAG) ---
    print("--- Building Knowledge Base (RAG) ---")
    knowledge_docs = [doc for doc in generic_travel_data]
    create_vector_store(knowledge_docs, embeddings, KNOWLEDGE_BASE_PATH)

    # --- 2. Create User Preferences Store (Memory) ---
    print("\n--- Building User Preferences Store (Memory) ---")
    user_pref_docs = [
        {
            "content": "This is a placeholder for user preferences.",
            "source": "system_init"
        }
    ]
    create_vector_store(user_pref_docs, embeddings, USER_PREFS_PATH)
    
    print("\nAll vector stores created successfully using the LOCAL model.")

if __name__ == "__main__":
    main()