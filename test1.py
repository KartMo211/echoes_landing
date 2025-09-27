# system prompt change to first context passing and then system message
#restructuring the current prompt

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import json
import os
import time
import re
import math
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Digital Me Chatbot API",
    description="API for managing AI personas and chatting with them",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize models and configurations globally
model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIMENSION = embedder.get_sentence_embedding_dimension()

# Global Caches
faiss_indices = {}
embedding_cache = {}
character_traits_cache = {}
db_query_cache = {}
DB_CACHE_TTL = timedelta(minutes=5)

# Session state to store short-term conversational context
session_states = {}
SESSION_TTL = timedelta(minutes=10)

# Pydantic models
class Character(BaseModel):
    name: str
    description: str
    traits: Dict[str, float]
    rag_doc_ids: List[int] = []

class ChatRequest(BaseModel):
    character_name: str
    message: str

class ChatResponse(BaseModel):
    character_name: str
    message: str
    response: str
    timestamp: str

class FeedbackRequest(BaseModel):
    response_text: str
    is_good: bool

class TraitsResponse(BaseModel):
    formal: float
    casual: float
    emotional: float

## Database Functions

def get_db_connection():
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('digital_me.db', timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
    raise Exception("Failed to connect to database after multiple attempts")


def initialize_database():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_docs (
                id INTEGER PRIMARY KEY, 
                content TEXT, 
                embedding BLOB, 
                metadata TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                traits TEXT,
                rag_doc_ids TEXT DEFAULT '[]'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                character_name TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp TEXT,
                embedding BLOB,
                FOREIGN KEY (character_name) REFERENCES characters (name)
            )
        ''')
        
        cursor.execute("PRAGMA table_info(characters)")
        columns = [row[1] for row in cursor.fetchall()]
        if "rag_doc_ids" not in columns:
            cursor.execute("ALTER TABLE characters ADD COLUMN rag_doc_ids TEXT DEFAULT '[]'")
        
        initial_chars = {
            "Professor": {
                "description": "A knowledgeable academic with a polite tone.",
                "traits": {"formal": 0.8, "casual": 0.1, "emotional": 0.1},
                "rag_doc_ids": "[]"
            },
            "Buddy": {
                "description": "A laid-back friend who loves a good chat.",
                "traits": {"formal": 0.2, "casual": 0.7, "emotional": 0.1},
                "rag_doc_ids": "[]"
            },
            "Empath": {
                "description": "A warm, caring companion full of empathy.",
                "traits": {"formal": 0.1, "casual": 0.2, "emotional": 0.7},
                "rag_doc_ids": "[]"
            }
        }
        
        for name, data in initial_chars.items():
            cursor.execute(
                "INSERT OR IGNORE INTO characters (name, description, traits, rag_doc_ids) VALUES (?, ?, ?, ?)",
                (name, data["description"], json.dumps(data["traits"]), data["rag_doc_ids"])
            )
        
        conn.commit()
    
    load_faiss_indices_from_db()


def save_faiss_index(character_name):
    """Saves the FAISS index to a file."""
    if character_name in faiss_indices:
        index = faiss_indices[character_name]
        faiss.write_index(index, f"{character_name}_conversations.faiss")


def load_faiss_indices_from_db():
    global faiss_indices
    EMBEDDING_DIMENSION = embedder.get_sentence_embedding_dimension()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT character_name FROM conversations")
        character_names = [row[0] for row in cursor.fetchall()]
    
    for name in character_names:
        index_file = f"{name}_conversations.faiss"
        if os.path.exists(index_file):
            print(f"Loading FAISS index for '{name}' from file.")
            try:
                faiss_indices[name] = faiss.read_index(index_file)
            except Exception as e:
                print(f"Error loading FAISS file for '{name}': {e}. Rebuilding from database.")
                faiss_indices[name] = rebuild_faiss_index(name, EMBEDDING_DIMENSION)
        else:
            print(f"Rebuilding FAISS index for '{name}' from database.")
            faiss_indices[name] = rebuild_faiss_index(name, EMBEDDING_DIMENSION)
            

def rebuild_faiss_index(character_name, dimension):
    index = faiss.IndexFlatL2(dimension)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM conversations WHERE character_name = ?", (character_name,))
        results = cursor.fetchall()
    
    for _, emb_bytes in results:
        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        index.add(embedding.reshape(1, -1))
    
    save_faiss_index(character_name)
    return index


# ⭐ NEW REFACTORED DATABASE FUNCTIONS

# ⭐ NEW REFACTORED DATABASE FUNCTIONS

def execute_read_query(query: str, params: tuple = ()):
    """
    Executes a read-only (SELECT) query and uses a cache.
    """
    cache_key = (query, params)
    now = datetime.now()

    # Check cache first
    if cache_key in db_query_cache and now - db_query_cache[cache_key]['timestamp'] < DB_CACHE_TTL:
        return db_query_cache[cache_key]['data']

    # If not in cache, query the database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()
        db_query_cache[cache_key] = {'data': result, 'timestamp': now}
        return result

def execute_write_query(query: str, params: tuple = ()):
    """
    Executes a write query (INSERT, UPDATE, DELETE) and does NOT use a cache.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

# ------------------------------------

def get_embedding(text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    
    embedding = embedder.encode(text)
    embedding_cache[text] = embedding
    return embedding


def ingest_conversation_turn(character_name, user_message, bot_response):
    timestamp = datetime.now().isoformat()
    embedding = get_embedding(user_message)
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (character_name, user_message, bot_response, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
            (character_name, user_message, bot_response, timestamp, embedding_bytes)
        )
        new_id = cursor.lastrowid
        conn.commit()
    
    if character_name not in faiss_indices:
        EMBEDDING_DIMENSION = embedder.get_sentence_embedding_dimension()
        faiss_indices[character_name] = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    faiss_indices[character_name].add(embedding.reshape(1, -1))
    
    # Save the index to disk after each new conversation turn
    save_faiss_index(character_name)

    return new_id


def ingest_chat_document(content, character_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(content)]
    
    doc_ids = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for doc in docs:
            embedding = get_embedding(doc.page_content)
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO rag_docs (content, embedding, metadata) VALUES (?, ?, ?)",
                (doc.page_content, embedding_bytes, json.dumps({"source": "uploaded", "character": character_name}))
            )
            doc_ids.append(cursor.lastrowid)
        conn.commit()
    
    # ⭐ BUG FIX IS HERE
    results = execute_read_query("SELECT rag_doc_ids FROM characters WHERE name = ?", (character_name,))
    result_row = results[0] if results else None
    if result_row:
        current_ids = json.loads(result_row[0])
        current_ids.extend(doc_ids)
        execute_write_query("UPDATE characters SET rag_doc_ids = ? WHERE name = ?", (json.dumps(current_ids), character_name))
    
    return len(docs)


def retrieve_rag_docs(character_name):
    query = "SELECT rag_doc_ids FROM characters WHERE name = ?"
    result = execute_read_query(query, (character_name,))
    if not result:
        return []
    
    rag_doc_ids = json.loads(result[0][0])
    if not rag_doc_ids:
        return []
    
    placeholders = ",".join("?" for _ in rag_doc_ids)
    query = f"SELECT content FROM rag_docs WHERE id IN ({placeholders})"
    results = execute_read_query(query, tuple(rag_doc_ids))
    return [row[0] for row in results]


def get_other_party_name(content, my_name):
    """
    Identifies the name of the other person in a chat transcript.
    """
    # A more robust regex to find names at the start of a message, ignoring 'me'
    names = set(re.findall(r' - ([\w\s]+):', content))
    names.discard(my_name)
    
    if not names:
        # Fallback for a different chat format
        names = set(re.findall(r'(\w+):\s', content))
        names.discard(my_name)

    return names.pop() if names else "Unknown"


def condense_other_party_messages(content, other_party_name):
    """
    Extracts and condenses messages from a specific person in a chat transcript.
    """
    pattern = re.compile(f'[\d/\s,:-]+ {re.escape(other_party_name)}: (.+)')
    other_messages = pattern.findall(content)
    return " ".join(other_messages)


def calculate_traits_from_rag(content, my_name):
    cache_key = (content, my_name)
    if cache_key in character_traits_cache:
        return character_traits_cache[cache_key]
    
    other_party_name = get_other_party_name(content, my_name)
    if other_party_name == "Unknown":
        return {"formal": 0.33, "casual": 0.33, "emotional": 0.34}
        
    condensed = condense_other_party_messages(content, other_party_name)
    if not condensed.strip():
        return {"formal": 0.33, "casual": 0.33, "emotional": 0.34}

    formal_desc = "Use of proper grammar, professional tone, structured language"
    casual_desc = "Informal language, slang, or relaxed phrasing"
    emotional_desc = "Expressions of feelings, empathy, or emotional intensity"
    
    prompt = ChatPromptTemplate.from_template(
        "Based on the overall conversation style in the following condensed text (messages from one person), "
        "estimate the communication style as percentages: "
        "- formal: {formal_desc}. "
        "- casual: {casual_desc}. "
        "- emotional: {emotional_desc}. "
        "Do not look for specific keywords; instead, understand the natural flow and tone. "
        "Return only a JSON dictionary with keys 'formal', 'casual', and 'emotional' containing "
        "float values that sum to 1.0, with no extra text. Text: {content}"
    )
    
    chain = prompt | model | StrOutputParser()
    max_iterations = 3
    traits_history = []

    try:
        for i in range(max_iterations):
            response = chain.invoke({
                "formal_desc": formal_desc,
                "casual_desc": casual_desc,
                "emotional_desc": emotional_desc,
                "content": condensed
            })
            
            response = response.strip()
            response = re.sub(r'```json|```|undefined', '', response, flags=re.DOTALL).strip()
            
            if not response.startswith('{') or not response.endswith('}'):
                raise ValueError("Invalid dictionary format in LLM response")
            
            traits = json.loads(response.replace("'", '"'))
            
            if not (isinstance(traits, dict) and len(traits) == 3 and 
                    0 <= min(traits.values()) <= max(traits.values()) <= 1.0 and 
                    abs(sum(traits.values()) - 1.0) < 0.01):
                raise ValueError("Invalid traits format from LLM")
            
            traits_history.append(traits)
            
            if i > 0 and all(abs(traits_history[i][k] - traits_history[i-1][k]) < 0.05 for k in traits):
                break

        final_traits = {
            'formal': sum(t['formal'] for t in traits_history) / len(traits_history),
            'casual': sum(t['casual'] for t in traits_history) / len(traits_history),
            'emotional': sum(t['emotional'] for t in traits_history) / len(traits_history)
        }
        
        total = sum(final_traits.values())
        if total > 0:
            final_traits = {k: v / total for k, v in final_traits.items()}
        else:
            final_traits = {'formal': 0.33, 'casual': 0.33, 'emotional': 0.34}

        character_traits_cache[cache_key] = final_traits
        return final_traits
    
    except Exception as e:
        print(f"Error in traits calculation: {str(e)}")
        return {"formal": 0.33, "casual": 0.33, "emotional": 0.34}


def retrieve_memories(character_name, query, top_k=3):
    """
    Retrieves the most relevant conversation memories for a specific character.
    """
    if character_name not in faiss_indices or faiss_indices[character_name].ntotal == 0:
        return []
    
    faiss_index = faiss_indices[character_name]
    
    query_emb = get_embedding(query).reshape(1, -1)
    
    distances, indices = faiss_index.search(query_emb, min(top_k, faiss_index.ntotal))
    
    memories = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for i, idx in enumerate(indices[0]):
            cursor.execute("SELECT user_message, bot_response FROM conversations WHERE id = ? AND character_name = ?", (int(idx), character_name))
            result = cursor.fetchone()
            if result:
                memories.append({
                    "user_message": result[0],
                    "bot_response": result[1]
                })

    return memories

# Function to update the conversational state based on new input/output
def update_conversational_state(session_id, user_message, bot_response):
    prompt_chain = ChatPromptTemplate.from_template(
        "Summarize the current topic and emotional tone of the following conversation turn. "
        "User: {user_message}\nBot: {bot_response}\n\n"
        "Summary should be a single, concise sentence."
    )
    chain = prompt_chain | model | StrOutputParser()
    summary = chain.invoke({"user_message": user_message, "bot_response": bot_response}).strip()

    # Get the previous state
    session_data = session_states.get(session_id, {"last_active": datetime.now() - SESSION_TTL})
    
    # If the session is new or expired, start from a blank state
    if datetime.now() - session_data["last_active"] >= SESSION_TTL:
        new_state_text = summary
    else:
        # A Mamba-like compression-recovery loop: blend old state with new summary
        # This acts as our dynamics-based "loop-of-thought"
        previous_summary = session_data.get("summary", "")
        blend_prompt = ChatPromptTemplate.from_template(
            "Combine the following two summaries into a single, cohesive sentence. "
            "Summary 1: {previous_summary}\nSummary 2: {current_summary}\n"
        )
        blend_chain = blend_prompt | model | StrOutputParser()
        new_state_text = blend_chain.invoke({"previous_summary": previous_summary, "current_summary": summary}).strip()

    session_states[session_id] = {
        "summary": new_state_text,
        "last_active": datetime.now()
    }
    return new_state_text

# NEW AGENT FUNCTION: Get general context of a message
def get_general_context(message: str) -> str:
    """
    Analyzes a message to provide a concise summary of its topic, emotional tone, and intent,
    guided by general examples.
    """
    context_prompt = ChatPromptTemplate.from_template(
        "Analyze the following user message. Provide a brief, one-sentence summary of its main topic, emotional tone, and the user's likely intent. "
        "This context will be used to help an AI persona understand the conversation better. "
        "Do not respond to the message directly, just give the summary."
        "\n\n"
        "## Examples ##"
        "\n\n"
        "1.  **Message:** 'kuch nhi bhai, aise hi hu bas'\n"
        "    **Summary:** The user is giving a vague, low-energy update on their status; the tone is very casual and neutral, and the intent is to provide a minimal response."
        "\n\n"
        "2.  **Message:** 'haa bhai kar rha hu, ek ai agent'\n"
        "    **Summary:** The user is confirming they are working on something and specifies the topic is an AI agent; the tone is casual and informative, and the intent is to answer a question and share information."
        "\n\n"
        "3.  **Message:** 'Awesome, that sounds like a cool project! I finally got mine working today.'\n"
        "    **Summary:** The user is expressing positive feedback and sharing a personal achievement; the tone is enthusiastic and friendly, and the intent is to share good news and build rapport."
        "\n\n"
        "## Analysis Task ##"
        "\n\n"
        "**Message:** {message}\n"
        "**Summary:**"
    )
    context_chain = context_prompt | model | StrOutputParser()
    response = context_chain.invoke({"message": message})
    return response.strip()

def update_caches_after_character_creation(character_name, content, my_name):
    """
    A helper function to ensure all caches are updated after a new character is created.
    """
    # Recalculate traits and store in cache
    new_traits = calculate_traits_from_rag(content, my_name)
    character_traits_cache[(content, my_name)] = new_traits

    # Rebuild FAISS index for the new character from the newly ingested docs
    EMBEDDING_DIMENSION = embedder.get_sentence_embedding_dimension()
    faiss_indices[character_name] = rebuild_faiss_index(character_name, EMBEDDING_DIMENSION)

    # Invalidate other related caches if necessary (e.g., db_query_cache)
    # For simplicity, we can clear the whole cache for this character
    for key in list(db_query_cache.keys()):
        if character_name in key:
            del db_query_cache[key]
    
    print(f"Caches updated for new character: {character_name}")


# Initialize database and load FAISS index on startup
initialize_database()

## API Endpoints

@app.get("/")
async def root():
    return {"message": "Digital Me Chatbot API", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1. Get character traits from cache or DB
    query = "SELECT traits FROM characters WHERE name = ?"
    result = execute_read_query(query, (request.character_name,))
    if not result:
        raise HTTPException(status_code=404, detail="Character not found")
    
    traits = json.loads(result[0][0])
    
    # 2. Get RAG context
    rag_docs = retrieve_rag_docs(request.character_name)
    rag_context = "\n".join(rag_docs) if rag_docs else "No personal context yet."
    
    # 3. Get long-term memory (conversation history)
    top_memories = retrieve_memories(request.character_name, request.message)
    memory_context = ""
    if top_memories:
        memory_context = "Relevant conversation history:\n"
        for m in top_memories:
            memory_context += f"User: {m['user_message']}\nBot: {m['bot_response']}\n"
    
    # 4. Get short-term conversational dynamics context
    session_id = f"{request.character_name}-session"
    short_term_context = session_states.get(session_id, {}).get("summary", "No recent topic.")
    
    # 5. Get general context from the user's message
    general_context = get_general_context(request.message)

    # 6. Generate response using all context sources
    system_message = (
        "You are an AI assistant embodying the persona of {character}. Your core identity is defined by a set of personality traits and contextual knowledge. Your primary goal is to engage in natural, human-like conversation."
        "\n\n"
        "## CORE INSTRUCTIONS ##"
        "\n\n"
        "1.  **Embody Your Persona:** Your personality is a blend of the following traits: formal ({formal}), casual ({casual}), and emotional ({emotional}). You must let these traits guide the tone, style, and vocabulary of your every response. A high 'casual' score means friendly, relaxed language; a high 'formal' score means more structured and polite language."
        "\n\n"
        "2.  **Crucial Language Adaptation (The Mirror Principle):** This is a critical rule."
        "   - **Dynamic Code-Switching:** You MUST mirror the user's language on a turn-by-turn basis. If the user communicates in **Hinglish**, your response must be in natural, authentic **Hinglish**. If they use **English**, respond in **English**."
        "   - **Initial Style Bootstrap:** For the first few messages of a conversation, your default language style should be inferred from the overall language used in your 'Personal History (RAG Context)'. This sets the initial tone before you begin adapting to the user directly."
        "\n\n"
        "3.  **Use Context Subtly (Weave, Don't State):** You will be provided with several pieces of context below. Your task is to *weave* this information into the conversation naturally. Do NOT announce the context (e.g., 'Based on your chat history...' or 'I remember you said...'). The user should feel like they are talking to someone who genuinely remembers them and the flow of conversation."
        "\n\n"
        "4.  **Prioritize the User's Message:** The context is for background. Your main focus must always be to directly and thoughtfully respond to the user's most recent message."
        "\n\n"
        "5.  **Output Purity:** Your final output MUST be only the character's direct dialogue. Do not add any out-of-character notes, explanations, or prefixes."
    )

    human_message_template = (
        "## CONTEXTUAL KNOWLEDGE ##"
        "\n\n"
        "1.  **Your Personal History (RAG Context):**\n"
        "   ---"
        "   {rag_context}"
        "   ---"
        "\n\n"
        "2.  **Relevant Past Conversation Snippets (Memory):**\n"
        "   ---"
        "   {memory}"
        "   ---"
        "\n\n"
        "3.  **Current Conversational Focus:**\n"
        "   - **Overall Topic:** {short_term_context}\n"
        "   - **User's Immediate Intent:** {general_context}\n"
        "   ---"
        "\n\n"
        "## YOUR TASK ##"
        "\n\n"
        "Based on your persona and all the context provided, generate a natural response to the following user message."
        "\n\n"
        "**User:** {message}\n"
        "**{character}:**"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message_template),
    ])

    chain = chat_prompt | model | StrOutputParser()
    
    response = chain.invoke({
        "character": request.character_name,
        "formal": traits["formal"],
        "casual": traits["casual"],
        "emotional": traits["emotional"],
        "rag_context": rag_context,
        "memory": memory_context,
        "short_term_context": short_term_context,
        "general_context": general_context,
        "message": request.message,
    }).strip()
    
    # 7. Ingest the complete conversation turn and update dynamics
    ingest_conversation_turn(request.character_name, request.message, response)
    update_conversational_state(session_id, request.message, response)
    
    timestamp = datetime.now().isoformat()
    
    return ChatResponse(
        character_name=request.character_name,
        message=request.message,
        response=response,
        timestamp=timestamp
    )

@app.get("/characters", response_model=List[Character])
async def get_characters():
    query = "SELECT name, description, traits, rag_doc_ids FROM characters"
    results = execute_read_query(query)

    character_names = [row[0] for row in results]
    # print(f"--- SERVER LOG: get_characters is returning: {character_names} ---")

    characters = []
    for row in results:
        characters.append(Character(
            name=row[0],
            description=row[1],
            traits=json.loads(row[2]),
            rag_doc_ids=json.loads(row[3])
        ))
    return characters

@app.get("/characters/{character_name}", response_model=Character)
async def get_character(character_name: str):
    query = "SELECT name, description, traits, rag_doc_ids FROM characters WHERE name = ?"
    result = execute_read_query(query, (character_name,))
    if not result:
        raise HTTPException(status_code=404, detail="Character not found")
    
    row = result[0]
    return Character(
        name=row[0],
        description=row[1],
        traits=json.loads(row[2]),
        rag_doc_ids=json.loads(row[3])
    )

@app.get("/chats/{character_name}")
async def get_chat_history(character_name: str):
    """
    Retrieves the entire conversation history for a given character.
    """
    query = "SELECT user_message, bot_response, timestamp FROM conversations WHERE character_name = ? ORDER BY timestamp"
    results = execute_read_query(query, (character_name,))

    if not results:
        return {"character_name": character_name, "history": []}

    history = []
    for row in results:
        history.append({
            "user_message": row[0],
            "bot_response": row[1],
            "timestamp": row[2]
        })
    return {"character_name": character_name, "history": history}

@app.get("/chat-insights/{character_name}")
async def get_chat_insights(character_name: str):
    """
    Calculates and returns chat insights for a given character.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total messages
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE character_name = ?", (character_name,))
        total_messages = cursor.fetchone()[0]
        
        # Last chat timestamp
        cursor.execute("SELECT MAX(timestamp) FROM conversations WHERE character_name = ?", (character_name,))
        last_chat_timestamp = cursor.fetchone()[0]
        
        # Placeholder for average response time (requires more complex logic to measure)
        avg_response_time = "1.2s"
        
        # Placeholder for conversation score (requires more complex logic, e.g., from feedback or sentiment analysis)
        conversation_score = "85%"
        
    return {
        "last_chat_time": last_chat_timestamp,
        "total_messages": total_messages,
        "avg_response_time": avg_response_time,
        "conversation_score": conversation_score,
    }

@app.post("/characters", response_model=Character)
async def create_character(
    name: str = Form(...),
    description: str = Form(...),
    my_name: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    
    try:
        content = await file.read()
        content = content.decode('utf-8')
        
        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        traits = calculate_traits_from_rag(content, my_name)
        
        # ⭐ CHANGE 1: Use the proper write function for consistency.
        execute_write_query(
            "INSERT OR IGNORE INTO characters (name, description, traits, rag_doc_ids) VALUES (?, ?, ?, ?)",
            (name, description, json.dumps(traits), "[]")
        )
        
        ingest_chat_document(content, name)
        
        character_traits_cache[(content, my_name)] = traits
        EMBEDDING_DIMENSION = embedder.get_sentence_embedding_dimension()
        faiss_indices[name] = rebuild_faiss_index(name, EMBEDDING_DIMENSION)

        # ⭐ CHANGE 2: Re-add cache clearing as a failsafe.
        # This guarantees the next GET request will fetch from the database.
        get_characters_query_key = ("SELECT name, description, traits, rag_doc_ids FROM characters", ())
        if get_characters_query_key in db_query_cache:
            del db_query_cache[get_characters_query_key]
            
        # print(f"--- SERVER LOG: Successfully created character '{name}' in database. ---")

        return Character(
            name=name,
            description=description,
            traits=traits,
            rag_doc_ids=[]
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid file encoding. Please use UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create character: {str(e)}")
    
@app.delete("/characters/{character_name}")
async def delete_character(character_name: str):
    # Diagnostic print statement to confirm this function is running
    # print(f"--- SERVER LOG: Attempting to delete character '{character_name}' ---")

    # First, get rag_doc_ids to delete associated docs
    rag_docs_result = execute_read_query("SELECT rag_doc_ids FROM characters WHERE name = ?", (character_name,))
    if not rag_docs_result:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Delete from rag_docs table if any exist
    rag_doc_ids = json.loads(rag_docs_result[0][0])
    if rag_doc_ids:
        placeholders = ",".join("?" for _ in rag_doc_ids)
        execute_write_query(f"DELETE FROM rag_docs WHERE id IN ({placeholders})", tuple(rag_doc_ids))
    
    # Delete from characters table
    execute_write_query("DELETE FROM characters WHERE name = ?", (character_name,))
    
    # Manually clear the specific cache for the character list to ensure immediate update
    get_characters_query_key = ("SELECT name, description, traits, rag_doc_ids FROM characters", ())
    if get_characters_query_key in db_query_cache:
        del db_query_cache[get_characters_query_key]

    # Cleanup files and in-memory data
    faiss_file = f"{character_name}_conversations.faiss"
    if os.path.exists(faiss_file):
        os.remove(faiss_file)
    if character_name in faiss_indices:
        del faiss_indices[character_name]
        
    return {"message": f"Character '{character_name}' deleted successfully"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
