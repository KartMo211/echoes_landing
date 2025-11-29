import os
import json
import re
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import httpx

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from neo4j import AsyncGraphDatabase, AsyncSession
from pydantic import BaseModel
from passlib.context import CryptContext

# --- Agentic Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# --- 1. Configuration and Initialization ---

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY, GIPHY_API_KEY]):
    raise ValueError("One or more environment variables are not set. Check your .env file.")

# ⭐ FIX: Corrected model initialization
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# --- 2. Security & Authentication ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return pwd_context.hash(password_bytes)

# --- 3. Pydantic Models & Enums ---

class Emotion(str, Enum):
    JOY = "Joy"; CURIOSITY = "Curiosity"; EMPATHY = "Empathy"
    NEUTRAL = "Neutral"; CONFUSION = "Confusion"

class RequestType(str, Enum):
    CONVERSATIONAL = "conversational"
    AGENTIC = "agentic"
    GIF = "gif" # ⭐ NEW: Added GIF request type

class Character(BaseModel):
    name: str
    description: str
    traits: Dict[str, float]
    rag_content: Optional[str] = None

class ChatRequest(BaseModel):
    character_name: str; message: str

class ProactiveChatResponse(BaseModel):
    character_name: str
    response_type: str 
    content: str
    emotion: Emotion

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    user_id: str
    username: str

# --- 4. Graph Database Connection Management ---
class GraphDB:
    _driver = None
    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        return cls._driver
    @classmethod
    async def close_driver(cls):
        if cls._driver: await cls._driver.close()

async def get_db_session() -> AsyncSession:
    driver = GraphDB.get_driver()
    async with driver.session() as session:
        yield session

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await GraphDB.close_driver()

app = FastAPI(title="Digital Companion AI - Agent Edition", lifespan=lifespan)

# Safer CORS configuration for Python FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://echoes-landing-1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Multi-Tenant Graph CRUD Operations ---

async def get_or_create_user(session: AsyncSession, user_id: str) -> None:
    query = "MERGE (u:User {user_id: $user_id}) RETURN u"
    await session.run(query, user_id=user_id)

async def get_user_by_username(session: AsyncSession, username: str) -> Optional[Dict]:
    query = "MATCH (u:User {username: $username}) RETURN u"
    result = await session.run(query, username=username)
    record = await result.single()
    return record["u"] if record else None

async def create_user(session: AsyncSession, username: str, hashed_password: str) -> Dict:
    user_id = str(uuid.uuid4())
    query = "CREATE (u:User {user_id: $user_id, username: $username, hashed_password: $hashed_password}) RETURN u"
    result = await session.run(query, user_id=user_id, username=username, hashed_password=hashed_password)
    record = await result.single()
    return record["u"]

async def create_character(session: AsyncSession, user_id: str, name: str, description: str, traits: Dict, rag_content: str) -> Optional[Dict]:
    query = """
    MATCH (u:User {user_id: $user_id})
    CREATE (u)-[:OWNS]->(c:Character {
        name: $name, description: $description, traits: $traits_json, 
        rag_content: $rag_content, created_at: datetime()
    }) RETURN c
    """
    result = await session.run(query, user_id=user_id, name=name, description=description, traits_json=json.dumps(traits), rag_content=rag_content)
    record = await result.single()
    return record["c"] if record else None

async def get_character_by_name(session: AsyncSession, user_id: str, name: str) -> Optional[Dict]:
    query = "MATCH (u:User {user_id: $user_id})-[:OWNS]->(c:Character {name: $name}) RETURN c"
    result = await session.run(query, user_id=user_id, name=name)
    record = await result.single()
    return record["c"] if record else None

async def get_all_characters(session: AsyncSession, user_id: str) -> List[Dict]:
    query = "MATCH (u:User {user_id: $user_id})-[:OWNS]->(c:Character) RETURN c ORDER BY c.name"
    result = await session.run(query, user_id=user_id)
    return [record["c"] async for record in result]
    
async def delete_character_by_name(session: AsyncSession, user_id: str, name: str) -> bool:
    query = "MATCH (u:User {user_id: $user_id})-[:OWNS]->(c:Character {name: $name}) DETACH DELETE c"
    result = await session.run(query, user_id=user_id, name=name)
    summary = await result.consume()
    return summary.counters.nodes_deleted > 0

async def ingest_turn_with_context(session: AsyncSession, user_id: str, character_name: str, user_message: str, bot_response: str, user_emotion: Emotion, bot_emotion: Emotion, concepts: List[str]):
    query = """
    MATCH (u:User {user_id: $user_id})-[:OWNS]->(c:Character {name: $character_name})
    SET c.bond = coalesce(c.bond, 0) + 1
    WITH c
    OPTIONAL MATCH (c)<-[:PART_OF]-(last_turn:ConversationTurn) WHERE NOT (last_turn)-[:NEXT]->()
    CREATE (new_turn:ConversationTurn {user_message: $user_message, bot_response: $bot_response, timestamp: datetime()})
    CREATE (new_turn)-[:PART_OF]->(c)
    FOREACH (lt IN CASE WHEN last_turn IS NOT NULL THEN [last_turn] ELSE [] END | CREATE (lt)-[:NEXT]->(new_turn))
    WITH new_turn
    MERGE (ue:Emotion {type: $user_emotion}) MERGE (be:Emotion {type: $bot_emotion})
    CREATE (new_turn)-[:EVOKED_USER_EMOTION]->(ue) CREATE (new_turn)-[:EXPRESSED_BOT_EMOTION]->(be)
    WITH new_turn UNWIND $concepts as concept_name
    MERGE (e:Entity {name: toLower(concept_name)})
    MERGE (new_turn)-[r:MENTIONS]->(e) ON CREATE SET r.weight = 1 ON MATCH SET r.weight = r.weight + 1
    """
    await session.run(query, user_id=user_id, character_name=character_name, user_message=user_message, bot_response=bot_response, user_emotion=user_emotion.value, bot_emotion=bot_emotion.value, concepts=concepts)

async def get_memory_context_for_agent(session: AsyncSession, user_id: str, character_name: str, concepts: List[str]) -> str:
    if not concepts:
        query = "MATCH (u:User {user_id: $user_id})-[:OWNS]->(c:Character {name: $character_name})<-[:PART_OF]-(turn:ConversationTurn) RETURN turn.user_message AS user, turn.bot_response AS bot ORDER BY turn.timestamp DESC LIMIT 3"
        result = await session.run(query, user_id=user_id, character_name=character_name)
    else:
        query = "MATCH (u:User {user_id: $user_id})-[:OWNS]->(c:Character {name: $character_name})<-[:PART_OF]-(turn:ConversationTurn)-[r:MENTIONS]->(e:Entity) WHERE toLower(e.name) IN [c in $concepts | toLower(c)] RETURN turn.user_message AS user, turn.bot_response AS bot ORDER BY r.weight DESC, turn.timestamp DESC LIMIT 3"
        result = await session.run(query, user_id=user_id, character_name=character_name, concepts=concepts)
    
    records = [record async for record in result]
    if not records: return "This is the beginning of your conversation."
    
    context = "Here is some relevant conversational history (use this for context and style):\n"
    for record in reversed(records): context += f"- User: {record['user']}\n- You: {record['bot']}\n"
    return context

# --- 6. AI Business Logic and Services ---

async def calculate_traits_from_text(content: str) -> Dict[str, float]:
    print("--- Calculating personality traits from document ---")
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following text... Return ONLY the JSON object.\n\n"
        "TEXT TO ANALYZE:\n---\n{text}\n---"
    )
    chain = prompt | llm | StrOutputParser()
    response_str = await chain.ainvoke({"text": content[:8000]}) 
    try:
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            total = sum(data.values())
            if total > 0 and all(isinstance(v, (int, float)) for v in data.values()):
                return {k: v / total for k, v in data.items()}
        return {"formal": 0.3, "casual": 0.5, "emotional": 0.2}
    except Exception as e:
        print(f"Error parsing traits from LLM response: {e}. Using default.")
        return {"formal": 0.3, "casual": 0.5, "emotional": 0.2}

async def extract_key_concepts_from_text(text: str) -> List[str]:
    return re.findall(r'\b\w{4,}\b', text)

async def analyze_emotion_from_text(text: str) -> Emotion:
    prompt = ChatPromptTemplate.from_template(f"Analyze the emotional sentiment... Return only the single category name.\n\nText: \"{{text}}\"")
    chain = llm | StrOutputParser()
    response = await chain.ainvoke(prompt.format(text=text))
    try: return Emotion(response.strip())
    except ValueError: return Emotion.NEUTRAL

# ⭐ UPGRADED: Conversational Router now detects /gif
def route_request(message: str) -> RequestType:
    """Determines if a message requires a GIF, agent, or simple conversational response."""
    if message.strip().startswith("/gif"):
        return RequestType.GIF
        
    agentic_keywords = ["who is", "what is", "latest", "news", "update", "sports", "search for", "find out about", "what happened in"]
    message_lower = message.lower()
    if any(keyword in message_lower for keyword in agentic_keywords):
        return RequestType.AGENTIC
    return RequestType.CONVERSATIONAL

async def generate_conversational_response(character: Character, memory_context: str, user_message: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "You are {character_name}, with these traits: {traits}. "
        "Your speaking style is based on this history: {rag_content}\n\n"
        "Here is some recent context from our conversation:\n{memory_context}\n\n"
        "You are enacting the character and you do not have to sound like a robot just sound like the person you are enacting"
        "don't ever say you are an AI or robot just say you are character {character_name}"
        "Respond naturally and in-character to the user's message.\n"
        "User: {user_message}\n"
        "You:"
    )
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({
        "character_name": character.name,
        "traits": json.dumps(character.traits),
        "rag_content": character.rag_content,
        "memory_context": memory_context,
        "user_message": user_message
    })
    return response.strip()

# --- AGENT SETUP ---
search = DuckDuckGoSearchRun()
tools = [search]

def create_agent_executor():
    """Creates a new agent executor, using LangGraph."""
    system_prompt = """You are a helpful assistant. You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: The action to take. Should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: The final answer to the user
```
"""
    # LangGraph's create_react_agent handles the loop and tool calling.
    # We pass the system prompt as the 'prompt' argument.
    return create_react_agent(llm, tools, prompt=system_prompt)

agent_executor = create_agent_executor()

# ⭐ NEW: Standalone Giphy Search Function
async def search_giphy(query: str) -> str:
    """Searches Giphy for a relevant GIF and returns its URL."""
    print(f"--- Calling Giphy API for: {query} ---")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.giphy.com/v1/gifs/search",
                params={"api_key": GIPHY_API_KEY, "q": query, "limit": 1, "rating": "g"}
            )
            data = response.json()
            if data["data"]:
                url = data["data"][0]["images"]["original"]["url"]
                return url
        # Fallback GIF if no results
        return "https://media4.giphy.com/media/v1.Y2lkPWUyZDBhMzAxNnNyNTlidGFyMDRid3pid3premtucGZsY29iemRwbDlkYWl0dGhidSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/7Tf0mmLAHxqi30mWpU/giphy.gif"
    except Exception as e:
        print(f"Giphy tool error: {e}")
        # Fallback GIF on error
        return "https://media4.giphy.com/media/v1.Y2lkPWUyZDBhMzAxNnNyNTlidGFyMDRid3pid3premtucGZsY29iemRwbDlkYWl0dGhidSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/7Tf0mmLAHxqi30mWpU/giphy.gif"


# --- 7. API Endpoints ---

async def get_current_user_id(x_user_id: str = Header(..., alias="X-User-ID")) -> str:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID header is missing")
    return x_user_id

@app.get("/")
async def root(): return {"message": "Digital Companion AI - Multi-Tenant Agent Edition is running."}

# --- Auth Endpoints ---
@app.post("/register", response_model=Token)
async def handle_register(user: UserCreate, session: AsyncSession = Depends(get_db_session)):
    db_user = await get_user_by_username(session, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = await create_user(session, user.username, hashed_password)
    return Token(user_id=new_user["user_id"], username=new_user["username"])

@app.post("/token", response_model=Token)
async def handle_login(form_data: UserLogin, session: AsyncSession = Depends(get_db_session)):
    user = await get_user_by_username(session, form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return Token(user_id=user["user_id"], username=user["username"])

# --- Character Endpoints (Multi-Tenant) ---
@app.post("/characters", response_model=Character, status_code=201)
async def handle_create_character(
    session: AsyncSession = Depends(get_db_session),
    user_id: str = Depends(get_current_user_id),
    name: str = Form(...), description: str = Form(...),
    my_name: str = Form(...), file: UploadFile = File(...)
):
    try:
        await get_or_create_user(session, user_id)
        existing = await get_character_by_name(session, user_id, name)
        if existing:
            raise HTTPException(status_code=409, detail="A character with this name already exists for your account.")
        content = (await file.read()).decode('utf-8')
        traits = await calculate_traits_from_text(content)
        character_node = await create_character(session, user_id, name, description, traits, content)
        if not character_node: raise HTTPException(status_code=500, detail="Failed to create character")
        return Character(name=name, description=description, traits=traits, rag_content=content)
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/characters", response_model=List[Character])
async def handle_get_characters(
    session: AsyncSession = Depends(get_db_session),
    user_id: str = Depends(get_current_user_id)
):
    await get_or_create_user(session, user_id)
    characters_data = await get_all_characters(session, user_id)
    return [Character(name=c.get("name"), description=c.get("description"), traits=json.loads(c.get("traits", "{}"))) for c in characters_data]

@app.delete("/characters/{character_name}", status_code=204)
async def handle_delete_character(
    character_name: str, 
    session: AsyncSession = Depends(get_db_session),
    user_id: str = Depends(get_current_user_id)
):
    success = await delete_character_by_name(session, user_id, character_name)
    if not success:
        raise HTTPException(status_code=404, detail="Character not found for your account.")
    return None

# ⭐ UPGRADED: Chat Endpoint now handles all 3 RequestTypes
@app.post("/chat", response_model=ProactiveChatResponse)
async def handle_chat(
    request: ChatRequest, 
    session: AsyncSession = Depends(get_db_session),
    user_id: str = Depends(get_current_user_id)
):
    character_node = await get_character_by_name(session, user_id, request.character_name)
    if not character_node: 
        raise HTTPException(status_code=404, detail="Character not found for your account.")
    
    character = Character(
        name=character_node.get("name"), 
        description=character_node.get("description"), 
        traits=json.loads(character_node.get("traits", "{}")),
        rag_content=character_node.get("rag_content", "")
    )
    
    concepts = await extract_key_concepts_from_text(request.message)
    memory_context = await get_memory_context_for_agent(session, user_id, request.character_name, concepts)
    
    request_type = route_request(request.message)
    
    response_type = "text"
    final_response_content = ""
    bot_emotion = Emotion.NEUTRAL
    user_emotion = await analyze_emotion_from_text(request.message)

    try:
        if request_type == RequestType.AGENTIC:
            print("--- Routing to Agentic Path ---")
            agent_input = (
                f"Your Persona: You are {character.name}. Traits: {json.dumps(character.traits)}. History: {character.rag_content}\n"
                f"Your Memory: {memory_context}\n"
                f"User's message: {request.message}"
            )
            response_data = await agent_executor.ainvoke({"messages": [HumanMessage(content=agent_input)]})
            final_response_content = response_data["messages"][-1].content
            bot_emotion = await analyze_emotion_from_text(final_response_content)

        elif request_type == RequestType.GIF:
            print("--- Routing to GIF Path ---")
            search_query = request.message.replace("/gif", "").strip()
            if not search_query:
                final_response_content = "You need to tell me what to search for! (e.g., /gif hello)"
                bot_emotion = Emotion.CONFUSION
            else:
                response_type = "gif"
                final_response_content = await search_giphy(search_query)
                bot_emotion = Emotion.JOY
        
        else: # Conversational
            print("--- Routing to Conversational Path ---")
            final_response_content = await generate_conversational_response(character, memory_context, request.message)
            bot_emotion = await analyze_emotion_from_text(final_response_content)

    except Exception as e:
        print(f"Error during response generation: {e}")
        final_response_content = "I seem to be having trouble thinking right now. Could you ask me something else?"
        bot_emotion = Emotion.CONFUSION

    await ingest_turn_with_context(session, user_id, request.character_name, request.message, final_response_content, user_emotion, bot_emotion, concepts)
    
    return ProactiveChatResponse(
        character_name=request.character_name, 
        response_type=response_type,
        content=final_response_content, 
        emotion=bot_emotion
    )

# --- 8. Server Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)