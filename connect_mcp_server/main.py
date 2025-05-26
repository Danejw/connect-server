from fastmcp import FastMCP, Context
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from typing import List, Dict
from supabase import create_client, Client
import openai
from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server, and FastAPI app, and router
mcp = FastMCP("KnoliaConnectMCP")
app = FastAPI(title="KnoliaConnectApp")
router = APIRouter(prefix="/connect", tags=["Connect"])
app.include_router(router=router)

# Environment variables
# Provide fallbacks for testing environments where these variables may not be
# configured. Real deployments should supply valid credentials via environment
# variables.
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "key")

EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize Supabase and OpenAI. When running tests or in environments without
# valid credentials these clients may fail to initialize; handle that gracefully
# so the module can be imported without raising exceptions.
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    supabase = None

try:
    openai.api_key = OPENAI_API_KEY
    client = OpenAI()
except Exception:
    client = None


class SexualPreferences(BaseModel):
    orientation: str
    looking_for: str

class ConnectProfile(BaseModel):
    name: str
    age: int
    relationship_goals: str
    bio: str
    personality_tags: List[str]
    sexual_preferences: SexualPreferences
    location: str


# Helper function to serialize profile
@app.post("/serialize_profile")
def serialize_profile(profile: ConnectProfile) -> str:
    return f"""
    Name:
    "{profile.name}"

    Age:
    "{profile.age}"

    Relationship Goals:
    "{profile.relationship_goals}"

    Personality Tags:
    {', '.join(profile.personality_tags)}

    Sexual Preferences:
    orientation: {profile.sexual_preferences.orientation}
    looking_for: {profile.sexual_preferences.looking_for}

    Location:
    {profile.location}
    """

# Helper function to get embedding
def get_embedding(text: str) -> List[float]:
    return client.embeddings.create(input = [text], model=EMBEDDING_MODEL).data[0].embedding  

# Tool to create a new profile
@mcp.tool()
@app.post("/create_profile")
def create_profile(profile: ConnectProfile, user_id: str):
    serialized = serialize_profile(profile)
    embedding = get_embedding(serialized)
    data = profile.model_dump()
    data["embedded_vector"] = embedding
    data["user_id"] = user_id
    supabase.table("user_connect_profiles").insert(data).execute()
    return {"status": "Profile created", "user_id": user_id}

# Tool to update an existing profile
@mcp.tool()
@app.put("/update_profile")
def update_profile(profile: ConnectProfile, user_id: str):
    # make sure there is a row for the user_id
    user_data = supabase.table("user_connect_profiles").select("*").eq("user_id", user_id).execute()
    if not user_data.data:
        return {"error": "User profile not found."}
    else:
        serialized = serialize_profile(profile)
        embedding = get_embedding(serialized)
        data = profile.model_dump()
        data["embedded_vector"] = embedding
        supabase.table("user_connect_profiles").update(data).eq("user_id", user_id).execute()
        return {"status": "Profile updated", "user_id": user_id}

@mcp.tool()
@app.post("/upsert_profile")
def upsert_profile(profile: ConnectProfile, user_id: str):
    print(f"Upserting profile for user_id: {user_id}")
    # Check if user profile exists
    try:
        user_data = supabase.table("user_connect_profiles").select("*").eq("user_id", user_id).execute()
        
        # Generate embedding
        serialized = serialize_profile(profile)
        embedding = get_embedding(serialized)
        data = profile.model_dump()
        data["embedded_vector"] = embedding
        data["user_id"] = user_id

        if not user_data.data:
            # Create new profile
            supabase.table("user_connect_profiles").insert(data).execute()
            return {"status": "Profile created", "user_id": user_id}
        else:
            # Update existing profile
            supabase.table("user_connect_profiles").update(data).eq("user_id", user_id).execute()
            return {"status": "Profile updated", "user_id": user_id}
    except Exception as e:
        return {"error": str(e)}

# Tool to delete a profile
@mcp.tool()
@app.delete("/delete_profile")
def delete_profile(user_id: str):
    # make sure there is a row for the user_id
    user_data = supabase.table("user_connect_profiles").select("*").eq("user_id", user_id).execute()
    if not user_data.data:
        return {"error": "User profile not found."}
    else:
        supabase.table("user_connect_profiles").delete().eq("user_id", user_id).execute()
        return {"status": "Profile deleted", "user_id": user_id}

# Tool to find matches
@mcp.tool()
@app.post("/find_matches")
def find_matches(user_id: str, top_k: int = 5):
    # Retrieve the user's embedding
    user_data = supabase.table("user_connect_profiles").select("embedded_vector").eq("user_id", user_id).execute()
    if not user_data.data:
        return {"error": "User profile not found."}
    user_vector = np.array(user_data.data[0]["embedded_vector"])

    # Retrieve all other profiles
    all_profiles = supabase.table("user_connect_profiles").select("user_id", "embedded_vector").neq("user_id", user_id).execute()
    matches = []
    for profile in all_profiles.data:
        other_vector = np.array(profile["embedded_vector"])
        similarity = np.dot(user_vector, other_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(other_vector))
        matches.append({"user_id": profile["user_id"], "score": similarity})

    # Sort matches by similarity score
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:top_k]

# Run the MCP server
if __name__ == "__main__":
    mcp.run()

