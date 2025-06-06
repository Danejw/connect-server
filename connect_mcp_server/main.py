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
import json

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server, and FastAPI app, and router
mcp = FastMCP("KnoliaConnectMCP")
app = FastAPI(title="KnoliaConnectApp")
router = APIRouter(prefix="/connect", tags=["Connect"])
app.include_router(router=router)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize Supabase and OpenAI
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
client = OpenAI()


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
    """Get all profiles, calculate similarity scores, sort by similarity score, and return the top k profiles"""
    try:
        # Get the reference user's embedding
        user_data = supabase.table("user_connect_profiles").select("embedded_vector").eq("user_id", user_id).execute()
        if not user_data.data:
            return {"error": "Reference user profile not found."}
        
        # Convert string embedding to numpy array
        user_embedding = user_data.data[0]["embedded_vector"]
        if isinstance(user_embedding, str):
            user_vector = np.array(json.loads(user_embedding))
        else:
            user_vector = np.array(user_embedding)
        
        # Get random profiles (excluding the reference user)
        all_profiles = supabase.table("user_connect_profiles").select("*").neq("user_id", user_id).execute()
        if not all_profiles.data:
            return {"profiles": [], "count": 0}
                        
        # Calculate similarity scores and prepare response
        profiles = []
        for profile in all_profiles.data:
            profile_copy = profile.copy()
            
            # Convert string embedding to numpy array and calculate similarity score
            other_embedding = profile_copy["embedded_vector"]
            if isinstance(other_embedding, str):
                other_vector = np.array(json.loads(other_embedding))
            else:
                other_vector = np.array(other_embedding)
                
            similarity = np.dot(user_vector, other_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(other_vector))
            
            # Remove embedded_vector and add similarity score
            profile_copy.pop("embedded_vector", None)
            profile_copy["similarity_score"] = float(similarity)
            profiles.append(profile_copy)
            
            
            # Sort profiles by similarity score
        profiles.sort(key=lambda x: x["similarity_score"], reverse=True)
                            
        return {"profiles": profiles[:top_k], "count": len(profiles), "reference_user_id": user_id}
    except Exception as e:
        return {"error": str(e)}

# Tool to get a list of profiles
@mcp.tool()
@app.get("/get_profiles")
def get_profiles(user_id: str, limit: int = 10):
    """Get a random list of profiles with similarity scores relative to the given user_id"""
    try:
        # Get the reference user's embedding
        user_data = supabase.table("user_connect_profiles").select("embedded_vector").eq("user_id", user_id).execute()
        if not user_data.data:
            return {"error": "Reference user profile not found."}
        
        # Convert string embedding to numpy array
        user_embedding = user_data.data[0]["embedded_vector"]
        if isinstance(user_embedding, str):
            user_vector = np.array(json.loads(user_embedding))
        else:
            user_vector = np.array(user_embedding)
        
        # Get random profiles (excluding the reference user)
        profiles_data = supabase.table("user_connect_profiles").select("*").neq("user_id", user_id).limit(limit).execute()
        if not profiles_data.data:
            return {"profiles": [], "count": 0}
                
        # Calculate similarity scores and prepare response
        profiles = []
        for profile in profiles_data.data:
            profile_copy = profile.copy()
            
            # Convert string embedding to numpy array and calculate similarity score
            other_embedding = profile_copy["embedded_vector"]
            if isinstance(other_embedding, str):
                other_vector = np.array(json.loads(other_embedding))
            else:
                other_vector = np.array(other_embedding)
                
            similarity = np.dot(user_vector, other_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(other_vector))
            
            # Remove embedded_vector and add similarity score
            profile_copy.pop("embedded_vector", None)
            profile_copy["similarity_score"] = float(similarity)
            profiles.append(profile_copy)
                    
        return {"profiles": profiles, "count": len(profiles), "reference_user_id": user_id}
    except Exception as e:
        return {"error": str(e)}

# Run the MCP server
if __name__ == "__main__":
    mcp.run()

