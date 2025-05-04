from fastmcp import FastMCP, Context
from pydantic import BaseModel
from typing import List, Dict
from supabase import create_client, Client
import openai
import os
import numpy as np

# Initialize FastMCP server
mcp = FastMCP("KnoliaConnectMCP")

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize Supabase and OpenAI
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY

# Pydantic model for user profile
class ConnectProfile(BaseModel):
    user_id: str
    relationship_goals: str
    personality_tags: List[str]
    sexual_preferences: Dict[str, str]
    location: str

# Helper function to serialize profile
def serialize_profile(profile: ConnectProfile) -> str:
    return f"""
    Relationship Goals:
    "{profile.relationship_goals}"

    Personality Tags:
    {', '.join(profile.personality_tags)}

    Sexual Preferences:
    Gender: {profile.sexual_preferences.get('gender')}
    Preference: {profile.sexual_preferences.get('preference')}

    Location:
    {profile.location}
    """

# Helper function to get embedding
def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    return response["data"][0]["embedding"]

# Tool to create a new profile
@mcp.tool()
def create_profile(profile: ConnectProfile, ctx: Context):
    serialized = serialize_profile(profile)
    embedding = get_embedding(serialized)
    data = profile.dict()
    data["embedded_vector"] = embedding
    supabase.table("user_connect_profiles").insert(data).execute()
    return {"status": "Profile created", "user_id": profile.user_id}

# Tool to update an existing profile
@mcp.tool()
def update_profile(profile: ConnectProfile, ctx: Context):
    serialized = serialize_profile(profile)
    embedding = get_embedding(serialized)
    data = profile.dict()
    data["embedded_vector"] = embedding
    supabase.table("user_connect_profiles").update(data).eq("user_id", profile.user_id).execute()
    return {"status": "Profile updated", "user_id": profile.user_id}

# Tool to delete a profile
@mcp.tool()
def delete_profile(user_id: str, ctx: Context):
    supabase.table("user_connect_profiles").delete().eq("user_id", user_id).execute()
    return {"status": "Profile deleted", "user_id": user_id}

# Tool to find matches
@mcp.tool()
def find_matches(user_id: str, top_k: int = 5, ctx: Context):
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
