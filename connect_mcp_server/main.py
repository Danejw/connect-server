import logging
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
    gender: str
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
    try:
        return client.embeddings.create(input = [text], model=EMBEDDING_MODEL).data[0].embedding  
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

# Tool to create a new profile
@mcp.tool()
@app.post("/create_profile")
def create_profile(profile: ConnectProfile, user_id: str):
    try:
        serialized = serialize_profile(profile)
        embedding = get_embedding(serialized)    
        data = profile.model_dump()
        data["embedded_vector"] = embedding
        data["user_id"] = user_id
        supabase.table("user_connect_profiles").insert(data).execute()
        return {"status": "Profile created", "user_id": user_id}
    except Exception as e:
        logging.error(f"Error creating profile: {e}")
        return {"error": str(e)}

# Tool to update an existing profile
@mcp.tool()
@app.put("/update_profile")
def update_profile(profile: ConnectProfile, user_id: str):
    # make sure there is a row for the user_id
    try:
        user_data = supabase.table("user_connect_profiles").select("*").eq("user_id", user_id).execute()
        if not user_data.data:
            logging.warning(f"User profile not found for user_id: {user_id}")
            return {"error": "User profile not found."}
        else:
            serialized = serialize_profile(profile)
            embedding = get_embedding(serialized)
            data = profile.model_dump()
            data["embedded_vector"] = embedding
            supabase.table("user_connect_profiles").update(data).eq("user_id", user_id).execute()
            return {"status": "Profile updated", "user_id": user_id}
    except Exception as e:
        logging.error(f"Error updating profile: {e}")
        return {"error": str(e)}

@mcp.tool()
@app.post("/upsert_profile")
def upsert_profile(profile: ConnectProfile, user_id: str):
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
        logging.error(f"Error upserting profile: {e}")
        return {"error": str(e)}

# Tool to delete a profile
@mcp.tool()
@app.delete("/delete_profile")
def delete_profile(user_id: str):
    # make sure there is a row for the user_id
    user_data = supabase.table("user_connect_profiles").select("*").eq("user_id", user_id).execute()
    if not user_data.data:
        logging.error(f"User profile not found for user_id: {user_id}")
        return {"error": "User profile not found."}
    else:
        supabase.table("user_connect_profiles").delete().eq("user_id", user_id).execute()
        return {"status": "Profile deleted", "user_id": user_id}

# Tool to get a single profile
@mcp.tool()
@app.get("/get_profile")
def get_profile(user_id: str):
    """Get a profile by user_id"""
    try:
        profile = supabase.table("user_connect_profiles").select("*").eq("user_id", user_id).execute()
        if not profile.data:
            logging.warning(f"Profile not found for user_id: {user_id}")
            return {"error": "Profile not found."}
        else:
            profile_data = profile.data[0].copy()
            # Remove embedded_vector from response
            profile_data.pop("embedded_vector", None)
            return profile_data
    except Exception as e:
        logging.error(f"Error getting profile: {e}")
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
            logging.warning(f"Reference user profile not found for user_id: {user_id}")
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
        logging.error(f"Error getting profiles: {e}")
        return {"error": str(e)}

# Tool to get filtered profiles
@mcp.tool()
@app.get("/get_profiles_filtered")
def get_profiles_filtered(
    user_id: str,
    limit: int = 10,
    age_min: int = None,
    age_max: int = None,
    location: str = None,
    orientation: str = None,
    looking_for: str = None,
    personality_tags: str = None,  # Comma-separated string
    personality_match_type: str = "any",
    sort_by: str = "similarity",
    sort_order: str = "desc",
    min_similarity_score: float = None
):
    """Get filtered profiles with advanced filtering options"""
    try:
        # Get the reference user's embedding
        user_data = supabase.table("user_connect_profiles").select("embedded_vector").eq("user_id", user_id).execute()
        if not user_data.data:
            logging.warning(f"Reference user profile not found for user_id: {user_id}")
            return {"error": "Reference user profile not found."}
        
        # Convert string embedding to numpy array
        user_embedding = user_data.data[0]["embedded_vector"]
        if isinstance(user_embedding, str):
            user_vector = np.array(json.loads(user_embedding))
        else:
            user_vector = np.array(user_embedding)
        
        # Get all profiles first, then filter in Python to avoid JSON query issues
        query = supabase.table("user_connect_profiles").select("*").neq("user_id", user_id)
        
        # Apply basic filters that work with direct column queries
        if age_min is not None:
            query = query.gte("age", age_min)
        if age_max is not None:
            query = query.lte("age", age_max)
        if location:
            query = query.ilike("location", f"%{location}%")
        
        # Execute query with larger limit to account for post-filtering
        profiles_data = query.limit(limit * 3).execute()
        
        if not profiles_data.data:
            return {"profiles": [], "count": 0, "reference_user_id": user_id}
        
        # Filter profiles in Python to handle JSON fields and personality tags
        filtered_profiles = []
        for profile in profiles_data.data:
            # Filter by orientation
            if orientation and profile.get("sexual_preferences", {}).get("orientation") != orientation:
                continue
            
            # Filter by looking_for
            if looking_for and profile.get("sexual_preferences", {}).get("looking_for") != looking_for:
                continue
            
            # Filter by personality tags
            if personality_tags:
                tags_list = [tag.strip() for tag in personality_tags.split(",")]
                profile_tags = profile.get("personality_tags", [])
                
                if personality_match_type == "all":
                    # Must have all tags
                    if not all(tag in profile_tags for tag in tags_list):
                        continue
                else:
                    # Must have any of the tags
                    if not any(tag in profile_tags for tag in tags_list):
                        continue
            
            filtered_profiles.append(profile)
        
        # Calculate similarity scores and prepare response
        profiles = []
        for profile in filtered_profiles:
            profile_copy = profile.copy()
            
            # Convert string embedding to numpy array and calculate similarity score
            other_embedding = profile_copy["embedded_vector"]
            if isinstance(other_embedding, str):
                other_vector = np.array(json.loads(other_embedding))
            else:
                other_vector = np.array(other_embedding)
                
            similarity = np.dot(user_vector, other_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(other_vector))
            
            # Filter by minimum similarity score
            if min_similarity_score is not None and similarity < min_similarity_score:
                continue
            
            # Remove embedded_vector and add similarity score
            profile_copy.pop("embedded_vector", None)
            profile_copy["similarity_score"] = float(similarity)
            profiles.append(profile_copy)
        
        # Sort profiles
        if sort_by == "similarity":
            profiles.sort(key=lambda x: x["similarity_score"], reverse=(sort_order == "desc"))
        elif sort_by == "age":
            profiles.sort(key=lambda x: x["age"], reverse=(sort_order == "desc"))
        elif sort_by == "created_at":
            profiles.sort(key=lambda x: x.get("created_at", ""), reverse=(sort_order == "desc"))
        
        # Return only the requested limit
        return {"profiles": profiles[:limit], "count": len(profiles), "reference_user_id": user_id}
        
    except Exception as e:
        logging.error(f"Error getting filtered profiles: {e}")
        return {"error": str(e)}

# Tool to find matches
@mcp.tool()
@app.post("/find_matches")
def find_matches(user_id: str, top_k: int = 5):
    """Get all profiles, calculate similarity scores, sort by similarity score, and return the top k profiles"""
    try:
        # Get the reference user's embedding
        user_data = supabase.table("user_connect_profiles").select("embedded_vector").eq("user_id", user_id).execute()
        if not user_data.data:
            logging.warning(f"Reference user profile not found for user_id: {user_id}")
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
        logging.error(f"Error finding matches: {e}")
        return {"error": str(e)}


# Run the MCP server
if __name__ == "__main__":
    mcp.run()

