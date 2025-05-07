@mcp.tool()
@app.post("/upsert_profile")
def upsert_profile(profile: ConnectProfile, user_id: str):
    try:
        # Check if user profile exists
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
        import logging
        logging.error(f"Error in upsert_profile: {str(e)}")
        return {"status": "error", "message": f"Failed to upsert profile: {str(e)}", "user_id": user_id} 