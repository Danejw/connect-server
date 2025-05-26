# Connect Server Overview

This repository hosts the code for **Knolia Connect**, a small API server that matches users for relationships using AI embeddings. The server is built with FastMCP and FastAPI, stores data in Supabase, and relies on OpenAI for embeddings.

## Purpose
- Provide endpoints to create, update, or delete a user's connection profile.
- Embed profile data as a 1536‑dimension vector using OpenAI.
- Store profiles in Supabase (`user_connect_profiles` table).
- Find compatible matches by comparing vector similarity.

## Data
A connection profile contains:
- Name and age
- Relationship goals and short bio
- Personality tags
- Sexual preferences
- Location

Before saving to Supabase, the profile is serialized to text and embedded with OpenAI. Environment variables (`SUPABASE_URL`, `SUPABASE_KEY`, and `OPENAI_API_KEY`) are loaded from a `.env` file.

## Folder Structure
```
connect_mcp_server/
    main.py         FastAPI/FastMCP server and endpoints
    __init__.py

tests/
    test_connect_server.py  Pytest unit tests

Dockerfile            Build instructions for a Docker image
docker-compose.yml    Compose configuration
requirements.txt      Python package requirements
pyproject.toml        Minimal project metadata
setup.py              Package setup script
README.md             Project overview and usage
```

## Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Run the server locally: `uvicorn connect_mcp_server.main:app --reload`
3. Execute tests with `pytest`
