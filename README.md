# knolia-connect-server

MCP server to connect users with compatible companions for romantic and emotional relationships using AI-powered vector matching.

## Overview

The Knolia Connect MCP server powers deep and context-aware matchmaking for users seeking meaningful relationships. Built using FastMCP and backed by Supabase with pgvector, it allows users to create rich connection profiles that are embedded using OpenAI's model and matched semantically to other users.

This service is part of Knolia's mission to help users combat loneliness and emotional disconnection by fostering intentional and high-quality relationships.

## Core Components

### 📌 Connect Profiles

Users who opt into Knolia Connect create a structured profile that captures their goals, personality, preferences, and values. This profile is embedded as a 1536-dimensional vector and stored for fast similarity-based matching.

#### User Connect Profile Example

```json
{
  "user_id": "521cb145-d40a-4e8b-92a3-7e3cf03f7c00",
  "relationship_goals": "Looking to build a deep emotional connection with someone who shares my values and growth mindset.",
  "personality_tags": ["introverted", "empathetic", "spiritual"],
  "sexual_preferences": {
    "gender": "female",
    "preference": "heterosexual"
  },
  "location": "Honolulu, HI",
  "embedded_vector": [/* 1536-dim embedding */],
  "opted_in": true,
  "created_at": "2025-05-03T09:30:00Z",
  "updated_at": "2025-05-03T09:30:00Z"
}
```

### 🧠 Embedding Format

Before storage, all profile data is transformed into natural language for better semantic understanding.

#### Embedded Connect Profile Example

```text
Relationship Goals:
Looking to build a deep emotional connection with someone who shares my values and growth mindset.

Personality Tags:
introverted, empathetic, spiritual

Sexual Preferences:
Gender: female
Orientation: heterosexual

Location:
Honolulu, HI
```

## Endpoints (via FastMCP tools)

* `create_profile`
* `update_profile`
* `delete_profile`
* `find_matches`

## Matching Logic

Matches are found by retrieving all other embedded connect profiles and calculating the cosine similarity between the requesting user and other users. Matches are filtered and sorted by similarity score to find the most aligned companions.

## Local Dev

```bash
pip install -r requirements.txt
python connect_mcp_server.main.py
```

## Docker

```bash
docker build -t connect-mcp .
docker run -p 8002:8000 --env-file .env connect-mcp
```

## Testing

Run Tests:

```bash
pytest
```


## Usage in Knolia

Noelle, Knolia's orchestrator, activates the Connect flow when a user opts into relationship discovery. Once opted in, their profile is embedded and compared against other matchable users. Noelle can then surface matches, initiate introductions, and facilitate conversation.

---

Built as part of the Knolia platform to bring meaningful connection to people ready for growth, companionship, and love.
