import pytest
from unittest.mock import patch, MagicMock
from connect_mcp_server.main import (
    create_profile,
    update_profile,
    delete_profile,
    find_matches,
    ConnectProfile,
    Context
)

@pytest.fixture
def mock_embed():
    try:
        with patch("connect_mcp_server.main.openai.Embedding.create") as mock_embed:
            mock_embed.return_value = {"data": [{"embedding": [0.1] * 1536}]}
            yield mock_embed
    except Exception as e:
        print(f"Error in mock_embed: {e}")
        raise e

@pytest.fixture
def mock_supabase():
    try:
        with patch("connect_mcp_server.main.supabase") as mock_sb:
            mock_sb.table.return_value.upsert.return_value.execute.return_value = MagicMock()
            mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
            mock_sb.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock()
            yield mock_sb
    except Exception as e:
        print(f"Error in mock_supabase: {e}")
        raise e

def sample_profile():
    try:
        return ConnectProfile(
            user_id="user1",
            relationship_goals="Long term connection",
            personality_tags=["empathetic", "introverted"],
            sexual_preferences={"gender": "female", "preference": "hetero"},
            location="Honolulu"
        )
    except Exception as e:
        print(f"Error in sample_profile: {e}")
        raise e

def test_create_profile(mock_embed, mock_supabase):
    try:
        result = create_profile(sample_profile(), Context())
        assert result["status"] == "Profile created"
    except Exception as e:
        print(f"Error in test_create_profile: {e}")
        raise e

def test_update_profile(mock_embed, mock_supabase):
    try:
        result = update_profile(sample_profile(), Context())
        assert result["status"] == "Profile updated"
    except Exception as e:
        print(f"Error in test_update_profile: {e}")
        raise e

def test_delete_profile(mock_supabase):
    try:
        result = delete_profile("user1", Context())
        assert result["status"] == "Profile deleted"
    except Exception as e:
        print(f"Error in test_delete_profile: {e}")
        raise e

def test_find_matches(mock_supabase):
    try:
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"embedded_vector": [0.1] * 1536}
        ]
        mock_supabase.table.return_value.select.return_value.neq.return_value.execute.return_value.data = [
            {"user_id": "user2", "embedded_vector": [0.1] * 1536}
        ]
        result = find_matches("user1", Context(), 1)
        assert isinstance(result, list)
        assert result[0]["user_id"] == "user2"
    except Exception as e:
        print(f"Error in test_find_matches: {e}")
        raise e
