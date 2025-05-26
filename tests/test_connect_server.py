import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure environment variables exist during test collection so that the
# coverage plugin can import the package without errors.
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_KEY", "key")
os.environ.setdefault("OPENAI_API_KEY", "key")


@pytest.fixture()
def main_module(monkeypatch):
    # Provide dummy environment variables so the module imports cleanly
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    with patch("supabase.create_client") as mock_create_client, patch("openai.OpenAI") as mock_openai:
        supabase_client = MagicMock()
        mock_create_client.return_value = supabase_client
        mock_openai.return_value = MagicMock()
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        import connect_mcp_server.main as main
        importlib.reload(main)
        # Avoid hitting the OpenAI API when computing embeddings
        monkeypatch.setattr(main, "get_embedding", lambda text: [0.1] * 1536)
        yield main, supabase_client


def sample_profile(main):
    module, _ = main
    return module.ConnectProfile(
        name="Alice",
        age=30,
        relationship_goals="Long term connection",
        bio="Example bio",
        personality_tags=["empathetic", "introverted"],
        sexual_preferences=module.SexualPreferences(orientation="hetero", looking_for="female"),
        location="Honolulu",
    )


def test_create_profile(main_module):
    module, sb = main_module
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
    result = module.create_profile(sample_profile(main_module), "user1")
    assert result["status"] == "Profile created"
    sb.table.return_value.insert.return_value.execute.assert_called_once()


def test_update_profile(main_module):
    module, sb = main_module
    table = sb.table.return_value
    # Simulate existing record
    table.select.return_value.eq.return_value.execute.return_value.data = ["row"]
    table.update.return_value.eq.return_value.execute.return_value = MagicMock()
    result = module.update_profile(sample_profile(main_module), "user1")
    assert result["status"] == "Profile updated"
    table.update.return_value.eq.return_value.execute.assert_called_once()


def test_delete_profile(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = ["row"]
    table.delete.return_value.eq.return_value.execute.return_value = MagicMock()
    result = module.delete_profile("user1")
    assert result["status"] == "Profile deleted"
    table.delete.return_value.eq.return_value.execute.assert_called_once()


def test_find_matches(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = [
        {"embedded_vector": [0.1] * 1536}
    ]
    table.select.return_value.neq.return_value.execute.return_value.data = [
        {"user_id": "user2", "embedded_vector": [0.1] * 1536}
    ]
    matches = module.find_matches("user1", 1)
    assert matches[0]["user_id"] == "user2"


def test_upsert_profile_insert(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = []
    table.insert.return_value.execute.return_value = MagicMock()
    result = module.upsert_profile(sample_profile(main_module), "user1")
    assert result["status"] == "Profile created"
    table.insert.return_value.execute.assert_called_once()


def test_upsert_profile_update(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = ["row"]
    table.update.return_value.eq.return_value.execute.return_value = MagicMock()
    result = module.upsert_profile(sample_profile(main_module), "user1")
    assert result["status"] == "Profile updated"
    table.update.return_value.eq.return_value.execute.assert_called_once()


def test_explain_match(main_module):
    module, sb = main_module
    table = sb.table.return_value

    select_one = MagicMock()
    select_two = MagicMock()
    table.select.side_effect = [select_one, select_two]

    eq_one = MagicMock()
    eq_two = MagicMock()
    select_one.eq.return_value = eq_one
    select_two.eq.return_value = eq_two

    eq_one.execute.return_value.data = [
        {"embedded_vector": [0.1] * 1536, "personality_tags": ["kind", "fun"]}
    ]
    eq_two.execute.return_value.data = [
        {"embedded_vector": [0.1] * 1536, "personality_tags": ["fun", "smart"]}
    ]

    explanation = module.explain_match("user1", "user2")
    assert explanation["similarity"] == pytest.approx(1.0)
    assert explanation["shared_tags"] == ["fun"]
