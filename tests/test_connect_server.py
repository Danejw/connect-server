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
        sexual_preferences=module.SexualPreferences(
            gender="female",
            orientation="hetero", 
            looking_for="male"
        ),
        location="Honolulu",
    )


def test_serialize_profile(main_module):
    module, _ = main_module
    profile = sample_profile(main_module)
    result = module.serialize_profile(profile)
    assert "Alice" in result
    assert "30" in result
    assert "Long term connection" in result
    assert "empathetic, introverted" in result
    assert "hetero" in result
    assert "Honolulu" in result


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
        {"user_id": "user2", "name": "Bob", "age": 25, "embedded_vector": [0.1] * 1536}
    ]
    result = module.find_matches("user1", 1)
    assert result["profiles"][0]["user_id"] == "user2"
    assert result["count"] == 1
    assert result["reference_user_id"] == "user1"
    assert "similarity_score" in result["profiles"][0]


def test_get_profiles(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = [
        {"embedded_vector": [0.1] * 1536}
    ]
    table.select.return_value.neq.return_value.limit.return_value.execute.return_value.data = [
        {"user_id": "user2", "name": "Bob", "age": 25, "embedded_vector": [0.1] * 1536},
        {"user_id": "user3", "name": "Carol", "age": 28, "embedded_vector": [0.2] * 1536}
    ]
    result = module.get_profiles("user1", limit=5)
    assert result["count"] == 2
    assert result["reference_user_id"] == "user1"
    assert len(result["profiles"]) == 2
    assert "similarity_score" in result["profiles"][0]
    assert "embedded_vector" not in result["profiles"][0]


def test_get_profiles_filtered(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data - first call to select
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "age": 30, "embedded_vector": [0.1] * 1536}
    ]
    
    # Mock the filtered profiles query - second call to select
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.gte.return_value = query_mock
    query_mock.lte.return_value = query_mock  
    query_mock.ilike.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob", 
            "age": 25, 
            "location": "Honolulu",
            "personality_tags": ["fun", "outgoing"],
            "embedded_vector": [0.1] * 1536
        }
    ]
    
    # Set up the side_effect to return different mocks for each call
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        limit=5,
        age_min=20,
        age_max=35,
        location="Honolulu"
    )
    
    assert result["count"] == 1
    assert result["reference_user_id"] == "user1"
    assert len(result["profiles"]) == 1
    assert "similarity_score" in result["profiles"][0]
    assert "embedded_vector" not in result["profiles"][0]


def test_get_profiles_filtered_personality_tags(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "age": 30, "embedded_vector": [0.1] * 1536}
    ]
    
    # Mock the filtered profiles query
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob", 
            "age": 25,
            "personality_tags": ["fun", "outgoing"],
            "embedded_vector": [0.1] * 1536
        },
        {
            "user_id": "user3", 
            "name": "Carol", 
            "age": 28,
            "personality_tags": ["serious", "academic"],
            "embedded_vector": [0.2] * 1536
        }
    ]
    
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        personality_tags="fun,outgoing",
        personality_match_type="any"
    )
    
    assert result["count"] == 1  # Only Bob should match
    assert result["profiles"][0]["user_id"] == "user2"


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


# Add this test to debug what's available in the module
def test_debug_available_functions(main_module):
    module, _ = main_module
    available_functions = [attr for attr in dir(module) if not attr.startswith('_')]
    print("Available functions:", available_functions)
    
    # Check specifically for get_profiles_filtered
    has_get_profiles_filtered = hasattr(module, 'get_profiles_filtered')
    print("Has get_profiles_filtered:", has_get_profiles_filtered)
    
    # This test will always pass, it's just for debugging
    assert True


def test_get_profile(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = [
        {
            "user_id": "user1",
            "name": "Alice", 
            "age": 30,
            "relationship_goals": "Long term connection",
            "bio": "Example bio",
            "personality_tags": ["empathetic", "introverted"],
            "sexual_preferences": {
                "gender": "female",
                "orientation": "hetero", 
                "looking_for": "male"
            },
            "location": "Honolulu",
            "embedded_vector": [0.1] * 1536
        }
    ]
    result = module.get_profile("user1")
    assert result["user_id"] == "user1"
    assert result["name"] == "Alice"
    assert "embedded_vector" not in result  # Should be removed from response
    table.select.return_value.eq.return_value.execute.assert_called_once()


def test_get_profile_not_found(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = []
    result = module.get_profile("nonexistent_user")
    assert "error" in result
    assert result["error"] == "Profile not found."


def test_get_profiles_filtered_orientation_filter(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "embedded_vector": [0.1] * 1536}
    ]
    
    # Mock the filtered profiles query
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob",
            "sexual_preferences": {"orientation": "hetero", "looking_for": "female"},
            "embedded_vector": [0.1] * 1536
        },
        {
            "user_id": "user3", 
            "name": "Carol",
            "sexual_preferences": {"orientation": "homo", "looking_for": "female"},
            "embedded_vector": [0.2] * 1536
        }
    ]
    
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        orientation="hetero"
    )
    
    assert result["count"] == 1  # Only Bob should match
    assert result["profiles"][0]["user_id"] == "user2"


def test_get_profiles_filtered_looking_for_filter(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "embedded_vector": [0.1] * 1536}
    ]
    
    # Mock the filtered profiles query
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob",
            "sexual_preferences": {"orientation": "hetero", "looking_for": "female"},
            "embedded_vector": [0.1] * 1536
        },
        {
            "user_id": "user3", 
            "name": "Carol",
            "sexual_preferences": {"orientation": "hetero", "looking_for": "male"},
            "embedded_vector": [0.2] * 1536
        }
    ]
    
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        looking_for="female"
    )
    
    assert result["count"] == 1  # Only Bob should match
    assert result["profiles"][0]["user_id"] == "user2"


def test_get_profiles_filtered_personality_tags_all_match(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "embedded_vector": [0.1] * 1536}
    ]
    
    # Mock the filtered profiles query
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob",
            "personality_tags": ["fun", "outgoing", "adventurous"],
            "embedded_vector": [0.1] * 1536
        },
        {
            "user_id": "user3", 
            "name": "Carol",
            "personality_tags": ["fun", "serious"],
            "embedded_vector": [0.2] * 1536
        }
    ]
    
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        personality_tags="fun,outgoing",
        personality_match_type="all"
    )
    
    assert result["count"] == 1  # Only Bob should match (has both fun and outgoing)
    assert result["profiles"][0]["user_id"] == "user2"


def test_get_profiles_filtered_min_similarity_score(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "embedded_vector": [1.0] + [0.0] * 1535}  # Vector pointing in one direction
    ]
    
    # Mock the filtered profiles query
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob",
            "embedded_vector": [1.0] + [0.0] * 1535  # Same direction - high similarity (1.0)
        },
        {
            "user_id": "user3", 
            "name": "Carol",
            "embedded_vector": [0.0] + [1.0] + [0.0] * 1534  # Orthogonal direction - low similarity (0.0)
        }
    ]
    
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        min_similarity_score=0.5  # Threshold that should filter out Carol
    )
    
    assert result["count"] == 1  # Only Bob should pass the similarity threshold
    assert result["profiles"][0]["user_id"] == "user2"
    assert result["profiles"][0]["similarity_score"] >= 0.5


def test_get_profiles_filtered_sort_by_age(main_module):
    module, sb = main_module
    table = sb.table.return_value
    
    # Mock the reference user data
    select_mock1 = MagicMock()
    select_mock1.eq.return_value.execute.return_value.data = [
        {"user_id": "user1", "embedded_vector": [0.1] * 1536}
    ]
    
    # Mock the filtered profiles query
    select_mock2 = MagicMock()
    query_mock = MagicMock()
    select_mock2.neq.return_value = query_mock
    query_mock.limit.return_value.execute.return_value.data = [
        {
            "user_id": "user2", 
            "name": "Bob",
            "age": 25,
            "embedded_vector": [0.1] * 1536
        },
        {
            "user_id": "user3", 
            "name": "Carol",
            "age": 35,
            "embedded_vector": [0.2] * 1536
        }
    ]
    
    table.select.side_effect = [select_mock1, select_mock2]
    
    result = module.get_profiles_filtered(
        user_id="user1",
        sort_by="age",
        sort_order="asc"
    )
    
    assert result["count"] == 2
    assert result["profiles"][0]["age"] == 25  # Bob should be first (younger)
    assert result["profiles"][1]["age"] == 35  # Carol should be second (older)


def test_get_profiles_filtered_no_reference_user(main_module):
    module, sb = main_module
    table = sb.table.return_value
    table.select.return_value.eq.return_value.execute.return_value.data = []
    
    result = module.get_profiles_filtered("nonexistent_user")
    
    assert "error" in result
    assert result["error"] == "Reference user profile not found."
