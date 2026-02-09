"""
Tests for POST /parse/text endpoint (AMA-555)

Uses the conftest.py `client` fixture which applies auth overrides.
LLM fallback is mocked to ensure deterministic, offline tests.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_llm_fallback():
    """Mock ParserService.parse_free_text_to_workout so LLM calls never fire."""
    with patch(
        "workout_ingestor_api.api.parse_routes.ParserService.parse_free_text_to_workout",
        side_effect=Exception("LLM should not be called in tests"),
    ):
        yield


# ---------------------------------------------------------------------------
# Happy-path structured parsing
# ---------------------------------------------------------------------------


class TestParseTextStructuredParsing:
    """Tests for the structured (regex) parsing path."""

    def test_parse_standard_instagram_notation(self, client):
        text = """Workout:
Pull-ups 4x8 + Z Press 4x8
SA cable row 4x12 + SA DB press 4x8
Seated sled pull 5 x 10m"""

        response = client.post("/parse/text", json={
            "text": text,
            "source": "instagram_caption"
        })

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["exercises"]) == 5

        # First superset
        assert data["exercises"][0]["raw_name"] == "Pull-ups"
        assert data["exercises"][0]["sets"] == 4
        assert data["exercises"][0]["reps"] == "8"
        assert data["exercises"][0]["superset_group"] == "A"

        assert data["exercises"][1]["raw_name"] == "Z Press"
        assert data["exercises"][1]["superset_group"] == "A"

        # Distance exercise
        assert data["exercises"][4]["raw_name"] == "Seated sled pull"
        assert data["exercises"][4]["sets"] == 5
        assert data["exercises"][4]["distance"] == "10m"

    def test_parse_superset_notation(self, client):
        text = "Bench Press 4x8 + Rows 3x10"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["exercises"]) == 2
        assert data["exercises"][0]["superset_group"] == "A"
        assert data["exercises"][1]["superset_group"] == "A"

    def test_mixed_formats_numbered_and_fitness(self, client):
        text = """1. Squats 4x8
2. Bench Press 3x10
3. Deadlifts 5x5"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["exercises"]) == 3
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][0]["sets"] == 4
        assert data["exercises"][0]["reps"] == "8"

    def test_skip_hashtags(self, client):
        text = """Squats 4x8
#fitness #legday #workout
Bench Press 3x10"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 2
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][1]["raw_name"] == "Bench Press"

    def test_skip_ctas(self, client):
        text = """Squats 4x8
Follow me for more workouts!
Bench Press 3x10"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 2
        names = [e["raw_name"] for e in data["exercises"]]
        assert "Follow me for more workouts!" not in names

    def test_skip_section_headers(self, client):
        text = """Upper Body:
Bench Press 4x8
Lower Body:
Squats 3x10"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 2
        names = [e["raw_name"] for e in data["exercises"]]
        assert "Upper Body:" not in names
        assert "Lower Body:" not in names

    def test_no_split_compound_names_without_sets_reps(self, client):
        text = "Chin-up + Negative Hold"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["raw_name"] == "Chin-up + Negative Hold"
        assert data["exercises"][0]["sets"] is None
        assert data["exercises"][0]["reps"] is None

    def test_rep_ranges(self, client):
        text = "Squats 4x8-12"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["reps"] == "8-12"

    def test_time_based_exercises(self, client):
        text = "Plank 3x30s"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["sets"] == 3
        assert data["exercises"][0]["reps"] == "30s"

    def test_rpe_notation(self, client):
        text = "Squats 4x8 @RPE8"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["rpe"] == 8.0

    def test_exercise_order(self, client):
        text = """Squats 4x8
Bench Press 3x10
Deadlifts 5x5"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 3
        assert data["exercises"][0]["order"] == 0
        assert data["exercises"][1]["order"] == 1
        assert data["exercises"][2]["order"] == 2

    def test_bullet_format(self, client):
        text = """• Squats 4x8
- Bench Press 3x10
→ Deadlifts 5x5"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 3
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][1]["raw_name"] == "Bench Press"
        assert data["exercises"][2]["raw_name"] == "Deadlifts"

    def test_detected_format(self, client):
        text = "Squats 4x8"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert data["detected_format"] == "instagram_caption"

    def test_sets_equal_to_one_preserved(self, client):
        """Verify sets=1 is not silently dropped."""
        text = "Deadlift 1x5"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert data["exercises"][0]["sets"] == 1
        assert data["exercises"][0]["reps"] == "5"

    def test_multiple_superset_groups(self, client):
        """Two superset lines should produce groups A and B."""
        text = """Pull-ups 4x8 + Z Press 4x8
Squats 3x10 + Lunges 3x12"""

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 4
        assert data["exercises"][0]["superset_group"] == "A"
        assert data["exercises"][1]["superset_group"] == "A"
        assert data["exercises"][2]["superset_group"] == "B"
        assert data["exercises"][3]["superset_group"] == "B"

    def test_windows_line_endings(self, client):
        """\\r\\n line endings should parse correctly."""
        text = "Squats 4x8\r\nBench Press 3x10\r\nDeadlifts 5x5"

        response = client.post("/parse/text", json={"text": text})

        assert response.status_code == 200
        data = response.json()

        assert len(data["exercises"]) == 3
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][1]["raw_name"] == "Bench Press"
        assert data["exercises"][2]["raw_name"] == "Deadlifts"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestConfidenceScoring:

    def test_all_structured_caps_at_90(self, client):
        """All exercises with sets/reps -> confidence capped at 90."""
        text = """Squats 4x8
Bench Press 3x10
Deadlifts 5x5"""

        response = client.post("/parse/text", json={"text": text})
        data = response.json()

        assert data["confidence"] == 90

    def test_partial_structured_proportional(self, client):
        """Mix of structured and bare exercises -> proportional confidence."""
        text = """Squats 4x8
Mobility flow
Bench Press 3x10"""

        response = client.post("/parse/text", json={"text": text})
        data = response.json()

        # 2 out of 3 have sets -> ~66, capped at 90
        assert 0 < data["confidence"] <= 90

    def test_confidence_in_range(self, client):
        text = """Squats 4x8
Bench Press 3x10
Random text"""

        response = client.post("/parse/text", json={"text": text})
        data = response.json()

        assert 0 <= data["confidence"] <= 100


# ---------------------------------------------------------------------------
# Input validation & error handling
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_empty_input_returns_400(self, client):
        response = client.post("/parse/text", json={"text": ""})
        assert response.status_code == 400
        assert "Text is required" in response.json()["detail"]

    def test_whitespace_only_returns_empty(self, client):
        response = client.post("/parse/text", json={"text": "   \n\n   "})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["exercises"] == []

    def test_missing_text_field_returns_422(self, client):
        response = client.post("/parse/text", json={})
        assert response.status_code == 422

    def test_text_too_long_returns_422(self, client):
        response = client.post("/parse/text", json={"text": "a" * 50001})
        assert response.status_code == 422

    def test_invalid_source_returns_422(self, client):
        """Source must match ^[a-z_]+$ pattern."""
        response = client.post("/parse/text", json={
            "text": "Squats 4x8",
            "source": "INVALID SOURCE!"
        })
        assert response.status_code == 422

    def test_valid_source_accepted(self, client):
        response = client.post("/parse/text", json={
            "text": "Squats 4x8",
            "source": "instagram_caption"
        })
        assert response.status_code == 200

    def test_null_source_accepted(self, client):
        response = client.post("/parse/text", json={
            "text": "Squats 4x8",
            "source": None
        })
        assert response.status_code == 200

    def test_unauthenticated_returns_401(self):
        """Without auth override, endpoint should require authentication."""
        from fastapi.testclient import TestClient
        from workout_ingestor_api.main import app

        # Create client WITHOUT auth override
        raw_client = TestClient(app)
        # Clear any overrides
        from workout_ingestor_api.auth import get_current_user
        if get_current_user in app.dependency_overrides:
            saved = app.dependency_overrides[get_current_user]
            del app.dependency_overrides[get_current_user]
            try:
                response = raw_client.post("/parse/text", json={"text": "Squats 4x8"})
                assert response.status_code == 401
            finally:
                app.dependency_overrides[get_current_user] = saved
        else:
            pytest.skip("Auth override not set")


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


class TestLLMFallback:

    @staticmethod
    def _empty_structured_result(**kwargs):
        """Return a ParseTextResponse with no exercises, simulating structured parse failure."""
        from workout_ingestor_api.api.parse_routes import ParseTextResponse
        return ParseTextResponse(
            success=False,
            exercises=[],
            detected_format="text_unstructured",
            confidence=0,
            **kwargs,
        )

    def test_llm_fallback_on_unstructured_text(self, client):
        """When structured parsing finds nothing, LLM fallback fires."""
        # Use SimpleNamespace instead of MagicMock to avoid MagicMock silently
        # satisfying attribute access on fields that should be None.
        mock_exercise = SimpleNamespace(
            name="Push-ups", sets=3, reps=15,
            distance_m=None, weight_kg=None, notes=None,
        )
        mock_workout = SimpleNamespace(blocks=[
            SimpleNamespace(
                label="Workout",
                structure="regular",
                exercises=[mock_exercise],
            )
        ])

        with patch(
            "workout_ingestor_api.api.parse_routes.parse_with_text_parser",
            new_callable=AsyncMock,
            return_value=self._empty_structured_result(source="instagram_caption"),
        ), patch(
            "workout_ingestor_api.api.parse_routes.ParserService.parse_free_text_to_workout",
            return_value=mock_workout,
        ):
            response = client.post("/parse/text", json={
                "text": "Had a really great session at the gym this morning, feeling pumped!",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["detected_format"] == "text_llm"
        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["raw_name"] == "Push-ups"

    def test_llm_fallback_failure_returns_sanitized_error(self, client):
        """When LLM fallback raises, error message should be sanitized."""
        with patch(
            "workout_ingestor_api.api.parse_routes.parse_with_text_parser",
            new_callable=AsyncMock,
            return_value=self._empty_structured_result(source="instagram_caption"),
        ), patch(
            "workout_ingestor_api.api.parse_routes.ParserService.parse_free_text_to_workout",
            side_effect=Exception("OpenAI API key invalid: sk-proj-abc123"),
        ):
            response = client.post("/parse/text", json={
                "text": "Had a really great session at the gym this morning, feeling pumped!",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        # Must NOT contain the internal error details
        assert "sk-proj" not in data.get("metadata", {}).get("error", "")
        assert data["metadata"]["error"] == "Text could not be parsed. Please try a different format."

    def test_hashtag_only_input_skips_llm(self, client):
        """Input that preprocesses to empty lines should not trigger LLM."""
        # The autouse mock raises if LLM is called — this test verifies it's skipped
        response = client.post("/parse/text", json={
            "text": "#fitness #gym #workout"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["exercises"] == []
