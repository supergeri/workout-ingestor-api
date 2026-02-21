import json
from unittest.mock import patch, MagicMock
import pytest

from workout_ingestor_api.services.unified_parser import UnifiedParser, UnifiedParserError
from workout_ingestor_api.services.adapters.base import MediaContent


SAMPLE_MEDIA = MediaContent(
    primary_text="4 rounds: Squats 20 reps, Push-ups 15 reps",
    title="HYROX Session",
    media_metadata={"video_duration_sec": 120},
)

VALID_LLM_RESPONSE = json.dumps({
    "title": "HYROX Session",
    "workout_type": "circuit",
    "workout_type_confidence": 0.9,
    "blocks": [
        {"label": "Main", "structure": "circuit", "rounds": 4,
         "exercises": [{"name": "Squats", "reps": 20, "type": "strength"}],
         "supersets": []}
    ],
})


def _mock_openai_client(content: str):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_parse_returns_dict_with_blocks():
    mock_client = _mock_openai_client(VALID_LLM_RESPONSE)
    with patch(
        "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
        return_value=mock_client,
    ):
        parser = UnifiedParser()
        result = parser.parse(SAMPLE_MEDIA, platform="instagram")
        assert "blocks" in result
        assert isinstance(result["blocks"], list)


def test_parse_raises_on_invalid_json():
    mock_client = _mock_openai_client("not valid json {{{")
    with patch(
        "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
        return_value=mock_client,
    ):
        parser = UnifiedParser()
        with pytest.raises(UnifiedParserError, match="invalid JSON"):
            parser.parse(SAMPLE_MEDIA, platform="instagram")
        # Assert no spurious retries on bad JSON — should fail fast
        assert mock_client.chat.completions.create.call_count == 1


def test_parse_feature_name_contains_platform():
    captured_context = {}

    def capture_context(context):
        captured_context["feature_name"] = context.feature_name
        return _mock_openai_client(VALID_LLM_RESPONSE)

    with patch(
        "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
        side_effect=capture_context,
    ):
        parser = UnifiedParser()
        parser.parse(SAMPLE_MEDIA, platform="tiktok")
        assert "tiktok" in captured_context["feature_name"]


def test_parse_calls_spacy_corrector():
    mock_client = _mock_openai_client(VALID_LLM_RESPONSE)
    with patch(
        "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
        return_value=mock_client,
    ), patch(
        "workout_ingestor_api.services.unified_parser.SpacyCorrector.correct",
        wraps=lambda data, raw_text: data,
    ) as mock_correct:
        parser = UnifiedParser()
        result = parser.parse(SAMPLE_MEDIA, platform="instagram")
        mock_correct.assert_called_once()
        _, kwargs = mock_correct.call_args
        # raw_text should be the primary_text from MediaContent
        assert SAMPLE_MEDIA.primary_text in str(mock_correct.call_args)
        assert "blocks" in result  # verify parse() returns the corrected dict, not self


# ---------------------------------------------------------------------------
# Confidence scoring pass-through tests
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE_LLM_RESPONSE = json.dumps({
    "title": "HYROX Session",
    "workout_type": "circuit",
    "workout_type_confidence": 0.9,
    "blocks": [
        {
            "label": "Main",
            "structure": "circuit",
            "rounds": 4,
            "exercises": [{"name": "Squats", "reps": 20, "type": "strength"}],
            "supersets": [],
            "structure_confidence": 1.0,
            "structure_options": [],
        }
    ],
})

_LOW_CONFIDENCE_LLM_RESPONSE = json.dumps({
    "title": "Ambiguous Workout",
    "workout_type": "strength",
    "workout_type_confidence": 0.6,
    "blocks": [
        {
            "label": "Block A",
            "structure": "circuit",
            "rounds": None,
            "exercises": [{"name": "Push-ups", "reps": 10, "type": "strength"}],
            "supersets": [],
            "structure_confidence": 0.4,
            "structure_options": ["circuit", "straight_sets"],
        }
    ],
})

_NO_CONFIDENCE_LLM_RESPONSE = json.dumps({
    "title": "Old Style Workout",
    "workout_type": "strength",
    "workout_type_confidence": 0.8,
    "blocks": [
        {
            "label": "Main",
            "structure": None,
            "exercises": [{"name": "Deadlifts", "sets": 3, "reps": 5, "type": "strength"}],
            "supersets": [],
        }
    ],
})


class TestUnifiedParserConfidenceOutput:
    """Parser passes through structure_confidence and structure_options from LLM."""

    def test_high_confidence_block_passes_through(self):
        """When LLM returns confidence=1.0, parser preserves it."""
        mock_client = _mock_openai_client(_HIGH_CONFIDENCE_LLM_RESPONSE)
        with patch(
            "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
            return_value=mock_client,
        ):
            parser = UnifiedParser()
            result = parser.parse(SAMPLE_MEDIA, platform="instagram")
            block = result["blocks"][0]
            assert block["structure_confidence"] == 1.0
            assert block["structure_options"] == []

    def test_low_confidence_block_passes_through(self):
        """When LLM returns low confidence, parser preserves options list."""
        mock_client = _mock_openai_client(_LOW_CONFIDENCE_LLM_RESPONSE)
        with patch(
            "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
            return_value=mock_client,
        ):
            parser = UnifiedParser()
            result = parser.parse(SAMPLE_MEDIA, platform="instagram")
            block = result["blocks"][0]
            assert block["structure_confidence"] == 0.4
            assert block["structure_options"] == ["circuit", "straight_sets"]

    def test_missing_confidence_defaults_gracefully(self):
        """If LLM omits confidence (old-style response), no crash and defaults apply."""
        mock_client = _mock_openai_client(_NO_CONFIDENCE_LLM_RESPONSE)
        with patch(
            "workout_ingestor_api.services.unified_parser.AIClientFactory.create_openai_client",
            return_value=mock_client,
        ):
            parser = UnifiedParser()
            # Should not raise
            result = parser.parse(SAMPLE_MEDIA, platform="instagram")
            block = result["blocks"][0]
            # When absent, treat as fully confident (1.0)
            assert block.get("structure_confidence", 1.0) == 1.0
