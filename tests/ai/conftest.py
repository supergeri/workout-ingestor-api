"""Shared fixtures for AI module tests."""
import pytest
from unittest.mock import patch, MagicMock
import os


@pytest.fixture
def mock_settings_helicone_enabled():
    """Mock settings with Helicone enabled."""
    with patch("workout_ingestor_api.ai.client_factory.settings") as mock:
        mock.OPENAI_API_KEY = "sk-test-openai"
        mock.ANTHROPIC_API_KEY = "sk-test-anthropic"
        mock.HELICONE_ENABLED = True
        mock.HELICONE_API_KEY = "sk-test-helicone"
        mock.ENVIRONMENT = "test"
        yield mock


@pytest.fixture
def mock_settings_helicone_disabled():
    """Mock settings with Helicone disabled."""
    with patch("workout_ingestor_api.ai.client_factory.settings") as mock:
        mock.OPENAI_API_KEY = "sk-test-openai"
        mock.ANTHROPIC_API_KEY = "sk-test-anthropic"
        mock.HELICONE_ENABLED = False
        mock.HELICONE_API_KEY = None
        mock.ENVIRONMENT = "development"
        yield mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"title": "Test Workout"}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"title": "Test Workout"}')]
    mock_client.messages.create.return_value = mock_response
    return mock_client


# Error simulation fixtures


@pytest.fixture
def rate_limit_error():
    """Simulate OpenAI rate limit error."""
    return Exception("Error code: 429 - Rate limit reached for requests")


@pytest.fixture
def server_error_503():
    """Simulate 503 Service Unavailable."""
    return Exception("Error code: 503 - Service temporarily unavailable")


@pytest.fixture
def timeout_error():
    """Simulate timeout error."""
    import httpx

    return httpx.ReadTimeout("Connection read timed out")


@pytest.fixture
def auth_error():
    """Simulate authentication error."""
    return Exception("Error code: 401 - Invalid API key provided")


@pytest.fixture
def bad_request_error():
    """Simulate bad request error."""
    return Exception("Error code: 400 - Invalid request: missing 'messages' field")
