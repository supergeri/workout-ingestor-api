"""
End-to-end tests for Helicone integration.

These tests make REAL API calls and should only run:
- Nightly in CI (not on every PR)
- Manually during development
- In staging environment

Requires real API keys:
- OPENAI_API_KEY
- HELICONE_API_KEY
- HELICONE_ENABLED=true
"""
import pytest
import os


# Skip E2E tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E_TESTS", "false").lower() != "true",
    reason="E2E tests disabled. Set RUN_E2E_TESTS=true to run.",
)


@pytest.fixture
def real_api_keys():
    """Verify real API keys are available."""
    openai_key = os.getenv("OPENAI_API_KEY")
    helicone_key = os.getenv("HELICONE_API_KEY")

    if not openai_key or openai_key.startswith("test"):
        pytest.skip("Real OPENAI_API_KEY required for E2E tests")
    if not helicone_key:
        pytest.skip("Real HELICONE_API_KEY required for E2E tests")

    return {"openai": openai_key, "helicone": helicone_key}


class TestHeliconeE2E:
    """End-to-end tests with real Helicone integration."""

    @pytest.mark.slow
    def test_openai_call_through_helicone_proxy(self, real_api_keys):
        """
        Smoke test: Make a real OpenAI call through Helicone.

        Validates:
        - Helicone proxy is reachable
        - Authentication works
        - Request completes successfully
        """
        # Ensure Helicone is enabled for this test
        os.environ["HELICONE_ENABLED"] = "true"

        # Reload settings to pick up env var
        from importlib import reload
        from workout_ingestor_api import config

        reload(config)

        from workout_ingestor_api.ai.client_factory import (
            AIClientFactory,
            AIRequestContext,
        )

        context = AIRequestContext(
            user_id="e2e-test-user",
            feature_name="e2e_smoke_test",
            custom_properties={"test_run": "true"},
        )

        client = AIClientFactory.create_openai_client(context=context)

        # Make a minimal API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test passed' in 2 words"}],
            max_tokens=10,
        )

        assert response.choices[0].message.content is not None
        # Note: Verify in Helicone dashboard that request appears with correct metadata

    @pytest.mark.slow
    def test_openai_direct_call_when_helicone_disabled(self, real_api_keys):
        """
        Verify direct OpenAI calls work when Helicone is disabled.

        This ensures the fallback path (no proxy) is functional.
        """
        os.environ["HELICONE_ENABLED"] = "false"

        # Force reload to pick up new env var
        from importlib import reload
        from workout_ingestor_api import config

        reload(config)

        from workout_ingestor_api.ai.client_factory import AIClientFactory

        client = AIClientFactory.create_openai_client()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'direct test' in 2 words"}],
            max_tokens=10,
        )

        assert response.choices[0].message.content is not None


class TestHeliconeMetadataVerification:
    """Tests to verify metadata appears in Helicone dashboard."""

    @pytest.mark.slow
    @pytest.mark.manual
    def test_verify_user_id_in_helicone_dashboard(self, real_api_keys):
        """
        Manual verification test - check Helicone dashboard after running.

        Steps:
        1. Run this test
        2. Go to Helicone dashboard
        3. Find request with user_id = "e2e-verify-user-123"
        4. Verify custom properties are visible
        """
        os.environ["HELICONE_ENABLED"] = "true"

        from importlib import reload
        from workout_ingestor_api import config

        reload(config)

        from workout_ingestor_api.ai.client_factory import (
            AIClientFactory,
            AIRequestContext,
        )

        context = AIRequestContext(
            user_id="e2e-verify-user-123",
            session_id="e2e-session-456",
            feature_name="metadata_verification_test",
            request_id="e2e-req-789",
            custom_properties={
                "test_type": "manual_verification",
                "workout_source": "youtube",
            },
        )

        client = AIClientFactory.create_openai_client(context=context)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Echo: metadata test"}],
            max_tokens=10,
        )

        print(f"\n[MANUAL VERIFICATION REQUIRED]")
        print(f"Check Helicone dashboard for request with:")
        print(f"  - User ID: e2e-verify-user-123")
        print(f"  - Session ID: e2e-session-456")
        print(f"  - Feature: metadata_verification_test")
        print(f"  - Request ID: e2e-req-789")
        print(f"  - Custom properties: test_type, workout_source")


class TestHeliconeProxyResilience:
    """Test resilience when Helicone proxy has issues."""

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires intentional network manipulation")
    def test_behavior_when_helicone_unreachable(self, real_api_keys):
        """
        Test what happens when Helicone proxy is unreachable.

        This test is skipped by default as it requires network manipulation.
        Run manually to verify failure behavior.
        """
        # This would require DNS manipulation or firewall rules
        # to simulate Helicone being unreachable
        pass
