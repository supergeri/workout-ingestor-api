from workout_ingestor_api.services.adapters import (
    register_adapter, get_adapter, PlatformAdapter, MediaContent
)


class _FakeAdapter(PlatformAdapter):
    @staticmethod
    def platform_name() -> str:
        return "fake_platform"

    def fetch(self, url: str, source_id: str) -> MediaContent:
        return MediaContent(primary_text="hello", title="test")


def test_register_and_get_adapter():
    register_adapter(_FakeAdapter)
    adapter = get_adapter("fake_platform")
    assert isinstance(adapter, _FakeAdapter)

def test_get_unregistered_raises():
    import pytest
    with pytest.raises(KeyError):
        get_adapter("nonexistent_platform_xyz")

def test_fetch_returns_media_content():
    register_adapter(_FakeAdapter)
    adapter = get_adapter("fake_platform")
    result = adapter.fetch("https://fake.com/video/1", "1")
    assert isinstance(result, MediaContent)
    assert result.primary_text == "hello"
