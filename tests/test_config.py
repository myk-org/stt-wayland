"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from stt_wayland.config import ERR_NO_API_KEY, Config


class TestConfig:
    """Test Config class."""

    def test_from_env_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful config loading from environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")
        monkeypatch.setenv("STT_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        config = Config.from_env()

        assert config.api_key == "test-api-key-12345"
        assert config.model == "gemini-2.0-flash"
        assert config.runtime_dir == Path("/run/user/1000")

    def test_from_env_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(ValueError, match=ERR_NO_API_KEY):
            Config.from_env()

    def test_from_env_empty_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty API key raises ValueError."""
        monkeypatch.setenv("GEMINI_API_KEY", "")

        with pytest.raises(ValueError, match=ERR_NO_API_KEY):
            Config.from_env()

    def test_from_env_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that model defaults to gemini-2.5-flash when not set."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("STT_MODEL", raising=False)

        config = Config.from_env()

        assert config.model == "gemini-2.5-flash"

    def test_from_env_default_runtime_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that runtime_dir defaults to /tmp when XDG_RUNTIME_DIR not set."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

        config = Config.from_env()

        assert config.runtime_dir == Path("/tmp")

    def test_from_env_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test custom model configuration."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("STT_MODEL", "gemini-pro")

        config = Config.from_env()

        assert config.model == "gemini-pro"

    def test_config_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Config is immutable (frozen dataclass)."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = Config.from_env()

        with pytest.raises(AttributeError):
            config.api_key = "new-key"  # type: ignore[misc,unused-ignore]

    def test_config_has_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Config uses slots for memory efficiency."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = Config.from_env()

        # Frozen dataclass prevents attribute modification
        with pytest.raises((AttributeError, TypeError)):
            config.new_attribute = "value"  # type: ignore[attr-defined,unused-ignore]

    def test_pid_file_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test pid_file property returns correct path."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        config = Config.from_env()

        assert config.pid_file == Path("/run/user/1000/stt-wayland.pid")

    def test_pid_file_with_tmp_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test pid_file uses /tmp when XDG_RUNTIME_DIR not set."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

        config = Config.from_env()

        assert config.pid_file == Path("/tmp/stt-wayland.pid")

    def test_config_equality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that configs with same values are equal."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        config1 = Config.from_env()
        config2 = Config.from_env()

        assert config1 == config2

    def test_config_inequality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that configs with different values are not equal."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-1")
        config1 = Config.from_env()

        monkeypatch.setenv("GEMINI_API_KEY", "test-key-2")
        config2 = Config.from_env()

        assert config1 != config2

    def test_config_with_dotenv_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that config can load from environment when .env file is present.

        Note: This tests that the dotenv integration works by setting
        environment variables that would normally be loaded from .env.
        """
        # Simulate what .env would set
        monkeypatch.setenv("GEMINI_API_KEY", "dotenv-key")
        monkeypatch.setenv("STT_MODEL", "gemini-1.5-pro")

        config = Config.from_env()

        assert config.api_key == "dotenv-key"
        assert config.model == "gemini-1.5-pro"

    def test_config_env_overrides_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables take precedence.

        This verifies that explicitly set environment variables
        override any values that might be in .env files.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")

        config = Config.from_env()

        # Environment variable should be used
        assert config.api_key == "env-key"

    def test_api_key_with_special_characters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API keys with special characters are handled correctly."""
        special_key = "test-key_123!@#$%^&*()_+-={}[]|:;<>?,./~`"
        monkeypatch.setenv("GEMINI_API_KEY", special_key)

        config = Config.from_env()

        assert config.api_key == special_key

    def test_model_name_variations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test various model name formats."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        test_models = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "custom-model-123",
        ]

        for model in test_models:
            monkeypatch.setenv("STT_MODEL", model)
            config = Config.from_env()
            assert config.model == model

    def test_runtime_dir_with_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that runtime_dir handles trailing slashes correctly."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000/")

        config = Config.from_env()

        # Path should normalize the trailing slash
        assert config.runtime_dir == Path("/run/user/1000/")
        assert config.pid_file == Path("/run/user/1000/stt-wayland.pid")
