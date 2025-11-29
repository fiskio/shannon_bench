"""Tests for the configuration module."""

import pytest
from pydantic import ValidationError

from shannon_bench.config import Config


def test_config_defaults() -> None:
  """Test that default values are set correctly."""
  config = Config(name="test_run")
  assert config.name == "test_run"
  expected_iterations = 10
  assert config.iterations == expected_iterations


def test_config_validation() -> None:
  """Test that validation works correctly."""
  with pytest.raises(ValidationError):
    Config(name="test_run", iterations=0)


def test_config_missing_name() -> None:
  """Test that missing required fields raises an error."""
  with pytest.raises(ValidationError):
    Config(iterations=5)  # type: ignore[call-arg]
