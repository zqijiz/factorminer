import os
from pathlib import Path

import pytest

from factorminer.utils.config import Config, load_config

def test_config_to_dict():
    """Test that Config can be serialized to a dictionary."""
    config = Config()
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)

    # Check that top-level sections exist
    expected_sections = [
        "mining",
        "evaluation",
        "data",
        "llm",
        "memory",
        "phase2",
        "benchmark",
        "research",
    ]
    for section in expected_sections:
        assert section in config_dict
        assert isinstance(config_dict[section], dict)

    # Check some nested values to ensure correct serialization
    assert config_dict["mining"]["target_library_size"] == 110
    assert config_dict["evaluation"]["backend"] == "gpu"
    assert config_dict["data"]["market"] == "a_shares"
    assert config_dict["llm"]["provider"] == "google"
    assert config_dict["memory"]["policy"] == "paper"
    assert config_dict["phase2"]["causal"]["enabled"] is False
    assert config_dict["benchmark"]["mode"] == "paper"
    assert config_dict["research"]["enabled"] is False


def test_config_save(tmp_path: Path):
    """Test that Config can be saved to a YAML file and reloaded."""
    config = Config()

    # Modify a value to ensure we're not just loading defaults
    config.mining.batch_size = 999
    config.llm.model = "test-model-123"

    save_path = tmp_path / "test_config.yaml"
    config.save(save_path)

    assert save_path.exists()

    # Load the saved config
    loaded_config = load_config(config_path=save_path)

    # Verify the modified values are preserved
    assert loaded_config.mining.batch_size == 999
    assert loaded_config.llm.model == "test-model-123"

    # Verify that the entire dictionary representation matches
    assert config.to_dict() == loaded_config.to_dict()
