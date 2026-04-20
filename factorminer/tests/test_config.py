import pytest
from factorminer.utils.config import MiningConfig

def test_mining_config_validation_target_library_size():
    with pytest.raises(ValueError, match="target_library_size must be >= 1"):
        MiningConfig(target_library_size=0).validate()

def test_mining_config_validation_batch_size():
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        MiningConfig(batch_size=0).validate()

def test_mining_config_validation_max_iterations():
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        MiningConfig(max_iterations=0).validate()

def test_mining_config_validation_ic_threshold():
    with pytest.raises(ValueError, match=r"ic_threshold must be in \(0, 1\)"):
        MiningConfig(ic_threshold=0.0).validate()
    with pytest.raises(ValueError, match=r"ic_threshold must be in \(0, 1\)"):
        MiningConfig(ic_threshold=1.0).validate()

def test_mining_config_validation_icir_threshold():
    with pytest.raises(ValueError, match=r"icir_threshold must be in \(0, 10\)"):
        MiningConfig(icir_threshold=0.0).validate()
    with pytest.raises(ValueError, match=r"icir_threshold must be in \(0, 10\)"):
        MiningConfig(icir_threshold=10.0).validate()

def test_mining_config_validation_correlation_threshold():
    with pytest.raises(ValueError, match=r"correlation_threshold must be in \(0, 1\]"):
        MiningConfig(correlation_threshold=0.0).validate()
    with pytest.raises(ValueError, match=r"correlation_threshold must be in \(0, 1\]"):
        MiningConfig(correlation_threshold=1.1).validate()

def test_mining_config_validation_replacement_ic_min():
    with pytest.raises(ValueError, match="replacement_ic_min must be > ic_threshold"):
        # Default ic_threshold is 0.04
        MiningConfig(replacement_ic_min=0.04).validate()

def test_mining_config_validation_replacement_ic_ratio():
    with pytest.raises(ValueError, match="replacement_ic_ratio must be >= 1.0"):
        MiningConfig(replacement_ic_ratio=0.9).validate()
