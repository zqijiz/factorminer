import pytest
from factorminer.utils.config import ResearchRegimesConfig

def test_research_regimes_config_validate_missing_error():
    config = ResearchRegimesConfig(definition="invalid_definition")
    with pytest.raises(ValueError, match="research.regimes.definition must be one of: return_volatility, return_volatility_liquidity"):
        config.validate()
