"""
Configuration settings for the equity factor backtesting framework.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Data settings
    DATA_DIR: str = "data"
    CACHE_DIR: str = "data/cache"
    
    # Market data providers
    DEFAULT_DATA_PROVIDER: str = "yfinance"
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    IEX_CLOUD_API_KEY: Optional[str] = None
    
    # Default universes
    SP500_SYMBOLS_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Backtesting defaults
    DEFAULT_START_DATE: str = "2010-01-01"
    DEFAULT_END_DATE: str = "2023-12-31"
    DEFAULT_REBALANCE_FREQ: str = "M"  # Monthly
    DEFAULT_BENCHMARK: str = "SPY"
    
    # Transaction cost assumptions
    DEFAULT_TRANSACTION_COST: float = 0.001  # 10 bps
    DEFAULT_MARKET_IMPACT: float = 0.0005   # 5 bps
    
    # Risk management
    MAX_POSITION_SIZE: float = 0.05  # 5% max per position
    MAX_LEVERAGE: float = 1.0        # No leverage by default
    
    # Performance thresholds
    MIN_OBSERVATIONS: int = 252  # Minimum trading days for valid backtest
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator("DATA_DIR", "CACHE_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v


# Global settings instance
settings = Settings()

# Factor categories and definitions
FACTOR_CATEGORIES: Dict[str, List[str]] = {
    "value": [
        "price_to_earnings",
        "price_to_book", 
        "price_to_sales",
        "ev_to_ebitda",
        "price_to_cash_flow"
    ],
    "momentum": [
        "price_momentum_1m",
        "price_momentum_3m", 
        "price_momentum_6m",
        "price_momentum_12m",
        "earnings_momentum"
    ],
    "quality": [
        "return_on_equity",
        "return_on_assets",
        "debt_to_equity",
        "current_ratio",
        "earnings_stability"
    ],
    "size": [
        "market_cap",
        "enterprise_value",
        "revenue"
    ],
    "low_volatility": [
        "realized_volatility",
        "beta",
        "idiosyncratic_volatility"
    ],
    "profitability": [
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roa",
        "roe"
    ]
}

# Rebalancing frequencies
REBALANCE_FREQUENCIES: Dict[str, str] = {
    "D": "Daily",
    "W": "Weekly", 
    "M": "Monthly",
    "Q": "Quarterly",
    "A": "Annually"
}
