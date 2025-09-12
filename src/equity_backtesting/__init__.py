"""
Equity Factor Backtesting Framework

A comprehensive toolkit for equity factor research and backtesting with ML integration.
"""

__version__ = "0.1.0"
__author__ = "Rohan"

from .backtesting.engine import BacktestEngine
from .factors.library import FactorLibrary
from .data.provider import DataProvider
from .analytics.performance import PerformanceAnalytics

__all__ = [
    "BacktestEngine",
    "FactorLibrary", 
    "DataProvider",
    "PerformanceAnalytics"
]
