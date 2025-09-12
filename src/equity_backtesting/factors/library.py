"""
Factor library containing common equity factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod

from ..data.provider import DataProvider
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseFactor(ABC):
    """Base class for all factors."""
    
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate factor values."""
        pass
    
    def normalize(self, factor_data: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """
        Normalize factor values.
        
        Args:
            factor_data: Raw factor values
            method: Normalization method ('zscore', 'rank', 'minmax')
            
        Returns:
            Normalized factor values
        """
        if method == "zscore":
            return (factor_data - factor_data.mean()) / factor_data.std()
        elif method == "rank":
            return factor_data.rank(pct=True) - 0.5
        elif method == "minmax":
            return (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min()) - 0.5
        else:
            return factor_data


class MomentumFactor(BaseFactor):
    """Price momentum factors."""
    
    def __init__(self, lookback_periods: int = 63):
        super().__init__(f"momentum_{lookback_periods}d", "momentum")
        self.lookback_periods = lookback_periods
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate momentum factor (cumulative return over lookback period)."""
        adj_close = data.xs('Adj Close', level=1, axis=1)
        
        # Calculate momentum as cumulative return over lookback period
        momentum = adj_close.pct_change(periods=self.lookback_periods)
        
        return momentum.dropna()


class ValueFactor(BaseFactor):
    """Value factors based on price ratios."""
    
    def __init__(self, metric: str = "pe_ratio"):
        super().__init__(f"value_{metric}", "value")
        self.metric = metric
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate value factor.
        Note: This is a simplified implementation. In practice, you'd need
        fundamental data which requires additional data sources.
        """
        # For demonstration, we'll use a price-based proxy
        adj_close = data.xs('Adj Close', level=1, axis=1)
        volume = data.xs('Volume', level=1, axis=1)
        
        # Simple price-to-volume ratio as proxy for value
        # In practice, you'd use actual fundamental ratios
        value_proxy = adj_close / volume.rolling(window=252).mean()
        
        # Invert so lower values (cheaper stocks) have higher factor scores
        return -value_proxy.dropna()


class VolatilityFactor(BaseFactor):
    """Low volatility factor."""
    
    def __init__(self, lookback_periods: int = 63):
        super().__init__(f"low_vol_{lookback_periods}d", "low_volatility")
        self.lookback_periods = lookback_periods
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate volatility factor (negative of realized volatility)."""
        adj_close = data.xs('Adj Close', level=1, axis=1)
        
        # Calculate daily returns
        returns = adj_close.pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.lookback_periods).std() * np.sqrt(252)
        
        # Return negative volatility (so low vol stocks have high scores)
        return -volatility.dropna()


class SizeFactor(BaseFactor):
    """Size factor based on market capitalization."""
    
    def __init__(self):
        super().__init__("size", "size")
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate size factor.
        Note: This uses price * volume as a proxy for market cap.
        In practice, you'd use actual shares outstanding data.
        """
        adj_close = data.xs('Adj Close', level=1, axis=1)
        volume = data.xs('Volume', level=1, axis=1)
        
        # Market cap proxy: price * average volume
        market_cap_proxy = adj_close * volume.rolling(window=63).mean()
        
        # Take log and invert (so smaller companies have higher scores)
        size_factor = -np.log(market_cap_proxy)
        
        return size_factor.dropna()


class QualityFactor(BaseFactor):
    """Quality factor based on price stability."""
    
    def __init__(self, lookback_periods: int = 252):
        super().__init__(f"quality_{lookback_periods}d", "quality")
        self.lookback_periods = lookback_periods
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate quality factor.
        Uses price trend stability as a proxy for quality.
        """
        adj_close = data.xs('Adj Close', level=1, axis=1)
        returns = adj_close.pct_change()
        
        # Quality as negative of return standard deviation
        return_stability = -returns.rolling(window=self.lookback_periods).std()
        
        return return_stability.dropna()


class FactorLibrary:
    """
    Main factory class for creating and managing factors.
    """
    
    def __init__(self):
        self.factors: Dict[str, BaseFactor] = {}
        self._register_default_factors()
        logger.info("Initialized FactorLibrary with default factors")
    
    def _register_default_factors(self):
        """Register default factor implementations."""
        # Momentum factors
        self.factors["momentum_1m"] = MomentumFactor(21)
        self.factors["momentum_3m"] = MomentumFactor(63)
        self.factors["momentum_6m"] = MomentumFactor(126)
        self.factors["momentum_12m"] = MomentumFactor(252)
        
        # Value factors (simplified)
        self.factors["value"] = ValueFactor("pe_ratio")
        
        # Volatility factor
        self.factors["low_volatility"] = VolatilityFactor(63)
        
        # Size factor
        self.factors["size"] = SizeFactor()
        
        # Quality factor
        self.factors["quality"] = QualityFactor(252)
    
    def register_factor(self, factor: BaseFactor):
        """Register a custom factor."""
        self.factors[factor.name] = factor
        logger.info(f"Registered custom factor: {factor.name}")
    
    def get_factor(self, name: str) -> BaseFactor:
        """Get a factor by name."""
        if name not in self.factors:
            raise ValueError(f"Factor '{name}' not found. Available: {list(self.factors.keys())}")
        return self.factors[name]
    
    def list_factors(self) -> List[str]:
        """Get list of available factor names."""
        return list(self.factors.keys())
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """Get factor names by category."""
        return [name for name, factor in self.factors.items() if factor.category == category]
    
    def calculate_factor(
        self, 
        factor_name: str, 
        data: pd.DataFrame,
        normalize: bool = True,
        normalization_method: str = "zscore"
    ) -> pd.DataFrame:
        """
        Calculate a specific factor.
        
        Args:
            factor_name: Name of the factor to calculate
            data: Price/volume data
            normalize: Whether to normalize the factor values
            normalization_method: Method for normalization
            
        Returns:
            DataFrame with factor values
        """
        factor = self.get_factor(factor_name)
        
        logger.info(f"Calculating factor: {factor_name}")
        factor_data = factor.calculate(data)
        
        if normalize:
            # Normalize cross-sectionally at each date
            normalized_data = factor_data.apply(
                lambda row: factor.normalize(row, normalization_method), 
                axis=1
            )
            return normalized_data
        
        return factor_data
    
    def calculate_multiple_factors(
        self,
        factor_names: List[str],
        data: pd.DataFrame,
        normalize: bool = True,
        normalization_method: str = "zscore"
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate multiple factors at once.
        
        Args:
            factor_names: List of factor names to calculate
            data: Price/volume data
            normalize: Whether to normalize factor values
            normalization_method: Method for normalization
            
        Returns:
            Dictionary mapping factor names to their values
        """
        results = {}
        
        for factor_name in factor_names:
            try:
                results[factor_name] = self.calculate_factor(
                    factor_name, data, normalize, normalization_method
                )
                logger.info(f"Successfully calculated {factor_name}")
            except Exception as e:
                logger.error(f"Failed to calculate {factor_name}: {str(e)}")
                continue
        
        return results
    
    def create_composite_factor(
        self,
        factor_names: List[str],
        weights: Optional[List[float]] = None,
        data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create a composite factor from multiple factors.
        
        Args:
            factor_names: List of factor names to combine
            weights: Weights for each factor (default: equal weight)
            data: Price/volume data (required if factors not pre-calculated)
            
        Returns:
            Composite factor values
        """
        if weights is None:
            weights = [1.0 / len(factor_names)] * len(factor_names)
        
        if len(weights) != len(factor_names):
            raise ValueError("Number of weights must match number of factors")
        
        if data is None:
            raise ValueError("Data is required to calculate composite factor")
        
        # Calculate individual factors
        factor_data = self.calculate_multiple_factors(factor_names, data)
        
        # Combine factors with weights
        composite = None
        for i, (factor_name, weight) in enumerate(zip(factor_names, weights)):
            if factor_name in factor_data:
                if composite is None:
                    composite = weight * factor_data[factor_name]
                else:
                    # Align indices and add
                    aligned_data = factor_data[factor_name].reindex_like(composite).fillna(0)
                    composite = composite + weight * aligned_data
        
        logger.info(f"Created composite factor from {factor_names} with weights {weights}")
        return composite