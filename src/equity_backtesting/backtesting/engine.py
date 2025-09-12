"""
Core backtesting engine for equity factor strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..data.provider import DataProvider
from ..factors.library import FactorLibrary
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters."""
    start_date: str
    end_date: str
    universe: List[str]
    benchmark: str = "SPY"
    rebalance_freq: str = "M"  # D, W, M, Q, A
    transaction_cost: float = 0.001  # 10 bps
    market_impact: float = 0.0005   # 5 bps
    max_position_size: float = 0.05  # 5%
    leverage: float = 1.0
    min_observations: int = 252


@dataclass
class StrategyConfig:
    """Configuration for factor strategy."""
    factors: Dict[str, float]  # factor_name -> weight
    strategy_type: str = "long_short"  # long_short, long_only, rank_based
    top_percentile: float = 0.2  # For long positions
    bottom_percentile: float = 0.2  # For short positions
    neutralization: Optional[str] = None  # sector, size, etc.


class Portfolio:
    """Portfolio state tracking."""
    
    def __init__(self):
        self.positions: pd.Series = pd.Series(dtype=float)
        self.cash: float = 1.0
        self.value: float = 1.0
        self.returns: List[float] = []
        self.turnover: List[float] = []
        self.transaction_costs: List[float] = []
    
    def update_positions(
        self, 
        new_positions: pd.Series, 
        prices: pd.Series,
        transaction_cost: float = 0.001
    ):
        """Update portfolio positions and calculate costs."""
        if len(self.positions) == 0:
            # Initial portfolio
            self.positions = new_positions.copy()
            return
        
        # Calculate turnover
        position_changes = (new_positions - self.positions.reindex(new_positions.index, fill_value=0)).abs()
        turnover = position_changes.sum()
        self.turnover.append(turnover)
        
        # Calculate transaction costs
        cost = turnover * transaction_cost
        self.transaction_costs.append(cost)
        
        # Update positions
        self.positions = new_positions.copy()
    
    def calculate_return(self, current_prices: pd.Series, previous_prices: pd.Series) -> float:
        """Calculate portfolio return for the period."""
        if len(self.positions) == 0:
            return 0.0
        
        # Calculate returns for each position
        price_returns = (current_prices / previous_prices - 1).fillna(0)
        
        # Portfolio return is weighted sum of position returns
        portfolio_return = (self.positions * price_returns).sum()
        
        # Subtract transaction costs (already applied to previous period)
        if len(self.transaction_costs) > 0:
            portfolio_return -= self.transaction_costs[-1]
        
        self.returns.append(portfolio_return)
        return portfolio_return


class BacktestEngine:
    """
    Main backtesting engine for equity factor strategies.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_provider = DataProvider()
        self.factor_library = FactorLibrary()
        self.results = None
        logger.info(f"Initialized BacktestEngine for {config.start_date} to {config.end_date}")
    
    def run_backtest(self, strategy: StrategyConfig) -> Dict:
        """
        Run a complete backtest for the given strategy.
        
        Args:
            strategy: Strategy configuration
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Starting backtest execution")
        
        # Get market data
        market_data = self._load_market_data()
        
        # Calculate factors
        factor_data = self._calculate_factors(strategy.factors, market_data)
        
        # Generate rebalancing dates
        rebalance_dates = self._generate_rebalance_dates(market_data.index)
        
        # Run simulation
        portfolio_results = self._simulate_strategy(
            strategy, factor_data, market_data, rebalance_dates
        )
        
        # Get benchmark data
        benchmark_returns = self._get_benchmark_returns()
        
        # Package results
        results = {
            'portfolio_returns': pd.Series(
                portfolio_results['returns'], 
                index=portfolio_results['dates']
            ),
            'positions_history': portfolio_results['positions'],
            'turnover': portfolio_results['turnover'],
            'transaction_costs': portfolio_results['costs'],
            'benchmark_returns': benchmark_returns,
            'factor_data': factor_data,
            'config': self.config,
            'strategy': strategy
        }
        
        self.results = results
        logger.info("Backtest completed successfully")
        return results
    
    def _load_market_data(self) -> pd.DataFrame:
        """Load market data for the universe."""
        logger.info(f"Loading market data for {len(self.config.universe)} symbols")
        
        data = self.data_provider.get_stock_data(
            symbols=self.config.universe,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        return data
    
    def _calculate_factors(
        self, 
        factor_weights: Dict[str, float], 
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate composite factor from individual factors."""
        logger.info(f"Calculating factors: {list(factor_weights.keys())}")
        
        factor_names = list(factor_weights.keys())
        weights = list(factor_weights.values())
        
        composite_factor = self.factor_library.create_composite_factor(
            factor_names=factor_names,
            weights=weights,
            data=market_data
        )
        
        return composite_factor
    
    def _generate_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Generate rebalancing dates based on frequency."""
        freq_map = {
            'D': 'D',
            'W': 'W-FRI',  # Weekly on Fridays
            'M': 'BM',     # Business month end
            'Q': 'BQ',     # Business quarter end
            'A': 'BA'      # Business year end
        }
        
        freq = freq_map.get(self.config.rebalance_freq, 'BM')
        
        # Generate date range with specified frequency
        rebalance_dates = pd.date_range(
            start=date_index[0],
            end=date_index[-1],
            freq=freq
        )
        
        # Filter to only include dates that exist in our data
        rebalance_dates = [d for d in rebalance_dates if d in date_index]
        
        logger.info(f"Generated {len(rebalance_dates)} rebalancing dates")
        return rebalance_dates
    
    def _simulate_strategy(
        self,
        strategy: StrategyConfig,
        factor_data: pd.DataFrame,
        market_data: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp]
    ) -> Dict:
        """Simulate the strategy over time."""
        portfolio = Portfolio()
        
        returns = []
        dates = []
        positions_history = {}
        
        # Get adjusted close prices for return calculation
        adj_close = market_data.xs('Adj Close', level=1, axis=1)
        
        previous_prices = None
        
        for i, date in enumerate(rebalance_dates):
            if date not in factor_data.index:
                continue
            
            logger.debug(f"Rebalancing on {date}")
            
            # Get factor scores for this date
            factor_scores = factor_data.loc[date].dropna()
            
            if len(factor_scores) < 10:  # Need minimum stocks
                logger.warning(f"Insufficient factor scores on {date}: {len(factor_scores)}")
                continue
            
            # Construct portfolio positions
            new_positions = self._construct_portfolio(factor_scores, strategy)
            
            # Get current prices
            current_prices = adj_close.loc[date]
            
            # Calculate return since last rebalancing
            if previous_prices is not None and len(portfolio.positions) > 0:
                portfolio_return = portfolio.calculate_return(current_prices, previous_prices)
                returns.append(portfolio_return)
                dates.append(date)
            
            # Update portfolio positions
            portfolio.update_positions(
                new_positions, 
                current_prices, 
                self.config.transaction_cost
            )
            
            # Store positions
            positions_history[date] = new_positions.copy()
            
            # Update previous prices
            previous_prices = current_prices.copy()
        
        return {
            'returns': returns,
            'dates': dates[1:],  # Skip first date since no return calculated
            'positions': positions_history,
            'turnover': portfolio.turnover,
            'costs': portfolio.transaction_costs
        }
    
    def _construct_portfolio(
        self, 
        factor_scores: pd.Series, 
        strategy: StrategyConfig
    ) -> pd.Series:
        """Construct portfolio positions based on factor scores."""
        positions = pd.Series(0.0, index=factor_scores.index)
        
        if strategy.strategy_type == "long_short":
            # Long-short strategy
            n_stocks = len(factor_scores)
            n_long = int(n_stocks * strategy.top_percentile)
            n_short = int(n_stocks * strategy.bottom_percentile)
            
            # Sort by factor scores
            sorted_scores = factor_scores.sort_values(ascending=False)
            
            # Long positions (highest scores)
            long_stocks = sorted_scores.head(n_long).index
            positions[long_stocks] = 1.0 / n_long
            
            # Short positions (lowest scores)
            short_stocks = sorted_scores.tail(n_short).index
            positions[short_stocks] = -1.0 / n_short
            
        elif strategy.strategy_type == "long_only":
            # Long-only strategy
            n_stocks = len(factor_scores)
            n_long = int(n_stocks * strategy.top_percentile)
            
            # Sort by factor scores
            sorted_scores = factor_scores.sort_values(ascending=False)
            
            # Long positions only (highest scores)
            long_stocks = sorted_scores.head(n_long).index
            positions[long_stocks] = 1.0 / n_long
            
        elif strategy.strategy_type == "rank_based":
            # Position size based on factor rank
            ranks = factor_scores.rank(pct=True) - 0.5  # Center around 0
            
            # Scale positions by rank (with leverage constraint)
            positions = ranks * self.config.leverage / ranks.abs().sum()
        
        # Apply position size limits
        positions = positions.clip(-self.config.max_position_size, self.config.max_position_size)
        
        return positions
    
    def _get_benchmark_returns(self) -> pd.Series:
        """Get benchmark returns for comparison."""
        benchmark_data = self.data_provider.get_benchmark_data(
            benchmark=self.config.benchmark,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        return benchmark_data
    
    def get_results(self) -> Optional[Dict]:
        """Get the results of the last backtest run."""
        return self.results


def create_default_backtest(
    factors: Dict[str, float],
    universe: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31"
) -> Tuple[BacktestConfig, StrategyConfig]:
    """
    Create default backtest and strategy configurations.
    
    Args:
        factors: Dictionary of factor names and weights
        universe: List of stock symbols (default: S&P 500 sample)
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        Tuple of (BacktestConfig, StrategyConfig)
    """
    if universe is None:
        # Use a sample of large-cap stocks for demonstration
        universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JNJ', 'UNH', 'JPM', 'PG', 'HD', 'BAC', 'ABBV', 'PFE', 'KO',
            'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'WMT', 'CRM', 'LIN',
            'NFLX', 'ADBE', 'XOM', 'CVX', 'AMD', 'QCOM', 'ACN', 'MRK', 'TXN'
        ]
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        benchmark="SPY",
        rebalance_freq="M",
        transaction_cost=0.001,
        market_impact=0.0005,
        max_position_size=0.05,
        leverage=1.0,
        min_observations=252
    )
    
    strategy_config = StrategyConfig(
        factors=factors,
        strategy_type="long_short",
        top_percentile=0.2,
        bottom_percentile=0.2
    )
    
    return backtest_config, strategy_config