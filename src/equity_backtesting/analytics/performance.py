"""
Performance analytics module for calculating risk and return metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceAnalytics:
    """
    Performance analytics for equity factor backtests.
    """
    
    def __init__(self, backtest_results: Dict):
        """
        Initialize with backtest results.
        
        Args:
            backtest_results: Results dictionary from BacktestEngine
        """
        self.results = backtest_results
        self.portfolio_returns = backtest_results['portfolio_returns']
        self.benchmark_returns = backtest_results['benchmark_returns']
        
        # Align returns by date
        self._align_returns()
        
        logger.info("Initialized PerformanceAnalytics")
    
    def _align_returns(self):
        """Align portfolio and benchmark returns by date."""
        if isinstance(self.benchmark_returns, pd.Series):
            # Find common dates
            common_dates = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
            
            if len(common_dates) > 0:
                self.portfolio_returns = self.portfolio_returns.loc[common_dates]
                self.benchmark_returns = self.benchmark_returns.loc[common_dates]
            else:
                logger.warning("No common dates found between portfolio and benchmark returns")
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Returns:
            Dictionary of basic metrics
        """
        returns = self.portfolio_returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Annualization factor (assume daily returns)
        trading_days = 252
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': (1 + returns.mean()) ** trading_days - 1,
            'annualized_volatility': returns.std() * np.sqrt(trading_days),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(trading_days) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': (returns.mean() * trading_days) / abs(self._calculate_max_drawdown(returns)) if self._calculate_max_drawdown(returns) != 0 else 0,
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'hit_rate': (returns > 0).mean(),
            'average_win': returns[returns > 0].mean(),
            'average_loss': returns[returns < 0].mean(),
            'win_loss_ratio': returns[returns > 0].mean() / abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else np.inf
        }
        
        return metrics
    
    def calculate_relative_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics relative to benchmark.
        
        Returns:
            Dictionary of relative metrics
        """
        if self.benchmark_returns is None or len(self.benchmark_returns) == 0:
            logger.warning("No benchmark data available for relative metrics")
            return {}
        
        portfolio_returns = self.portfolio_returns.dropna()
        benchmark_returns = self.benchmark_returns.reindex(portfolio_returns.index).dropna()
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        if len(excess_returns) == 0:
            return {}
        
        trading_days = 252
        
        # Beta calculation
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_var = benchmark_returns.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Jensen's Alpha
        risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
        alpha = (portfolio_returns.mean() - risk_free_rate) - beta * (benchmark_returns.mean() - risk_free_rate)
        
        metrics = {
            'beta': beta,
            'alpha': alpha * trading_days,  # Annualized
            'correlation': portfolio_returns.corr(benchmark_returns),
            'tracking_error': excess_returns.std() * np.sqrt(trading_days),
            'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days) if excess_returns.std() > 0 else 0,
            'excess_return': excess_returns.mean() * trading_days,
            'up_capture': self._calculate_capture_ratio(portfolio_returns, benchmark_returns, up=True),
            'down_capture': self._calculate_capture_ratio(portfolio_returns, benchmark_returns, up=False)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (return/downside deviation)."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        annualized_return = returns.mean() * 252
        
        return annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_capture_ratio(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series, 
        up: bool = True
    ) -> float:
        """Calculate up/down capture ratio."""
        if up:
            condition = benchmark_returns > 0
        else:
            condition = benchmark_returns < 0
        
        portfolio_subset = portfolio_returns[condition]
        benchmark_subset = benchmark_returns[condition]
        
        if len(portfolio_subset) == 0 or len(benchmark_subset) == 0:
            return 0
        
        portfolio_capture = portfolio_subset.mean()
        benchmark_capture = benchmark_subset.mean()
        
        return portfolio_capture / benchmark_capture if benchmark_capture != 0 else 0
    
    def generate_performance_report(self) -> pd.DataFrame:
        """
        Generate comprehensive performance report.
        
        Returns:
            DataFrame with all performance metrics
        """
        basic_metrics = self.calculate_basic_metrics()
        relative_metrics = self.calculate_relative_metrics()
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **relative_metrics}
        
        # Convert to DataFrame for nice formatting
        report = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Value'])
        report.index.name = 'Metric'
        
        # Format certain values as percentages
        percentage_metrics = [
            'total_return', 'annualized_return', 'annualized_volatility', 
            'max_drawdown', 'excess_return', 'tracking_error', 'alpha'
        ]
        
        for metric in percentage_metrics:
            if metric in report.index:
                report.loc[metric, 'Formatted'] = f"{report.loc[metric, 'Value']:.2%}"
        
        return report
    
    def plot_cumulative_returns(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot cumulative returns vs benchmark.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        
        ax.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                label='Strategy', linewidth=2, color='blue')
        
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='Benchmark', linewidth=2, color='red', alpha=0.7)
        
        ax.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y-1)))
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot drawdown over time.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate drawdown
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        
        ax.set_title('Portfolio Drawdown Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        # Resample to monthly returns
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns.name = 'returns'
        
        pivot_table = monthly_returns.to_frame().assign(
            year=monthly_returns.index.year,
            month=monthly_returns.index.month_name()
        ).pivot(index='year', columns='month', values='returns')
        
        # Reorder months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        pivot_table = pivot_table.reindex(columns=[m for m in month_order if m in pivot_table.columns])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
        
        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_return_scatter(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot risk-return scatter (annual return vs volatility).
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate annual metrics
        basic_metrics = self.calculate_basic_metrics()
        portfolio_ret = basic_metrics.get('annualized_return', 0)
        portfolio_vol = basic_metrics.get('annualized_volatility', 0)
        
        # Plot portfolio
        ax.scatter(portfolio_vol, portfolio_ret, s=200, color='blue', 
                  label='Strategy', alpha=0.8, edgecolors='black')
        
        # Plot benchmark if available
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            benchmark_ret = self.benchmark_returns.mean() * 252
            benchmark_vol = self.benchmark_returns.std() * np.sqrt(252)
            
            ax.scatter(benchmark_vol, benchmark_ret, s=200, color='red', 
                      label='Benchmark', alpha=0.8, edgecolors='black')
        
        ax.set_title('Risk-Return Profile', fontsize=16, fontweight='bold')
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format axes as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        return fig
    
    def analyze_factor_attribution(self) -> pd.DataFrame:
        """
        Analyze factor attribution of returns.
        
        Returns:
            DataFrame with factor attribution analysis
        """
        if 'factor_data' not in self.results:
            logger.warning("No factor data available for attribution analysis")
            return pd.DataFrame()
        
        factor_data = self.results['factor_data']
        strategy_config = self.results['strategy']
        
        # This is a simplified attribution analysis
        # In practice, you'd want more sophisticated factor attribution
        attribution = pd.DataFrame({
            'Factor': list(strategy_config.factors.keys()),
            'Weight': list(strategy_config.factors.values()),
            'Description': ['Factor contribution to strategy'] * len(strategy_config.factors)
        })
        
        return attribution
    
    def generate_tearsheet(self, save_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive performance tearsheet.
        
        Args:
            save_path: Path to save the tearsheet (optional)
        """
        # Create subplots
        fig = plt.figure(figsize=(16, 20))
        
        # 1. Cumulative returns
        ax1 = plt.subplot(4, 2, 1)
        self.plot_cumulative_returns()
        
        # 2. Drawdown
        ax2 = plt.subplot(4, 2, 2)
        self.plot_drawdown()
        
        # 3. Monthly returns heatmap
        ax3 = plt.subplot(4, 1, 3)
        self.plot_monthly_returns_heatmap()
        
        # 4. Risk-return scatter
        ax4 = plt.subplot(4, 2, 7)
        self.plot_risk_return_scatter()
        
        plt.suptitle('Strategy Performance Tearsheet', fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tearsheet saved to {save_path}")
        
        plt.show()