#!/usr/bin/env python3
"""
Simple example: Momentum strategy backtesting

This script demonstrates how to:
1. Create a simple momentum strategy
2. Run a backtest
3. Display basic performance metrics
"""

import sys
import os

# Add src to path if running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from equity_backtesting.backtesting.engine import BacktestEngine, create_default_backtest
from equity_backtesting.analytics.performance import PerformanceAnalytics


def run_momentum_strategy():
    """Run a simple momentum strategy example."""
    
    print("Equity Factor Backtesting - Momentum Strategy Example")
    print("=" * 55)
    
    # 1. Define strategy
    strategy_factors = {
        'momentum_3m': 0.6,  # 60% weight to 3-month momentum
        'momentum_6m': 0.4   # 40% weight to 6-month momentum
    }
    
    print(f"Strategy factors: {strategy_factors}")
    
    # 2. Create backtest configuration
    backtest_config, strategy_config = create_default_backtest(
        factors=strategy_factors,
        start_date="2021-01-01",
        end_date="2023-12-31"
    )
    
    print(f"Universe size: {len(backtest_config.universe)} stocks")
    print(f"Period: {backtest_config.start_date} to {backtest_config.end_date}")
    print(f"Rebalancing: {backtest_config.rebalance_freq}")
    
    # 3. Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(backtest_config)
    
    try:
        results = engine.run_backtest(strategy_config)
        print("✓ Backtest completed successfully!")
        
        # 4. Analyze performance
        print("\nPerformance Analysis:")
        print("-" * 30)
        
        analytics = PerformanceAnalytics(results)
        
        # Get basic metrics
        basic_metrics = analytics.calculate_basic_metrics()
        
        # Display key metrics
        print(f"Total Return:     {basic_metrics.get('total_return', 0):.1%}")
        print(f"Annual Return:    {basic_metrics.get('annualized_return', 0):.1%}")
        print(f"Annual Vol:       {basic_metrics.get('annualized_volatility', 0):.1%}")
        print(f"Sharpe Ratio:     {basic_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:     {basic_metrics.get('max_drawdown', 0):.1%}")
        print(f"Calmar Ratio:     {basic_metrics.get('calmar_ratio', 0):.2f}")
        
        # Relative metrics if available
        relative_metrics = analytics.calculate_relative_metrics()
        if relative_metrics:
            print(f"\nVs Benchmark:")
            print(f"Alpha:            {relative_metrics.get('alpha', 0):.1%}")
            print(f"Beta:             {relative_metrics.get('beta', 0):.2f}")
            print(f"Info Ratio:       {relative_metrics.get('information_ratio', 0):.2f}")
        
        # Portfolio statistics
        returns = results['portfolio_returns']
        print(f"\nPortfolio Stats:")
        print(f"Trading days:     {len(returns)}")
        print(f"Winning days:     {(returns > 0).sum()} ({(returns > 0).mean():.1%})")
        print(f"Average return:   {returns.mean():.3%}")
        
        print(f"\nTransaction Costs:")
        avg_turnover = sum(results['turnover']) / len(results['turnover']) if results['turnover'] else 0
        avg_costs = sum(results['transaction_costs']) / len(results['transaction_costs']) if results['transaction_costs'] else 0
        print(f"Avg turnover:     {avg_turnover:.1%}")
        print(f"Avg costs:        {avg_costs:.3%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtest failed: {str(e)}")
        return False


def main():
    """Main function."""
    success = run_momentum_strategy()
    
    if success:
        print("\n" + "=" * 55)
        print("Example completed successfully!")
        print("\nNext steps:")
        print("- Check out notebooks/equity_factor_backtesting_demo.ipynb")
        print("- Experiment with different factor combinations")
        print("- Try different time periods and universes")
    else:
        print("Example failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())