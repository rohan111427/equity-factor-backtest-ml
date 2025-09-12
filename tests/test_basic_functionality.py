"""
Basic unit tests for the equity factor backtesting framework.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from equity_backtesting.data.provider import DataProvider
from equity_backtesting.factors.library import FactorLibrary, MomentumFactor
from equity_backtesting.backtesting.engine import BacktestConfig, StrategyConfig


class TestDataProvider(unittest.TestCase):
    """Test DataProvider functionality."""
    
    def setUp(self):
        self.provider = DataProvider()
    
    def test_data_provider_initialization(self):
        """Test DataProvider initializes correctly."""
        self.assertEqual(self.provider.provider, "yfinance")
        self.assertIsNone(self.provider.api_key)
    
    def test_get_sp500_symbols(self):
        """Test S&P 500 symbol retrieval."""
        try:
            symbols = self.provider.get_sp500_symbols()
            self.assertIsInstance(symbols, list)
            self.assertGreater(len(symbols), 100)  # Should have many symbols
            self.assertIn('AAPL', symbols)  # Apple should be there
        except Exception:
            # If web scraping fails, should return fallback
            symbols = self.provider.get_sp500_symbols()
            self.assertIsInstance(symbols, list)
            self.assertGreater(len(symbols), 10)


class TestFactorLibrary(unittest.TestCase):
    """Test FactorLibrary functionality."""
    
    def setUp(self):
        self.factor_lib = FactorLibrary()
    
    def test_factor_library_initialization(self):
        """Test FactorLibrary initializes with default factors."""
        factors = self.factor_lib.list_factors()
        self.assertIsInstance(factors, list)
        self.assertGreater(len(factors), 5)
        
        # Check for expected factors
        self.assertIn('momentum_3m', factors)
        self.assertIn('value', factors)
        self.assertIn('low_volatility', factors)
    
    def test_get_factors_by_category(self):
        """Test getting factors by category."""
        momentum_factors = self.factor_lib.get_factors_by_category('momentum')
        self.assertIsInstance(momentum_factors, list)
        self.assertGreater(len(momentum_factors), 0)
        
        # All returned factors should contain 'momentum' in name
        for factor in momentum_factors:
            self.assertIn('momentum', factor)
    
    def test_get_factor(self):
        """Test getting individual factors."""
        factor = self.factor_lib.get_factor('momentum_3m')
        self.assertIsInstance(factor, MomentumFactor)
        self.assertEqual(factor.category, 'momentum')
        
        # Test invalid factor
        with self.assertRaises(ValueError):
            self.factor_lib.get_factor('invalid_factor')
    
    def test_factor_calculation_with_mock_data(self):
        """Test factor calculation with mock data."""
        # Create mock market data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create mock price data that increases over time
        price_data = {}
        for symbol in symbols:
            # Create trending prices with some randomness
            base_price = 100 + np.random.random() * 50
            trend = np.linspace(0, 20, 100)  # Upward trend
            noise = np.random.normal(0, 2, 100)  # Random noise
            prices = base_price + trend + noise
            
            for field in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                if field == 'Volume':
                    price_data[(symbol, field)] = np.random.randint(1000000, 5000000, 100)
                else:
                    price_data[(symbol, field)] = prices * (1 + np.random.normal(0, 0.01, 100))
        
        # Create DataFrame with MultiIndex columns
        columns = pd.MultiIndex.from_tuples(price_data.keys())
        mock_data = pd.DataFrame(price_data, index=dates, columns=columns)
        
        # Test momentum factor calculation
        momentum_factor = self.factor_lib.calculate_factor('momentum_3m', mock_data)
        
        self.assertIsInstance(momentum_factor, pd.DataFrame)
        self.assertEqual(list(momentum_factor.columns), symbols)
        
        # Should have fewer rows due to lookback period
        self.assertLess(len(momentum_factor), len(mock_data))


class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig and StrategyConfig."""
    
    def test_backtest_config_creation(self):
        """Test BacktestConfig creation."""
        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2023-12-31",
            universe=['AAPL', 'MSFT', 'GOOGL']
        )
        
        self.assertEqual(config.start_date, "2020-01-01")
        self.assertEqual(config.end_date, "2023-12-31")
        self.assertEqual(len(config.universe), 3)
        self.assertEqual(config.benchmark, "SPY")  # Default
        self.assertEqual(config.rebalance_freq, "M")  # Default
    
    def test_strategy_config_creation(self):
        """Test StrategyConfig creation."""
        factors = {'momentum_3m': 0.5, 'value': 0.5}
        config = StrategyConfig(factors=factors)
        
        self.assertEqual(config.factors, factors)
        self.assertEqual(config.strategy_type, "long_short")  # Default
        self.assertEqual(config.top_percentile, 0.2)  # Default


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        factor_lib = FactorLibrary()
        
        # Empty DataFrame
        empty_data = pd.DataFrame()
        
        # Should handle gracefully
        try:
            result = factor_lib.calculate_factor('momentum_3m', empty_data)
            # If it doesn't raise an error, result should be empty or None-like
            self.assertTrue(result.empty if hasattr(result, 'empty') else result is None)
        except Exception as e:
            # It's ok if it raises an appropriate error
            self.assertIsInstance(e, (ValueError, KeyError, IndexError))
    
    def test_invalid_date_ranges(self):
        """Test invalid date range handling."""
        with self.assertRaises(Exception):
            # End date before start date should cause issues somewhere
            config = BacktestConfig(
                start_date="2023-12-31",
                end_date="2020-01-01",  # Before start date
                universe=['AAPL']
            )


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataProvider))
    test_suite.addTest(unittest.makeSuite(TestFactorLibrary))
    test_suite.addTest(unittest.makeSuite(TestBacktestConfig))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)