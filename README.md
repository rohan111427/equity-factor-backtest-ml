# Equity Factor Strategy Backtesting with ML

A comprehensive Python framework for equity factor research and backtesting with machine learning integration.

## 🚀 Features

- **Factor Library**: Pre-built momentum, value, quality, size, and volatility factors
- **Flexible Backtesting**: Support for long-short, long-only, and rank-based strategies
- **Performance Analytics**: Comprehensive risk and return metrics with professional visualizations
- **Data Integration**: Yahoo Finance integration with support for custom data sources
- **Transaction Costs**: Realistic modeling of transaction costs and market impact
- **Extensible Design**: Easy to add custom factors and strategies
- **Professional Reporting**: Generate publication-ready performance tearsheets

## 📦 Installation

### Prerequisites
- Python 3.9+
- Git

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/rohan111427/equity-factor-backtest-ml.git
cd equity-factor-backtest-ml
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the package in development mode:**
```bash
pip install -e .
```

4. **Verify installation:**
```bash
python examples/simple_momentum_strategy.py
```

## 🎯 Quick Start

### Basic Momentum Strategy

```python
from equity_backtesting.backtesting.engine import BacktestEngine, create_default_backtest
from equity_backtesting.analytics.performance import PerformanceAnalytics

# Define strategy
strategy_factors = {
    'momentum_3m': 0.6,  # 60% weight to 3-month momentum
    'momentum_6m': 0.4   # 40% weight to 6-month momentum
}

# Create configuration
backtest_config, strategy_config = create_default_backtest(
    factors=strategy_factors,
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Run backtest
engine = BacktestEngine(backtest_config)
results = engine.run_backtest(strategy_config)

# Analyze performance
analytics = PerformanceAnalytics(results)
performance_report = analytics.generate_performance_report()
print(performance_report)

# Create visualizations
analytics.plot_cumulative_returns()
analytics.plot_drawdown()
```

## 📊 Available Factors

### Momentum Factors
- `momentum_1m`: 1-month price momentum
- `momentum_3m`: 3-month price momentum
- `momentum_6m`: 6-month price momentum
- `momentum_12m`: 12-month price momentum

### Value Factors
- `value`: Price-based value proxy (simplified implementation)

### Quality Factors
- `quality`: Price stability-based quality measure

### Size Factors
- `size`: Market capitalization proxy

### Low Volatility Factors
- `low_volatility`: Realized volatility-based factor

## 🛠️ Advanced Usage

### Custom Factor Implementation

```python
from equity_backtesting.factors.library import BaseFactor
import pandas as pd
import numpy as np

class CustomMeanReversionFactor(BaseFactor):
    def __init__(self, lookback_periods: int = 20):
        super().__init__(f"mean_reversion_{lookback_periods}d", "mean_reversion")
        self.lookback_periods = lookback_periods
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        adj_close = data.xs('Adj Close', level=1, axis=1)
        
        # Calculate rolling mean
        rolling_mean = adj_close.rolling(window=self.lookback_periods).mean()
        
        # Mean reversion signal: negative of (price - rolling_mean) / rolling_mean
        mean_reversion = -(adj_close - rolling_mean) / rolling_mean
        
        return mean_reversion.dropna()

# Register custom factor
from equity_backtesting.factors.library import FactorLibrary
factor_lib = FactorLibrary()
custom_factor = CustomMeanReversionFactor(20)
factor_lib.register_factor(custom_factor)
```

### Multi-Factor Strategy

```python
# Define a balanced multi-factor strategy
balanced_strategy = {
    'momentum_3m': 0.25,      # 25% momentum
    'value': 0.25,            # 25% value
    'low_volatility': 0.25,   # 25% low volatility
    'quality': 0.25           # 25% quality
}

backtest_config, strategy_config = create_default_backtest(
    factors=balanced_strategy,
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Modify strategy parameters
strategy_config.strategy_type = "long_only"  # Long-only instead of long-short
strategy_config.top_percentile = 0.1         # Top 10% of stocks

engine = BacktestEngine(backtest_config)
results = engine.run_backtest(strategy_config)
```

### Performance Analysis

```python
analytics = PerformanceAnalytics(results)

# Get detailed metrics
basic_metrics = analytics.calculate_basic_metrics()
relative_metrics = analytics.calculate_relative_metrics()

print(f"Sharpe Ratio: {basic_metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {basic_metrics['max_drawdown']:.1%}")
print(f"Information Ratio: {relative_metrics['information_ratio']:.2f}")

# Generate comprehensive tearsheet
analytics.generate_tearsheet(save_path="strategy_tearsheet.png")

# Create individual plots
fig1 = analytics.plot_cumulative_returns()
fig2 = analytics.plot_monthly_returns_heatmap()
fig3 = analytics.plot_risk_return_scatter()
```

## 📁 Project Structure

```
equity-factor-backtest-ml/
├── src/equity_backtesting/          # Main package
│   ├── data/                        # Data acquisition and processing
│   ├── factors/                     # Factor library and calculations
│   ├── backtesting/                 # Core backtesting engine
│   ├── analytics/                   # Performance analytics
│   └── utils/                       # Utilities (logging, caching)
├── notebooks/                       # Jupyter notebooks with examples
├── examples/                        # Python script examples
├── tests/                          # Unit tests
├── config/                         # Configuration files
└── data/                          # Data storage (created automatically)
```

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python -m pytest tests/ -v
```

Or run tests directly:

```bash
python tests/test_basic_functionality.py
```

## 📓 Examples and Tutorials

### Jupyter Notebooks
1. **[Equity Factor Backtesting Demo](notebooks/equity_factor_backtesting_demo.ipynb)**: Comprehensive tutorial covering all framework features

### Python Scripts
1. **[Simple Momentum Strategy](examples/simple_momentum_strategy.py)**: Basic momentum strategy example

## ⚙️ Configuration

The framework uses a configuration system for easy customization:

```python
from config.settings import settings, FACTOR_CATEGORIES

# View current settings
print(f"Default data provider: {settings.DEFAULT_DATA_PROVIDER}")
print(f"Transaction cost: {settings.DEFAULT_TRANSACTION_COST}")

# View available factor categories
print(f"Available factors: {FACTOR_CATEGORIES}")
```

## 🚦 Performance Considerations

- **Data Caching**: Market data is automatically cached to improve performance
- **Vectorized Operations**: All calculations use vectorized pandas/numpy operations
- **Memory Management**: Efficient handling of large datasets
- **Parallel Processing**: Multi-threaded data downloads where supported

## 🔧 Customization

### Adding New Data Sources

```python
from equity_backtesting.data.provider import DataProvider

class CustomDataProvider(DataProvider):
    def __init__(self, api_key):
        super().__init__(provider="custom", api_key=api_key)
    
    def _get_custom_data(self, symbols, start_date, end_date, fields):
        # Implement your custom data source logic
        pass
```

### Strategy Customization

```python
from equity_backtesting.backtesting.engine import StrategyConfig

# Long-only strategy with sector neutralization
strategy_config = StrategyConfig(
    factors={'momentum_3m': 1.0},
    strategy_type="long_only",
    top_percentile=0.15,
    neutralization="sector"  # Sector-neutral positions
)
```

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

**Basic Metrics:**
- Total Return, Annualized Return
- Volatility, Sharpe Ratio
- Maximum Drawdown, Calmar Ratio
- Sortino Ratio, Skewness, Kurtosis
- Value at Risk (VaR), Conditional VaR

**Relative Metrics (vs Benchmark):**
- Alpha, Beta, Correlation
- Tracking Error, Information Ratio
- Up/Down Capture Ratios
- Excess Return

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure tests pass: `python -m pytest`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Format code
black src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the documentation in the notebooks/
- Review example scripts in examples/

## 🔮 Future Enhancements

- [ ] Machine learning factor selection and optimization
- [ ] Real-time portfolio tracking and alerts
- [ ] Additional asset classes (bonds, commodities)
- [ ] Advanced risk models (Fama-French, Barra)
- [ ] Web-based dashboard interface
- [ ] Database integration (PostgreSQL, MongoDB)
- [ ] Alternative data sources integration
- [ ] Options and derivatives support

## 📚 References

- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model.
- Ang, A. (2014). Asset Management: A Systematic Approach to Factor Investing.
- Chincarini, L. B., & Kim, D. (2006). Quantitative Equity Portfolio Management.

---

**Disclaimer**: This software is for educational and research purposes only. It is not intended as investment advice. Past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.
