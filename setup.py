from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="equity-factor-backtesting",
    version="0.1.0",
    author="Rohan",
    author_email="rohan@example.com",
    description="A comprehensive equity factor backtesting framework with machine learning integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohan111427/equity-factor-backtest-ml",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "viz": ["dash", "plotly"],
        "ml": ["scikit-learn", "xgboost"],
    },
)
