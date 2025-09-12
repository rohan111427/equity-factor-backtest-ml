"""
Caching utilities for data storage and retrieval.
"""

import os
import pickle
import hashlib
from functools import wraps
from typing import Any, Callable
import pandas as pd


def cache_data(cache_dir: str = "data/cache", expire_hours: int = 24):
    """
    Decorator to cache function results to disk.
    
    Args:
        cache_dir: Directory to store cache files
        expire_hours: Hours after which cache expires
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if cache file exists and is not expired
            if os.path.exists(cache_file):
                file_age = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache_file), unit='s'))
                if file_age.total_seconds() < expire_hours * 3600:
                    try:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                    except:
                        pass  # Fall through to recalculate
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass  # Don't fail if we can't cache
                
            return result
        return wrapper
    return decorator


def clear_cache(cache_dir: str = "data/cache"):
    """Clear all cache files."""
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(cache_dir, file))