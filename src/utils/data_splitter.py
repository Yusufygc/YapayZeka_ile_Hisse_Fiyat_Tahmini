# -*- coding: utf-8 -*-
"""
data_splitter.py — Strict Time Series Splitter.
Prevents data leakage by ensuring train bounds strictly precede test bounds.
"""

import pandas as pd
from typing import Tuple, List, Dict

class TimeSeriesSplitter:
    """
    Handles robust train/test splitting for time series to prevent data leakage.
    Provides methods for both a single hold-out split and walk-forward rolling window splits.
    """
    
    @staticmethod
    def single_split(df: pd.DataFrame, target_col: str = "Close", test_ratio: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data chronologically into a single train and test set.
        """
        # Ensure data is sorted by date
        if "Date" in df.columns:
            df = df.sort_values(by="Date").reset_index(drop=True)
            
        n = len(df)
        train_size = int(n * (1 - test_ratio))
        
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")
            
        y_train = train_df[target_col].copy()
        y_test = test_df[target_col].copy()
        
        return train_df, test_df, y_train, y_test

    @staticmethod
    def walk_forward_splits(df: pd.DataFrame, n_splits: int = 3, min_train_size: int = 100, test_size: int = 30) -> List[Dict]:
        """
        Creates multiple chronological train/test splits for walk-forward validation.
        Yields a list of dictionaries with 'train' and 'test' DataFrames.
        """
        if "Date" in df.columns:
            df = df.sort_values(by="Date").reset_index(drop=True)
            
        n = len(df)
        splits = []
        
        # We start from the end and work backwards to create n_splits.
        # Alternatively, we can start from n - (n_splits * test_size)
        total_test_size = n_splits * test_size
        if n < min_train_size + total_test_size:
            print(f"[WARNING] Not enough data for {n_splits} splits with test_size={test_size} and min_train_size={min_train_size}.")
            # Adjust n_splits
            n_splits = max(1, (n - min_train_size) // test_size)
            print(f"[WARNING] Adjusted n_splits to {n_splits}.")
        
        for i in range(n_splits, 0, -1):
            train_end = n - (i * test_size)
            test_end = train_end + test_size
            
            # Use rolling window or expanding window for train
            # Using expanding window here (from 0 to train_end)
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            splits.append({
                "split_idx": n_splits - i + 1,
                "train": train_df,
                "test": test_df
            })
            
        return splits
