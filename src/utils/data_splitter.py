# -*- coding: utf-8 -*-
"""
data_splitter.py — Strict Time Series Splitter.
Prevents data leakage by ensuring train bounds strictly precede test bounds.
"""

import pandas as pd
from typing import Tuple, List, Dict


# Sprint 0 (2026-05-25): WF embargo auto-default. None veya 0 verilirse
# `max(200, time_steps)` kullanilir. Sebep: Market_Regime_SMA200 ve diger
# rolling-200 feature'lar train/test arasinda sizinti yaratir; tampon en az
# 200 olmalidir. Bu helper data_manager.py'dan buraya tasindi ki agir
# import zincirleri (joblib, tensorflow vb.) olmayan test ortamlarinda da
# import edilebilsin.
_MIN_AUTO_EMBARGO_SIZE = 200


def _resolve_wf_embargo_size(raw_value, time_steps: int) -> int:
    """Plan v1.0 Sprint 0 A0.2: None/0/negative → auto max(200, time_steps)."""
    if raw_value is None:
        return max(_MIN_AUTO_EMBARGO_SIZE, int(time_steps))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return max(_MIN_AUTO_EMBARGO_SIZE, int(time_steps))
    if value <= 0:
        return max(_MIN_AUTO_EMBARGO_SIZE, int(time_steps))
    return value

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
    def walk_forward_splits(
        df:             pd.DataFrame,
        n_splits:       int           = 3,
        min_train_size: int           = 100,
        test_size:      int           = 30,
        max_train_size: int | None    = None,
        embargo_size:   int           = 0,
    ) -> List[Dict]:
        """
        Creates multiple chronological train/test splits for walk-forward validation.

        Args:
            df             : Tam veri seti (Date sütunu varsa kronolojik sıralama yapılır).
            n_splits       : Kaç pencere oluşturulacağı.
            min_train_size : Eğitim setinin minimum uzunluğu.
            test_size      : Her pencerenin test uzunluğu (gün sayısı).
            max_train_size : None → expanding window (tüm geçmiş kullanılır).
                             int  → sliding window: her pencerede yalnızca
                                    son ``max_train_size`` satır eğitim için
                                    kullanılır.  Durağan olmayan fiyat serilerinde
                                    (örn. BIST hisseleri) systematic bias'ı önler.
        """
        if "Date" in df.columns:
            df = df.sort_values(by="Date").reset_index(drop=True)

        n = len(df)
        splits = []
        embargo_size = max(0, int(embargo_size))

        total_test_size = n_splits * test_size
        min_required = min_train_size + embargo_size + total_test_size
        if n < min_required:
            print(f"[WARNING] Not enough data for {n_splits} splits with test_size={test_size} and min_train_size={min_train_size}.")
            max_possible_splits = (n - min_train_size - embargo_size) // test_size
            if max_possible_splits < 1:
                print(
                    "[WARNING] No valid walk-forward split can be created "
                    f"(rows={n}, required_for_one_split={min_train_size + embargo_size + test_size})."
                )
                return []
            n_splits = min(n_splits, max_possible_splits)
            print(f"[WARNING] Adjusted n_splits to {n_splits}.")

        for i in range(n_splits, 0, -1):
            test_start = n - (i * test_size)
            train_end = max(0, test_start - embargo_size)
            embargo_start = train_end
            embargo_end = test_start
            test_end = test_start + test_size

            if max_train_size is not None:
                # Sliding window: yalnızca son max_train_size satırı kullan
                train_start = max(0, train_end - max_train_size)
            else:
                # Expanding window: 0'dan train_end'e kadar tüm geçmiş
                train_start = 0

            train_df = df.iloc[train_start:train_end].copy()
            embargo_df = df.iloc[embargo_start:embargo_end].copy()
            test_df  = df.iloc[test_start:test_end].copy()
            if len(train_df) < min_train_size or len(test_df) < test_size:
                continue

            splits.append({
                "split_idx":   n_splits - i + 1,
                "train":       train_df,
                "embargo_context": embargo_df,
                "test":        test_df,
                "train_start": train_start,
                "train_end":   train_end,
                "effective_train_end": train_end,
                "embargo_start": embargo_start,
                "embargo_end": embargo_end,
                "test_start": test_start,
                "test_end":    test_end,
                "embargo_size": embargo_size,
                "train_date_start": train_df["Date"].iloc[0] if "Date" in train_df.columns and not train_df.empty else None,
                "train_date_end": train_df["Date"].iloc[-1] if "Date" in train_df.columns and not train_df.empty else None,
                "embargo_date_start": embargo_df["Date"].iloc[0] if "Date" in embargo_df.columns and not embargo_df.empty else None,
                "embargo_date_end": embargo_df["Date"].iloc[-1] if "Date" in embargo_df.columns and not embargo_df.empty else None,
                "test_date_start": test_df["Date"].iloc[0] if "Date" in test_df.columns and not test_df.empty else None,
                "test_date_end": test_df["Date"].iloc[-1] if "Date" in test_df.columns and not test_df.empty else None,
            })

        return splits
