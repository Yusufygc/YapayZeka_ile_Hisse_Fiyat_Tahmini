# -*- coding: utf-8 -*-
"""
data_splitter.py — Strict Time Series Splitter.
Prevents data leakage by ensuring train bounds strictly precede test bounds.
"""

import pandas as pd
from typing import Tuple, List, Dict


# P1 derin denetim (2026-11): WF embargo varsayilani `target_horizon + 5`.
# Eski varsayilan `max(200, time_steps)` "SMA200 sizintisi" gerekcesiyle konmustu;
# bu yanlis: ozellikler nedensel (causal) ve test penceresi etiketsiz baglam satirlari
# (`embargo_context` + train kuyrugu) alir. Yalnizca ETIKET ortusmesi (ufuk h) sizdirir:
# train son satirinin hedefi t+h kapanisini kullanir -> en az h satir tampon gerekir.
# +5 guvenlik payi pooled CV'deki E=h+5 ile ayni.
# Bu helper data_manager.py'dan buraya tasindi ki agir import zincirleri
# (joblib, tensorflow vb.) olmayan test ortamlarinda da import edilebilsin.
_AUTO_EMBARGO_BUFFER = 5
# Kullanimdan kalkti (eski 200 tamponu). data_manager.py hala import ettigi icin
# uyumluluk adina tutuluyor; embargo tabani artik target_horizon'dur.
_MIN_AUTO_EMBARGO_SIZE = 1


def _validate_wf_embargo(embargo_size: int, target_horizon: int) -> int:
    """Explicit embargo < target_horizon etiket ortusmesi (sizinti) yaratir -> ValueError."""
    h = max(1, int(target_horizon or 1))
    if int(embargo_size) < h:
        raise ValueError(
            f"wf_embargo_size={int(embargo_size)} < target_horizon={h}: train son "
            f"satirlarinin t+{h} hedefi test doneminin fiyatlarini kullanir (etiket "
            f"ortusmesi). En az {h} (onerilen {h + _AUTO_EMBARGO_BUFFER}) olmali."
        )
    return int(embargo_size)


def _resolve_wf_embargo_size(raw_value, time_steps: int, target_horizon: int = 1) -> int:
    """None/0/negatif/gecersiz -> auto `target_horizon + 5`; explicit deger < h -> ValueError.

    ``time_steps`` artik embargoyu etkilemez (sequence baglami embargodan degil,
    train kuyruğu + embargo satirlarindan gelir); imza geriye uyum icin korunur.
    """
    h = max(1, int(target_horizon or 1))
    auto = h + _AUTO_EMBARGO_BUFFER
    if raw_value is None:
        return auto
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return auto
    if value <= 0:
        return auto
    return _validate_wf_embargo(value, h)


def _resolve_split_count(
    n: int, n_splits: int, min_train_size: int, test_size: int, embargo_size: int
) -> int:
    """Veri yetmezse ``n_splits``'i mümkün olan en yükseğe indirir.

    Hiç geçerli pencere kurulamıyorsa 0 döner. Davranış orijinaldekiyle aynı —
    sadece yetersizlik dalı ana döngüden ayrıldı (karmaşıklık azaltma).
    """
    min_required = min_train_size + embargo_size + (n_splits * test_size)
    if n >= min_required:
        return n_splits
    print(
        f"[WARNING] Not enough data for {n_splits} splits with test_size={test_size} "
        f"and min_train_size={min_train_size}."
    )
    max_possible_splits = (n - min_train_size - embargo_size) // test_size
    if max_possible_splits < 1:
        print(
            "[WARNING] No valid walk-forward split can be created "
            f"(rows={n}, required_for_one_split={min_train_size + embargo_size + test_size})."
        )
        return 0
    adjusted = min(n_splits, max_possible_splits)
    print(f"[WARNING] Adjusted n_splits to {adjusted}.")
    return adjusted


def _window_bounds(
    n: int, i: int, test_size: int, embargo_size: int, max_train_size: int | None
) -> Tuple[int, int, int, int]:
    """Sondan ``i``'inci pencerenin (train_start, train_end, test_start, test_end)
    indekslerini döner. ``max_train_size`` None → expanding, int → sliding window."""
    test_start = n - (i * test_size)
    train_end = max(0, test_start - embargo_size)
    test_end = test_start + test_size
    if max_train_size is not None:
        train_start = max(0, train_end - max_train_size)  # sliding: son N satır
    else:
        train_start = 0  # expanding: tüm geçmiş
    return train_start, train_end, test_start, test_end


def _first_last_date(frame: pd.DataFrame):
    """(ilk, son) Date değeri; Date kolonu yoksa ya da frame boşsa (None, None)."""
    if "Date" not in frame.columns or frame.empty:
        return None, None
    return frame["Date"].iloc[0], frame["Date"].iloc[-1]


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
        target_horizon: int | None    = None,
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
            embargo_size   : train ile test arasindaki tampon (satir).
            target_horizon : verilirse ``embargo_size < target_horizon`` ValueError
                             (etiket ortusmesi sizintisi).  Test penceresinin sequence
                             baglami ``embargo_context`` (+ train kuyrugu) ile saglanir;
                             context satirlari yalniz ozellik olarak kullanilir.
        """
        if "Date" in df.columns:
            df = df.sort_values(by="Date").reset_index(drop=True)

        n = len(df)
        embargo_size = max(0, int(embargo_size))
        if target_horizon is not None:
            _validate_wf_embargo(embargo_size, target_horizon)
        n_splits = _resolve_split_count(n, n_splits, min_train_size, test_size, embargo_size)
        if n_splits < 1:
            return []

        splits = []
        for i in range(n_splits, 0, -1):
            train_start, train_end, test_start, test_end = _window_bounds(
                n, i, test_size, embargo_size, max_train_size
            )
            embargo_start, embargo_end = train_end, test_start

            train_df = df.iloc[train_start:train_end].copy()
            embargo_df = df.iloc[embargo_start:embargo_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            if len(train_df) < min_train_size or len(test_df) < test_size:
                continue

            train_date_start, train_date_end = _first_last_date(train_df)
            embargo_date_start, embargo_date_end = _first_last_date(embargo_df)
            test_date_start, test_date_end = _first_last_date(test_df)

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
                "train_date_start": train_date_start,
                "train_date_end": train_date_end,
                "embargo_date_start": embargo_date_start,
                "embargo_date_end": embargo_date_end,
                "test_date_start": test_date_start,
                "test_date_end": test_date_end,
            })

        return splits
