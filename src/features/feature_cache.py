# -*- coding: utf-8 -*-
"""
feature_cache.py - Ozellik muhendisligi pickle/parquet cache katmani (Faz 2.4).

Motivasyon:
  FeaturePipeline + MacroPipeline her calistirmada ~2-10 sn alir.
  Ayni veri dosyasi + konfigurasyonu ile tekrar eden calistirmalarda
  bu maliyeti ortadan kaldirmak icin disk tabanli bir cache kullanilir.

Cache anahtari (MD5):
  data_file absolutepath + mtime + feature_mode + use_macro +
  macro_rate_lag_days + macro_cpi_lag_days + prune_correlated_features +
  correlation_threshold + lag_feature_count + training_window_years

Cache formati:
  {key}.pkl   -- (df, meta) tuple pickle dosyasi
  (pyarrow kurulu ortamlarda gelecekte .parquet'e gec gecilebilir)

TTL:
  Varsayilan 24 saat. Veri dosyasinin mtime'i degisirse hash farklilasir
  ve eski cache kaydi otomatik olarak gozardi edilir.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class FeatureCache:
    """
    Disk tabanli ozellik muhendisligi cache'i.

    Kullanim:
        cache = FeatureCache(cache_dir="data/feature_cache")
        key   = cache.make_key(data_file, data_cfg)
        hit   = cache.get(key)
        if hit is not None:
            df, meta = hit
        else:
            df, meta = _run_feature_engineering(...)
            cache.put(key, df, meta)
    """

    _VERSION = "1"  # Bump to invalidate all existing cache entries.

    def __init__(self, cache_dir: str, ttl_hours: float = 24.0) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600.0
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Key construction                                                    #
    # ------------------------------------------------------------------ #

    def make_key(self, data_file: str, data_cfg: Any) -> str:
        """
        Konfigurasyona gore deterministik bir MD5 cache anahtari olusturur.

        Parameters
        ----------
        data_file : str
            CSV/veri dosyasinin yolu (mutlak hale cevirilir).
        data_cfg  : DataConfig veya benzer nesneler; su alanlari okunur:
            feature_mode, use_macro, macro_rate_lag_days, macro_cpi_lag_days,
            prune_correlated_features, correlation_threshold, lag_feature_count,
            training_window_years
        """
        abs_path = os.path.abspath(data_file)
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0

        fingerprint = {
            "_cache_version": self._VERSION,
            "abs_path": abs_path,
            "mtime": round(mtime, 3),
            "feature_mode": getattr(data_cfg, "feature_mode", "stationary_features"),
            "use_macro": bool(getattr(data_cfg, "use_macro", True)),
            "macro_rate_lag_days": int(getattr(data_cfg, "macro_rate_lag_days", 1)),
            "macro_cpi_lag_days": int(getattr(data_cfg, "macro_cpi_lag_days", 15)),
            "prune_correlated_features": bool(getattr(data_cfg, "prune_correlated_features", False)),
            "correlation_threshold": float(getattr(data_cfg, "correlation_threshold", 0.98)),
            "lag_feature_count": int(getattr(data_cfg, "lag_feature_count", 5)),
            "training_window_years": (
                int(data_cfg.training_window_years)
                if getattr(data_cfg, "training_window_years", None) is not None
                else None
            ),
        }
        raw = json.dumps(fingerprint, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def _pkl_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def get(self, key: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Cache'den oku.

        Returns
        -------
        (df, meta) tuple'i yoksa None.
        meta su alanlari icerir: feature_names, feature_groups, feature_pruning_report.
        """
        pkl_path = self._pkl_path(key)
        if not os.path.exists(pkl_path):
            return None

        # TTL kontrolu (ttl_seconds < 0 ise TTL devre disi; 0 ise her zaman suresi dolmus)
        if self.ttl_seconds >= 0:
            age = time.time() - os.path.getmtime(pkl_path)
            if age > self.ttl_seconds:
                self._evict(key)
                return None

        try:
            with open(pkl_path, "rb") as fh:
                df, meta = pickle.load(fh)
            if not isinstance(df, pd.DataFrame) or not isinstance(meta, dict):
                raise ValueError("Beklenmeyen cache formati")
        except Exception:
            self._evict(key)
            return None

        return df, meta

    def put(self, key: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        """
        Cache'e yaz. Yazma basarisiz olursa sessizce devam et.
        """
        pkl_path = self._pkl_path(key)
        try:
            with open(pkl_path, "wb") as fh:
                pickle.dump((df, meta), fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            self._evict(key)
            print(f"  [CACHE] Yazma hatasi, cache atlanacak: {exc}")

    # ------------------------------------------------------------------ #
    #  Maintenance                                                         #
    # ------------------------------------------------------------------ #

    def _evict(self, key: str) -> None:
        """Cache girdisini sil."""
        try:
            pkl_path = self._pkl_path(key)
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
        except OSError:
            pass

    def purge_expired(self) -> int:
        """
        Suresi dolmus tum cache girdilerini temizle.
        Silinen girdi sayisini doner.
        """
        removed = 0
        try:
            for fname in os.listdir(self.cache_dir):
                if not fname.endswith(".pkl"):
                    continue
                fpath = os.path.join(self.cache_dir, fname)
                age = time.time() - os.path.getmtime(fpath)
                if self.ttl_seconds >= 0 and age > self.ttl_seconds:
                    key = fname[:-4]
                    self._evict(key)
                    removed += 1
        except OSError:
            pass
        return removed

    def clear(self) -> int:
        """Tum cache girdilerini sil. Silinen girdi sayisini doner."""
        removed = 0
        try:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(self.cache_dir, fname))
                        removed += 1
                    except OSError:
                        pass
        except OSError:
            pass
        return removed

    def stats(self) -> Dict[str, Any]:
        """Cache dizinindeki girdi sayisi ve toplam boyutunu doner."""
        total_size = 0
        count = 0
        try:
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(".pkl"):
                    total_size += os.path.getsize(fpath)
                    count += 1
        except OSError:
            pass
        return {
            "entries": count,
            "total_size_mb": round(total_size / 1024**2, 2),
            "cache_dir": self.cache_dir,
            "ttl_hours": self.ttl_seconds / 3600.0,
        }


# ------------------------------------------------------------------ #
#  JSON serialization helper (meta icin kullanilir)                  #
# ------------------------------------------------------------------ #

def _make_json_serializable(obj: Any) -> Any:
    """numpy/pandas tiplerini JSON-serializasyona hazirlar (recursive)."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj
