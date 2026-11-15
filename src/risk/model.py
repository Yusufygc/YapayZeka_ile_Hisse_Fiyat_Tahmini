# -*- coding: utf-8 -*-
"""Havuzlu oynaklik modeli (HAR-X Ridge) — urun karari "A".

Ridge(alpha=1) on standardized features, hedef = log gunluk vol (features.py `y`).
Log-vol regresyonu MEDYANI tahmin eder; varyans tahmini icin Jensen duzeltmesi
gerekir:

    sigma2_daily = exp(2 * pred) * c,   c = mean(exp(2 * resid))

`c` yalniz ONCEKI katlarin OOS artiklarindan (validation.py) hesaplanir. Gunluk
sigma ve h-gun (sqrt(h) kuralli) sigma:

    sigma_daily = exp(pred) * sqrt(c),   sigma_h = sqrt(h) * sigma_daily

%80 aralik: merkez +- z * sigma_h; Gaussal z=1.28 yetersiz kapsar (~%72), bu
yuzden z, |r_h - merkez| / sigma_h'nin tarihsel konformal %80 kuantilidir (yine
yalniz onceki OOS satirlarindan). Model joblib ile metadata'siyla saklanir.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.risk.features import RISK_FEATURE_COLS, RISK_HORIZON

METHOD_MODEL = "har_x_ridge"
METHOD_NAIVE = "naive_fallback"
BAND_LEVEL = 0.8


@dataclass(frozen=True)
class VolModelConfig:
    alpha: float = 1.0
    feature_cols: tuple[str, ...] = RISK_FEATURE_COLS
    horizon: int = RISK_HORIZON
    level: float = BAND_LEVEL


def bias_factor(resid: np.ndarray) -> float:
    """c = mean(exp(2 * resid)): log-vol artiklarindan varyans (Jensen) duzeltmesi.

    `resid = y - pred` ONCEKI katlarin OOS artiklari olmali. Bos/sonlu olmayan -> 1.0.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1.0
    c = float(np.mean(np.exp(2.0 * r)))
    return c if np.isfinite(c) and c > 0 else 1.0


def conformal_z(
    ret_h: np.ndarray, sigma_h: np.ndarray, level: float = BAND_LEVEL,
    center: Optional[np.ndarray] = None,
) -> float:
    """z = |ret_h - center| / sigma_h'nin `level` kuantili (havuzlanmis satirlar).

    Tarih-basina kuantil medyani YETERSIZ kapsar; tum pencere satirlarindan tek
    kuantil alinir. Gecerli satir yoksa NaN.
    """
    ret = np.asarray(ret_h, dtype=float)
    sig = np.asarray(sigma_h, dtype=float)
    ctr = np.zeros_like(ret) if center is None else np.asarray(center, dtype=float)
    ok = np.isfinite(ret) & np.isfinite(sig) & (sig > 0) & np.isfinite(ctr)
    if not ok.any():
        return float("nan")
    return float(np.quantile(np.abs(ret[ok] - ctr[ok]) / sig[ok], float(level)))


class PooledVolModel:
    """Ridge(standardize(X)) -> log gunluk vol; bias `c` ve konformal `z` tasir."""

    def __init__(self, cfg: Optional[VolModelConfig] = None) -> None:
        self.cfg = cfg or VolModelConfig()
        self._pipe: Optional[Pipeline] = None
        self.c: float = 1.0
        self.z: float = float("nan")
        self.z_naive: float = float("nan")
        self.meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ fit
    @property
    def feature_cols(self) -> tuple[str, ...]:
        return tuple(self.cfg.feature_cols)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[list(self.cfg.feature_cols)].to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> "PooledVolModel":
        """`df` ozellik kolonlarini tasimali; y verilmezse df['y']. NaN satir -> hata."""
        target = df["y"].to_numpy(dtype=float) if y is None else np.asarray(y, dtype=float)
        X = self._X(df)
        if not (np.isfinite(X).all() and np.isfinite(target).all()):
            raise ValueError("PooledVolModel.fit: ozellik/hedefte NaN/inf var")
        if len(X) < 10:
            raise ValueError("PooledVolModel.fit: yetersiz satir")
        self._pipe = Pipeline([("sc", StandardScaler()),
                               ("ridge", Ridge(alpha=float(self.cfg.alpha)))])
        self._pipe.fit(X, target)
        dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else None
        self.meta = {
            "feature_cols": list(self.cfg.feature_cols),
            "horizon": int(self.cfg.horizon),
            "level": float(self.cfg.level),
            "alpha": float(self.cfg.alpha),
            "n_train": int(len(X)),
            "fit_date_min": None if dates is None else str(dates.min().date()),
            "fit_date_max": None if dates is None else str(dates.max().date()),
            "fitted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        return self

    def set_calibration(self, c: float, z: float, z_naive: float = float("nan")) -> None:
        """Onceki OOS'tan gelen bias faktoru ve konformal z (model + naive) ata."""
        self.c, self.z, self.z_naive = float(c), float(z), float(z_naive)

    # -------------------------------------------------------------- predict
    def predict_logvol(self, df: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("PooledVolModel fit edilmedi")
        return self._pipe.predict(self._X(df))

    def sigma_daily(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_logvol(df)) * np.sqrt(self.c)

    def sigma_h(self, df: pd.DataFrame) -> np.ndarray:
        return np.sqrt(float(self.cfg.horizon)) * self.sigma_daily(df)

    # ------------------------------------------------------------- persist
    def metadata(self) -> dict[str, Any]:
        return {**self.meta, "c": self.c, "z": self.z, "z_naive": self.z_naive}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"cfg": self.cfg, "pipe": self._pipe, "c": self.c, "z": self.z,
                     "z_naive": self.z_naive, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str) -> "PooledVolModel":
        blob = joblib.load(path)
        m = cls(blob["cfg"])
        m._pipe, m.c, m.z, m.z_naive, m.meta = (
            blob["pipe"], blob["c"], blob["z"], blob["z_naive"], blob["meta"])
        return m


def naive_sigma_h(df: pd.DataFrame, horizon: int = RISK_HORIZON,
                  cols: Sequence[str] = ("rv_21",)) -> np.ndarray:
    """Naive persistence: sigma_daily = exp(rv_21) (son 21g RMS), sqrt(h) ile olceklenir."""
    return np.sqrt(float(horizon)) * np.exp(df[cols[0]].to_numpy(dtype=float))
