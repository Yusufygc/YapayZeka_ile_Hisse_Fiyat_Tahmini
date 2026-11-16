# -*- coding: utf-8 -*-
"""Risk modeli purged expanding walk-forward dogrulama + kabul kapisi.

Bolme: tarih-bazli purged walk-forward (`src/validation/pooled_cv.py`; expanding,
63 gunluk test bloklari, embargo = h + 5 pozisyon + kesin `target_date < test_basi`
purge). Her blokta model yalniz gecmis bloklarla fit edilir; OOS tahminleri toplanir.

Kalibrasyon SIZINTISIZ: blok k icin bias `c` ve konformal `z`, yalniz hedefi blok
baslangicindan ONCE gerceklesmis (`target_date < test_basi`) ONCEKI bloklarin OOS
satirlarindan hesaplanir (z: son `conformal_window_days` tarih penceresi). Ilk blok
kalibre edilemez -> QLIKE/kapsama yalniz kalibre bloklarda raporlanir; log-vol MSE
ve DM tum OOS satirlarinda (kalibrasyon gerektirmez).

Aralik merkezi 0 (validasyonda yon tahmini yok); serving'de merkez Kol-B beklenen
getirisidir (h-gun sigma'ya gore kucuk). Kapsama = P(|r_h - merkez| <= z * sigma_h).

KABUL KAPISI (on-kayitli; sabitler asagida, sessizce degistirilmez):
  model sunulur  <=>  mse_ratio <= GATE_MAX_MSE_RATIO  VE  dm_t >= GATE_MIN_DM_T
                      VE  GATE_COVERAGE_MIN <= kapsama <= GATE_COVERAGE_MAX.
  Aksi halde `naive_fallback`: naive persistence sigma + KENDI konformal z'si;
  `risk_method` ve metrikler PeerStore'a yazilir (sessiz degil).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.risk.features import RISK_FEATURE_COLS, RISK_HORIZON, labeled_rows
from src.risk.model import (
    BAND_LEVEL, METHOD_MODEL, METHOD_NAIVE, PooledVolModel, VolModelConfig,
    bias_factor, conformal_z)
from src.validation.pooled_cv import PooledCVConfig, PooledPurgedWalkForward

# --- on-kayitli kabul kapisi (docstring'e bkz) ---
GATE_MAX_MSE_RATIO = 0.95
GATE_MIN_DM_T = 2.0
GATE_COVERAGE_MIN = 0.75
GATE_COVERAGE_MAX = 0.85

QLIKE_FLOOR = 1e-8   # gerceklesen varyans tabani (y'deki EPS=1e-4 ile tutarli: EPS^2)


@dataclass(frozen=True)
class RiskValidationConfig:
    horizon: int = RISK_HORIZON
    level: float = BAND_LEVEL
    embargo_buffer: int = 5
    window_len: int = 63
    min_train_days: int = 504
    conformal_window_days: int = 252      # z icin son N tarih (onceki OOS)
    alpha: float = 1.0
    feature_cols: tuple[str, ...] = RISK_FEATURE_COLS


@dataclass
class RiskValidationResult:
    oos: pd.DataFrame
    metrics: dict[str, Any]
    folds: list[dict[str, Any]]
    method: str
    passed: bool
    c_final: float
    z_final: float
    z_naive_final: float
    cfg: RiskValidationConfig = field(default_factory=RiskValidationConfig)


def newey_west_t(d: np.ndarray, lag: int) -> float:
    """Ortalama(d)'nin Newey-West (Bartlett) t-istatistigi; n<3 veya LRV<=0 -> NaN."""
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    lrv = float(e @ e) / n
    for k in range(1, min(int(lag), n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (int(lag) + 1.0)) * float(e[k:] @ e[:-k]) / n
    if not np.isfinite(lrv) or lrv <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(lrv / n))


def qlike(rv2: np.ndarray, sigma2: np.ndarray) -> float:
    """Ortalama QLIKE = mean(rv2/s2 - log(rv2/s2) - 1) (>=0; kucuk iyi)."""
    rv = np.maximum(np.asarray(rv2, dtype=float), QLIKE_FLOOR)
    s2 = np.maximum(np.asarray(sigma2, dtype=float), QLIKE_FLOOR)
    ratio = rv / s2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def calibrate_from_prior(
    prior: pd.DataFrame, cutoff: pd.Timestamp, cfg: RiskValidationConfig,
) -> dict[str, float]:
    """`cutoff` ONCESI gerceklesmis (`target_date < cutoff`) onceki OOS satirlarindan
    bias `c`, konformal `z` (model) ve `c_naive`, `z_naive`.

    prior kolonlari: Date, target_date, y, pred, rv_21, fwd_ret. Bos -> NaN (kalibre degil).
    """
    nan = {"c": float("nan"), "z": float("nan"), "c_naive": float("nan"),
           "z_naive": float("nan"), "n_prior": 0, "n_window": 0}
    if prior is None or prior.empty:
        return nan
    p = prior[prior["target_date"] < pd.Timestamp(cutoff)]
    if p.empty:
        return nan
    h = float(cfg.horizon)
    c = bias_factor((p["y"] - p["pred"]).to_numpy())
    c_naive = bias_factor((p["y"] - p["rv_21"]).to_numpy())
    dates = np.sort(p["Date"].unique())
    thr = dates[max(0, len(dates) - int(cfg.conformal_window_days))]
    w = p[p["Date"] >= thr]
    sig_m = np.sqrt(h) * np.exp(w["pred"].to_numpy()) * np.sqrt(c)
    sig_n = np.sqrt(h) * np.exp(w["rv_21"].to_numpy())
    ret = w["fwd_ret"].to_numpy()
    return {"c": c, "c_naive": c_naive, "n_prior": int(len(p)), "n_window": int(len(w)),
            "z": conformal_z(ret, sig_m, cfg.level),
            "z_naive": conformal_z(ret, sig_n, cfg.level)}


def _make_folds(lab: pd.DataFrame, cfg: RiskValidationConfig):
    n_dates = int(lab["Date"].nunique())
    cv = PooledCVConfig(
        target_horizon=cfg.horizon, embargo_buffer=cfg.embargo_buffer,
        window_len=cfg.window_len, n_windows=n_dates // cfg.window_len + 1,
        min_train_days=cfg.min_train_days, final_holdout=False)
    return PooledPurgedWalkForward(cv).split(lab)


def _liq_quintile(lab: pd.DataFrame) -> np.ndarray:
    pct = lab.groupby("Date")["liq_log"].rank(pct=True, method="first")
    return np.clip(np.floor(pct.to_numpy() * 5.0), 0, 4).astype(int)


def run_risk_validation(frame: pd.DataFrame, cfg: Optional[RiskValidationConfig] = None,
                        ) -> RiskValidationResult:
    """OOS walk-forward + metrikler + kapi. `frame` = features.build_risk_frame ciktisi."""
    cfg = cfg or RiskValidationConfig()
    lab = labeled_rows(frame, cfg.feature_cols)
    lab = lab.sort_values(["Date", "symbol"]).reset_index(drop=True)
    if lab.empty:
        raise ValueError("risk dogrulama: etiketli satir yok")
    folds = _make_folds(lab, cfg)
    if not folds:
        raise ValueError("risk dogrulama: yeterli gecmis yok (fold uretilemedi)")
    lab["liq_q"] = _liq_quintile(lab)
    fit_cols = ["Date", "y", *cfg.feature_cols]
    mcfg = VolModelConfig(alpha=cfg.alpha, feature_cols=cfg.feature_cols,
                          horizon=cfg.horizon, level=cfg.level)
    keep = ["symbol", "Date", "target_date", "y", "rv_21", "fwd_ret", "fwd_rv2", "liq_log", "liq_q"]
    parts: list[pd.DataFrame] = []
    fold_info: list[dict[str, Any]] = []
    for k, fold in enumerate(folds):          # folds zaman sirali (test_basi artan)
        train = lab.loc[fold.train_mask, fit_cols]
        test = lab.loc[fold.test_mask, list(dict.fromkeys(keep + list(cfg.feature_cols)))].copy()
        model = PooledVolModel(mcfg).fit(train)
        test["pred"] = model.predict_logvol(test)
        prior = pd.concat(parts) if parts else None
        cal = calibrate_from_prior(prior, fold.test_date_start, cfg)
        ok = bool(np.isfinite(cal["c"]) and np.isfinite(cal["z"]))
        h = float(cfg.horizon)
        test["fold"] = k
        test["calibrated"] = ok
        test["c"], test["z"] = cal["c"], cal["z"]
        test["c_naive"], test["z_naive"] = cal["c_naive"], cal["z_naive"]
        test["sigma_h"] = np.sqrt(h) * np.exp(test["pred"]) * np.sqrt(cal["c"]) if ok else np.nan
        test["sigma_h_naive"] = np.sqrt(h) * np.exp(test["rv_21"])
        parts.append(test[["symbol", "Date", "target_date", "fold", "y", "pred", "rv_21", "fwd_ret",
                           "fwd_rv2", "liq_q", "calibrated", "c", "z", "c_naive", "z_naive",
                           "sigma_h", "sigma_h_naive"]])
        fold_info.append({
            "fold": k, "test_start": str(fold.test_date_start.date()),
            "test_end": str(fold.test_date_end.date()), "n_train": int(fold.n_train),
            "n_test": int(fold.n_test), "c": cal["c"], "z": cal["z"],
            "z_naive": cal["z_naive"], "n_prior": cal["n_prior"], "calibrated": ok})
    oos = pd.concat(parts, ignore_index=True)
    metrics = compute_metrics(oos, cfg)
    passed = gate_passed(metrics)
    final = calibrate_from_prior(
        oos[["Date", "target_date", "y", "pred", "rv_21", "fwd_ret"]],
        pd.Timestamp(oos["target_date"].max()) + pd.Timedelta(days=1), cfg)
    metrics["gate_passed"] = passed
    metrics["method"] = METHOD_MODEL if passed else METHOD_NAIVE
    return RiskValidationResult(
        oos=oos, metrics=metrics, folds=fold_info, method=metrics["method"], passed=passed,
        c_final=final["c"], z_final=final["z"], z_naive_final=final["z_naive"], cfg=cfg)


def gate_passed(m: dict[str, Any]) -> bool:
    """Pre-registered kapi. NaN metrik -> False (kapi muhafazakar)."""
    r, t, cov = m.get("mse_ratio"), m.get("dm_t"), m.get("coverage")
    vals = (r, t, cov)
    if any(v is None or not np.isfinite(v) for v in vals):
        return False
    return bool(r <= GATE_MAX_MSE_RATIO and t >= GATE_MIN_DM_T
                and GATE_COVERAGE_MIN <= cov <= GATE_COVERAGE_MAX)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) and b > 0 else float("nan")


def compute_metrics(oos: pd.DataFrame, cfg: RiskValidationConfig) -> dict[str, Any]:
    """OOS tablosundan kapi + rapor metrikleri."""
    e_m = (oos["y"] - oos["pred"]).to_numpy()
    e_n = (oos["y"] - oos["rv_21"]).to_numpy()
    mse_m, mse_n = float(np.mean(e_m ** 2)), float(np.mean(e_n ** 2))
    d = pd.Series(e_n ** 2 - e_m ** 2, index=oos.index).groupby(oos["Date"].to_numpy()).mean()
    out: dict[str, Any] = {
        "mse_model": mse_m, "mse_naive": mse_n, "mse_ratio": _ratio(mse_m, mse_n),
        "dm_t": newey_west_t(d.sort_index().to_numpy(), cfg.horizon),
        "n_oos_rows": int(len(oos)), "n_oos_dates": int(oos["Date"].nunique()),
        "n_folds": int(oos["fold"].nunique()),
        "oos_start": str(pd.Timestamp(oos["Date"].min()).date()),
        "oos_end": str(pd.Timestamp(oos["Date"].max()).date()),
        "level": float(cfg.level), "horizon": int(cfg.horizon),
    }
    cal = oos[oos["calibrated"]]
    out["n_calibrated_rows"] = int(len(cal))
    if cal.empty:
        out.update(qlike_model=float("nan"), qlike_naive=float("nan"),
                   qlike_naive_corrected=float("nan"), coverage=float("nan"),
                   coverage_naive=float("nan"), coverage_by_year={}, coverage_naive_by_year={},
                   mse_by_liq_quintile={})
        return out
    rv2 = cal["fwd_rv2"].to_numpy()
    s2_m = np.exp(2.0 * cal["pred"].to_numpy()) * cal["c"].to_numpy()
    s2_n = np.exp(2.0 * cal["rv_21"].to_numpy())
    out["qlike_model"] = qlike(rv2, s2_m)
    out["qlike_naive"] = qlike(rv2, s2_n)
    out["qlike_naive_corrected"] = qlike(rv2, s2_n * cal["c_naive"].to_numpy())
    hit_m = (cal["fwd_ret"].abs() <= cal["z"] * cal["sigma_h"]).to_numpy()
    hit_n = (cal["fwd_ret"].abs() <= cal["z_naive"] * cal["sigma_h_naive"]).to_numpy()
    out["coverage"], out["coverage_naive"] = float(hit_m.mean()), float(hit_n.mean())
    # keskinlik: ayni kapsamada dar bant daha iyi (ort. yari-genislik, log-getiri birimi)
    out["halfwidth_model"] = float((cal["z"] * cal["sigma_h"]).mean())
    out["halfwidth_naive"] = float((cal["z_naive"] * cal["sigma_h_naive"]).mean())
    year = pd.to_datetime(cal["Date"]).dt.year.to_numpy()
    out["coverage_by_year"] = {int(y): float(hit_m[year == y].mean()) for y in np.unique(year)}
    out["coverage_naive_by_year"] = {int(y): float(hit_n[year == y].mean()) for y in np.unique(year)}
    q: dict[str, dict[str, float]] = {}
    for k, g in oos.groupby("liq_q"):
        mm = float(np.mean((g["y"] - g["pred"]) ** 2))
        mn = float(np.mean((g["y"] - g["rv_21"]) ** 2))
        q[f"Q{int(k) + 1}"] = {"mse_model": mm, "mse_naive": mn, "ratio": _ratio(mm, mn),
                               "n": int(len(g))}
    out["mse_by_liq_quintile"] = q
    return out
