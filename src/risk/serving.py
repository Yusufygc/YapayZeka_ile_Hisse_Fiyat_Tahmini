# -*- coding: utf-8 -*-
"""Gecelik risk servisi: dogrulama + final fit + son gunu skorlama.

`compute_risk_serving(panel)`:
  1. features.build_risk_frame(panel)           (panel nightly'de zaten yuklu; tekrar yuklenmez)
  2. validation.run_risk_validation             (OOS metrikleri + kabul kapisi)
  3. kapi gecerse HAR-X Ridge TUM etiketli gecmise fit; aksi halde naive persistence
  4. as_of gunu sembol satirlari -> risk_* kolonlari (PeerStore `peer_scores`)

Kapi basarisizsa `risk_method="naive_fallback"` yazilir ve metrikler run config'ine
girer; sessiz dusus yoktur. Skorlama satirlari etiketsiz son h gunu icerir; bunlar
egitime GIRMEZ (labeled_rows hedefi sonlu olmayi sart kosar).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.risk.features import build_risk_frame, labeled_rows
from src.risk.model import METHOD_MODEL, PooledVolModel, VolModelConfig, naive_sigma_h
from src.risk.validation import (
    GATE_COVERAGE_MAX, GATE_COVERAGE_MIN, GATE_MAX_MSE_RATIO, GATE_MIN_DM_T,
    RiskValidationConfig, RiskValidationResult, run_risk_validation)

logger = logging.getLogger(__name__)

RISK_STATUS_OK, RISK_STATUS_FALLBACK, RISK_STATUS_FAILED = "ok", "fallback", "failed"
RISK_LEVELS = ("düşük", "orta", "yüksek")
RISK_COLUMNS = (
    "risk_sigma_5d", "risk_sigma_naive_5d", "risk_vol_percentile", "risk_level",
    "risk_z", "risk_band_level", "risk_horizon_days", "risk_method")


def risk_levels_from_sigma(sigma: pd.Series) -> pd.Series:
    """Evren tertile'ina gore 'düşük' | 'orta' | 'yüksek' (NaN -> None)."""
    s = pd.to_numeric(sigma, errors="coerce")
    out = pd.Series([None] * len(s), index=s.index, dtype=object)
    ok = s.notna()
    if ok.sum() < 3:
        return out
    lo, hi = s[ok].quantile([1 / 3, 2 / 3])
    out[ok & (s <= lo)] = RISK_LEVELS[0]
    out[ok & (s > lo) & (s <= hi)] = RISK_LEVELS[1]
    out[ok & (s > hi)] = RISK_LEVELS[2]
    return out


def score_risk_table(
    frame: pd.DataFrame, as_of: pd.Timestamp, result: RiskValidationResult,
    model: Optional[PooledVolModel],
) -> pd.DataFrame:
    """as_of gunu sembol satirlari -> symbol + RISK_COLUMNS.

    Yontem `result.method`: model (har_x_ridge) -> sigma = sqrt(h)*exp(pred)*sqrt(c),
    z = konformal; naive_fallback -> sigma = sqrt(h)*exp(rv_21), kendi konformal z'si.
    Her iki durumda `risk_sigma_naive_5d` doludur. Gerekli girdisi NaN olan sembol -> NaN.
    """
    cfg = result.cfg
    day = frame[frame["Date"] == pd.Timestamp(as_of)].reset_index(drop=True)
    if day.empty:
        return pd.DataFrame(columns=["symbol", *RISK_COLUMNS])
    naive = naive_sigma_h(day, cfg.horizon)
    if result.method == METHOD_MODEL and model is not None:
        feats = list(cfg.feature_cols)
        ok = day[feats].notna().all(axis=1).to_numpy()
        sigma = np.full(len(day), np.nan)
        if ok.any():
            sigma[ok] = model.sigma_h(day.loc[ok])
        z = result.z_final
    else:
        sigma, z = naive.copy(), result.z_naive_final
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    sig = pd.Series(sigma)
    out = pd.DataFrame({
        "symbol": day["symbol"].to_numpy(),
        "risk_sigma_5d": sigma,
        "risk_sigma_naive_5d": np.where(np.isfinite(naive), naive, np.nan),
        "risk_vol_percentile": (sig.rank(pct=True, method="average") * 100.0).to_numpy(),
        "risk_level": risk_levels_from_sigma(sig).to_numpy(),
        "risk_z": z if np.isfinite(z) else np.nan,
        "risk_band_level": float(cfg.level),
        "risk_horizon_days": int(cfg.horizon),
        "risk_method": result.method,
    })
    bad = out["risk_sigma_5d"].isna()
    out.loc[bad, ["risk_z", "risk_band_level", "risk_horizon_days", "risk_method"]] = np.nan
    out["risk_method"] = out["risk_method"].astype(object).where(~bad, None)
    return out


def validation_summary(result: RiskValidationResult) -> dict[str, Any]:
    """Run config_json'a yazilacak ozet (kapi sabitleriyle birlikte)."""
    m = result.metrics
    keep = ("mse_model", "mse_naive", "mse_ratio", "dm_t", "qlike_model", "qlike_naive",
            "qlike_naive_corrected", "coverage", "coverage_naive", "halfwidth_model",
            "halfwidth_naive", "coverage_by_year", "mse_by_liq_quintile", "n_oos_rows",
            "n_oos_dates", "n_folds", "oos_start", "oos_end", "gate_passed", "method")
    out = {k: m.get(k) for k in keep}
    out.update(
        c=result.c_final, z=result.z_final, z_naive=result.z_naive_final,
        level=float(result.cfg.level), horizon=int(result.cfg.horizon),
        gate={"max_mse_ratio": GATE_MAX_MSE_RATIO, "min_dm_t": GATE_MIN_DM_T,
              "coverage_min": GATE_COVERAGE_MIN, "coverage_max": GATE_COVERAGE_MAX})
    return out


def compute_risk_serving(
    panel: pd.DataFrame, as_of: pd.Timestamp,
    cfg: Optional[RiskValidationConfig] = None,
) -> tuple[pd.DataFrame, dict[str, Any], RiskValidationResult]:
    """Panel -> (risk_table, config ozeti, dogrulama sonucu). Bkz. modul dokumani."""
    cfg = cfg or RiskValidationConfig()
    frame = build_risk_frame(panel, cfg.horizon)
    result = run_risk_validation(frame, cfg)
    model = None
    if result.method == METHOD_MODEL:
        lab = labeled_rows(frame, cfg.feature_cols)
        model = PooledVolModel(VolModelConfig(
            alpha=cfg.alpha, feature_cols=cfg.feature_cols, horizon=cfg.horizon,
            level=cfg.level)).fit(lab)
        model.set_calibration(result.c_final, result.z_final, result.z_naive_final)
    table = score_risk_table(frame, as_of, result, model)
    return table, validation_summary(result), result


def compute_risk_safe(
    panel: pd.DataFrame, as_of: pd.Timestamp, log: Callable[[str], None] = print,
    cfg: Optional[RiskValidationConfig] = None,
) -> tuple[Optional[pd.DataFrame], dict[str, Any]]:
    """Nightly sarmalayicisi: `compute_risk_serving` + log. Panel yeniden yuklenmez.

    Hata -> (None, {"error": ...}): risk_*/fiyat bandi bos kalir, hata run config'ine
    yazilir (Kol-B ciktisi korunur; sessiz degil). Kapi gecmezse UYARI loglanir.
    """
    t0 = time.time()
    try:
        table, summary, _ = compute_risk_serving(panel, as_of, cfg)
    except Exception as exc:  # risk hatasi Kol-B skorlamasini dusurmesin; log + config'e gecer
        log(f"HATA risk modeli hesaplanamadi ({type(exc).__name__}: {str(exc)[:150]}); "
            "risk_* ve fiyat bandi bos")
        return None, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    log(f"risk: method={summary['method']} kapi={'GECTI' if summary['gate_passed'] else 'GECMEDI'} "
        f"mse_ratio={summary['mse_ratio']:.3f} dm_t={summary['dm_t']:.1f} "
        f"kapsama={summary['coverage']:.3f} ({time.time() - t0:.0f}s)")
    if not summary["gate_passed"]:
        log("UYARI risk kabul kapisi gecmedi -> naive_fallback (naive persistence sigma + konformal z)")
    return table, summary


def risk_run_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Risk adiminin makine-okunur durumu (nightly_status.json `risk` alani).

    `failed`: hesaplama patladi (summary['error']); risk_*/fiyat bandi BOS kalir.
    `fallback`: kabul kapisi gecmedi, naive persistence servis edildi. `ok`: HAR-X Ridge.
    """
    err = summary.get("error")
    if err:
        return {"status": RISK_STATUS_FAILED, "method": None, "error": str(err)[:300]}
    method = summary.get("method")
    status = RISK_STATUS_OK if method == METHOD_MODEL else RISK_STATUS_FALLBACK
    return {"status": status, "method": method, "error": None}


def previous_risk_method(prev_run: Optional[dict[str, Any]]) -> Optional[str]:
    """Bir onceki kosunun config_json['risk']['method'] degeri (yok/bozuk/hata kosusu -> None)."""
    if not prev_run:
        return None
    try:
        cfg = json.loads(prev_run.get("config_json") or "{}")
    except (ValueError, TypeError) as exc:
        logger.warning("onceki kosunun config_json'u okunamadi: %s", exc)
        return None
    risk = cfg.get("risk") if isinstance(cfg, dict) else None
    method = risk.get("method") if isinstance(risk, dict) else None
    return str(method) if method else None


def gate_flip_warning(prev_method: Optional[str], new_method: Optional[str]) -> Optional[str]:
    """Kabul kapisi bir onceki geceye gore yon degistirdiyse uyari metni; aksi halde None.

    Gece-gece har_x_ridge <-> naive_fallback gecisi sessiz olmamali (esige yakin metrik).
    Onceki/yeni yontem bilinmiyorsa (ilk kosu, hata kosusu) karsilastirma yapilmaz.
    """
    if not prev_method or not new_method or prev_method == new_method:
        return None
    return f"risk kapisi degisti: {prev_method} -> {new_method} (onceki gece -> bu gece)"
