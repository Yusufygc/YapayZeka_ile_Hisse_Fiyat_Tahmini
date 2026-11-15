# -*- coding: utf-8 -*-
"""Nightly universe scoring orchestration — E2 Faz 5.

Akis (kilitli karar): per-query egitim YOK. Gecelik batch:
  1. pooled panel + cross-sectional hedef + cs-features.
  2. GlobalPooledModel'i TUM gecmise egit (serving = en guncel modele kadar).
  3. en guncel tarihin evren satirlarini skorla -> peer_score/percentile/label.
  4. her sembole segment (likidite/vol/sektor) + segment ICIR ata.
  5. confidence = segment_ICIR x tradability kapilari.
  6. peer_scores satirlari PeerStore'a yazilir; API okur.

Segment ICIR referansi Faz 6 OOS calismasindan gelir (likidite kovasi birincil
ayirici). Burada `segment_icir_map` (liq_bucket -> ICIR) DISARIDAN verilir;
boylece bu fonksiyon saf/testable kalir, agir OOS hesabi CLI'de yapilir.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from src.risk.serving import RISK_COLUMNS
from src.serving.confidence import ConfidenceThresholds, peer_confidence, select_segment_icir
from src.serving.peer_scoring import PeerScoringConfig, score_latest_universe
from src.serving.trend_tendency import TrendCalibration, trend_from_peer


# (b) harman: eksen agirliklari (likidite baskin ayirici).
_DEFAULT_AXIS_WEIGHTS = {"liq": 0.5, "vol": 0.3, "sector": 0.2}


def composite_icir(
    liq_b: Optional[str], vol_b: Optional[str], sec: Optional[str],
    icir_maps: Mapping[str, Mapping[str, float]],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Cok-eksenli (liq/vol/sektor) segment ICIR'larini agirlikli harmanla.

    Eksik/NaN eksen atlanir, kalan agirliklar yeniden normalize edilir. Tek
    eksen (or. yalniz liq) verilince o eksenin ICIR'ini dondurur -> geriye uyumlu.
    """
    weights = weights or _DEFAULT_AXIS_WEIGHTS
    axis_bucket = {"liq": liq_b, "vol": vol_b, "sector": sec}
    num = 0.0
    wsum = 0.0
    for axis, bucket in axis_bucket.items():
        m = icir_maps.get(axis)
        if m is None or bucket is None:
            continue
        v = m.get(bucket)
        if v is None or not np.isfinite(v):
            continue
        w = float(weights.get(axis, 0.0))
        num += w * float(v)
        wsum += w
    return (num / wsum) if wsum > 0 else float("nan")


def composite_n_eff(
    liq_b: Optional[str], vol_b: Optional[str], sec: Optional[str],
    neff_maps: Optional[Mapping[str, Mapping[str, Optional[int]]]],
) -> Optional[int]:
    """Harman ICIR'in dayandigi ortusmeyen tarih sayisi (n_eff): katkida bulunan
    eksenlerin MINIMUMU (muhafazakar). Bilinen n_eff yoksa None (kapi uygulanmaz)."""
    if not neff_maps:
        return None
    vals = []
    for axis, bucket in (("liq", liq_b), ("vol", vol_b), ("sector", sec)):
        m = neff_maps.get(axis)
        if m is None or bucket is None:
            continue
        v = m.get(bucket)
        if v is not None and v == v:
            vals.append(int(v))
    return min(vals) if vals else None


def attach_risk_and_band(
    merged: pd.DataFrame, risk_table: Optional[pd.DataFrame], last_closes: Mapping[str, float],
) -> pd.DataFrame:
    """risk_* kolonlarini ekle + fiyat bandi: Close * exp(exp_ret -/+ risk_z * risk_sigma_5d).

    p50 = Close * exp(exp_ret) (exp_ret Kol-B'den). Band genisligi YALNIZ risk modelinden
    (src/risk): eski quintile-std tablosu ve vol-orani kaldirildi. Close/exp_ret/risk eksikse
    ilgili band alanlari None (p50 yalniz Close+exp_ret ile dolar).
    """
    out = merged.drop(columns=[c for c in RISK_COLUMNS if c in merged.columns])
    if risk_table is not None and not risk_table.empty:
        out = out.merge(risk_table[["symbol", *RISK_COLUMNS]], on="symbol", how="left")
    else:
        for c in RISK_COLUMNS:
            out[c] = None
    close = pd.to_numeric(out["symbol"].map(last_closes), errors="coerce")
    exp_ret = pd.to_numeric(out["trend_expected_return"], errors="coerce")
    half = (pd.to_numeric(out["risk_z"], errors="coerce")
            * pd.to_numeric(out["risk_sigma_5d"], errors="coerce"))
    ok_p = (close > 0) & np.isfinite(exp_ret)
    ok_b = ok_p & np.isfinite(half)
    out["kolb_price_p50"] = (close * np.exp(exp_ret)).where(ok_p)
    out["kolb_price_low"] = (close * np.exp(exp_ret - half)).where(ok_b)
    out["kolb_price_high"] = (close * np.exp(exp_ret + half)).where(ok_b)
    out["kolb_horizon_days"] = pd.to_numeric(out["risk_horizon_days"], errors="coerce").where(ok_b)
    out["kolb_band_level"] = pd.to_numeric(out["risk_band_level"], errors="coerce").where(ok_b)
    return out


def assemble_peer_table(
    model,
    panel_latest: pd.DataFrame,
    feature_cols: list[str],
    seg_table: pd.DataFrame,
    segment_icir_map: Optional[Mapping[str, float]] = None,
    *,
    icir_maps: Optional[Mapping[str, Mapping[str, float]]] = None,
    axis_weights: Optional[Mapping[str, float]] = None,
    tradable_for: Optional[Callable[[str], bool]] = None,
    stale_for: Optional[Callable[[str], bool]] = None,
    icir_segment_col: str = "liq_bucket",
    scoring_cfg: Optional[PeerScoringConfig] = None,
    thr: Optional[ConfidenceThresholds] = None,
    trend_cfg: Optional[TrendCalibration] = None,
    neff_maps: Optional[Mapping[str, Mapping[str, Optional[int]]]] = None,
    risk_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """En guncel evreni skorla + segment + confidence -> PeerStore'a hazir tablo.

    Segment ICIR kaynagi (oncelik sirasi):
      - `icir_maps` verilirse (b) HARMAN: {axis: {bucket: icir}} liq/vol/sector
        agirlikli ortalamasi (composite_icir).
      - yoksa `segment_icir_map` + `icir_segment_col` tek-eksen (geriye uyumlu).
    tradable_for/stale_for : symbol -> bool (varsayilan: hepsi tradable, taze).
    neff_maps : {axis: {bucket: n_eff}} segment ICIR'inin ortusmeyen tarih sayisi;
        verilirse peer_confidence'a n_eff olarak gecer ('high' icin n_eff kapisi).
    risk_table : src.risk.serving.score_risk_table ciktisi (symbol + risk_* kolonlari);
        sembol bazinda birlestirilir. Fiyat bandi genisligi buradan gelir:
        low/high = Close * exp(exp_ret -/+ risk_z * risk_sigma_5d). Yoksa bant None.
    """
    scoring_cfg = scoring_cfg or PeerScoringConfig()
    thr = thr or ConfidenceThresholds()
    tradable_for = tradable_for or (lambda s: True)
    stale_for = stale_for or (lambda s: False)

    peer = score_latest_universe(model, panel_latest, feature_cols, scoring_cfg)
    if peer.empty:
        return peer

    seg = seg_table.rename(columns={
        "liq_bucket": "segment_liq", "vol_bucket": "segment_vol",
        "sector": "segment_sector"})
    merged = peer.merge(seg, on="symbol", how="left")

    if icir_maps is not None:
        merged["segment_icir"] = [
            composite_icir(r["segment_liq"], r["segment_vol"], r["segment_sector"],
                           icir_maps, axis_weights)
            for _, r in merged.iterrows()
        ]
    else:
        smap = segment_icir_map or {}
        seg_key_col = {"liq_bucket": "segment_liq", "vol_bucket": "segment_vol",
                       "sector": "segment_sector"}.get(icir_segment_col, "segment_liq")
        merged["segment_icir"] = merged[seg_key_col].map(
            lambda b: float(smap.get(b, float("nan"))) if b is not None else float("nan"))

    universe_ok = int(merged["universe_size"].iloc[0]) >= scoring_cfg.min_names
    labels, reasons, warns = [], [], []
    for _, r in merged.iterrows():
        c = peer_confidence(
            r["segment_icir"], r["peer_label"],
            tradable=bool(tradable_for(r["symbol"])),
            stale=bool(stale_for(r["symbol"])),
            universe_ok=universe_ok, thr=thr,
            n_eff=(composite_n_eff(r["segment_liq"], r["segment_vol"],
                                   r["segment_sector"], neff_maps)
                   if icir_maps is not None else None))
        labels.append(c.label)
        reasons.append(c.reasons)
        warns.append(c.warnings)
    merged["confidence_label"] = labels
    merged["confidence_reasons"] = reasons
    merged["confidence_warnings"] = warns

    # --- mutlak trend eğilimi (Faz 7 kalibrasyonu) ---
    tcfg = trend_cfg or TrendCalibration()
    t_labels, t_pup, t_exp, t_src = [], [], [], []
    for _, r in merged.iterrows():
        t = trend_from_peer(r["peer_percentile"], r["universe_size"], tcfg)
        t_src.append(t.probability_source)
        t_labels.append(t.label)
        t_pup.append(t.prob_up)
        t_exp.append(t.expected_return)
    merged["trend_label"] = t_labels
    merged["trend_prob_up"] = t_pup
    merged["trend_expected_return"] = t_exp
    # PeerStore'da kolon yok (peer_store.py bu gorevde degistirilmez): tablo kolonu +
    # run config'inde saklanir; store eksik kolonlari yok sayar.
    merged["probability_source"] = t_src

    # --- risk (havuzlu oynaklik) + Kol-B fiyat bandi ---
    last_closes = {}
    if "Close" in panel_latest.columns:
        last_closes = panel_latest.set_index("symbol")["Close"].to_dict()
    merged = attach_risk_and_band(merged, risk_table, last_closes)
    return merged


def liqlog_floor_from_turnover(min_turnover_tl: float) -> float:
    """TL/gun ciro tabanini loader liq_log birimine cevir (liq_log=log1p(ciro)).

    Tradability kapisi icin: sembolun medyan liq_log'u bu esigin altinda ise
    'islem yapilamaz' -> confidence low (Faz 6 gerginligi: en az likit = en guclu
    sinyal ama islem zor). 0/negatif -> 0 (kapi etkisiz)."""
    return float(np.log1p(max(0.0, float(min_turnover_tl))))


def latest_scoring_rows(
    full_panel: pd.DataFrame, date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """En guncel PIYASA gununun sembol satirlari + as_of tarihi (K4).

    `full_panel` loader'dan `keep_unlabeled=True` ile gelmeli: hedefi henuz
    gerceklesmemis son h satir panelde kalir. Etiketli-only panelden max(Date)
    almak skorlamayi hedefi gerceklesmis son gune (~h is gunu geriye) cekerdi.
    """
    if full_panel.empty or date_col not in full_panel.columns:
        return full_panel.iloc[0:0], pd.NaT
    as_of = pd.Timestamp(full_panel[date_col].max())
    return full_panel[full_panel[date_col] == as_of].copy(), as_of


def coverage_warnings(
    n_loaded: int, n_universe: int, latest_market_day: pd.Timestamp,
    today: pd.Timestamp, stale_days: int = 10,
) -> list[str]:
    """Gece kosusu sonunda operatore gosterilecek kapsama/tazelik uyarilari (K6/K7).

    - panel evrenden kucukse (ornegin semboller yalniz yedek klasorunde kalmissa)
    - en guncel piyasa gunu `stale_days` takvim gunden eskiyse
    """
    out: list[str] = []
    if n_universe > 0 and n_loaded < n_universe:
        out.append(f"kapsama: panelde {n_loaded}/{n_universe} sembol (eksik {n_universe - n_loaded})")
    if pd.notna(latest_market_day):
        age = int((pd.Timestamp(today).normalize() - pd.Timestamp(latest_market_day).normalize()).days)
        if age > int(stale_days):
            out.append(f"veri yasi: en guncel piyasa gunu {age} gun once "
                       f"({pd.Timestamp(latest_market_day).date()}), esik {stale_days}")
    return out


def segment_icir_from_table(seg_ic_table: pd.DataFrame) -> dict[str, float]:
    """Faz 6 segment_cross_sectional_ic ciktisindan {segment: icir} sozlugu.

    Ortusmeyen ICIR (`icir_nonoverlap`) kolonu varsa ve gecerliyse o tercih edilir
    (confidence.select_segment_icir); yoksa eski ortusmeli `icir`.
    """
    out = {}
    for _, r in seg_ic_table.iterrows():
        v, _n = select_segment_icir(r.to_dict())
        out[str(r["segment"])] = float(v) if v is not None and np.isfinite(v) else float("nan")
    return out


def segment_neff_from_table(seg_ic_table: pd.DataFrame) -> dict[str, Optional[int]]:
    """{segment: n_eff}: `segment_icir_from_table`'in ICIR'inin dayandigi ortusmeyen
    tarih sayisi; ortusmeli geri-dusus (veya kolon yok) -> None (kapi uygulanmaz)."""
    out: dict[str, Optional[int]] = {}
    for _, r in seg_ic_table.iterrows():
        _v, n = select_segment_icir(r.to_dict())
        out[str(r["segment"])] = n
    return out
