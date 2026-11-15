# -*- coding: utf-8 -*-
"""
feature_dictionary.py - Human readable descriptions and feature groups.
"""

from __future__ import annotations

import re
from typing import Iterable


def describe_feature(feature_name: str) -> str:
    """Return a plain Turkish explanation for a feature name."""
    # E2 Kol-B (pooled cross-sectional) — gün içi akran dönüşümleri ve meta sinyaller.
    if feature_name.endswith("_csr"):
        return f"{describe_feature(feature_name[:-4])} — gün içi akran sıralaması (cross-sectional sıra)"
    if feature_name.endswith("_csz"):
        return f"{describe_feature(feature_name[:-4])} — gün içi akran standardı (cross-sectional z-skor)"
    if feature_name == "symbol_id":
        return "sembol kimliği: modelin sembole özgü öğrendiği eğilim (gömme)"
    if feature_name in {"sector_code", "sector"}:
        return "sektör kodu: hissenin ait olduğu sektörün etkisi"
    if feature_name == "liq_log":
        return "likidite: log devir hacmi (işlem yoğunluğu)"
    if feature_name == "vol":
        return "gerçekleşen oynaklık: hissenin yakın dönem dalgalanması"

    if feature_name == "WalkForward_Summary":
        return "walk-forward pencerelerindeki genel model davranışı"
    if feature_name == "RSI_14":
        return "RSI 14: hissenin aşırı alım veya aşırı satım bölgesine yaklaşması"
    if feature_name == "Return":
        return "son fiyat getirisi: hissenin yakın dönem fiyat değişimi"
    if feature_name == "Relative_Strength":
        return "BIST100 göreli güç: hissenin endekse göre güçlü veya zayıf kalması"

    ma_match = re.match(r"^SMA_(\d+)_rel$", feature_name)
    if ma_match:
        window = ma_match.group(1)
        return f"SMA {window}: fiyatın {window} günlük basit hareketli ortalamasına göre konumu"

    if feature_name == "day_of_week":
        return "haftanın günü: gün bazlı takvim etkisi"
    if feature_name == "momentum_60d":
        return "60 günlük momentum: hissenin son 60 iş günündeki getirisi"
    if feature_name == "market_momentum_60d":
        return "piyasa momentumu: BIST100'ün son 60 iş günündeki getirisi"
    if feature_name == "relative_to_market_60d":
        return "piyasaya göre 60 günlük momentum: hissenin endeksten güçlü veya zayıf kalması"

    if re.match(r"^RollStd_\d+_norm$", feature_name):
        window = feature_name.split("_")[1]
        return f"volatilite {window}: son {window} gündeki oynaklığın artıp azalması"

    if feature_name.startswith("LogRet_Lag"):
        return "gecikmeli getiri: geçmiş gün getirilerinin bugünkü tahmine etkisi"
    if feature_name.startswith("Market_Regime"):
        return "piyasa rejimi: fiyatın SMA-200'e göre trend konumu"
    if feature_name.startswith("MACD"):
        return "MACD: momentumun güçlenip zayıflaması"
    if feature_name.startswith("USDTRY"):
        return "USDTRY: kur hareketinin hisse üzerindeki olası etkisi"
    if feature_name.startswith("BIST100"):
        return "BIST100: genel endeks hareketinin hisseye etkisi"
    if feature_name.startswith("ATR"):
        return "ATR: fiyat oynaklığı ve günlük hareket aralığı"
    if feature_name.startswith("NATR"):
        return "NATR: normalize ATR (fiyata oranlı oynaklık)"
    if feature_name.startswith("Stoch"):
        return "Stokastik osilatör: kısa vadeli momentum ve aşırı bölge sinyali"
    if feature_name.startswith("Signal_"):
        return f"sinyal kuralı: {feature_name.replace('Signal_', '').replace('_', ' ').lower()}"

    labels = {
        "Open": "açılış fiyat seviyesi",
        "High": "gün içi en yüksek fiyat seviyesi",
        "Low": "gün içi en düşük fiyat seviyesi",
        "Volume": "işlem hacmindeki değişim",
    }
    if feature_name in labels:
        return labels[feature_name]

    return f"okunabilir etiketi olmayan model sinyali: {feature_name}"


def has_readable_label(feature_name: str) -> bool:
    """Return True when the feature dictionary has a specific product label."""
    return "okunabilir etiketi olmayan" not in describe_feature(feature_name).lower()


def compute_dictionary_coverage(feature_names: Iterable[str]) -> dict:
    """Compute readable-label coverage for a feature list."""
    names = [str(name) for name in feature_names if str(name)]
    if not names:
        return {"total": 0, "covered": 0, "ratio": 1.0, "missing": []}
    missing = [name for name in names if not has_readable_label(name)]
    covered = len(names) - len(missing)
    return {
        "total": len(names),
        "covered": covered,
        "ratio": covered / len(names),
        "missing": missing,
    }


_GROUP_LABELS = {
    "macro": "Makro ekonomik sinyaller",
    "technical": "Teknik gostergeler",
    "market_relative": "Piyasa ve sektor goreli guc",
    "volume": "Hacim ve para akisi",
    "calendar": "Takvim etkileri",
    "volatility": "Volatilite ve fiyat araligi",
    "regime": "Piyasa rejimi",
    "lag": "Gecikmeli fiyat davranisi",
    "cross_sectional": "Akran goreli sinyaller",
    "meta": "Sembol ve likidite meta sinyalleri",
    "signal": "Sinyal kurallari",
    "model_summary": "Model ozeti",
    "other": "Diger model sinyalleri",
}


def feature_group_label(group: str) -> str:
    """Return the product-facing label for a feature group."""
    key = str(group or "other").strip()
    return _GROUP_LABELS.get(key, _GROUP_LABELS["other"])


def feature_group_reason(group: str, direction: str, context: str = "forecast") -> str:
    """Return a short, non-causal product reason for a grouped XAI signal."""
    label = feature_group_label(group)
    direction_text = {
        "yukari": "yukari yonde",
        "asagi": "asagi yonde",
        "notr": "notr yonde",
        "dikkat": "dikkat edilmesi gereken",
    }.get(str(direction or "dikkat"), "dikkat edilmesi gereken")
    if context == "peer":
        if direction in {"yukari", "asagi"}:
            return f"{label} akran siralamasini {direction_text} etkileyen faktorler arasinda."
        return f"{label} akran siralamasinda izlenen faktorler arasinda."
    if direction in {"yukari", "asagi"}:
        return f"{label} model tahminini {direction_text} etkileyen faktorler arasinda."
    return f"{label} model tahmininde izlenen faktorler arasinda."


def feature_group(feature_name: str) -> str:
    """Map a feature to the Phase 5 feature group taxonomy."""
    # E2 Kol-B — cross-sectional & meta grupları (prefix kontrollerinden ÖNCE).
    if feature_name.endswith(("_csr", "_csz")):
        return "cross_sectional"
    if feature_name in {"symbol_id", "sector_code", "sector", "liq_log"}:
        return "meta"
    if feature_name == "vol":
        return "volatility"
    if feature_name.startswith("LogRet_Lag") or feature_name.endswith("_Lag") or "_Lag_" in feature_name:
        return "lag"
    if feature_name.startswith("Market_Regime"):
        return "regime"
    if feature_name.startswith(("RollStd", "ATR", "NATR", "Volatility")):
        return "volatility"
    if feature_name == "day_of_week":
        return "calendar"
    if feature_name in {"Relative_Strength", "market_momentum_60d", "relative_to_market_60d"} or (
        feature_name.startswith("BIST100")
    ):
        return "market_relative"
    if feature_name.startswith("USDTRY"):
        return "macro"
    if feature_name.startswith("Signal_"):
        return "signal"
    if feature_name == "WalkForward_Summary":
        return "model_summary"
    if feature_name.startswith(("SMA", "EMA", "MACD", "RSI", "Return", "Stoch")) or feature_name == "momentum_60d":
        return "technical"
    if feature_name in {"Open", "High", "Low", "Volume"}:
        return "technical"
    return "other"
