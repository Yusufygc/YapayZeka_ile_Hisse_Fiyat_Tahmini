# -*- coding: utf-8 -*-
"""
feature_dictionary.py - Human readable descriptions and feature groups.
"""

from __future__ import annotations

import re


def describe_feature(feature_name: str) -> str:
    """Return a plain Turkish explanation for a feature name."""
    if feature_name == "WalkForward_Summary":
        return "walk-forward pencerelerindeki genel model davranışı"
    if feature_name == "RSI_14":
        return "RSI 14: hissenin aşırı alım veya aşırı satım bölgesine yaklaşması"
    if feature_name in {"Return", "Log_Return"}:
        return "son fiyat getirisi: hissenin yakın dönem fiyat değişimi"
    if feature_name == "Relative_Strength":
        return "BIST100 göreli güç: hissenin endekse göre güçlü veya zayıf kalması"

    ma_match = re.match(r"^(SMA|EMA)_(\d+)_rel$", feature_name)
    if ma_match:
        ma_type, window = ma_match.groups()
        avg_name = "basit hareketli ortalamasına" if ma_type == "SMA" else "üssel hareketli ortalamasına"
        return f"{ma_type} {window}: fiyatın {window} günlük {avg_name} göre konumu"

    if re.match(r"^RollStd_\d+_norm$", feature_name):
        window = feature_name.split("_")[1]
        return f"volatilite {window}: son {window} gündeki oynaklığın artıp azalması"

    if re.match(r"^BB_Width_\d+$", feature_name):
        window = feature_name.split("_")[-1]
        return f"Bollinger bant genişliği {window}: fiyat bandının genişleyip daralması"

    if feature_name.startswith("LogRet_Lag"):
        return "gecikmeli getiri: geçmiş gün getirilerinin bugünkü tahmine etkisi"
    if feature_name.startswith("OBV"):
        return "OBV hacim akışı: hacim ile fiyat yönünün birlikte güçlenmesi"
    if feature_name.startswith("VWAP"):
        return "VWAP: fiyatın hacim ağırlıklı ortalamaya göre konumu"
    if feature_name.startswith("Market_Regime"):
        return "piyasa rejimi: fiyatın SMA-200'e göre trend konumu"
    if feature_name.startswith("MACD"):
        return "MACD: momentumun güçlenip zayıflaması"
    if feature_name.startswith("USDTRY"):
        return "USDTRY: kur hareketinin hisse üzerindeki olası etkisi"
    if feature_name.startswith("BIST100"):
        return "BIST100: genel endeks hareketinin hisseye etkisi"
    if feature_name.startswith("Rate"):
        return "faiz: faiz seviyesi veya faiz değişiminin piyasa baskısı"
    if feature_name.startswith("CPI"):
        return "enflasyon: TÜFE değişiminin piyasa algısına etkisi"
    if feature_name == "Real_Rate":
        return "reel faiz: reel faiz koşullarının risk iştahına etkisi"
    if feature_name.startswith("ATR"):
        return "ATR: fiyat oynaklığı ve günlük hareket aralığı"
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


def feature_group(feature_name: str) -> str:
    """Map a feature to the Phase 5 feature group taxonomy."""
    if feature_name.startswith("LogRet_Lag") or feature_name.endswith("_Lag") or "_Lag_" in feature_name:
        return "lag"
    if feature_name.startswith(("OBV", "VWAP")):
        return "volume"
    if feature_name.startswith("Market_Regime"):
        return "regime"
    if feature_name.startswith(("RollStd", "BB_Width", "ATR", "Volatility")):
        return "volatility"
    if feature_name == "Relative_Strength" or feature_name.startswith("BIST100"):
        return "market_relative"
    if feature_name.startswith(("USDTRY", "Rate", "CPI")) or feature_name == "Real_Rate":
        return "macro"
    if feature_name.startswith("Signal_"):
        return "signal"
    if feature_name == "WalkForward_Summary":
        return "model_summary"
    if feature_name.startswith(("SMA", "EMA", "MACD", "RSI", "Return", "Log_Return", "Stoch")):
        return "technical"
    if feature_name in {"Open", "High", "Low", "Volume"}:
        return "technical"
    return "other"
