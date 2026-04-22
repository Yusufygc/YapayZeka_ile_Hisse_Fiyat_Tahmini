# -*- coding: utf-8 -*-
"""
feature_dictionary.py - Human readable descriptions and feature groups.
"""

from __future__ import annotations

import re


def describe_feature(feature_name: str) -> str:
    """Return a plain Turkish explanation for a feature name."""
    if feature_name == "RSI_14":
        return "hissenin asiri alim veya asiri satim bolgesine yaklasip yaklasmadigi"
    if feature_name in {"Return", "Log_Return"}:
        return "hissenin son donemdeki fiyat degisimi"
    if feature_name == "Relative_Strength":
        return "hissenin BIST100'e gore daha guclu veya zayif hareket etmesi"

    ma_match = re.match(r"^(SMA|EMA)_(\d+)_rel$", feature_name)
    if ma_match:
        ma_type, window = ma_match.groups()
        avg_name = "basit ortalamasina" if ma_type == "SMA" else "agirlikli ortalamasina"
        return f"fiyatin son {window} gunluk {avg_name} gore konumu"

    if re.match(r"^RollStd_\d+_norm$", feature_name):
        window = feature_name.split("_")[1]
        return f"son {window} gundeki oynakligin artip azalmadigi"

    if re.match(r"^BB_Width_\d+$", feature_name):
        window = feature_name.split("_")[-1]
        return f"son {window} gunde fiyat bandinin genisleyip daralmasi"

    if feature_name.startswith("LogRet_Lag"):
        return "gecmis gun getirilerinin bugunku tahmine etkisi"
    if feature_name.startswith("MACD"):
        return "momentumun guclenip zayifladigi"
    if feature_name.startswith("USDTRY"):
        return "kur tarafindaki hareketin hisse uzerindeki olasi etkisi"
    if feature_name.startswith("BIST100"):
        return "genel BIST100 piyasa hareketinin hisseye etkisi"
    if feature_name.startswith("Rate"):
        return "faiz seviyesindeki veya faiz degisimindeki baski"
    if feature_name.startswith("CPI"):
        return "enflasyon tarafindaki degisimin piyasa algisina etkisi"
    if feature_name == "Real_Rate":
        return "reel faiz kosullarinin risk istahina etkisi"

    labels = {
        "Open": "gunun acilis fiyat seviyesi",
        "High": "gun icinde gorulen en yuksek fiyat seviyesi",
        "Low": "gun icinde gorulen en dusuk fiyat seviyesi",
        "Volume": "islem hacmindeki degisim",
    }
    if feature_name in labels:
        return labels[feature_name]

    return "modelin kullandigi teknik veya makro sinyal"


def feature_group(feature_name: str) -> str:
    """Map a feature to the Phase 5 feature group taxonomy."""
    if feature_name.startswith("LogRet_Lag") or feature_name.endswith("_Lag") or "_Lag_" in feature_name:
        return "lag"
    if feature_name.startswith(("RollStd", "BB_Width", "ATR", "Volatility")):
        return "volatility"
    if feature_name == "Relative_Strength" or feature_name.startswith("BIST100"):
        return "market_relative"
    if feature_name.startswith(("USDTRY", "Rate", "CPI")) or feature_name == "Real_Rate":
        return "macro"
    if feature_name.startswith(("SMA", "EMA", "MACD", "RSI", "Return", "Log_Return", "Stoch")):
        return "technical"
    if feature_name in {"Open", "High", "Low", "Volume"}:
        return "technical"
    return "other"
