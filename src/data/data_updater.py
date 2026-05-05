# -*- coding: utf-8 -*-
"""
data_updater.py - CSV tabanli hisse verisi guncelleme servisi.

Egitimden once hedef CSV'nin son tarihini kontrol eder. Veri gerideyse
yfinance uzerinden eksik gunleri indirir ve mevcut CSV semasina uygun
satirlari dosyaya ekler. Hata durumunda pipeline'i durdurmaz; mevcut veriyle
devam edilmesini saglar.
"""

from __future__ import annotations

import builtins
import os
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf


class DataUpdater:
    """Veri seti guncelligini denetleyen ve tamamlayan sinif."""

    @staticmethod
    def check_and_update(csv_path: str, stock_symbol: str, interactive: bool = True) -> None:
        try:
            df_raw = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  [UYARI] {csv_path} dosyasi bulunamadi. Lutfen kontrol edin.")
            return
        except Exception as exc:
            print(f"  [UYARI] {csv_path} okunamadi ({exc}); veri guncellemesi atlandi.")
            return

        if df_raw.empty:
            print(f"  [UYARI] {csv_path} bos; veri guncellemesi atlandi.")
            return

        date_col = _find_column(df_raw.columns, ["Date", "Tarih"])
        if date_col is None:
            print("  [UYARI] CSV'de Date/Tarih kolonu yok; veri guncellemesi atlandi.")
            return

        last_date = _parse_last_date(df_raw[date_col])
        if pd.isna(last_date):
            print("  [UYARI] Son tarih parse edilemedi; veri guncellemesi atlandi.")
            return

        last_date = pd.Timestamp(last_date).normalize()
        today = pd.Timestamp(datetime.today().date())
        diff_days = int((today - last_date).days)
        if diff_days <= 1:
            print("  [INFO] CSV veri setiniz halihazirda son gune ait. Guncellemeye gerek yok.")
            return

        print(f"\n  [INFO] Veri setinizdeki son tarih : {last_date.strftime('%d/%m/%Y')}")
        print(f"  [INFO] Bugunun tarihi             : {today.strftime('%d/%m/%Y')}")

        ask = builtins.input if interactive else (lambda *_args, **_kwargs: "e")
        answer = ask(
            f"  [SORU] Veri seti bugunden {diff_days} gun geride. "
            f"Eksik veriler yfinance ({stock_symbol}.IS) ile tamamlansin mi? (e/h): "
        )
        if str(answer).strip().lower() != "e":
            print("  [INFO] Veri guncellemesi atlandi. Mevcut eski veriler kullaniliyor.")
            return

        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ticker = f"{stock_symbol}.IS"
        print(f"  [INFO] yfinance indiriliyor: {ticker} {start} -> {end}")

        try:
            new_data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as exc:
            print(f"  [UYARI] yfinance guncellemesi basarisiz ({exc}); mevcut CSV ile devam ediliyor.")
            return

        if new_data is None or new_data.empty:
            print("  [INFO] Indirilecek yeni islem gunu bulunamadi.")
            return

        new_data = _normalize_yfinance_frame(new_data)
        rows = _map_to_existing_schema(new_data, df_raw.columns, date_col)
        rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.normalize()
        rows = rows[rows[date_col] > last_date].copy()
        rows.dropna(subset=[date_col], inplace=True)
        rows.drop_duplicates(subset=[date_col], keep="last", inplace=True)
        rows.sort_values(date_col, inplace=True)

        if rows.empty:
            print("  [INFO] Indirilen veri mevcut CSV'den daha yeni satir icermiyor.")
            return

        rows[date_col] = rows[date_col].dt.strftime(_infer_date_format(str(df_raw[date_col].iloc[-1])))
        rows = rows[list(df_raw.columns)]
        try:
            rows.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        except Exception as exc:
            print(f"  [UYARI] Yeni satirlar CSV'ye yazilamadi ({exc}); mevcut veriyle devam ediliyor.")
            return

        print(f"  [OK] {len(rows)} yeni islem gunu {os.path.basename(csv_path)} dosyasina eklendi.")


def _find_column(columns: Any, candidates: list[str]) -> str | None:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _parse_last_date(values: pd.Series) -> pd.Timestamp:
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    parsed = parsed.dropna()
    return pd.NaT if parsed.empty else pd.Timestamp(parsed.max())


def _normalize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data.columns = [str(col).replace("Datetime", "Date") for col in data.columns]
    return data


def _map_to_existing_schema(new_data: pd.DataFrame, existing_columns: Any, date_col: str) -> pd.DataFrame:
    source_map = {
        "date": "Date",
        "tarih": "Date",
        "open": "Open",
        "açılış": "Open",
        "acilis": "Open",
        "high": "High",
        "yüksek": "High",
        "yuksek": "High",
        "low": "Low",
        "düşük": "Low",
        "dusuk": "Low",
        "close": "Close",
        "kapanış": "Close",
        "kapanis": "Close",
        "adj_close": "Adj Close",
        "adj close": "Adj Close",
        "düzeltilmiş_kapanış": "Adj Close",
        "duzeltilmis_kapanis": "Adj Close",
        "volume": "Volume",
        "hacim": "Volume",
    }
    available = {str(col).strip().lower(): str(col) for col in new_data.columns}
    result = pd.DataFrame(index=new_data.index)
    for target_col in existing_columns:
        key = str(target_col).strip().lower()
        source_name = source_map.get(key)
        source_col = available.get(str(source_name).lower()) if source_name else None
        if source_col is None and source_name == "Adj Close":
            source_col = available.get("close")
        if source_col is None:
            result[target_col] = pd.NA
        else:
            result[target_col] = new_data[source_col]
    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    return result


def _infer_date_format(sample: str) -> str:
    sample = sample.strip()
    if "-" in sample:
        return "%Y-%m-%d" if len(sample.split("-")[0]) == 4 else "%d-%m-%Y"
    if "/" in sample:
        return "%Y/%m/%d" if len(sample.split("/")[0]) == 4 else "%d/%m/%Y"
    if "." in sample:
        return "%Y.%m.%d" if len(sample.split(".")[0]) == 4 else "%d.%m.%Y"
    return "%Y-%m-%d"
