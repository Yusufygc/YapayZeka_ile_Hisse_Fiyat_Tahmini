# -*- coding: utf-8 -*-
"""
macro_pipeline.py — Makroekonomik & Piyasa Bağlamı Veri Katmanı
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2026-11 IC denetimi sonrasi KURATE makro katman. Model ozellikleri:

  yfinance (günlük, otomatik)
     • USD/TRY : "USDTRY=X"  -> USDTRY_Return, USDTRY_Volatility7
     • BIST100 : "XU100.IS"  -> BIST100_Return, BIST100_MA7

Kaldirilanlar: EURTRY, VIX, Altin, Petrol, DXY, US10Y, sektor endeksleri
(XBANK...), FRED CPI ve faiz/enflasyon FEATURE'lari (Rate_Level, Rate_Change,
CPI_MoM, CPI_YoY, Real_Rate) + ABD-seansi 1 gun gecikme mantigi (yalniz onlara
hizmet ediyordu).

ISTISNA — INTEREST_RATE.csv: ``src/utils/risk_free_rate.py`` risk-free oran
kaynagi olarak ``data/macro/INTEREST_RATE.csv``'yi okur (Sharpe/Sortino). Bu
yuzden TCMB EVDS faiz cache yenilemesi (``refresh_interest_rate_cache``) korunur;
``get_macro_features`` her cagrida cache'i tazeler AMA faiz artik model feature'i
uretmez. EVDS erisilemezse manuel CSV fallback (kolonlar: Date, INTEREST_RATE).

Sonuc: makro matris USDTRY/BIST100 serilerinin basladigi tarihten (2014) baslar;
eski aylik CPI YoY (12 ay geriye bakis) + dropna hisse gecmisini 2021-07'ye
kesiyordu — artik kesmez.

Look-ahead notlari:
  * USDTRY ve BIST100 BIST seansi sirasinda gozlenebilir -> gecikmesiz.
  * Faiz cache'inin yayim kurali risk_free_rate.get_risk_free_rate_series icinde
    (macro_transforms.lag_monthly_features: ay sonu + lag) uygulanir.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.dates import parse_dates
from src.features import macro_feature_engineering, macro_transforms

try:
    import requests
except ImportError:  # pragma: no cover - minimal validation runtimes
    class _MissingRequests:
        @staticmethod
        def get(*args, **kwargs):
            raise ImportError("requests yüklü değil -> pip install requests")

    requests = _MissingRequests()

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────

# yfinance ile çekilen günlük seriler (yalniz model feature'i uretenler)
_YFINANCE_TICKERS = {
    "USDTRY":  "USDTRY=X",
    "BIST100": "XU100.IS",
}

_EVDS_BASE_URL = "https://evds2.tcmb.gov.tr/service/evds/"
_DEFAULT_EVDS_RATE_SERIES = "TP.PPK.H01"
_RATE_KEY = "INTEREST_RATE"  # risk-free oran kaynagi (feature DEGIL)

# Cache dosya adları
_CACHE_FILES = {
    "USDTRY":        "USDTRY.csv",
    "BIST100":       "BIST100.csv",
    _RATE_KEY:       "INTEREST_RATE.csv",
}

_STALE_DAYS_DAILY   = 1   # Günlük veri için yenileme eşiği
_STALE_DAYS_MONTHLY = 28  # Aylık veri için yenileme eşiği (yaklaşık 1 ay)


# ─────────────────────────────────────────────────────────────────────────────
# MacroPipeline
# ─────────────────────────────────────────────────────────────────────────────
class MacroPipeline:
    """
    Günlük makro özellik pipeline'ı (USDTRY + BIST100) + faiz cache yenileme.

    Args:
        cache_dir : Ham veri CSV'lerinin saklanacağı dizin.
        rate_release_lag_days : Politika faizi (risk-free kaynagi) icin AY SONUNDAN
            sonraki gecikme (gün). Artik model feature'i uretmez; parametre
            ``risk_free_rate.get_risk_free_rate_series`` ile ayni yayim kuralini
            (ay sonu + lag) belgelemek/geriye uyumluluk icin korunur.
    """

    def __init__(
        self,
        cache_dir: str = "data/macro",
        rate_release_lag_days: int = 1,
    ):
        self.cache_dir = cache_dir
        self.rate_release_lag_days = max(0, int(rate_release_lag_days))
        os.makedirs(cache_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # A. Günlük Veri — yfinance
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _download_yfinance(
        ticker: str,
        start: str,
        end: str,
        value_name: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """yfinance'tan günlük kapanış indirir."""
        try:
            import yfinance as yf
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.reset_index(inplace=True)
            raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()
            col = value_name or ticker.replace("=X", "").replace(".IS", "")
            return raw[["Date", "Close"]].rename(columns={"Close": col})
        except Exception as exc:
            print(f"  [MACRO] yfinance {ticker}: {exc}")
            return None

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, _CACHE_FILES[key])

    @staticmethod
    def _ticker_aliases(key: str) -> list[str]:
        ticker = _YFINANCE_TICKERS.get(key, "")
        aliases = [
            key,
            ticker,
            ticker.replace("=X", "").replace(".IS", ""),
            ticker.replace("=X", ""),
        ]
        if key == "BIST100":
            aliases.extend(["XU100", "XU100.IS"])
        return list(dict.fromkeys([alias for alias in aliases if alias]))

    def _normalize_daily_cache_schema(self, key: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Older cache files may store yfinance ticker names such as XU100.IS,
        ^VIX or GC=F. Collapse aliases into the canonical macro key expected
        by downstream feature engineering.
        """
        if key not in _YFINANCE_TICKERS or df is None or df.empty:
            return df

        normalized = df.copy()
        aliases = [alias for alias in self._ticker_aliases(key) if alias in normalized.columns]
        value_cols = [col for col in normalized.columns if col != "Date"]
        if aliases:
            canonical = normalized[aliases].bfill(axis=1).iloc[:, 0]
        elif value_cols:
            canonical = normalized[value_cols].bfill(axis=1).iloc[:, 0]
        else:
            return normalized[["Date"]].copy()

        normalized[key] = pd.to_numeric(canonical, errors="coerce")
        normalized = normalized[["Date", key]].copy()
        normalized.dropna(subset=[key], inplace=True)
        normalized.drop_duplicates("Date", keep="last", inplace=True)
        normalized.sort_values("Date", inplace=True)
        normalized.reset_index(drop=True, inplace=True)
        return normalized

    def _is_stale(self, key: str, threshold_days: int) -> bool:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return True
        df = pd.read_csv(path, parse_dates=["Date"])
        if df.empty:
            return True
        last = pd.to_datetime(df["Date"].max())
        return (datetime.today() - last).days > threshold_days

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df = self._normalize_daily_cache_schema(key, df)
        return df

    def _update_daily_cache(self, key: str, start: str) -> None:
        """yfinance günlük cache'i güncelle."""
        ticker   = _YFINANCE_TICKERS[key]
        existing = self._load_cache(key)

        fetch_start = start
        if existing is not None and not existing.empty:
            fetch_start = (existing["Date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")

        fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  [MACRO] {key} güncelleniyor: {fetch_start} -> bugün ...")

        new_data = self._download_yfinance(ticker, fetch_start, fetch_end, value_name=key)
        if new_data is not None and not new_data.empty:
            new_data = self._normalize_daily_cache_schema(key, new_data)
            combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
            combined.drop_duplicates("Date", inplace=True)
            combined.sort_values("Date", inplace=True)
            combined.to_csv(self._cache_path(key), index=False)
            print(f"  [MACRO] {key}: {len(combined)} gün cache'lendi.")
        else:
            print(f"  [MACRO] {key}: yeni veri alınamadı, mevcut cache korunuyor.")

    # ══════════════════════════════════════════════════════════════════════════
    # B. Aylık Faiz — TCMB EVDS + CSV Fallback (risk-free kaynagi; feature DEGIL)
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_evds_series(
        self,
        series_id: str,
        start: str,
        end: str,
        value_name: str,
    ) -> Optional[pd.DataFrame]:
        """
        TCMB EVDS web servisinden tek seri ceker.

        API anahtari TCMB_EVDS_API_KEY ortam degiskeninden okunur. Faiz seri
        kodu TCMB_EVDS_RATE_SERIES ile override edilebilir.
        """
        api_key = os.getenv("TCMB_EVDS_API_KEY", "").strip()
        if not api_key:
            print("  [MACRO] TCMB_EVDS_API_KEY yok; EVDS faiz guncellemesi atlandi.")
            return None

        params = {
            "series": series_id,
            "startDate": pd.to_datetime(start).strftime("%d-%m-%Y"),
            "endDate": pd.to_datetime(end).strftime("%d-%m-%Y"),
            "type": "json",
            "aggregationTypes": "avg",
            "formulas": "0",
            "frequency": "5",
        }
        try:
            response = requests.get(
                _EVDS_BASE_URL,
                params=params,
                headers={"key": api_key},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            print(f"  [MACRO] EVDS {series_id}: {exc}")
            return None

        if isinstance(payload, dict):
            items = payload.get("items", [])
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        if not items:
            return None

        rows = []
        ignored_cols = {"Tarih", "Date", "UNIXTIME", "YEARWEEK"}
        for item in items:
            date_value = item.get("Tarih", item.get("Date"))
            value_key = next((k for k in item.keys() if k not in ignored_cols), None)
            if date_value is None or value_key is None:
                continue
            value = item.get(value_key)
            if value in (None, "", "-"):
                continue
            try:
                numeric_value = float(str(value).replace(",", "."))
            except ValueError:
                continue
            rows.append({
                "Date": parse_dates(date_value).iloc[0],
                value_name: numeric_value,
            })

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df.sort_values("Date", inplace=True)
        return df[["Date", value_name]]

    def _update_interest_rate_cache(self, start: str) -> None:
        """
        INTEREST_RATE.csv'yi TCMB EVDS'ten gunceller (risk_free_rate.py kaynagi).
        Başarısız olursa manual CSV olup olmadığını kontrol eder.
        """
        key = _RATE_KEY
        path = self._cache_path(key)
        existing = self._load_cache(key)

        # Var olan veriden ileriye devam et
        if existing is not None and not existing.empty:
            last_cached = existing["Date"].max()
            fetch_start = (last_cached - timedelta(days=60)).strftime("%Y-%m-%d")  # üst üste biraz al
        else:
            fetch_start = start

        fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        series_id = os.getenv("TCMB_EVDS_RATE_SERIES", _DEFAULT_EVDS_RATE_SERIES).strip()
        print(f"  [MACRO] {key} (EVDS:{series_id}) guncelleniyor ...")
        new_data = self._fetch_evds_series(series_id, fetch_start, fetch_end, key)

        if new_data is not None and not new_data.empty:
            # Sütun adını normalize et: series_id → key (INTEREST_RATE)
            val_col = [c for c in new_data.columns if c != "Date"][0]
            new_data = new_data.rename(columns={val_col: key})

            combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
            combined.drop_duplicates("Date", inplace=True)
            combined.sort_values("Date", inplace=True)
            combined.to_csv(path, index=False)
            print(f"  [MACRO] {key}: {len(combined)} aylık kayıt cache'lendi.")
        else:
            # Manual CSV fallback kontrolü
            self._check_manual_csv(key)

    def _check_manual_csv(self, key: str) -> None:
        """
        Kullanıcının manuel olarak yerleştirdiği CSV dosyasını kontrol eder.
        Yoksa ne yapılması gerektiğini açıklar.
        """
        path = self._cache_path(key)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                print(f"  [MACRO] {key}: manuel CSV bulundu ({len(df)} satır), kullanılıyor.")
                return

        # Rehber mesajı
        print("\n  [MACRO] ⚠ Faiz oranı verisi otomatik alınamadı.")
        print(f"  Manuel yol: '{self.cache_dir}/INTEREST_RATE.csv' dosyasını oluştur.")
        print("  Format örneği:\n    Date,INTEREST_RATE\n    2020-01-01,12.0\n    2020-02-01,10.75\n    ...")
        print("  Kaynak: https://www.tcmb.gov.tr → Para Politikası → Politika Faizi\n")

    # ══════════════════════════════════════════════════════════════════════════
    # C. Ana Metod
    # ══════════════════════════════════════════════════════════════════════════

    def get_macro_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Belirtilen tarih aralığı için makro özellik matrisini döndürür.

        Akış:
          1. Faiz cache'ini tazele (risk-free kaynagi; feature uretmez)
          2. Günlük USD/TRY ve BIST100 cache'lerini tazele + yükle
          3. Outer join + ffill, USDTRY_* / BIST100_* özelliklerini türet

        Returns:
            pd.DataFrame  — Date sütunu + 4 makro özellik
                            (USDTRY_Return, USDTRY_Volatility7, BIST100_Return,
                            BIST100_MA7). Boş döner → feature_pipeline makroyu atlar.
        """
        window = self._macro_date_window(start_date, end_date)
        # Faiz cache'i feature uretmese de risk_free_rate.py okur -> yenileme korunur.
        self.refresh_interest_rate_cache(window["buf_monthly_str"])
        self._refresh_daily_caches(window["buf_daily_str"])
        usdtry_df, bist100_df = self._load_required_daily_frames()

        if usdtry_df is None or bist100_df is None:
            print("  [MACRO] Döviz/endeks verisi yüklenemedi, makro özellikler atlanacak.")
            return pd.DataFrame()

        s = window["start"]
        e = window["end"]
        buf = window["buf_daily"]
        usdtry_df = self._filter_macro_frame(usdtry_df, buf, e)
        bist100_df = self._filter_macro_frame(bist100_df, buf, e)

        macro = self._build_base_daily_macro(usdtry_df, bist100_df)
        macro = self._engineer_daily(macro)

        # Tampon dönemini çıkar
        macro = macro[macro["Date"] >= s].reset_index(drop=True)
        return macro

    @staticmethod
    def _macro_date_window(start_date: str, end_date: str) -> dict[str, pd.Timestamp | str]:
        return macro_transforms.macro_date_window(start_date, end_date)

    def _refresh_daily_caches(self, buffer_start: str) -> None:
        for key in _YFINANCE_TICKERS:
            if self._is_stale(key, _STALE_DAYS_DAILY):
                self._update_daily_cache(key, buffer_start)

    def refresh_interest_rate_cache(self, buffer_start: str) -> None:
        """INTEREST_RATE.csv bayatsa EVDS'ten yeniler (risk-free kaynagi).

        Hata makro feature uretimini DURDURMAZ (faiz artik feature degil) ama
        loglanir; sessiz yutulmaz.
        """
        try:
            if self._is_stale(_RATE_KEY, _STALE_DAYS_MONTHLY):
                self._update_interest_rate_cache(buffer_start)
        except Exception as exc:  # noqa: BLE001 - risk-free cache hatasi feature'lari bozmamali
            logger.warning("INTEREST_RATE cache yenilemesi basarisiz: %s: %s", type(exc).__name__, exc)

    def _load_required_daily_frames(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return self._load_cache("USDTRY"), self._load_cache("BIST100")

    @staticmethod
    def _filter_macro_frame(
        df: Optional[pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        return macro_transforms.filter_macro_frame(df, start, end)

    @staticmethod
    def _build_base_daily_macro(
        usdtry_df: pd.DataFrame,
        bist100_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return macro_transforms.build_base_daily_macro(usdtry_df, bist100_df)

    @staticmethod
    def _engineer_daily(df: pd.DataFrame) -> pd.DataFrame:
        return macro_feature_engineering.engineer_daily(df)

    # ── Özellik İsimleri ──────────────────────────────────────────────────────
    @staticmethod
    def macro_feature_names() -> list:
        """MacroPipeline'in urettigi makro ozellik isimleri (kurate set)."""
        return [
            "USDTRY_Return",
            "USDTRY_Volatility7",
            "BIST100_Return",
            "BIST100_MA7",
        ]
