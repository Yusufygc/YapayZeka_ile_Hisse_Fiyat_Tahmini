# -*- coding: utf-8 -*-
"""
macro_pipeline.py — Makroekonomik & Piyasa Bağlamı Veri Katmanı
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BIST hisse tahminlerini desteklemek için üç farklı kaynaği birleştirir:

  1. yfinance  (günlük, otomatik)
     • USD/TRY : "USDTRY=X"
     • BIST100  : "XU100.IS"

  2. FRED / pandas_datareader  (aylık, otomatik)
     • TCMB Faizi proxy : "INTDSRTRM193N"  — IMF İskonto Faizi (% yıllık)
     • Türkiye TÜFE     : "TURCPIALLMINMEI" — CPI Endeksi (2015=100)

     Kurulum (FRED için):
       pip install pandas-datareader

  3. Manuel CSV fallback  (FRED erişilemezse)
     • data/macro/INTEREST_RATE.csv  →  kolonlar: Date, Rate
     • data/macro/CPI.csv            →  kolonlar: Date, CPI
     İki kolon yeterli; Date "YYYY-MM-DD" formatında olmalı.

Türetilen Özellikler:
  Döviz/Endeks (günlük):
    USDTRY_Return, USDTRY_MA7, USDTRY_Volatility7
    BIST100_Norm, BIST100_Return, BIST100_MA7

  Faiz & Enflasyon (aylık → günlük ffill):
    Rate_Level       — politika faizi düzeyi (%)
    Rate_Change      — aylık faiz değişimi (artış/indirim sinyali)
    CPI_YoY          — yıllık enflasyon oranı (%)
    CPI_MoM          — aylık CPI değişimi (%)
    Real_Rate        — Reel faiz = Rate_Level − CPI_YoY
                       (negatif reel faiz → hisse piyasası için olumlu)

Metodolojik not:
  Aylık faiz/CPI feature'ları ÖNCE kendi aylık frekanslarında hesaplanır
  (diff, pct_change), SONRA günlük takvime ffill ile taşınır.

  Yanlış: raw_monthly → ffill → daily → diff/pct_change
    → Rate_Change ay içinde hep 0, CPI_YoY 12 günlük fark olur.

  Doğru: raw_monthly → diff/pct_change (aylık) → ffill → daily
    → Tüm türetilmiş değerler gerçek aylık değişimleri yansıtır.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover - minimal validation runtimes
    class _MissingRequests:
        @staticmethod
        def get(*args, **kwargs):
            raise ImportError("requests yüklü değil -> pip install requests")

    requests = _MissingRequests()

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────

# yfinance ile çekilen günlük seriler
_YFINANCE_TICKERS = {
    "USDTRY":  "USDTRY=X",
    "BIST100": "XU100.IS",
    # Tier 1 — Kola eklenen yüksek etkili global göstergeler (Faz 4.1)
    "EURTRY":  "EURTRY=X",   # EUR/TRY kuru — BIST ile güçlü korelasyon
    "VIX":     "^VIX",        # CBOE Volatility Index — küresel risk iştahı
    "GOLD_USD": "GC=F",       # Altın vadeli (USD/oz) — TL varlıklarıyla ters korelasyon
    "OIL_USD":  "BZ=F",       # Brent petrol (USD/bbl) — enerji hisseleri için kritik
    "DXY":     "DX-Y.NYB",   # Dolar endeksi — küresel EM etkisi
    "US10Y":   "^TNX",        # ABD 10-yıllık faiz — carry trade sinyali
}

_EVDS_BASE_URL = "https://evds2.tcmb.gov.tr/service/evds/"
_DEFAULT_EVDS_RATE_SERIES = "TP.PPK.H01"
_MONTHLY_SERIES_KEYS = ("INTEREST_RATE", "CPI")

# FRED'den çekilen aylık seriler
_FRED_SERIES = {
    "CPI":           "TURCPIALLMINMEI",  # Türkiye TÜFE endeksi (aylık)
}

# Cache dosya adları
_CACHE_FILES = {
    "USDTRY":        "USDTRY.csv",
    "BIST100":       "BIST100.csv",
    "INTEREST_RATE": "INTEREST_RATE.csv",
    "CPI":           "CPI.csv",
    # Faz 4.1 eklentileri
    "EURTRY":        "EURTRY.csv",
    "VIX":           "VIX.csv",
    "GOLD_USD":      "GOLD_USD.csv",
    "OIL_USD":       "OIL_USD.csv",
    "DXY":           "DXY.csv",
    "US10Y":         "US10Y.csv",
}

_STALE_DAYS_DAILY   = 1   # Günlük veri için yenileme eşiği
_STALE_DAYS_MONTHLY = 28  # Aylık veri için yenileme eşiği (yaklaşık 1 ay)


# ─────────────────────────────────────────────────────────────────────────────
# MacroPipeline
# ─────────────────────────────────────────────────────────────────────────────
class MacroPipeline:
    """
    Tüm makro veri kaynaklarını birleştiren merkezi pipeline.

    Args:
        cache_dir : Ham veri CSV'lerinin saklanacağı dizin.
    """

    def __init__(
        self,
        cache_dir: str = "data/macro",
        rate_release_lag_days: int = 1,
        cpi_release_lag_days: int = 15,
    ):
        self.cache_dir = cache_dir
        self.rate_release_lag_days = max(0, int(rate_release_lag_days))
        self.cpi_release_lag_days = max(0, int(cpi_release_lag_days))
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
        print(f"  [MACRO] {key} güncelleniyor: {fetch_start} → bugün ...")

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
    # B. Aylık Veri — FRED (pandas_datareader) + CSV Fallback
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_fred(self, series_id: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        pandas_datareader üzerinden FRED verisini çeker.
        pandas_datareader yüklü değilse ya da bağlantı yoksa None döner.
        """
        try:
            import pandas_datareader.data as web
            df = web.DataReader(series_id, "fred",
                                start=pd.to_datetime(start),
                                end=pd.to_datetime(end))
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = ["Date", series_id]
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            return df
        except ImportError:
            print("  [MACRO] pandas_datareader yüklü değil → pip install pandas-datareader")
            return None
        except Exception as exc:
            print(f"  [MACRO] FRED {series_id}: {exc}")
            return None

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
                "Date": pd.to_datetime(date_value, dayfirst=True, errors="coerce"),
                value_name: numeric_value,
            })

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df.sort_values("Date", inplace=True)
        return df[["Date", value_name]]

    def _update_monthly_cache(self, key: str, start: str) -> None:
        """
        Aylık FRED verisini günceller.
        Başarısız olursa manual CSV olup olmadığını kontrol eder.
        """
        path      = self._cache_path(key)
        existing  = self._load_cache(key)

        # Var olan veriden ileriye devam et
        if existing is not None and not existing.empty:
            last_cached = existing["Date"].max()
            fetch_start = (last_cached - timedelta(days=60)).strftime("%Y-%m-%d")  # üst üste biraz al
        else:
            fetch_start = start

        fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        if key == "INTEREST_RATE":
            series_id = os.getenv("TCMB_EVDS_RATE_SERIES", _DEFAULT_EVDS_RATE_SERIES).strip()
            print(f"  [MACRO] {key} (EVDS:{series_id}) guncelleniyor ...")
            new_data = self._fetch_evds_series(series_id, fetch_start, fetch_end, key)
        else:
            series_id = _FRED_SERIES[key]
            print(f"  [MACRO] {key} (FRED:{series_id}) güncelleniyor ...")
            new_data = self._fetch_fred(series_id, fetch_start, fetch_end)

        if new_data is not None and not new_data.empty:
            # Sütun adını normalize et: series_id → key (INTEREST_RATE veya CPI)
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
        label_map = {
            "INTEREST_RATE": (
                "Faiz oranı",
                "INTEREST_RATE.csv",
                "Date,Rate\n2020-01-01,12.0\n2020-02-01,10.75\n...",
                "https://www.tcmb.gov.tr → Para Politikası → Politika Faizi",
            ),
            "CPI": (
                "TÜFE/Enflasyon",
                "CPI.csv",
                "Date,CPI\n2020-01-01,450.5\n2020-02-01,452.1\n...",
                "https://data.tuik.gov.tr → Tüketici Fiyat İstatistikleri",
            ),
        }
        if key in label_map:
            label, fname, sample, source = label_map[key]
            print(f"\n  [MACRO] ⚠ {label} verisi otomatik alınamadı.")
            print(f"  Manuel yol: '{self.cache_dir}/{fname}' dosyasını oluştur.")
            print(f"  Format örneği:\n    {sample}")
            print(f"  Kaynak: {source}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # C. Ana Metod
    # ══════════════════════════════════════════════════════════════════════════

    def get_macro_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Belirtilen tarih aralığı için tam makro özellik matrisini döndürür.

        Akış (metodolojik olarak doğru):
          1. Aylık ham veriyi kendi frekansında feature'lara çevir
             (Rate_Change=diff, CPI_MoM/YoY=pct_change aylık seride)
          2. Aylık feature'ları günlük takvime ffill ile taşı
          3. Günlük USD/TRY ve BIST100 feature'larını türet

        Returns:
            pd.DataFrame  — Date sütunu + tüm makro özellikler.
                            Boş döner → feature_pipeline makroyu atlar.
        """
        buf_daily   = (pd.to_datetime(start_date) - timedelta(days=90)).strftime("%Y-%m-%d")
        # CPI_YoY için 12 aylık geriye bakış gerekir; aylık veriler daha erken başlamalı
        buf_monthly = (pd.to_datetime(start_date) - timedelta(days=395)).strftime("%Y-%m-%d")

        # ── Günlük veriler (yfinance) ─────────────────────────────────────────
        for key in _YFINANCE_TICKERS:
            if self._is_stale(key, _STALE_DAYS_DAILY):
                self._update_daily_cache(key, buf_daily)

        usdtry_df  = self._load_cache("USDTRY")
        bist100_df = self._load_cache("BIST100")

        if usdtry_df is None or bist100_df is None:
            print("  [MACRO] Döviz/endeks verisi yüklenemedi, makro özellikler atlanacak.")
            return pd.DataFrame()

        # ── Tarih filtreleri ───────────────────────────────────────────────────
        s   = pd.to_datetime(start_date)
        e   = pd.to_datetime(end_date)
        buf = pd.to_datetime(buf_daily)
        buf_m = pd.to_datetime(buf_monthly)

        def _filter_daily(df):
            if df is None or df.empty:
                return df
            return df[(df["Date"] >= buf) & (df["Date"] <= e)].copy()

        def _filter_monthly(df):
            # Aylık veriler için daha geniş buffer (CPI_YoY 12 ay geriye bakar)
            if df is None or df.empty:
                return df
            return df[(df["Date"] >= buf_m) & (df["Date"] <= e)].copy()

        usdtry_df  = _filter_daily(usdtry_df)
        bist100_df = _filter_daily(bist100_df)

        # Faz 4.1: Genişletilmiş global göstergeler (try/except — her biri opsiyonel)
        _global_keys = ["EURTRY", "VIX", "GOLD_USD", "OIL_USD", "DXY", "US10Y"]
        _global_dfs = {}
        for _gkey in _global_keys:
            try:
                if self._is_stale(_gkey, _STALE_DAYS_DAILY):
                    self._update_daily_cache(_gkey, buf_daily)
                _gdf = self._load_cache(_gkey)
                if _gdf is not None and not _gdf.empty:
                    _global_dfs[_gkey] = _filter_daily(_gdf)
            except Exception as _exc:
                print(f"  [MACRO] {_gkey} atlanıyor: {_exc}")

        # ── Aylık veriler (EVDS/FRED) ─────────────────────────────────────────
        for key in _MONTHLY_SERIES_KEYS:
            if self._is_stale(key, _STALE_DAYS_MONTHLY):
                self._update_monthly_cache(key, buf_monthly)

        interest_df = self._load_cache("INTEREST_RATE")
        cpi_df      = self._load_cache("CPI")
        interest_df = _filter_monthly(interest_df) if interest_df is not None else None
        cpi_df      = _filter_monthly(cpi_df)      if cpi_df      is not None else None

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 1: Aylık feature'ları kendi frekansında hesapla (ffill'den ÖNCE)
        # ══════════════════════════════════════════════════════════════════════
        monthly_rate_feats = None
        if interest_df is not None and not interest_df.empty:
            monthly_rate_feats = self._engineer_monthly_rate(interest_df)
            monthly_rate_feats["Rate_Raw_Date"] = monthly_rate_feats["Date"]
            monthly_rate_feats["Date"] = monthly_rate_feats["Date"] + pd.to_timedelta(
                self.rate_release_lag_days,
                unit="D",
            )

        monthly_cpi_feats = None
        if cpi_df is not None and not cpi_df.empty:
            monthly_cpi_feats = self._engineer_monthly_cpi(cpi_df)
            monthly_cpi_feats["CPI_Raw_Date"] = monthly_cpi_feats["Date"]
            monthly_cpi_feats["Date"] = monthly_cpi_feats["Date"] + pd.to_timedelta(
                self.cpi_release_lag_days,
                unit="D",
            )

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 2: Günlük takvim kur (USD/TRY + BIST100 outer join)
        # ══════════════════════════════════════════════════════════════════════
        macro = pd.merge(usdtry_df, bist100_df, on="Date", how="outer")
        macro.sort_values("Date", inplace=True)
        macro.ffill(inplace=True)

        # Faz 4.1: Global göstergeleri left-merge ile ekle (yoksa sütun oluşmaz)
        for _gkey, _gdf in _global_dfs.items():
            try:
                macro = pd.merge(macro, _gdf, on="Date", how="left")
                # Boşlukları önceki gün değeriyle doldur (tatil vb.)
                close_col = _gdf.columns[-1]
                if close_col in macro.columns:
                    macro[close_col] = macro[close_col].ffill()
            except Exception as _exc:
                print(f"  [MACRO] {_gkey} merge atlandı: {_exc}")

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 3: Aylık feature dataframe'lerini günlük takvime ffill ile taşı
        # ══════════════════════════════════════════════════════════════════════
        if monthly_rate_feats is not None:
            # Left merge: aylık güne denk gelen satıra değer yazar, diğerleri NaN
            macro = pd.merge(macro, monthly_rate_feats, on="Date", how="left")
            # ffill: her günlük satır son bilinen aylık değeri taşır
            for col in ["Rate_Level", "Rate_Change"]:
                if col in macro.columns:
                    macro[col] = macro[col].ffill()

        if monthly_cpi_feats is not None:
            macro = pd.merge(macro, monthly_cpi_feats, on="Date", how="left")
            for col in ["CPI_MoM", "CPI_YoY"]:
                if col in macro.columns:
                    macro[col] = macro[col].ffill()

        # Real_Rate: her iki sütun ffill'den geçtikten sonra basit çıkarma
        if "Rate_Level" in macro.columns and "CPI_YoY" in macro.columns:
            macro["Real_Rate"] = macro["Rate_Level"] - macro["CPI_YoY"]

        raw_date_cols = [c for c in macro.columns if c.endswith("_Raw_Date")]
        if raw_date_cols:
            macro.drop(columns=raw_date_cols, inplace=True)

        macro.reset_index(drop=True, inplace=True)

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 4: Günlük USD/TRY & BIST100 feature'larını türet
        # ══════════════════════════════════════════════════════════════════════
        macro = self._engineer_daily(macro)

        # Tampon dönemini çıkar
        macro = macro[macro["Date"] >= s].reset_index(drop=True)

        return macro

    # ── Özellik Mühendisliği ──────────────────────────────────────────────────

    @staticmethod
    def _engineer_monthly_rate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Faiz feature'larını AYLLIK seri üzerinde hesaplar.

        Günlük takvime ffill'den ÖNCE çağrılmalıdır; bu sayede
        Rate_Change, ay içinde ffill edilmiş 0'lar değil, gerçek
        aylık faiz değişimini temsil eder.

        Giriş  : Date + INTEREST_RATE sütunları olan aylık DataFrame.
        Çıkış  : Date + Rate_Level + Rate_Change
        """
        df = df.copy().sort_values("Date").reset_index(drop=True)
        rate_col = "INTEREST_RATE" if "INTEREST_RATE" in df.columns else "Rate"
        df["Rate_Level"]  = df[rate_col]
        df["Rate_Change"] = df[rate_col].diff()   # gerçek aylık fark (ör. 42 → 45 → +3)
        df.drop(columns=[rate_col], inplace=True)
        return df[["Date", "Rate_Level", "Rate_Change"]]

    @staticmethod
    def _engineer_monthly_cpi(df: pd.DataFrame) -> pd.DataFrame:
        """
        CPI feature'larını AYLLIK seri üzerinde hesaplar.

        Günlük takvime ffill'den ÖNCE çağrılmalıdır; bu sayede:
          • CPI_MoM = pct_change(1)  → gerçek aylık enflasyon değişimi
          • CPI_YoY = pct_change(12) → gerçek yıllık enflasyon (12 ay öncesiyle kıyaslama)

        Günlük ffill sonrasında bu hesaplar yapılsaydı:
          • pct_change(1)  → günlük fark (anlamsız, ay içinde hep 0)
          • pct_change(12) → 12 günlük fark (yıllık değil, ~2 haftalık değişim)

        Giriş  : Date + CPI sütunları olan aylık DataFrame.
        Çıkış  : Date + CPI_MoM + CPI_YoY
        Önkoşul: CPI_YoY için en az 13 aylık veri bulunmalıdır.
        """
        df = df.copy().sort_values("Date").reset_index(drop=True)
        df["CPI_MoM"] = df["CPI"].pct_change(periods=1)  * 100   # gerçek aylık değişim %
        df["CPI_YoY"] = df["CPI"].pct_change(periods=12) * 100   # gerçek yıllık değişim %
        df.drop(columns=["CPI"], inplace=True)
        return df[["Date", "CPI_MoM", "CPI_YoY"]]

    @staticmethod
    def _engineer_daily(df: pd.DataFrame) -> pd.DataFrame:
        """
        Günlük USD/TRY ve BIST100 feature'larını türetir.

        Bu metod, aylık feature'lar zaten ffill ile günlük takvime
        taşındıktan SONRA çağrılır; yalnızca günlük frekansa özgü
        hesaplamalar (pct_change, rolling) burada yapılır.

        Giriş  : USDTRY, BIST100 sütunlarını içeren günlük DataFrame.
                 Rate_Level, Rate_Change, CPI_MoM, CPI_YoY, Real_Rate
                 sütunları zaten mevcut olabilir (dokunulmaz).
        Çıkış  : Ham USDTRY/BIST100 sütunları kaldırılmış, feature sütunları eklenmiş DataFrame.
        """
        df = df.copy()

        # ── USD/TRY ───────────────────────────────────────────────────────────
        if "USDTRY" in df.columns:
            df["USDTRY_Return"]      = df["USDTRY"].pct_change()
            df["USDTRY_MA7"]         = df["USDTRY"].rolling(7).mean()
            df["USDTRY_Volatility7"] = df["USDTRY_Return"].rolling(7).std()
            df.drop(columns=["USDTRY"], inplace=True)

        # ── BIST100 ───────────────────────────────────────────────────────────
        if "BIST100" in df.columns:
            first_bist = df["BIST100"].iloc[0] if df["BIST100"].iloc[0] != 0 else 1.0
            df["BIST100_Norm"]   = df["BIST100"] / first_bist
            df["BIST100_Return"] = df["BIST100"].pct_change()
            df["BIST100_MA7"]    = df["BIST100_Norm"].rolling(7).mean()
            df.drop(columns=["BIST100"], inplace=True)

        # ── EUR/TRY ───────────────────────────────────────────────────────────
        if "EURTRY" in df.columns:
            df["EURTRY_Return"]      = df["EURTRY"].pct_change()
            df["EURTRY_Volatility7"] = df["EURTRY_Return"].rolling(7).std()
            df.drop(columns=["EURTRY"], inplace=True)

        # ── VIX (küresel risk iştahı) ─────────────────────────────────────────
        if "VIX" in df.columns:
            df["VIX_Level"]   = df["VIX"]
            df["VIX_Change"]  = df["VIX"].diff()
            df.drop(columns=["VIX"], inplace=True)

        # ── Altın (TRY getirisi için USDTRY ile çarp) ────────────────────────
        if "GOLD_USD" in df.columns:
            df["Gold_USD_Return"] = df["GOLD_USD"].pct_change()
            # Altın/TRY = Altın/USD * USD/TRY
            if "USDTRY_Return" in df.columns:
                df["Gold_TRY_Return"] = (
                    (1 + df["Gold_USD_Return"]) * (1 + df["USDTRY_Return"]) - 1
                )
            df.drop(columns=["GOLD_USD"], inplace=True)

        # ── Brent Petrol ──────────────────────────────────────────────────────
        if "OIL_USD" in df.columns:
            df["Oil_USD_Return"] = df["OIL_USD"].pct_change()
            df.drop(columns=["OIL_USD"], inplace=True)

        # ── DXY (Dolar Endeksi) ───────────────────────────────────────────────
        if "DXY" in df.columns:
            df["DXY_Return"]      = df["DXY"].pct_change()
            df["DXY_Volatility7"] = df["DXY_Return"].rolling(7).std()
            df.drop(columns=["DXY"], inplace=True)

        # ── ABD 10Y Faizi ─────────────────────────────────────────────────────
        if "US10Y" in df.columns:
            df["US10Y_Level"]  = df["US10Y"]
            df["US10Y_Change"] = df["US10Y"].diff()
            df.drop(columns=["US10Y"], inplace=True)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── Özellik İsimleri ──────────────────────────────────────────────────────
    @staticmethod
    def macro_feature_names(include_rates: bool = True) -> list:
        """
        Pipeline'ın beklediği tüm makro özellik isimlerini döndürür.

        Args:
            include_rates: False ise faiz/enflasyon sütunları listeye eklenmez
                           (bu veriler yokken feature_names ile tutarlılık için).
        """
        base = [
            "USDTRY_Return",
            "USDTRY_MA7",
            "USDTRY_Volatility7",
            "BIST100_Norm",
            "BIST100_Return",
            "BIST100_MA7",
            # Faz 4.1 — Global göstergeler (opsiyonel; yoksa feature pipeline atlar)
            "EURTRY_Return",
            "EURTRY_Volatility7",
            "VIX_Level",
            "VIX_Change",
            "Gold_USD_Return",
            "Gold_TRY_Return",
            "Oil_USD_Return",
            "DXY_Return",
            "DXY_Volatility7",
            "US10Y_Level",
            "US10Y_Change",
        ]
        rate_inflation = [
            "Rate_Level",
            "Rate_Change",
            "CPI_YoY",
            "CPI_MoM",
            "Real_Rate",
        ]
        return base + rate_inflation if include_rates else base
