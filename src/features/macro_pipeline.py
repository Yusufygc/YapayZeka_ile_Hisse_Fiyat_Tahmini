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
    USDTRY_Return, USDTRY_Volatility7
    BIST100_Return, BIST100_MA7

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

import pandas as pd

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
    # Sektörel Endeksler
    "XBANK":   "XBANK.IS",
    "XUSIN":   "XUSIN.IS",
    "XHOLD":   "XHOLD.IS",
    "XULAS":   "XULAS.IS",
    "XTCRT":   "XTCRT.IS",
    "XTEK":    "XTEK.IS",
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
    # Sektörel Endeksler
    "XBANK":         "XBANK.csv",
    "XUSIN":         "XUSIN.csv",
    "XHOLD":         "XHOLD.csv",
    "XULAS":         "XULAS.csv",
    "XTCRT":         "XTCRT.csv",
    "XTEK":          "XTEK.csv",
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
        window = self._macro_date_window(start_date, end_date)
        self._refresh_daily_caches(window["buf_daily_str"])
        usdtry_df, bist100_df = self._load_required_daily_frames()
        buf_daily = window["buf_daily_str"]
        # CPI_YoY için 12 aylık geriye bakış gerekir; aylık veriler daha erken başlamalı
        buf_monthly = window["buf_monthly_str"]

        # ── Günlük veriler (yfinance) ─────────────────────────────────────────
        if usdtry_df is None or bist100_df is None:
            print("  [MACRO] Döviz/endeks verisi yüklenemedi, makro özellikler atlanacak.")
            return pd.DataFrame()

        # ── Tarih filtreleri ───────────────────────────────────────────────────
        s = window["start"]
        e = window["end"]
        buf = window["buf_daily"]
        buf_m = window["buf_monthly"]

        usdtry_df = self._filter_macro_frame(usdtry_df, buf, e)
        bist100_df = self._filter_macro_frame(bist100_df, buf, e)

        # Faz 4.1: Genişletilmiş global göstergeler (try/except — her biri opsiyonel)
        _global_dfs = self._refresh_global_daily_frames(buf_daily, buf, e)

        # ── Aylık veriler (EVDS/FRED) ─────────────────────────────────────────
        self._refresh_monthly_caches(buf_monthly)

        interest_df = self._load_cache("INTEREST_RATE")
        cpi_df      = self._load_cache("CPI")
        interest_df = self._filter_macro_frame(interest_df, buf_m, e)
        cpi_df      = self._filter_macro_frame(cpi_df, buf_m, e)

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 1: Aylık feature'ları kendi frekansında hesapla (ffill'den ÖNCE)
        # ══════════════════════════════════════════════════════════════════════
        monthly_rate_feats, monthly_cpi_feats = self._build_lagged_monthly_features(
            interest_df=interest_df,
            cpi_df=cpi_df,
        )

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 2: Günlük takvim kur (USD/TRY + BIST100 outer join)
        # ══════════════════════════════════════════════════════════════════════
        macro = self._build_base_daily_macro(usdtry_df, bist100_df)

        # Faz 4.1: Global göstergeleri left-merge ile ekle (yoksa sütun oluşmaz)
        macro = self._merge_global_daily_frames(macro, _global_dfs)

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 3: Aylık feature dataframe'lerini günlük takvime ffill ile taşı
        # ══════════════════════════════════════════════════════════════════════
        macro = self._merge_monthly_feature_frames(
            macro,
            monthly_rate_feats=monthly_rate_feats,
            monthly_cpi_feats=monthly_cpi_feats,
        )

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 4: Günlük USD/TRY & BIST100 feature'larını türet
        # ══════════════════════════════════════════════════════════════════════
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

    def _refresh_monthly_caches(self, buffer_start: str) -> None:
        for key in _MONTHLY_SERIES_KEYS:
            if self._is_stale(key, _STALE_DAYS_MONTHLY):
                self._update_monthly_cache(key, buffer_start)

    def _refresh_global_daily_frames(self, buf_daily: str, buf, e) -> dict:
        """Faz 4.1 genişletilmiş global günlük göstergeleri yeniler + filtreler.

        EURTRY/VIX/GOLD_USD/OIL_USD/DXY/US10Y; her biri opsiyonel — biri
        başarısız olursa atlanır (sütun oluşmaz). Returns: {key: filtreli df}.
        """
        global_keys = ["EURTRY", "VIX", "GOLD_USD", "OIL_USD", "DXY", "US10Y"]
        global_dfs: dict = {}
        for gkey in global_keys:
            try:
                if self._is_stale(gkey, _STALE_DAYS_DAILY):
                    self._update_daily_cache(gkey, buf_daily)
                gdf = self._load_cache(gkey)
                if gdf is not None and not gdf.empty:
                    global_dfs[gkey] = self._filter_macro_frame(gdf, buf, e)
            except Exception as exc:
                print(f"  [MACRO] {gkey} atlanıyor: {exc}")
        return global_dfs

    def _load_required_daily_frames(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return self._load_cache("USDTRY"), self._load_cache("BIST100")

    @staticmethod
    def _filter_macro_frame(
        df: Optional[pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        return macro_transforms.filter_macro_frame(df, start, end)

    def _build_lagged_monthly_features(
        self,
        *,
        interest_df: Optional[pd.DataFrame],
        cpi_df: Optional[pd.DataFrame],
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return (
            self._lag_monthly_rate_features(interest_df),
            self._lag_monthly_cpi_features(cpi_df),
        )

    def _lag_monthly_rate_features(
        self,
        interest_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if interest_df is None or interest_df.empty:
            return None
        monthly_rate_feats = self._engineer_monthly_rate(interest_df)
        return macro_transforms.lag_monthly_features(
            monthly_rate_feats,
            lag_days=self.rate_release_lag_days,
            raw_date_column="Rate_Raw_Date",
        )

    def _lag_monthly_cpi_features(
        self,
        cpi_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if cpi_df is None or cpi_df.empty:
            return None
        monthly_cpi_feats = self._engineer_monthly_cpi(cpi_df)
        return macro_transforms.lag_monthly_features(
            monthly_cpi_feats,
            lag_days=self.cpi_release_lag_days,
            raw_date_column="CPI_Raw_Date",
        )

    @staticmethod
    def _build_base_daily_macro(
        usdtry_df: pd.DataFrame,
        bist100_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return macro_transforms.build_base_daily_macro(usdtry_df, bist100_df)

    @staticmethod
    def _merge_global_daily_frames(
        macro: pd.DataFrame,
        global_dfs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return macro_transforms.merge_global_daily_frames(macro, global_dfs)

    def _merge_monthly_feature_frames(
        self,
        macro: pd.DataFrame,
        *,
        monthly_rate_feats: Optional[pd.DataFrame],
        monthly_cpi_feats: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        return macro_transforms.merge_monthly_feature_frames(
            macro,
            monthly_rate_feats=monthly_rate_feats,
            monthly_cpi_feats=monthly_cpi_feats,
        )

    @staticmethod
    def _merge_ffill_monthly_features(
        macro: pd.DataFrame,
        monthly_feats: Optional[pd.DataFrame],
        *,
        columns: list[str],
    ) -> pd.DataFrame:
        return macro_transforms.merge_ffill_monthly_features(
            macro,
            monthly_feats,
            columns=columns,
        )

    # ── Özellik Mühendisliği ──────────────────────────────────────────────────

    @staticmethod
    def _engineer_monthly_rate(df: pd.DataFrame) -> pd.DataFrame:
        return macro_feature_engineering.engineer_monthly_rate(df)

    @staticmethod
    def _engineer_monthly_cpi(df: pd.DataFrame) -> pd.DataFrame:
        return macro_feature_engineering.engineer_monthly_cpi(df)

    @staticmethod
    def _engineer_daily(df: pd.DataFrame) -> pd.DataFrame:
        return macro_feature_engineering.engineer_daily(df)

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
            "USDTRY_Volatility7",
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
            # Sektörel Getiriler
            "XBANK_Return",
            "XUSIN_Return",
            "XHOLD_Return",
            "XULAS_Return",
            "XTCRT_Return",
            "XTEK_Return",
        ]
        rate_inflation = [
            "Rate_Level",
            "Rate_Change",
            "CPI_YoY",
            "CPI_MoM",
            "Real_Rate",
        ]
        return base + rate_inflation if include_rates else base
