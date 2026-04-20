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

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────

# yfinance ile çekilen günlük seriler
_YFINANCE_TICKERS = {
    "USDTRY": "USDTRY=X",
    "BIST100": "XU100.IS",
}

# FRED'den çekilen aylık seriler
_FRED_SERIES = {
    "INTEREST_RATE": "INTDSRTRM193N",    # IMF iskonto faizi (aylık, % yıllık)
    "CPI":           "TURCPIALLMINMEI",  # Türkiye TÜFE endeksi (aylık)
}

# Cache dosya adları
_CACHE_FILES = {
    "USDTRY":        "USDTRY.csv",
    "BIST100":       "BIST100.csv",
    "INTEREST_RATE": "INTEREST_RATE.csv",
    "CPI":           "CPI.csv",
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

    def __init__(self, cache_dir: str = "data/macro"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # A. Günlük Veri — yfinance
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _download_yfinance(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
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
            col = ticker.replace("=X", "").replace(".IS", "")
            return raw[["Date", "Close"]].rename(columns={"Close": col})
        except Exception as exc:
            print(f"  [MACRO] yfinance {ticker}: {exc}")
            return None

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, _CACHE_FILES[key])

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

        new_data = self._download_yfinance(ticker, fetch_start, fetch_end)
        if new_data is not None and not new_data.empty:
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

    def _update_monthly_cache(self, key: str, start: str) -> None:
        """
        Aylık FRED verisini günceller.
        Başarısız olursa manual CSV olup olmadığını kontrol eder.
        """
        series_id = _FRED_SERIES[key]
        path      = self._cache_path(key)
        existing  = self._load_cache(key)

        # Var olan veriden ileriye devam et
        if existing is not None and not existing.empty:
            last_cached = existing["Date"].max()
            fetch_start = (last_cached - timedelta(days=60)).strftime("%Y-%m-%d")  # üst üste biraz al
        else:
            fetch_start = start

        fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
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

        # ── Aylık veriler (FRED) ──────────────────────────────────────────────
        for key in _FRED_SERIES:
            if self._is_stale(key, _STALE_DAYS_MONTHLY):
                self._update_monthly_cache(key, buf_monthly)

        interest_df = self._load_cache("INTEREST_RATE")
        cpi_df      = self._load_cache("CPI")

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
        interest_df = _filter_monthly(interest_df) if interest_df is not None else None
        cpi_df      = _filter_monthly(cpi_df)      if cpi_df      is not None else None

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 1: Aylık feature'ları kendi frekansında hesapla (ffill'den ÖNCE)
        # ══════════════════════════════════════════════════════════════════════
        monthly_rate_feats = None
        if interest_df is not None and not interest_df.empty:
            monthly_rate_feats = self._engineer_monthly_rate(interest_df)

        monthly_cpi_feats = None
        if cpi_df is not None and not cpi_df.empty:
            monthly_cpi_feats = self._engineer_monthly_cpi(cpi_df)

        # ══════════════════════════════════════════════════════════════════════
        # ADIM 2: Günlük takvim kur (USD/TRY + BIST100 outer join)
        # ══════════════════════════════════════════════════════════════════════
        macro = pd.merge(usdtry_df, bist100_df, on="Date", how="outer")
        macro.sort_values("Date", inplace=True)
        macro.ffill(inplace=True)

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
        df["Rate_Level"]  = df["INTEREST_RATE"]
        df["Rate_Change"] = df["INTEREST_RATE"].diff()   # gerçek aylık fark (ör. 42 → 45 → +3)
        df.drop(columns=["INTEREST_RATE"], inplace=True)
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
        ]
        rate_inflation = [
            "Rate_Level",
            "Rate_Change",
            "CPI_YoY",
            "CPI_MoM",
            "Real_Rate",
        ]
        return base + rate_inflation if include_rates else base
