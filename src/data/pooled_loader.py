# -*- coding: utf-8 -*-
"""Pooled (panel) loader for the E2 global model.

Tasarim: docs/wiki/e2-faz2-pooled-cv-design.md.

Tum hisse CSV'lerini tek uzun panele yukler:
    symbol, Date, <FeaturePipeline ozellikleri>, target, target_date,
    sector, symbol_id, liq_log, vol

- Ozellikler her sembol icin AYRI uretilir (capraz-sembol kontaminasyon yok),
  sonra birlestirilir.
- target = log(close[t+h]/close[t]); target_date = sembolun t+h tarihi (kesin
  purge icin -> pooled_cv). Son h satir (NaN target) varsayilan olarak dusulur;
  `keep_unlabeled=True` ile skorlama icin tutulur (target/target_date NaN).
- Kosullandirma: sector (universe, GICS), symbol_id (stabil), liq_log (trailing
  TL ciro medyani, causal), vol (trailing realized-vol, causal). Bucketleme/
  olcekleme CV/model asamasina birakilir (loader'da global quantile = lookahead).
- Survivorship: `bist_universe.csv` `Delisted_Date` dolu semboller `include_delisted`
  ile kontrol edilir (True: gecmis DAHIL, delisting sonrasi bayat satirlar atilir;
  False: sembol disarida). Evrende hic delisting yoksa panelin yalniz bugun hayatta
  kalanlari icerdigi (survivorship yanlilik riski) `survivorship_status` ve uyari ile
  bildirilir — sessiz gecilmez. Delisting verisi `Delisted_Date` kolonuna elle/
  kaynaktan girilir (sablon: data/meta/delisted_template.csv).
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.clean_store import CleanStore
from src.data.freshness import DEFAULT_MAX_BUSINESS_DAYS, warn_if_stale
from src.data.ohlcv_clean import clean_ohlcv_csv
from src.data.panel_columns import PIPELINE_RAW_COLS
from src.features.feature_pipeline import FeaturePipeline
from src.features.pooled_feature_cache import PooledFeatureCache
from src.features.sector_mapping import normalize_symbol  # tests monkeypatch eder

logger = logging.getLogger(__name__)

_EXCLUDE_STEMS = {"bist_universe", "advisory_history"}
_NON_FEATURE = PIPELINE_RAW_COLS      # tek tanim: src/data/panel_columns.py


@dataclass(frozen=True)
class PooledLoaderConfig:
    data_dir: str
    universe_file: str
    target_horizon: int = 5
    feature_mode: str = "stationary_features"
    min_rows: int = 60            # bundan az ham satirli sembol atlanir
    include_delisted: bool = True
    liq_lookback: int = 63
    vol_lookback: int = 63
    min_usable_rows: int = 10     # ozellik+target sonrasi en az kullanilabilir satir
    # True: hedefi henuz gerceklesmemis son h satir (target/target_date NaN) panelde
    # kalir -> skorlama en guncel piyasa gunune yapilabilir (K4). Egitim/OOS icin
    # cagiran `panel[panel["target"].notna()]` ile etiketli alt kumeyi alir.
    keep_unlabeled: bool = False
    # K7: birincil `data_dir`'de olmayan semboller bu klasorlerden (sirayla) tamamlanir
    # (ornegin "data/yedek"). Birincil dosya HER ZAMAN onceliklidir; dosya tasinmaz.
    fallback_dirs: tuple[str, ...] = ()
    # K8: kurumsal eylem karantinasi. |1 gunluk log-getiri| > jump_threshold olan gun
    # (bolunme/duzeltilmemis olcek kirigi; BIST gunluk fiyat marji bu esigin cok
    # altinda) etrafindaki satirlar panelden atilir: ozellik penceresi (jump gunu ve
    # sonraki `quarantine_rows` satir) + hedefi sicramayi iceren satirlar (t..t+h).
    quarantine_jumps: bool = False
    jump_threshold: float = 0.30
    quarantine_rows: int = 63
    # Veri tazeligi: son tarihi bugunden `freshness_max_business_days` is gunuden fazla
    # geride olan semboller uyari ile raporlanir (panelden ATILMAZ; karar cagirana ait).
    freshness_max_business_days: int = DEFAULT_MAX_BUSINESS_DAYS
    freshness_today: pd.Timestamp | None = None   # None -> Istanbul bugunu (test icin enjekte)
    # Sembol basi Parquet ozellik onbellegi (src/features/pooled_feature_cache.py).
    # None = kapali (varsayilan; FeaturePipeline'i patch'leyen testler etkilenmez).
    feature_cache_dir: str | None = None
    # Temizlenmis OHLCV'nin Parquet deposu (src/data/clean_store.py). CSV'ler kaynak kalir;
    # depo yalniz degisen CSV'leri yeniden temizler. None = kapali (eski per-CSV yol).
    clean_store_dir: str | None = None


class PooledPanelLoader:
    def __init__(self, cfg: PooledLoaderConfig) -> None:
        self.cfg = cfg
        self.report: dict[str, str] = {}  # symbol -> skip reason / "ok"
        # "unknown" | "no_delisted_in_universe" | "delisted_handled"
        self.survivorship_status: str = "unknown"
        self.quarantined_days: dict[str, int] = {}   # symbol -> atilan satir sayisi (K8)
        self.source_dir: dict[str, str] = {}         # symbol -> yuklenen klasor (K7)
        self.rejected_rows: dict[str, int] = {}      # symbol -> karantinaya alinan bozuk satir
        self.stale_symbols: dict[str, str] = {}      # symbol -> bayatlik mesaji (veri tazeligi)
        self._paths: dict[str, str] | None = None
        self._fcache = (PooledFeatureCache(cfg.feature_cache_dir)
                        if cfg.feature_cache_dir else None)
        self._cstore = CleanStore(cfg.clean_store_dir) if cfg.clean_store_dir else None
        self._clean_frames: dict[str, pd.DataFrame] | None = None
        # {enabled, content_hash, refresh (ozet)}; depo kapaliyken yalniz enabled=False
        self.clean_store_info: dict[str, object] = {"enabled": self._cstore is not None}

    @property
    def feature_cache_stats(self) -> dict[str, int | bool]:
        """{enabled, hits, misses, errors}; onbellek kapaliyken sayaclar 0."""
        base = self._fcache.stats() if self._fcache else {"hits": 0, "misses": 0, "errors": 0}
        return {"enabled": self._fcache is not None, **base}

    # ----------------------------------------------------------------- public
    def load(self) -> pd.DataFrame:
        sector_map = self._sector_map()
        delisted = self._delisted_map()
        self._report_survivorship(delisted)
        frames: list[pd.DataFrame] = []
        self._prepare_clean_store()
        for sym in self._symbols():
            if sym in delisted and not self.cfg.include_delisted:
                self.report[sym] = "delisted_excluded"
                continue
            try:
                f = self._engineer_symbol(
                    sym, sector_map.get(normalize_symbol(sym), "Unknown"), delisted.get(sym))
            except Exception as exc:  # tek sembol hatasi paneli durdurmasin
                self.report[sym] = f"error:{str(exc)[:60]}"
                continue
            if f is None or f.empty:
                continue
            frames.append(f)
            self.report[sym] = "ok"
        if self._fcache is not None:
            logger.info("Pooled ozellik onbellegi: %d hit, %d miss, %d hata (%s)",
                        self._fcache.hits, self._fcache.misses, self._fcache.errors,
                        self.cfg.feature_cache_dir)
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, ignore_index=True)
        self._check_freshness(panel)
        panel = panel.sort_values(["Date", "symbol"]).reset_index(drop=True)
        codes = {s: i for i, s in enumerate(sorted(panel["symbol"].unique()))}
        panel["symbol_id"] = panel["symbol"].map(codes).astype(int)
        return panel

    # ---------------------------------------------------------------- helpers
    def _symbol_paths(self) -> dict[str, str]:
        """{sembol: csv yolu}. Birincil data_dir once; eksikler fallback_dirs'ten (K7)."""
        if self._paths is None:
            paths: dict[str, str] = {}
            for directory in (self.cfg.data_dir, *self.cfg.fallback_dirs):
                for p in sorted(glob.glob(os.path.join(directory, "*.csv"))):
                    stem = os.path.basename(p)[:-4]
                    if stem in _EXCLUDE_STEMS or stem in paths:
                        continue
                    paths[stem] = p
                    self.source_dir[stem] = directory
            self._paths = paths
        return self._paths

    def _symbols(self) -> list[str]:
        return sorted(self._symbol_paths())

    def _prepare_clean_store(self) -> None:
        """Depo aciksa TUM kaynak CSV'leri esitle (kalkanlari dusurur; `limit` depoyu daraltmaz)
        ve bu kosunun sembollerinin temiz cercevelerini + icerik hash'ini al."""
        if self._cstore is None or self._clean_frames is not None:
            return
        report = self._cstore.refresh(self._symbol_paths())
        symbols = self._symbols()
        self._clean_frames = self._cstore.load(symbols)
        self.clean_store_info.update(
            content_hash=self._cstore.content_hash(symbols), refresh=report.summary(),
            failed=list(report.failed))

    def _clean_frame(self, symbol: str, path: str) -> pd.DataFrame:
        """Sembolun temiz OHLCV'si: depodan (varsa) yoksa dogrudan CSV'den (ayni sonuc)."""
        self._prepare_clean_store()
        if self._clean_frames is not None and symbol in self._clean_frames:
            return self._clean_frames[symbol]
        return clean_ohlcv_csv(path, label=symbol)

    def _check_freshness(self, panel: pd.DataFrame) -> None:
        """Sembol bazinda son Date bugunden cok geride ise tek ozet uyari + kayit."""
        last = panel.groupby("symbol")["Date"].max()
        stale = warn_if_stale(
            last.to_dict(), "Pooled panel veri tazeligi", self.cfg.freshness_today,
            self.cfg.freshness_max_business_days)
        self.stale_symbols = {s: r.message for s, r in stale.items()}
        if self.rejected_rows:
            logger.warning(
                "Pooled panel: %d sembolde toplam %d bozuk satir karantinaya alindi "
                "(tools/repair_symbol_csv.py ile onarin).",
                len(self.rejected_rows), sum(self.rejected_rows.values()))

    def _sector_map(self) -> dict[str, str]:
        if not os.path.exists(self.cfg.universe_file):
            return {}
        uni = pd.read_csv(self.cfg.universe_file, encoding="utf-8-sig")
        if "Symbol" not in uni.columns or "Sector" not in uni.columns:
            return {}
        out = {}
        for _, r in uni.iterrows():
            val = r.get("Sector")
            sec = "" if pd.isna(val) else str(val).strip()
            out[normalize_symbol(r["Symbol"])] = sec or "Unknown"
        return out

    def _universe_frame(self) -> pd.DataFrame | None:
        if not os.path.exists(self.cfg.universe_file):
            return None
        uni = pd.read_csv(self.cfg.universe_file, encoding="utf-8-sig")
        return uni if "Symbol" in uni.columns else None

    def _delisted_map(self) -> dict[str, pd.Timestamp | None]:
        """{sembol: delisting tarihi (bilinmiyorsa None)} — yalniz delisted isaretliler.

        Delisted: `Delisted_Date` gecerli tarih VEYA `Status` 'delisted'.
        """
        uni = self._universe_frame()
        if uni is None:
            return {}
        dates = (pd.to_datetime(uni["Delisted_Date"], errors="coerce")
                 if "Delisted_Date" in uni.columns
                 else pd.Series(pd.NaT, index=uni.index))
        status = (uni["Status"].fillna("").astype(str).str.strip().str.lower()
                  if "Status" in uni.columns else pd.Series("", index=uni.index))
        out: dict[str, pd.Timestamp | None] = {}
        for sym, d, st in zip(uni["Symbol"].astype(str).str.strip().str.upper(), dates, status):
            if pd.notna(d):
                out[sym] = pd.Timestamp(d)
            elif st == "delisted":
                out[sym] = None
        return out

    def _report_survivorship(self, delisted: dict) -> None:
        if delisted:
            self.survivorship_status = "delisted_handled"
            return
        self.survivorship_status = "no_delisted_in_universe"
        logger.warning(
            "Pooled evren hayatta kalma yanliligi riski: %s icinde hic delisted sembol "
            "yok; panel yalniz bugun listelenen hisseleri icerir. Delisting verisini "
            "Delisted_Date kolonuna girin (sablon: data/meta/delisted_template.csv).",
            self.cfg.universe_file)

    def _engineer_symbol(
        self, symbol: str, sector: str, delisted_on: pd.Timestamp | None = None,
    ) -> pd.DataFrame | None:
        path = self._symbol_paths().get(symbol) or os.path.join(self.cfg.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            self.report[symbol] = "no_csv"
            return None
        # Tek-sembol loader ile ORTAK temizleme (src/data/ohlcv_clean.py): alan-sayisi
        # dogrulamasi (bozuk satir karantina, kolon kaymasi yok), parse_dates, tekrar
        # tarih (son tutulur), Open/High/Low'u Adj/Close ile olcekleme, sifir-hacim atma.
        raw = self._clean_frame(symbol, path)
        if raw.empty or "Date" not in raw.columns or "Close" not in raw.columns:
            self.report[symbol] = "bad_schema"
            return None
        rejected = int(raw.attrs.get("rejected_rows", 0))
        if rejected:
            self.rejected_rows[symbol] = rejected
        raw = raw.dropna(subset=["Date", "Close"]).reset_index(drop=True)
        if delisted_on is not None:
            # delisting sonrasi satirlar (bayat/dolgu) panelde yer almaz
            raw = raw[raw["Date"] <= delisted_on].reset_index(drop=True)
        if len(raw) < self.cfg.min_rows:
            self.report[symbol] = f"too_short:{len(raw)}"
            return None

        # Yalniz standart OHLCV'yi FeaturePipeline'a ver; eslenmemis ham kolonlar
        # (or. "Kapanış" = ham fiyat seviyesi) ozellik olarak sizmasin.
        std_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if c in raw.columns]
        raw = raw[std_cols]

        # Ozellikler (per-symbol). Tests modul-seviyesi FeaturePipeline'i patch'ler.
        feat = self._engineer_features(symbol, raw)
        if feat is None or feat.empty or "Date" not in feat.columns:
            self.report[symbol] = "no_features"
            return None

        # Causal kosullandirma + target ham seriden (Date indexli).
        cond = self._causal_columns(raw)

        feat_cols = [c for c in feat.columns if c not in _NON_FEATURE]
        feat_clean = feat.drop(columns=["Close"], errors="ignore")
        merged = feat_clean.merge(cond, on="Date", how="inner")
        required = feat_cols if self.cfg.keep_unlabeled else feat_cols + ["target"]
        merged = merged.dropna(subset=required).reset_index(drop=True)
        if len(merged) < self.cfg.min_usable_rows:
            self.report[symbol] = f"few_usable:{len(merged)}"
            return None

        if self.cfg.quarantine_jumps:
            merged = self._apply_jump_quarantine(symbol, raw, merged)
            if len(merged) < self.cfg.min_usable_rows:
                self.report[symbol] = f"few_usable:{len(merged)}"
                return None

        merged["symbol"] = symbol
        merged["sector"] = sector
        keep = ["symbol", "Date", "Close"] + feat_cols + [
            "target", "target_date", "sector", "liq_log", "vol"
        ]
        return merged[keep]

    def _engineer_features(self, symbol: str, raw: pd.DataFrame) -> pd.DataFrame | None:
        """FeaturePipeline (macro_df=None); `feature_cache_dir` verildiyse Parquet onbellekli."""
        def compute() -> pd.DataFrame | None:
            fp = FeaturePipeline(feature_mode=self.cfg.feature_mode,
                                 enable_calendar_features=False)
            return fp.engineer_features(raw, macro_df=None, symbol=symbol)

        if self._fcache is None:
            return compute()
        return self._fcache.get_or_compute(
            symbol, raw, feature_mode=self.cfg.feature_mode,
            enable_calendar_features=False, compute=compute)

    def _apply_jump_quarantine(
        self, symbol: str, raw: pd.DataFrame, merged: pd.DataFrame,
    ) -> pd.DataFrame:
        """K8: sicrama gunu etrafindaki satirlari at. `raw` Date-sirali, ham OHLCV."""
        close = pd.to_numeric(raw["Close"], errors="coerce")
        jump_pos = np.flatnonzero(
            (np.log(close).diff().abs() > float(self.cfg.jump_threshold)).to_numpy())
        if jump_pos.size == 0:
            return merged
        h = max(1, int(self.cfg.target_horizon))
        q = max(0, int(self.cfg.quarantine_rows))
        bad_pos: set[int] = set()
        for p in jump_pos:
            # hedefi sicramayi iceren satirlar [p-h, p-1], sicrama gunu p, ozellik penceresi [p, p+q-1]
            bad_pos.update(range(max(0, p - h), min(len(raw), p + max(q, 1))))
        bad_dates = set(raw["Date"].iloc[sorted(bad_pos)])
        keep = ~merged["Date"].isin(bad_dates)
        self.quarantined_days[symbol] = int((~keep).sum())
        return merged[keep].reset_index(drop=True)

    def _causal_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        h = max(1, int(self.cfg.target_horizon))
        close = pd.to_numeric(raw["Close"], errors="coerce")
        vol_sh = pd.to_numeric(raw.get("Volume", pd.Series(np.nan, index=raw.index)), errors="coerce")
        # target: y[t] = log(close[t+h]/close[t]); target_date = t+h tarihi
        target = np.log(close.shift(-h) / close)
        target_date = raw["Date"].shift(-h)
        # liq_log: trailing TL ciro medyani (causal); vol: trailing realized-vol
        turnover = (close * vol_sh).clip(lower=0)
        liq_log = np.log1p(turnover.rolling(self.cfg.liq_lookback, min_periods=5).median())
        logret = np.log(close).diff()
        vol = logret.rolling(self.cfg.vol_lookback, min_periods=5).std()
        return pd.DataFrame({
            "Date": raw["Date"],
            "Close": close.values,
            "target": target.values,
            "target_date": target_date.values,
            "liq_log": liq_log.values,
            "vol": vol.values,
        })
