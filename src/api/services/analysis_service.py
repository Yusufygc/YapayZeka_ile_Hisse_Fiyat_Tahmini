# -*- coding: utf-8 -*-
"""GET /analysis/{symbol} servis katmani (API v2).

Yanit YALNIZ gecelik havuz ciktisindan (`data/serving_pool.db`, PeerStore son kosu)
ve piyasa verisi tazeliginden uretilir: yon/goreli sinyal Kol-B (`peer_*`,
`trend_*`, `confidence_*`), risk ve fiyat araligi risk modeli (`risk_*`,
`kolb_price_*`). Per-sembol yon modeli (Kol-A) emekli oldu; istek sirasinda
egitim/forecast tetiklenmez.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

from src.api.constants import INVESTMENT_DISCLAIMER
from src.api.schemas.analysis import (
    AnalysisResponse,
    ConfidenceBlock,
    DataBlock,
    DataQualityBlock,
    PeerBlock,
    ServingRunBlock,
)
from src.api.services.analysis_freshness import compute_freshness, trading_days_between
from src.api.services.data_quality_monitor import compute_psi_30d
from src.api.services.peer_service import PeerEnrichmentService, PeerStoreUnavailable
from src.api.services.serving_health import serving_pool_status
from src.serving.paths import default_serving_db_path

logger = logging.getLogger(__name__)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ISTANBUL = timezone(timedelta(hours=3))  # Turkiye sabit UTC+3 (DST yok)
_LEVELS = ("low", "medium", "high")
# as_of (havuz) ile piyasa CSV son tarihi arasi izin verilen islem gunu farki (verinin
# kendi bayatlik esigi ile ayni: 1 islem gunu).
_AS_OF_LAG_MAX_TRADING_DAYS = 1
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,10}$")


def _now() -> datetime:
    return datetime.now(tz=_ISTANBUL)


def _lower(label: str, steps: int = 1) -> str:
    """Guven etiketini `steps` basamak asagi ceker (alt sinir: low)."""
    idx = _LEVELS.index(label) if label in _LEVELS else 0
    return _LEVELS[max(0, idx - steps)]


def _read_market_tail(csv_path: str) -> Tuple[Optional[str], Optional[float]]:
    """Sembol CSV'sinin son (tarih, kapanis) cifti; dosya yok/okunamaz -> (None, None)."""
    if not os.path.exists(csv_path):
        return None, None
    from src.data.ohlcv_clean import clean_ohlcv_csv

    try:
        df = clean_ohlcv_csv(csv_path)
    except (OSError, ValueError) as exc:
        logger.warning("piyasa verisi okunamadi (%s): %s", csv_path, exc)
        return None, None
    if df.empty or "Close" not in df.columns:
        return None, None
    last = df.iloc[-1]
    close = float(last["Close"]) if last["Close"] == last["Close"] else None
    return str(last["Date"])[:10], close


class AnalysisService:
    def __init__(
        self,
        serving_db_path: Optional[str] = None,
        project_root: Optional[str] = None,
        data_dir: Optional[str] = None,
        calendar_path: Optional[str] = None,
        now_fn=None,
    ) -> None:
        self._project_root = project_root or _PROJECT_ROOT
        self._serving_db_path = serving_db_path or default_serving_db_path()
        self._data_dir = data_dir or os.path.join(self._project_root, "data")
        self._calendar_path = calendar_path
        self._now_fn = now_fn or _now
        self._peers = PeerEnrichmentService(self._serving_db_path)

    @property
    def serving_db_path(self) -> str:
        return self._serving_db_path

    @property
    def data_dir(self) -> str:
        return self._data_dir

    def build(self, symbol: str) -> AnalysisResponse:
        """Sembol icin PeerStore son kosusundan analiz yaniti kurar.

        Raises:
            ValueError: Sembol formati gecersizse.
        """
        symbol = str(symbol).upper().strip()
        if not _SYMBOL_RE.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        now = self._now_fn()
        generated_at = now.isoformat(timespec="seconds")
        today = now.date()

        pool = serving_pool_status(self._serving_db_path, today=today)
        serving_run = self._serving_block(pool)
        try:
            peer = self._peers.get_peer_block(symbol)
        except PeerStoreUnavailable as exc:
            # DB yok/kilitli/bozuk: hizmet hatasi (HTTP 503), sembol yoklugu (no_data) DEGIL.
            logger.error("analiz kurulamadi, serving DB kullanilamiyor (%s): %s", symbol, exc)
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="unavailable",
                generated_at=generated_at,
                confidence=ConfidenceBlock(
                    label="low",
                    warnings=["Gecelik havuz veritabanı şu an okunamıyor (unavailable); "
                              "daha sonra tekrar deneyin."],
                ),
                disclaimer=INVESTMENT_DISCLAIMER,
                serving_run=serving_run,
            )
        if peer is None:
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="no_data",
                generated_at=generated_at,
                confidence=ConfidenceBlock(
                    label="low",
                    warnings=["Sembol son gecelik havuz koşusunda bulunamadı (no_data)."],
                ),
                disclaimer=INVESTMENT_DISCLAIMER,
                serving_run=serving_run,
            )

        csv_path = os.path.join(self._data_dir, f"{symbol}.csv")
        last_date, last_close = _read_market_tail(csv_path)
        csv_last_date = last_date  # fallback'ten ONCE: gercek piyasa verisi son tarihi
        if last_date is None:
            # Piyasa CSV'si yok: serving as_of_date'e dus (fiyat bilinmiyor).
            last_date = str(peer.as_of_date)[:10] if peer.as_of_date else None
        data_block = self._data_block(last_date, last_close, today.isoformat())
        dq = self._data_quality(symbol, csv_path)

        conf, status = self._confidence(peer, data_block, pool, dq, csv_last_date, symbol)
        return AnalysisResponse(
            symbol=symbol,
            analysis_status=status,
            generated_at=generated_at,
            data=data_block,
            confidence=conf,
            disclaimer=INVESTMENT_DISCLAIMER,
            serving_run=serving_run,
            data_quality=dq,
            peer=peer,
        )

    # ------------------------------------------------------------------ blocks
    def _serving_block(self, pool: dict) -> Optional[ServingRunBlock]:
        if not pool.get("available"):
            return None
        return ServingRunBlock(
            run_id=pool.get("run_id"),
            model_name=pool.get("model_name"),
            as_of_date=pool.get("as_of_date"),
            age_days=pool.get("age_days"),
            stale=bool(pool.get("stale")),
        )

    def _data_block(self, last_date: Optional[str], last_close: Optional[float], today: str) -> DataBlock:
        if not last_date:
            return DataBlock()
        fresh = compute_freshness(last_date, today=today, calendar_path=self._calendar_path)
        return DataBlock(
            last_observed_date=last_date,
            last_close=last_close,
            data_freshness=fresh.status,
            staleness_days=fresh.staleness_days,
        )

    def _data_quality(self, symbol: str, csv_path: str) -> DataQualityBlock:
        try:
            r = compute_psi_30d(csv_path)
        except Exception as exc:  # noqa: BLE001 - PSI monitoru yaniti bozmasin
            return DataQualityBlock(
                psi_status="unavailable", stale_warning=True,
                reason=f"monitor_failed:{type(exc).__name__}",
            )
        return DataQualityBlock(
            psi_30d=r.psi_30d, psi_status=r.psi_status,
            stale_warning=r.stale_warning, reason=r.reason,
        )

    # -------------------------------------------------------------- confidence
    def _confidence(
        self,
        peer: PeerBlock,
        data: DataBlock,
        pool: dict,
        dq: DataQualityBlock,
        csv_last_date: Optional[str] = None,
        symbol: str = "",
    ) -> Tuple[ConfidenceBlock, str]:
        """Ust-seviye guven: Kol-B etiketi; bayat veri / as_of geriligi / risk durumu / drift dusurur."""
        label = peer.confidence_label if peer.confidence_label in _LEVELS else "low"
        reasons: List[str] = list(peer.confidence_reasons)
        warnings: List[str] = list(peer.confidence_warnings)

        data_stale = data.data_freshness == "stale_data"
        pool_stale = bool(pool.get("stale"))
        if data_stale:
            warnings.append(
                f"Piyasa verisi {data.staleness_days} işlem günü eski; analiz dikkatli yorumlanmalı."
            )
        if pool_stale:
            warnings.append(
                f"Gecelik havuz koşusu {pool.get('age_days')} gün önce "
                f"({pool.get('as_of_date')}); sonuçlar güncel olmayabilir."
            )
        as_of_lag = self._as_of_lag_warning(peer, csv_last_date)
        if as_of_lag:
            warnings.append(as_of_lag)
        if data_stale or pool_stale or as_of_lag:
            label = "low"

        risk_method = getattr(peer, "risk_method", None)
        if risk_method is None:
            warnings.append(self._missing_risk_warning(peer, symbol))
            label = _lower(label)
        elif risk_method == "naive_fallback":
            warnings.append(
                "Risk modeli yedek (naive) yönteme düştü; risk seviyesi ve fiyat aralığı "
                "düşük güvenlidir."
            )
            label = _lower(label)

        if dq.psi_status == "major_drift" and dq.psi_30d is not None:
            warnings.append(f"Veri dağılımı belirgin kaydı (PSI 30g={dq.psi_30d:.2f}).")
            label = "low"
        elif dq.psi_status == "moderate_drift" and dq.psi_30d is not None:
            warnings.append(f"Veri dağılımı orta düzeyde kaydı (PSI 30g={dq.psi_30d:.2f}).")
            if label == "high":
                label = "medium"
        elif dq.psi_status == "stable" and label != "low":
            reasons.append("Veri dağılımı stabil (PSI 30g düşük).")

        risk_level: Any = getattr(peer, "risk_level", None)
        if risk_level and label != "low":
            reasons.append(f"Risk seviyesi: {risk_level}.")

        if data_stale or pool_stale or as_of_lag:
            status = "stale_data"
        elif label == "low":
            status = "low_confidence"
        else:
            status = "ok"
        return ConfidenceBlock(label=label, reasons=reasons, warnings=warnings), status

    def _as_of_lag_warning(self, peer: PeerBlock, csv_last_date: Optional[str]) -> Optional[str]:
        """Havuz as_of_date piyasa CSV'sinin son tarihinden > 1 islem gunu gerideyse uyari.

        Fiyat bandi ve fiyat capasi (kolb_price_*) as_of kapanisina aittir; CSV ilerlediyse
        band guncel fiyata gore yorumlanamaz. Tarih okunamazsa / karsilastirilamazsa None.
        """
        if not csv_last_date or not peer.as_of_date:
            return None
        as_of = str(peer.as_of_date)[:10]
        try:
            lag = trading_days_between(as_of, csv_last_date[:10], self._calendar_path)
        except (ValueError, TypeError) as exc:
            logger.warning("as_of/CSV tarih karsilastirmasi yapilamadi (%s, %s): %s",
                           as_of, csv_last_date, exc)
            return None
        if lag <= _AS_OF_LAG_MAX_TRADING_DAYS:
            return None
        return (
            f"Gecelik havuz {as_of} kapanışına göre üretildi; piyasa verisi {csv_last_date[:10]} "
            f"tarihine ilerledi ({lag} işlem günü). Fiyat aralığı ve fiyat çapası eski "
            f"{as_of} kapanışına aittir."
        )

    @staticmethod
    def _missing_risk_warning(peer: PeerBlock, symbol: str = "") -> str:
        """risk_method None: kosu-duzeyi hata mi, yoksa sembole ozgu neden mi (run config 'missing')?"""
        val = peer.risk_validation if isinstance(peer.risk_validation, dict) else {}
        if val.get("error"):
            return ("Risk modeli bu gece çalışmadı (koşu düzeyi hata); risk seviyesi ve "
                    "fiyat aralığı üretilemedi.")
        missing = val.get("missing") if isinstance(val.get("missing"), dict) else {}
        reason = missing.get(symbol)
        if reason == "quarantine":
            return ("Sembolde yakın zamanda ölçek kırığı (bölünme/düzeltilmemiş fiyat) saptandı; "
                    "ilgili günler karantinaya alındı ve risk modelinin gerektirdiği temiz geçmiş "
                    "(~63 işlem günü + pencere) henüz oluşmadı; risk seviyesi ve fiyat aralığı yok.")
        if reason == "short_history":
            return ("Sembolün fiyat geçmişi risk modeli için yetersiz (<64 işlem günü); "
                    "risk seviyesi ve fiyat aralığı üretilemedi.")
        if reason == "gap":
            return ("Sembolün son günlerinde eksik/işlemsiz günler var (risk penceresi dolmadı); "
                    "birkaç gün içinde düzelmesi beklenir. Risk seviyesi ve fiyat aralığı yok.")
        return ("Bu sembol için risk tahmini yok (yetersiz/kesintili fiyat geçmişi veya "
                "ölçek kırığı karantinası); fiyat aralığı üretilemedi.")
