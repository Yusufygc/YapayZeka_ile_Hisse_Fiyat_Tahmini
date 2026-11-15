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
from src.api.services.analysis_freshness import compute_freshness
from src.api.services.data_quality_monitor import compute_psi_30d
from src.api.services.peer_service import PeerEnrichmentService
from src.api.services.serving_health import default_serving_db_path, serving_pool_status

logger = logging.getLogger(__name__)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ISTANBUL = timezone(timedelta(hours=3))  # Turkiye sabit UTC+3 (DST yok)
_LEVELS = ("low", "medium", "high")
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
        peer = self._peers.get_peer_block(symbol)
        if peer is None:
            return AnalysisResponse(
                symbol=symbol,
                analysis_status="no_data",
                generated_at=generated_at,
                confidence=ConfidenceBlock(
                    label="low",
                    warnings=["Sembol son gecelik havuz kosusunda bulunamadi (no_data)."],
                ),
                disclaimer=INVESTMENT_DISCLAIMER,
                serving_run=serving_run,
            )

        csv_path = os.path.join(self._data_dir, f"{symbol}.csv")
        last_date, last_close = _read_market_tail(csv_path)
        if last_date is None:
            # Piyasa CSV'si yok: serving as_of_date'e dus (fiyat bilinmiyor).
            last_date = str(peer.as_of_date)[:10] if peer.as_of_date else None
        data_block = self._data_block(last_date, last_close, today.isoformat())
        dq = self._data_quality(symbol, csv_path)

        conf, status = self._confidence(peer, data_block, pool, dq)
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
    ) -> Tuple[ConfidenceBlock, str]:
        """Ust-seviye guven: Kol-B etiketi; bayat veri / yedek risk yontemi / drift dusurur."""
        label = peer.confidence_label if peer.confidence_label in _LEVELS else "low"
        reasons: List[str] = list(peer.confidence_reasons)
        warnings: List[str] = list(peer.confidence_warnings)

        data_stale = data.data_freshness == "stale_data"
        pool_stale = bool(pool.get("stale"))
        if data_stale:
            warnings.append(
                f"Piyasa verisi {data.staleness_days} islem gunu eski; analiz dikkatli yorumlanmali."
            )
        if pool_stale:
            warnings.append(
                f"Gecelik havuz kosusu {pool.get('age_days')} gun once "
                f"({pool.get('as_of_date')}); sonuclar guncel olmayabilir."
            )
        if data_stale or pool_stale:
            label = "low"

        if getattr(peer, "risk_method", None) == "naive_fallback":
            warnings.append(
                "Risk modeli yedek (naive) yonteme dustu; risk seviyesi ve fiyat araligi "
                "dusuk guvenlidir."
            )
            label = _lower(label)

        if dq.psi_status == "major_drift" and dq.psi_30d is not None:
            warnings.append(f"Veri dagilimi belirgin kaydi (PSI 30g={dq.psi_30d:.2f}).")
            label = "low"
        elif dq.psi_status == "moderate_drift" and dq.psi_30d is not None:
            warnings.append(f"Veri dagilimi orta duzeyde kaydi (PSI 30g={dq.psi_30d:.2f}).")
            if label == "high":
                label = "medium"
        elif dq.psi_status == "stable" and label != "low":
            reasons.append("Veri dagilimi stabil (PSI 30g dusuk).")

        risk_level: Any = getattr(peer, "risk_level", None)
        if risk_level and label != "low":
            reasons.append(f"Risk seviyesi: {risk_level}.")

        if data_stale or pool_stale:
            status = "stale_data"
        elif label == "low":
            status = "low_confidence"
        else:
            status = "ok"
        return ConfidenceBlock(label=label, reasons=reasons, warnings=warnings), status
