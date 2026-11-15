# -*- coding: utf-8 -*-
"""
data_updater.py - CSV tabanli hisse verisi guncelleme servisi.

Egitimden once hedef CSV'nin son tarihini kontrol eder. Veri, son TAMAMLANMIS
BIST seansinin gerisindeyse yfinance uzerinden eksik gunleri indirir ve mevcut
CSV semasina uygun satirlari dosyaya ekler. Seans kapanmadan (18:10 Istanbul)
gelen kismi bugunluk mum asla yazilmaz. Yazimlar sembol basina dosya kilidi
altinda ve atomiktir; tam yeniden indirme oncesi yedek alinir.
Hata durumunda pipeline'i durdurmaz; mevcut veriyle devam edilmesini saglar.

Toplu guncelleme icin ``DataUpdater.update_many`` (chunk + retry/backoff).
"""

from __future__ import annotations

import builtins
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.csv_safety import (
    DEFAULT_LOCK_TIMEOUT_S,
    CsvLockTimeout,
    atomic_append_text,
    atomic_write_csv,
    backup_before_rewrite,
    csv_lock,
)
from src.data.market_providers import (  # noqa: F401  (YFinanceProvider geriye uyumlu re-export)
    DEFAULT_BACKOFF_S,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REFRESH_DEADLINE_S,
    MarketDataProvider,
    YFinanceProvider,
    download_with_retries,
)
from src.data.session_clock import latest_completed_session, now_istanbul
from src.data.updater_frames import (
    _OVERLAP_DAYS,
    _OVERLAP_TOLERANCE,
    _csv_date_format,
    _find_column,
    _max_overlap_deviation,
    _parse_date_series,
    _parse_last_date,
    _prepare_provider_rows,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataUpdateResult:
    status: str
    latest_date_before: str | None = None
    latest_date_after: str | None = None
    rows_added: int = 0
    error: str | None = None
    full_refetch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Plan:
    csv_path: str
    symbol: str
    ticker: str
    last_date: pd.Timestamp
    latest_before: str
    start: str
    end: str
    today: pd.Timestamp
    completed: pd.Timestamp
    diff_days: int
    lock_timeout: float


def _result(status: str, before: str | None, error: str | None = None, **kw: Any) -> DataUpdateResult:
    return DataUpdateResult(
        status=status, latest_date_before=before,
        latest_date_after=kw.pop("after", before), error=error, **kw,
    )


def _read_csv_state(csv_path: str) -> tuple[pd.DataFrame, str, pd.Timestamp] | DataUpdateResult:
    """CSV'yi okur; (df, tarih_kolonu, son_tarih) ya da hata sonucu doner."""
    try:
        df_raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        msg = f"{csv_path} dosyasi bulunamadi."
        print(f"  [UYARI] {msg} Lutfen kontrol edin.")
        return DataUpdateResult(status="failed", error=msg)
    except Exception as exc:
        msg = f"{csv_path} okunamadi ({exc})"
        print(f"  [UYARI] {msg}; veri guncellemesi atlandi.")
        return DataUpdateResult(status="failed", error=msg)
    if df_raw.empty:
        msg = f"{csv_path} bos"
        print(f"  [UYARI] {msg}; veri guncellemesi atlandi.")
        return DataUpdateResult(status="failed", error=msg)
    date_col = _find_column(df_raw.columns, ["Date", "Tarih"])
    if date_col is None:
        msg = "CSV'de Date/Tarih kolonu yok"
        print(f"  [UYARI] {msg}; veri guncellemesi atlandi.")
        return DataUpdateResult(status="failed", error=msg)
    last_date = _parse_last_date(df_raw[date_col])
    if pd.isna(last_date):
        msg = "Son tarih parse edilemedi"
        print(f"  [UYARI] {msg}; veri guncellemesi atlandi.")
        return DataUpdateResult(status="failed", error=msg)
    return df_raw, date_col, pd.Timestamp(last_date).normalize()


class DataUpdater:
    """Veri seti guncelligini denetleyen ve tamamlayan sinif."""

    @staticmethod
    def _plan(
        csv_path: str,
        stock_symbol: str,
        now: datetime | None,
        calendar_path: str | None,
        lock_timeout: float,
    ) -> DataUpdateResult | _Plan:
        state = _read_csv_state(csv_path)
        if isinstance(state, DataUpdateResult):
            return state
        _df, _date_col, last_date = state
        latest_before = last_date.strftime("%Y-%m-%d")
        completed = latest_completed_session(now, calendar_path)
        if last_date >= completed:
            print("  [INFO] CSV veri setiniz son tamamlanmis seansa ait. Guncellemeye gerek yok.")
            return _result("up_to_date", latest_before)
        today = pd.Timestamp(now_istanbul(now).date())
        # Overlap'li indirme: son tarihten 7 takvim gunu geriden basla. Split/temettu
        # sonrasi auto_adjust=True gecmisi yeniden olcekler; ust uste binen satirlarda
        # eski/yeni fiyat farki bunu yakalar (yoksa append sonrasi seride "dikis" kalir).
        return _Plan(
            csv_path=csv_path, symbol=stock_symbol, ticker=f"{stock_symbol}.IS",
            last_date=last_date, latest_before=latest_before,
            start=(last_date - pd.Timedelta(days=_OVERLAP_DAYS)).strftime("%Y-%m-%d"),
            end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            today=today, completed=completed,
            diff_days=int((completed - last_date).days), lock_timeout=lock_timeout,
        )

    @staticmethod
    def check_and_update(
        csv_path: str,
        stock_symbol: str,
        interactive: bool = True,
        provider: MarketDataProvider | None = None,
        *,
        now: datetime | None = None,
        calendar_path: str | None = None,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> DataUpdateResult:
        """Tek sembol guncellemesi. ``now`` test icin enjekte edilebilir (naive -> Istanbul)."""
        provider = provider or YFinanceProvider()
        plan = DataUpdater._plan(csv_path, stock_symbol, now, calendar_path, lock_timeout)
        if isinstance(plan, DataUpdateResult):
            return plan

        print(f"\n  [INFO] Veri setinizdeki son tarih : {plan.last_date.strftime('%d/%m/%Y')}")
        print(f"  [INFO] Son tamamlanmis seans      : {plan.completed.strftime('%d/%m/%Y')}")
        ask = builtins.input if interactive else (lambda *_args, **_kwargs: "e")
        answer = ask(
            f"  [SORU] Veri seti son seanstan {plan.diff_days} gun geride. "
            f"Eksik veriler yfinance ({stock_symbol}.IS) ile tamamlansin mi? (e/h): "
        )
        if str(answer).strip().lower() != "e":
            print("  [INFO] Veri guncellemesi atlandi. Mevcut eski veriler kullaniliyor.")
            return _result("skipped", plan.latest_before, "user_declined")

        print(f"  [INFO] yfinance indiriliyor: {plan.ticker} {plan.start} -> {plan.end}")
        try:
            new_data = provider.download(plan.ticker, start=plan.start, end=plan.end)
        except Exception as exc:
            msg = f"yfinance guncellemesi basarisiz ({exc})"
            print(f"  [UYARI] {msg}; mevcut CSV ile devam ediliyor.")
            return _result("failed", plan.latest_before, msg)
        return DataUpdater._apply_download(plan, new_data, provider, now, calendar_path)

    @staticmethod
    def update_many(
        targets: Sequence[tuple[str, str]],
        *,
        provider: MarketDataProvider | None = None,
        now: datetime | None = None,
        calendar_path: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S,
        deadline_s: float | None = DEFAULT_REFRESH_DEADLINE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> dict[str, DataUpdateResult]:
        """Toplu guncelleme: ``targets`` = [(csv_path, sembol)]. Etkileşimsizdir.

        ``deadline_s``: tum indirme asamasi icin duvar-saati siniri (None = sinirsiz). Asilirsa
        kalan semboller ``failed`` ("deadline_exceeded") olur; mevcut CSV'ler korunur, kosu surer.

        Guncel olanlar indirilmez; gerisi baslangic tarihine gore siralanip ``chunk_size``'lik
        gruplar halinde ``yf.download(tickers=[...])`` ile indirilir, bos/basarisiz ticker'lar
        ustel geri cekilmeyle yeniden denenir. Her sembol icin dogrulama + yazim tekli yolla aynidir.
        """
        provider = provider or YFinanceProvider()
        results: dict[str, DataUpdateResult] = {}
        plans: list[_Plan] = []
        for csv_path, symbol in targets:
            plan = DataUpdater._plan(csv_path, symbol, now, calendar_path, lock_timeout)
            if isinstance(plan, DataUpdateResult):
                results[symbol] = plan
            else:
                plans.append(plan)
        plans.sort(key=lambda p: p.start)
        deadline = None if deadline_s is None else clock() + deadline_s
        step = max(1, chunk_size)
        for i in range(0, len(plans), step):
            group = plans[i:i + step]
            frames, errors = download_with_retries(
                provider, [p.ticker for p in group],
                start=group[0].start, end=max(p.end for p in group),
                chunk_size=step, max_retries=max_retries, backoff_s=backoff_s, sleep=sleep,
                deadline=deadline, clock=clock,
            )
            for plan in group:
                frame = frames.get(plan.ticker)
                if frame is None or plan.ticker in errors:
                    msg = f"yfinance toplu indirme basarisiz ({errors.get(plan.ticker, 'bilinmeyen')})"
                    results[plan.symbol] = _result("failed", plan.latest_before, msg)
                else:
                    results[plan.symbol] = DataUpdater._apply_download(
                        plan, frame, provider, now, calendar_path
                    )
        return results

    @staticmethod
    def _apply_download(
        plan: _Plan,
        new_data: pd.DataFrame | None,
        provider: MarketDataProvider,
        now: datetime | None,
        calendar_path: str | None,
    ) -> DataUpdateResult:
        """Indirilen veriyi dogrular ve sembol kilidi altinda yazar."""
        if new_data is None or new_data.empty:
            print("  [INFO] Indirilecek yeni islem gunu bulunamadi.")
            return _result("skipped", plan.latest_before, "no_new_rows_from_provider")
        try:
            with csv_lock(plan.csv_path, plan.lock_timeout):
                return DataUpdater._apply_locked(plan, new_data, provider, now, calendar_path)
        except CsvLockTimeout as exc:
            print(f"  [UYARI] {exc}; mevcut CSV ile devam ediliyor.")
            return _result("failed", plan.latest_before, str(exc))

    @staticmethod
    def _apply_locked(
        plan: _Plan,
        new_data: pd.DataFrame,
        provider: MarketDataProvider,
        now: datetime | None,
        calendar_path: str | None,
    ) -> DataUpdateResult:
        def _failed(msg: str) -> DataUpdateResult:
            print(f"  [UYARI] {msg}; mevcut CSV ile devam ediliyor.")
            return _result("failed", latest_before, msg)

        # Kilit alindiktan sonra yeniden oku: baska yazar arada ilerletmis olabilir.
        state = _read_csv_state(plan.csv_path)
        if isinstance(state, DataUpdateResult):
            return state
        df_raw, date_col, last_date = state
        latest_before = last_date.strftime("%Y-%m-%d")
        if last_date >= plan.completed:
            print("  [INFO] CSV baska bir surec tarafindan guncellenmis; yazim atlandi.")
            return _result("up_to_date", latest_before)

        rows_all = _prepare_provider_rows(new_data, df_raw.columns, date_col, plan.completed)
        date_fmt = _csv_date_format()

        # Overlap dogrulamasi (Adj Close varsa o, yoksa Close karsilastirilir).
        compare_col = _find_column(
            df_raw.columns, ["Adj Close", "Adj_Close", "Düzeltilmiş_Kapanış", "Duzeltilmis_Kapanis"]
        ) or _find_column(df_raw.columns, ["Close", "Kapanış", "Kapanis"])
        max_dev = _max_overlap_deviation(df_raw, rows_all, date_col, compare_col, last_date)
        if max_dev is None:
            print("  [UYARI] Overlap satiri yok; fiyat sureksizligi dogrulanamadi, sadece append.")
        elif max_dev > _OVERLAP_TOLERANCE:
            print(
                f"  [UYARI] Overlap fiyat farki {max_dev:.2%} > {_OVERLAP_TOLERANCE:.2%} "
                "(split/temettu duzeltmesi suphesi); tum gecmis yeniden indiriliyor."
            )
            return DataUpdater._full_refetch(
                provider, plan.ticker, plan.csv_path, df_raw, date_col, date_fmt,
                plan.completed, latest_before, plan.end, now,
            )

        rows = rows_all[rows_all[date_col] > last_date].copy()
        if rows.empty:
            print("  [INFO] Indirilen veri mevcut CSV'den daha yeni (tamamlanmis) satir icermiyor.")
            return _result("skipped", latest_before, "provider_rows_not_newer")

        latest_after = pd.Timestamp(rows[date_col].max()).strftime("%Y-%m-%d")
        rows[date_col] = rows[date_col].dt.strftime(date_fmt)
        rows = rows[list(df_raw.columns)]
        try:
            atomic_append_text(
                plan.csv_path, rows.to_csv(header=False, index=False, lineterminator="\n")
            )
        except Exception as exc:
            return _failed(f"Yeni satirlar CSV'ye yazilamadi ({exc})")

        print(f"  [OK] {len(rows)} yeni islem gunu {os.path.basename(plan.csv_path)} dosyasina eklendi.")
        return _result("updated", latest_before, after=latest_after, rows_added=int(len(rows)))

    @staticmethod
    def _full_refetch(
        provider: MarketDataProvider,
        ticker: str,
        csv_path: str,
        df_raw: pd.DataFrame,
        date_col: str,
        date_fmt: str,
        max_date: pd.Timestamp,
        latest_before: str,
        end: str,
        now: datetime | None = None,
    ) -> DataUpdateResult:
        """Tum gecmisi yeniden indirir; yedek alip CSV'yi atomik (gecici dosya + os.replace) yazar.

        Cagiran kilidi zaten tutuyor olmalidir.
        """
        first_date = _parse_date_series(df_raw[date_col]).min()
        start = pd.Timestamp(first_date).strftime("%Y-%m-%d")

        def _failed(msg: str) -> DataUpdateResult:
            print(f"  [UYARI] {msg}; mevcut CSV korunuyor.")
            return _result("failed", latest_before, msg)

        try:
            full = provider.download(ticker, start=start, end=end)
        except Exception as exc:
            return _failed(f"tam yeniden indirme basarisiz ({exc})")
        if full is None or full.empty:
            return _failed("tam yeniden indirme bos dondu")

        rows = _prepare_provider_rows(full, df_raw.columns, date_col, max_date)
        if rows.empty:
            return _failed("tam yeniden indirmede gecerli satir yok")
        # Saglayici daha kisa gecmis veriyorsa eski (uzun) CSV'yi ezip veri kaybetme.
        if pd.Timestamp(rows[date_col].min()) > pd.Timestamp(first_date) + pd.Timedelta(days=_OVERLAP_DAYS + 3):
            return _failed(
                "tam yeniden indirme CSV'nin ilk tarihini kapsamiyor "
                f"({pd.Timestamp(rows[date_col].min()).date()} > {pd.Timestamp(first_date).date()})"
            )

        latest_after = pd.Timestamp(rows[date_col].max()).strftime("%Y-%m-%d")
        out = rows.copy()
        out[date_col] = out[date_col].dt.strftime(date_fmt)
        out = out[list(df_raw.columns)]
        try:
            backup_path = backup_before_rewrite(csv_path, today=now_istanbul(now))
            print(f"  [INFO] Yedek alindi: {backup_path}")
        except OSError as exc:
            return _failed(f"yedek alinamadi, yeniden yazim iptal ({exc})")
        try:
            atomic_write_csv(out, csv_path)
        except Exception as exc:
            return _failed(f"CSV atomik yazilamadi ({exc})")

        print(
            f"  [OK] {os.path.basename(csv_path)} tam yeniden yazildi "
            f"({len(df_raw)} -> {len(out)} satir; split/temettu dikisi giderildi)."
        )
        return _result(
            "updated", latest_before, after=latest_after,
            rows_added=max(0, int(len(out) - len(df_raw))), full_refetch=True,
        )
