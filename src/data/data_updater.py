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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf


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


class MarketDataProvider:
    def download(self, ticker: str, *, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    """
    Sprint 2 (2026-05-25) Plan A2.1:
      ``auto_adjust=True`` HARD kontrol. yfinance close serisini split ve
      temettu icin otomatik duzeltir. Duzeltilmemis seride 1:2 split
      gunu log_return ≈ -0.69 olur ve tum modeller bozulur. Bu deger
      degistirilmemelidir; testler de bunu denetler.
    """

    AUTO_ADJUST: bool = True

    def download(self, ticker: str, *, start: str, end: str) -> pd.DataFrame:
        # auto_adjust=True ZORUNLU — split/temettu sizintisini onler.
        return yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=self.AUTO_ADJUST,
        )


class DataUpdater:
    """Veri seti guncelligini denetleyen ve tamamlayan sinif."""

    @staticmethod
    def check_and_update(
        csv_path: str,
        stock_symbol: str,
        interactive: bool = True,
        provider: MarketDataProvider | None = None,
    ) -> DataUpdateResult:
        provider = provider or YFinanceProvider()
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

        last_date = pd.Timestamp(last_date).normalize()
        latest_before = last_date.strftime("%Y-%m-%d")
        today = pd.Timestamp(datetime.today().date())
        diff_days = int((today - last_date).days)
        if diff_days <= 1:
            print("  [INFO] CSV veri setiniz halihazirda son gune ait. Guncellemeye gerek yok.")
            return DataUpdateResult(
                status="up_to_date",
                latest_date_before=latest_before,
                latest_date_after=latest_before,
            )

        print(f"\n  [INFO] Veri setinizdeki son tarih : {last_date.strftime('%d/%m/%Y')}")
        print(f"  [INFO] Bugunun tarihi             : {today.strftime('%d/%m/%Y')}")

        ask = builtins.input if interactive else (lambda *_args, **_kwargs: "e")
        answer = ask(
            f"  [SORU] Veri seti bugunden {diff_days} gun geride. "
            f"Eksik veriler yfinance ({stock_symbol}.IS) ile tamamlansin mi? (e/h): "
        )
        if str(answer).strip().lower() != "e":
            print("  [INFO] Veri guncellemesi atlandi. Mevcut eski veriler kullaniliyor.")
            return DataUpdateResult(
                status="skipped",
                latest_date_before=latest_before,
                latest_date_after=latest_before,
                error="user_declined",
            )

        # Overlap'li indirme: son tarihten 7 takvim gunu geriden basla. Split/temettu
        # sonrasi auto_adjust=True gecmisi yeniden olcekler; ust uste binen satirlarda
        # eski/yeni fiyat farki bunu yakalar (yoksa append sonrasi seride "dikis" kalir).
        start = (last_date - pd.Timedelta(days=_OVERLAP_DAYS)).strftime("%Y-%m-%d")
        end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ticker = f"{stock_symbol}.IS"
        print(f"  [INFO] yfinance indiriliyor: {ticker} {start} -> {end}")

        def _failed(msg: str) -> DataUpdateResult:
            print(f"  [UYARI] {msg}; mevcut CSV ile devam ediliyor.")
            return DataUpdateResult(
                status="failed",
                latest_date_before=latest_before,
                latest_date_after=latest_before,
                error=msg,
            )

        try:
            new_data = provider.download(ticker, start=start, end=end)
        except Exception as exc:
            return _failed(f"yfinance guncellemesi basarisiz ({exc})")

        if new_data is None or new_data.empty:
            print("  [INFO] Indirilecek yeni islem gunu bulunamadi.")
            return DataUpdateResult(
                status="skipped",
                latest_date_before=latest_before,
                latest_date_after=latest_before,
                error="no_new_rows_from_provider",
            )

        rows_all = _prepare_provider_rows(new_data, df_raw.columns, date_col, today)
        date_fmt = _infer_date_format(str(df_raw[date_col].iloc[-1]))

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
                provider, ticker, csv_path, df_raw, date_col, date_fmt,
                today, latest_before, end,
            )

        rows = rows_all[rows_all[date_col] > last_date].copy()
        if rows.empty:
            print("  [INFO] Indirilen veri mevcut CSV'den daha yeni satir icermiyor.")
            return DataUpdateResult(
                status="skipped",
                latest_date_before=latest_before,
                latest_date_after=latest_before,
                error="provider_rows_not_newer",
            )

        latest_after = pd.Timestamp(rows[date_col].max()).strftime("%Y-%m-%d")
        rows[date_col] = rows[date_col].dt.strftime(date_fmt)
        rows = rows[list(df_raw.columns)]
        try:
            rows.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        except Exception as exc:
            return _failed(f"Yeni satirlar CSV'ye yazilamadi ({exc})")

        print(f"  [OK] {len(rows)} yeni islem gunu {os.path.basename(csv_path)} dosyasina eklendi.")
        return DataUpdateResult(
            status="updated",
            latest_date_before=latest_before,
            latest_date_after=latest_after,
            rows_added=int(len(rows)),
        )

    @staticmethod
    def _full_refetch(
        provider: MarketDataProvider,
        ticker: str,
        csv_path: str,
        df_raw: pd.DataFrame,
        date_col: str,
        date_fmt: str,
        today: pd.Timestamp,
        latest_before: str,
        end: str,
    ) -> DataUpdateResult:
        """Tum gecmisi yeniden indirir; CSV'yi atomik (gecici dosya + os.replace) yazar."""
        first_date = _parse_date_series(df_raw[date_col]).min()
        start = pd.Timestamp(first_date).strftime("%Y-%m-%d")

        def _failed(msg: str) -> DataUpdateResult:
            print(f"  [UYARI] {msg}; mevcut CSV korunuyor.")
            return DataUpdateResult(
                status="failed",
                latest_date_before=latest_before,
                latest_date_after=latest_before,
                error=msg,
            )

        try:
            full = provider.download(ticker, start=start, end=end)
        except Exception as exc:
            return _failed(f"tam yeniden indirme basarisiz ({exc})")
        if full is None or full.empty:
            return _failed("tam yeniden indirme bos dondu")

        rows = _prepare_provider_rows(full, df_raw.columns, date_col, today)
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
        tmp_path = f"{csv_path}.tmp"
        try:
            out.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            os.replace(tmp_path, csv_path)
        except Exception as exc:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as rm_exc:
                    print(f"  [UYARI] gecici dosya silinemedi: {rm_exc}")
            return _failed(f"CSV atomik yazilamadi ({exc})")

        print(
            f"  [OK] {os.path.basename(csv_path)} tam yeniden yazildi "
            f"({len(df_raw)} -> {len(out)} satir; split/temettu dikisi giderildi)."
        )
        return DataUpdateResult(
            status="updated",
            latest_date_before=latest_before,
            latest_date_after=latest_after,
            rows_added=max(0, int(len(out) - len(df_raw))),
            full_refetch=True,
        )


_OVERLAP_DAYS = 7
_OVERLAP_TOLERANCE = 0.005  # |yeni/eski - 1| > %0.5 -> yeniden olcekleme suphesi


def _prepare_provider_rows(
    new_data: pd.DataFrame, columns: Any, date_col: str, today: pd.Timestamp
) -> pd.DataFrame:
    """Saglayici ciktisini CSV semasina cevirir; Date > bugun satirlari atilir."""
    frame = _normalize_yfinance_frame(new_data)
    rows = _map_to_existing_schema(frame, columns, date_col)
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=[date_col])
    rows = rows[rows[date_col] <= today]
    rows = rows.drop_duplicates(subset=[date_col], keep="last").sort_values(date_col)
    return rows.reset_index(drop=True)


def _max_overlap_deviation(
    df_raw: pd.DataFrame,
    rows_all: pd.DataFrame,
    date_col: str,
    compare_col: str | None,
    last_date: pd.Timestamp,
) -> float | None:
    """Ust uste binen tarihlerde max |yeni/eski - 1|; karsilastirilabilir satir yoksa None."""
    if compare_col is None or compare_col not in rows_all.columns:
        return None
    old = pd.DataFrame({
        "d": _parse_date_series(df_raw[date_col]).dt.normalize(),
        "old": pd.to_numeric(df_raw[compare_col], errors="coerce"),
    }).dropna()
    old = old[old["d"] >= last_date - pd.Timedelta(days=_OVERLAP_DAYS)].drop_duplicates("d", keep="last")
    new = pd.DataFrame({
        "d": rows_all[date_col],
        "new": pd.to_numeric(rows_all[compare_col], errors="coerce"),
    }).dropna()
    merged = old.merge(new, on="d")
    merged = merged[(merged["old"] > 0) & (merged["new"] > 0)]
    if merged.empty:
        return None
    return float((merged["new"] / merged["old"] - 1.0).abs().max())


def _find_column(columns: Any, candidates: list[str]) -> str | None:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _parse_date_series(values: pd.Series) -> pd.Series:
    """Tarih kolonunu ayristirir (yil-ilk ise dayfirst=False, degilse True); NaT'ler atilir."""
    vals_clean = values.dropna()
    if vals_clean.empty:
        return pd.Series([], dtype="datetime64[ns]")
    sample = str(vals_clean.iloc[0]).strip()
    is_year_first = False
    for separator in ["-", "/", "."]:
        if separator in sample:
            parts = sample.split(separator)
            if parts and len(parts[0]) == 4:
                is_year_first = True
                break
    parsed = pd.to_datetime(vals_clean, errors="coerce", dayfirst=not is_year_first)
    return parsed.dropna()


def _parse_last_date(values: pd.Series) -> pd.Timestamp:
    parsed = _parse_date_series(values)
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
