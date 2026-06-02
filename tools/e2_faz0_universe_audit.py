# -*- coding: utf-8 -*-
"""E2 Faz 0 — read-only universe/data audit across all stock CSVs in data/.

Pooled global model'e gecmeden once veri tabanini denetler:
  - geçmiş-uzunluğu dağılımı (cold-start / thin hisse sayisi)
  - delisted / stale (survivorship) tespiti: son tarihi global son tarihten cok geride
  - universe (data/bist_universe.csv) capraz kontrolu: eksik kayit, eksik CSV
  - kurumsal islem (split/temettu) anomali proxy'si: |log_return(adj)| >= 0.30
  - veri kalitesi: dup tarih, monoton olmayan, sifir-hacim, NaN, takvim bosluklari

Cikti: outputs/e2_faz0_symbol_stats.csv + outputs/e2_faz0_universe_audit.md
Hicbir sey degistirmez (read-only).
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
UNIVERSE = os.path.join(DATA_DIR, "bist_universe.csv")

CA_THRESHOLD = 0.30  # tools/audit_corporate_actions.py ile ayni
_EXCLUDE = ("bist_universe", "advisory_history")


def _stock_csvs() -> list[str]:
    paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    return sorted(p for p in paths if os.path.basename(p)[:-4] not in _EXCLUDE)


def _parse_dates(raw: pd.Series) -> tuple[pd.Series, str]:
    """Tarih sutununu parse eder. Universe'de iki format var:
    ISO `YYYY-mm-dd` ve TR `dd/mm/YYYY`. Dosya basina ilk gecerli degerden
    formati tespit eder.
    """
    s = raw.astype(str).str.strip()
    sample = next((v for v in s if v and v.lower() != "nan"), "")
    if "-" in sample and len(sample.split("-")[0]) == 4:
        fmt = "%Y-%m-%d"
    elif "/" in sample:
        fmt = "%d/%m/%Y"
    elif "." in sample:
        fmt = "%d.%m.%Y"
    else:
        return pd.to_datetime(s, dayfirst=True, errors="coerce"), "mixed/unknown"
    return pd.to_datetime(s, format=fmt, errors="coerce"), fmt


def _load(path: str) -> tuple[pd.DataFrame | None, str]:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None, "read_error"
    if "Tarih" not in df.columns:
        return None, "no_tarih_col"
    df["Tarih"], fmt = _parse_dates(df["Tarih"])
    df = df.dropna(subset=["Tarih"]).sort_values("Tarih").reset_index(drop=True)
    return df, fmt


def _audit_one(path: str) -> dict:
    sym = os.path.basename(path)[:-4]
    df, fmt = _load(path)
    rec: dict = {"symbol": sym, "loaded": df is not None and not df.empty, "date_fmt": fmt}
    if not rec["loaded"]:
        return rec
    n = len(df)
    adj_col = "Düzeltilmiş_Kapanış" if "Düzeltilmiş_Kapanış" in df.columns else "Kapanış"
    close = pd.to_numeric(df[adj_col], errors="coerce")
    vol = pd.to_numeric(df.get("Hacim", pd.Series([np.nan] * n)), errors="coerce")
    # log-return yalniz pozitif fiyatlarda; close<=0 -> divide-by-zero / sahte anomali
    pos = close.where(close > 0)
    logret = np.log(pos / pos.shift(1))
    dates = df["Tarih"]
    gaps = dates.diff().dt.days.dropna()
    rec.update({
        "rows": n,
        "first_date": dates.iloc[0].date().isoformat(),
        "last_date": dates.iloc[-1].date().isoformat(),
        "span_days": int((dates.iloc[-1] - dates.iloc[0]).days),
        "adj_col": adj_col,
        "dup_dates": int(dates.duplicated().sum()),
        "zero_vol_rows": int((vol == 0).sum()),
        "nan_close": int(close.isna().sum()),
        "zero_neg_close": int((close <= 0).sum()),
        "max_gap_days": int(gaps.max()) if len(gaps) else 0,
        "ca_anomalies": int((logret.abs() >= CA_THRESHOLD).sum()),
        "max_abs_logret": round(float(logret.abs().max(skipna=True)), 4) if n > 1 else 0.0,
    })
    return rec


def _hist_bucket(rows: int) -> str:
    if rows < 250:
        return "a:<250 (<1y)"
    if rows < 500:
        return "b:250-500 (1-2y)"
    if rows < 1250:
        return "c:500-1250 (2-5y)"
    if rows < 2000:
        return "d:1250-2000 (5-8y)"
    return "e:2000+ (8y+)"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = _stock_csvs()
    recs = [_audit_one(p) for p in paths]
    df = pd.DataFrame(recs)
    loaded = df[df["loaded"]].copy()

    # universe cross-ref
    uni = pd.read_csv(UNIVERSE, encoding="utf-8-sig") if os.path.exists(UNIVERSE) else pd.DataFrame()
    uni_syms = set(uni["Symbol"]) if "Symbol" in uni.columns else set()
    csv_syms = set(loaded["symbol"])
    missing_in_uni = sorted(csv_syms - uni_syms)
    missing_csv = sorted(uni_syms - csv_syms)
    if not uni.empty:
        sec = uni.set_index("Symbol")["Sector"].to_dict() if "Sector" in uni.columns else {}
        status = uni.set_index("Symbol")["Status"].to_dict() if "Status" in uni.columns else {}
        loaded["sector"] = loaded["symbol"].map(lambda s: sec.get(s, "") or "")
        loaded["uni_status"] = loaded["symbol"].map(lambda s: status.get(s, "MISSING"))
    else:
        loaded["sector"] = ""
        loaded["uni_status"] = "MISSING"

    global_last = pd.to_datetime(loaded["last_date"]).max()
    loaded["days_behind_last"] = (global_last - pd.to_datetime(loaded["last_date"])).dt.days
    loaded["bucket"] = loaded["rows"].map(_hist_bucket)
    # En yaygin son-tarih = toplu snapshot kesim tarihi (freshness, delisting DEGIL)
    snapshot_date = loaded["last_date"].value_counts().idxmax()
    snapshot_n = int((loaded["last_date"] == snapshot_date).sum())
    fresh_n = int((loaded["days_behind_last"] <= 5).sum())

    # ---- write per-symbol stats csv ----
    stats_path = os.path.join(OUT_DIR, "e2_faz0_symbol_stats.csv")
    loaded.sort_values("rows").to_csv(stats_path, index=False, encoding="utf-8-sig")

    # ---- aggregate ----
    n_total = len(df)
    n_loaded = len(loaded)
    n_failed = n_total - n_loaded
    bucket_counts = loaded["bucket"].value_counts().sort_index()
    thin = loaded[loaded["rows"] < 500]
    ca = loaded[loaded["ca_anomalies"] > 0]
    zneg = loaded[loaded["zero_neg_close"] > 0]
    dups = loaded[loaded["dup_dates"] > 0]
    big_gap = loaded[loaded["max_gap_days"] > 30]
    sector_missing = loaded[(loaded["sector"] == "")]

    lines = []
    lines.append("# E2 Faz 0 — Universe / Data Audit")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')} (read-only)_")
    lines.append(f"_Global last date across universe: {global_last.date()}_")
    lines.append("")
    lines.append("## Ozet")
    lines.append("")
    lines.append(f"- CSV (stock): **{n_total}** | yuklenebildi: **{n_loaded}** | yuklenemedi: **{n_failed}**")
    lines.append(f"- Toplam (symbol,date) satir: **{int(loaded['rows'].sum()):,}**")
    lines.append(f"- Medyan gecmis: **{int(loaded['rows'].median())}** satir | min {int(loaded['rows'].min())} | max {int(loaded['rows'].max())}")
    lines.append("")
    lines.append("## Tarih formati tutarliligi (ingestion riski)")
    lines.append("")
    fmt_counts = loaded["date_fmt"].value_counts()
    lines.append("| format | hisse |")
    lines.append("|---|---|")
    for k, v in fmt_counts.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("- Universe'de **karisik tarih formati** var (ISO `%Y-%m-%d` + TR `%d/%m/%Y`). Pooled loader iki formati da tanimak zorunda; tek format varsayan kod sessizce bos frame uretir.")
    lines.append("")
    lines.append("## Gecmis-uzunlugu dagilimi (cold-start riski)")
    lines.append("")
    lines.append("| Kova | Hisse |")
    lines.append("|---|---|")
    for k, v in bucket_counts.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(f"- **Thin (<500 satir / <2y): {len(thin)} hisse** -> tek basina egitilemez, cold-start global modele bagimli.")
    lines.append("")
    lines.append("## Veri freshness (KRITIK)")
    lines.append("")
    lines.append(f"- Global son tarih: **{global_last.date()}**. Guncel (<=5 gun): **{fresh_n} hisse**.")
    lines.append(f"- **{snapshot_n} hisse {snapshot_date}'te donmus** = tek seferlik toplu snapshot export. Pipeline yalniz birkac sembolu yeniliyor.")
    lines.append(f"- Bu **delisting DEGIL**, freshness sorunu: bulk dump ~{int(loaded['days_behind_last'].median())} gun eski. Pooled egitim oncesi tum universe yeniden cekilmeli / tek kesim tarihine hizalanmali.")
    lines.append("")
    lines.append("## Survivorship (cozulemiyor — metadata eksik)")
    lines.append("")
    lines.append(f"- Gercek delisting `bist_universe.csv` `Delisted_Date`'ten gelmeli ama universe yalniz **{len(uni_syms)} sembol** kataloglu; CSV'lerin {len(missing_in_uni)} tanesi universe'de yok.")
    lines.append("- last_date'ten delisting cikarilamiyor cunku hepsi ayni snapshot tarihinde kesiliyor. **Survivorship bias riski acik**: delisted hisseler dump'ta var mi belirsiz; yoksa pooled egitim hayatta-kalanlara yanli olur.")
    lines.append("- Aksiyon: delisting kaynagi (universe genisletme) Faz 2 leakage/CV oncesi netlesmeli.")
    lines.append("")
    lines.append("## Universe (bist_universe.csv) capraz kontrol")
    lines.append("")
    lines.append(f"- CSV var ama universe'de YOK: **{len(missing_in_uni)}** {missing_in_uni[:20]}{' ...' if len(missing_in_uni)>20 else ''}")
    lines.append(f"- Universe'de var ama CSV YOK: **{len(missing_csv)}** {missing_csv[:20]}{' ...' if len(missing_csv)>20 else ''}")
    lines.append(f"- Sektor etiketi bos: **{len(sector_missing)}** hisse (koşullandırma feature'i icin eksik).")
    lines.append("")
    lines.append("## Veri kalitesi")
    lines.append("")
    lines.append(f"- Kurumsal-islem anomali (|log_return(adj)| >= {CA_THRESHOLD}): **{len(ca)} hisse**. Adj close split-adjusted olmali; bunlar gercek olay ya da veri hatasi.")
    lines.append(f"- Sifir/negatif fiyat satiri iceren: **{len(zneg)} hisse** (log-return bozar; temizlik gerek).")
    lines.append(f"- Tekrar eden tarih iceren: **{len(dups)} hisse**")
    lines.append(f"- >30 gun takvim boslugu: **{len(big_gap)} hisse** (gec listeleme / uzun durdurma).")
    lines.append("")
    lines.append("## Faz 0 -> politikaya etki (taslak)")
    lines.append("")
    lines.append(f"- **Freshness blocker:** {snapshot_n} hisse {snapshot_date}'te donmus. Pooled egitim oncesi tum universe tek guncel kesim tarihine cekilmeli/hizalanmali.")
    lines.append(f"- **Tarih formati:** ingestion 3 formati ({fmt_counts.to_dict()}) tanimali; yoksa sessiz bos-frame.")
    lines.append(f"- Min-history: pooling'e dahil ama tek-symbol fine-tune yalniz >=1250 satir (~5y); <500 satir ({len(thin)} hisse) cold-start (global-only).")
    lines.append(f"- **Survivorship:** delisting metadata yok (universe {len(uni_syms)}/{n_loaded}); pooled egitim hayatta-kalanlara yanli olabilir -> delisting kaynagi gerek.")
    lines.append(f"- Sektor eksigi {len(sector_missing)}: kosullandirma icin sektor doldurma / 'unknown' bucket gerek.")
    lines.append(f"- CA anomalileri {len(ca)} + zero/neg fiyat {len(zneg)}: pooled egitimden once audit + clip/duzeltme politikasi.")
    lines.append("")

    md_path = os.path.join(OUT_DIR, "e2_faz0_universe_audit.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ---- console summary ----
    print(f"[FAZ0] CSV={n_total} loaded={n_loaded} failed={n_failed} rows_total={int(loaded['rows'].sum()):,}")
    print(f"[FAZ0] buckets:\n{bucket_counts.to_string()}")
    print(f"[FAZ0] thin(<500)={len(thin)} ca_anom={len(ca)} zero_neg_close={len(zneg)} dup={len(dups)} biggap={len(big_gap)} sector_missing={len(sector_missing)}")
    print(f"[FAZ0] freshness: global_last={global_last.date()} fresh(<=5d)={fresh_n} frozen@{snapshot_date}={snapshot_n}")
    print(f"[FAZ0] date_fmt:\n{loaded['date_fmt'].value_counts().to_string()}")
    print(f"[FAZ0] universe: csv_not_in_uni={len(missing_in_uni)} uni_not_in_csv={len(missing_csv)}")
    print(f"[FAZ0] report -> {md_path}")
    print(f"[FAZ0] stats  -> {stats_path}")


if __name__ == "__main__":
    main()
