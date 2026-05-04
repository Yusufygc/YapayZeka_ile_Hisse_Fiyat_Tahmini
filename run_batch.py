#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch.py - Multi-Stock Batch Pipeline (Faz 5.3)

Tek hisselik interaktif CLI'dan çok hisseli otomatik batch moda geçiş.

Kullanım:
    # Tek hisse (test):
    python run_batch.py --stocks TUPRS

    # Çoklu hisse, paralel 2 worker:
    python run_batch.py --stocks TUPRS,ASELS,THYAO --mode walk_forward --workers 2

    # Universe dosyasından tüm hisseler:
    python run_batch.py --universe data/bist_universe.csv --mode walk_forward --workers 4

    # Sonuçları özel dizine yaz:
    python run_batch.py --stocks TUPRS,SISE --output-dir my_outputs/

Çıktı:
    - Her hisse için outputs/{SYMBOL}/ altında model dosyaları ve raporlar
    - stock_models.db'de güncel kayıtlar
    - batch_summary_{timestamp}.csv — toplu sonuç özeti
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

# Proje kökünü path'e ekle
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# İşçi fonksiyonu (her hisse için ayrı process'te çalışır)
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_stock(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tek hisse için pipeline'ı çalıştırır.
    ProcessPoolExecutor'a geçirilecek top-level fonksiyon (pickle gereksinimi).
    """
    symbol: str = args_dict["symbol"]
    data_dir: str = args_dict["data_dir"]
    mode: str = args_dict["mode"]
    selected_models: Optional[List[str]] = args_dict.get("selected_models")
    output_dir_base: Optional[str] = args_dict.get("output_dir_base")

    result: Dict[str, Any] = {
        "symbol": symbol,
        "status": "pending",
        "error": None,
        "metrics": {},
        "duration_sec": 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    import time
    t0 = time.time()

    try:
        data_file = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(data_file):
            result["status"] = "skipped"
            result["error"] = f"Veri dosyası bulunamadı: {data_file}"
            return result

        from src.pipeline.orchestrator import ForecastingPipeline

        pipeline = ForecastingPipeline(
            data_file=data_file,
            validation_mode=mode,
            selected_models=selected_models,
        )
        pipeline.run_all()

        # En iyi model metriklerini topla
        db_path = os.path.join(_PROJECT_ROOT, "data", "stock_models.db")
        try:
            from src.database.stock_model_db import StockModelDB
            db = StockModelDB(db_path)
            best = db.get_best_model(symbol)
            if best:
                result["metrics"] = {
                    "model_name": best.get("model_name"),
                    "composite_score": best.get("composite_score"),
                    "dir_acc": best.get("dir_acc"),
                    "sharpe": best.get("sharpe"),
                    "rmse": best.get("rmse"),
                }
        except Exception:
            pass

        result["status"] = "ok"

    except KeyboardInterrupt:
        result["status"] = "cancelled"
        result["error"] = "KeyboardInterrupt"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    result["duration_sec"] = round(time.time() - t0, 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Özet rapor
# ─────────────────────────────────────────────────────────────────────────────

def _save_summary(results: List[Dict[str, Any]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"batch_summary_{ts}.csv")
    json_path = os.path.join(output_dir, f"batch_summary_{ts}.json")

    flat_rows = []
    for r in results:
        row = {
            "symbol": r["symbol"],
            "status": r["status"],
            "error": r.get("error") or "",
            "duration_sec": r.get("duration_sec", 0),
            "timestamp": r.get("timestamp", ""),
        }
        metrics = r.get("metrics", {})
        row.update({
            "model_name": metrics.get("model_name", ""),
            "composite_score": metrics.get("composite_score", ""),
            "dir_acc": metrics.get("dir_acc", ""),
            "sharpe": metrics.get("sharpe", ""),
            "rmse": metrics.get("rmse", ""),
        })
        flat_rows.append(row)

    if flat_rows:
        keys = list(flat_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ts_forecasting_lab — Multi-Stock Batch Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    stock_group = p.add_mutually_exclusive_group(required=True)
    stock_group.add_argument(
        "--stocks",
        type=str,
        help="Virgülle ayrılmış hisse kodları (ör. TUPRS,ASELS,THYAO)",
    )
    stock_group.add_argument(
        "--universe",
        type=str,
        help="Hisse kodları listesini içeren CSV dosyası (Symbol sütunu beklenir)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data"),
        help="Hisse CSV dosyalarının dizini (varsayılan: data/)",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["single_split", "walk_forward"],
        default="walk_forward",
        help="Validasyon modu (varsayılan: walk_forward)",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Virgülle ayrılmış model listesi (ör. XGBoost,LSTM). Varsayılan: tümü",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Paralel process sayısı (varsayılan: 1). GPU modellerle dikkatli kullan.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=_PROJECT_ROOT,
        help="batch_summary CSV/JSON'un yazılacağı dizin (varsayılan: proje kökü)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Pipeline çalıştırmadan hangi hisselerin işleneceğini göster",
    )
    return p.parse_args()


def _load_universe(universe_file: str) -> List[str]:
    import csv as _csv
    symbols = []
    with open(universe_file, newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            sym = row.get("Symbol") or row.get("symbol") or row.get("SYMBOL")
            if sym:
                symbols.append(sym.strip().upper())
    return symbols


def main() -> None:
    args = _parse_args()

    # Hisse listesi
    if args.stocks:
        symbols = [s.strip().upper() for s in args.stocks.split(",") if s.strip()]
    else:
        symbols = _load_universe(args.universe)

    if not symbols:
        print("[ERROR] Hisse listesi boş.")
        sys.exit(1)

    selected_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else None
    )

    print(f"\n{'=' * 60}")
    print(f"  ts_forecasting_lab — Batch Runner")
    print(f"{'=' * 60}")
    print(f"  Hisseler    : {', '.join(symbols)}")
    print(f"  Mod         : {args.mode}")
    print(f"  Modeller    : {selected_models or 'tümü'}")
    print(f"  Workers     : {args.workers}")
    print(f"  Data dizini : {args.data_dir}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("[DRY-RUN] Çalıştırılacak hisseler:")
        for sym in symbols:
            csv_path = os.path.join(args.data_dir, f"{sym}.csv")
            exists = "✓" if os.path.exists(csv_path) else "✗ (dosya yok)"
            print(f"  {sym:12s}  {csv_path}  {exists}")
        return

    # Worker argümanları hazırla
    worker_args = [
        {
            "symbol": sym,
            "data_dir": args.data_dir,
            "mode": args.mode,
            "selected_models": selected_models,
            "output_dir_base": args.output_dir,
        }
        for sym in symbols
    ]

    results: List[Dict[str, Any]] = []

    if args.workers == 1:
        # Sıralı mod — hata ayıklama için daha güvenli
        for wa in worker_args:
            print(f"\n[Batch] {wa['symbol']} işleniyor...")
            r = _run_single_stock(wa)
            results.append(r)
            status_icon = "✓" if r["status"] == "ok" else "✗"
            print(f"  {status_icon} {wa['symbol']:12s}  {r['status']:10s}  {r['duration_sec']:.0f}s")
    else:
        # Paralel mod
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_sym = {
                executor.submit(_run_single_stock, wa): wa["symbol"]
                for wa in worker_args
            }
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    r = future.result()
                except Exception as exc:
                    r = {
                        "symbol": sym,
                        "status": "error",
                        "error": str(exc),
                        "duration_sec": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {},
                    }
                results.append(r)
                status_icon = "✓" if r["status"] == "ok" else "✗"
                print(f"  {status_icon} {sym:12s}  {r['status']:10s}  {r['duration_sec']:.0f}s")

    # Özet
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    total_time = sum(r.get("duration_sec", 0) for r in results)

    print(f"\n{'=' * 60}")
    print(f"  Batch tamamlandı: {ok_count} başarılı, {err_count} hata, {skip_count} atlandı")
    print(f"  Toplam süre: {total_time:.0f}s ({total_time / 60:.1f} dk)")
    print(f"{'=' * 60}")

    if err_count > 0:
        print("\n  Hatalı hisseler:")
        for r in results:
            if r["status"] == "error":
                print(f"    {r['symbol']}: {r['error']}")

    csv_path = _save_summary(results, args.output_dir)
    print(f"\n  Özet rapor: {csv_path}")


if __name__ == "__main__":
    main()
