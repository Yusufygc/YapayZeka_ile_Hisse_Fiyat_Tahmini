# -*- coding: utf-8 -*-
"""e2_faz1_horizon_compare.py - PREDICTIVE-only horizon kiyaslamasi (E2 Faz 1).

Soru: haftalik (5-gun) ileri getiri, gunluk (1-gun) getiriden daha mi ongorulebilir?
Tek-hisse overfit + dusuk sinyal sorununun "horizon" kolundaki testidir.

Yontem (her sembol, her h icin):
  - Ozellikler BIR KEZ FeaturePipeline ile uretilir (h'den bagimsiz).
  - Hedef: y[t] = log(close[t+h]/close[t]); X = ozellikler[:-h].
  - Kronolojik %80/%20 split (son %20 test).
  - Ridge + LightGBM (deterministik) egitilir, test'te tahmin edilir.
  - Metrik: Dir_Acc (yon dogrulugu, h'ler arasi KIYASLANABILIR), RMSE/MAE
    (h buyudukce olcek buyur, h-ici karsilastirma icin), base-rate (% pozitif).

Backtest/Sharpe YOK — bu yalniz predictive sinyal taramasi. Backtest semantigi
h>1 icin henuz horizon-aware degil (Faz 1b).

Kullanim:
    python tools/e2_faz1_horizon_compare.py
    python tools/e2_faz1_horizon_compare.py --symbols EREGL,AKBNK --horizons 1,5
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import FeaturePipeline

_COL_MAP = {
    "Tarih": "Date", "Açılış": "Open", "Yüksek": "High",
    "Düşük": "Low", "Düzeltilmiş_Kapanış": "Close", "Hacim": "Volume",
}


def _load_features(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(_PROJECT_ROOT, "data", f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, encoding="utf-8-sig").rename(columns=_COL_MAP)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    fp = FeaturePipeline(feature_mode="stationary_features", enable_calendar_features=False)
    frame = fp.engineer_features(raw, macro_df=None, symbol=symbol)
    return frame.dropna().reset_index(drop=True)


def _fit_predict(model, Xtr, ytr, Xte):
    sx = StandardScaler().fit(Xtr)
    model.fit(sx.transform(Xtr), ytr)
    return model.predict(sx.transform(Xte))


def _metrics(pred, actual) -> dict:
    pred, actual = np.asarray(pred), np.asarray(actual)
    dir_acc = float(np.mean(np.sign(pred) == np.sign(actual)) * 100)
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    return {"dir_acc": dir_acc, "rmse": rmse, "mae": mae}


def _run_symbol(symbol: str, horizons: list[int]) -> list[dict]:
    frame = _load_features(symbol)
    if frame is None or len(frame) < 300:
        print(f"  [{symbol}] atlandi (frame yok / kisa)")
        return []
    feat_cols = [c for c in frame.columns if c not in {"Date", "Close"}]
    close = frame["Close"].to_numpy(dtype=float)
    rows = []
    try:
        import lightgbm as lgb
        lgbm_ok = True
    except Exception:
        lgbm_ok = False

    for h in horizons:
        if h >= len(close) - 50:
            continue
        target = np.log(close[h:] / close[:-h])
        X = frame[feat_cols].iloc[:-h].to_numpy(dtype=float)
        n = len(target)
        cut = int(n * 0.8)
        Xtr, Xte = X[:cut], X[cut:]
        ytr, yte = target[:cut], target[cut:]
        base_pos = float(np.mean(yte > 0) * 100)

        rid = _metrics(_fit_predict(Ridge(alpha=1.0), Xtr, ytr, Xte), yte)
        rows.append({"symbol": symbol, "h": h, "model": "Ridge", "n_test": len(yte),
                     "base_pos%": round(base_pos, 1), **{k: round(v, 4) for k, v in rid.items()}})
        if lgbm_ok:
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                  random_state=42, verbose=-1)
            lg = _metrics(_fit_predict(m, Xtr, ytr, Xte), yte)
            rows.append({"symbol": symbol, "h": h, "model": "LightGBM", "n_test": len(yte),
                         "base_pos%": round(base_pos, 1), **{k: round(v, 4) for k, v in lg.items()}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="EREGL,AKBNK,TUPRS,AEFES,SASA")
    ap.add_argument("--horizons", type=str, default="1,3,5,10")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    all_rows = []
    for sym in symbols:
        print(f"[H-COMPARE] {sym} ...")
        all_rows.extend(_run_symbol(sym, horizons))

    if not all_rows:
        print("Sonuc yok.")
        return
    df = pd.DataFrame(all_rows)
    print("\n=== Per (symbol, h, model) ===")
    print(df.to_string(index=False))
    print("\n=== Dir_Acc ortalamasi (model x h) — sinyal/gurultu ozeti ===")
    piv = df.pivot_table(index="h", columns="model", values="dir_acc", aggfunc="mean").round(2)
    print(piv.to_string())
    out = os.path.join(_PROJECT_ROOT, "outputs", "e2_faz1_horizon_compare.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[H-COMPARE] -> {out}")


if __name__ == "__main__":
    main()
