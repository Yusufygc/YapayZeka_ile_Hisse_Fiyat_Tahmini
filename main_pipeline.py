# -*- coding: utf-8 -*-
"""
main_pipeline.py — İnteraktif Orkestrasyon Girişi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Başlamadan önce bir menü sunarak hisse, validasyon modu ve
eğitilecek modellerin seçilmesine olanak tanır.
"""

import os
import glob
from src.pipeline.orchestrator import ForecastingPipeline

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

AVAILABLE_MODELS = [
    "Prophet",
    "XGBoost",
    "Random Forest",
    "LightGBM Return",
    "LSTM",
    "TFT",
    "Ridge Return",
    "ElasticNet Return",
    "DLinear",
    "NLinear",
    "PatchTST Experimental",
]
DEFAULT_TARGET_MODE = "log_return"
DEFAULT_FEATURE_MODE = "stationary_features"
DEFAULT_SCALING_MODE = "robust_x_standard_y_clip"
DEFAULT_SIGNAL_MODE = "professional"
DEFAULT_SIGNAL_ENTRY_COST_MULTIPLIER = 2.0
DEFAULT_SIGNAL_VOLATILITY_MULTIPLIER = 0.25
DEFAULT_MIN_HOLDING_BARS = 3
DEFAULT_MAX_HOLDING_BARS = 20
DEFAULT_TAKE_PROFIT_VOL_MULTIPLIER = 1.5
DEFAULT_STOP_LOSS_VOL_MULTIPLIER = 1.0
DEFAULT_MIN_DIRECTIONAL_ACCURACY = 52.0
DEFAULT_MAX_RMSE_VS_BENCHMARK = 1.05
DEFAULT_MIN_COMPOSITE_SCORE = 50.0
DEFAULT_EMERGENCY_STOP_OVERRIDES_MIN_HOLD = True
DEFAULT_MACRO_RATE_LAG_DAYS = 1
DEFAULT_MACRO_CPI_LAG_DAYS = 15
DEFAULT_WF_N_SPLITS = 12
DEFAULT_WF_MIN_TRAIN_SIZE = 504
DEFAULT_WF_TEST_SIZE = 21
DEFAULT_WF_MAX_TRAIN_SIZE = 756
DEFAULT_WF_WINDOW_TYPE = "sliding"
DEFAULT_FINAL_HOLDOUT_SIZE = 60
DEFAULT_PRUNE_CORRELATED_FEATURES = False
DEFAULT_CORRELATION_THRESHOLD = 0.98
DEFAULT_CLIP_SHIFT_WARNING_THRESHOLD_PCT = 1.0
DEFAULT_MODEL_CONFIG = {
    "arima": {"auto_order": False, "order": (1, 0, 0)},
    "deep_learning": {
        "min_sequence_samples": 64,
        "validation_ratio": 0.1,
        "min_validation_samples": 32,
        "lstm": {
            "epochs_single": 80,
            "epochs_wf": 50,
            "epochs_final": 50,
            "patience": 15,
            "lr_patience": 5,
            "dropout": 0.2,
            "batch_size": 32,
        },
        "tft": {
            "model_label": "TFT-like Quantile Sequence Model",
            "epochs_single": 80,
            "epochs_wf": 50,
            "epochs_final": 50,
            "patience_single": 15,
            "patience_wf": 12,
            "patience_final": 12,
            "lr_patience": 5,
            "dropout": 0.3,
            "batch_size": 32,
        },
    },
    "experimental_sequence_baselines": {
        "enabled_models": ["DLinear", "NLinear", "PatchTST Experimental"],
        "patchtst_status": "evaluation_path_prepared_not_production",
        "patchtst_config": {"lookback": 128, "patch_length": 16, "stride": 8, "alpha": 1.0},
    },
    "gradient_boosting": {"lightgbm_optional": True},
}

PRESETS = {
    "1": ("Tüm Modeller",            AVAILABLE_MODELS),
    "2": ("Yalnızca Ağaç Tabanlı",   ["XGBoost", "Random Forest"]),
    "3": ("Yalnızca Derin Öğrenme",  ["LSTM", "TFT"]),
    "4": ("Hızlı (XGBoost + LSTM)",  ["XGBoost", "LSTM"]),
    "5": ("Yalnızca Prophet",        ["Prophet"]),
    "6": ("Manuel Seçim",            None),
}

PRESETS["2"] = ("Agac Tabanli", ["XGBoost", "Random Forest", "LightGBM Return"])
PRESETS["6"] = ("Modern Baseline", ["Ridge Return", "ElasticNet Return", "LightGBM Return", "DLinear", "NLinear", "PatchTST Experimental"])
PRESETS["7"] = ("Manuel Secim", None)

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def _divider(char="─", width=58):
    print(char * width)

def _header(title: str):
    _divider("═")
    print(f"  {title}")
    _divider("═")

def _ask(prompt: str, valid: set | None = None) -> str:
    while True:
        val = input(f"  » {prompt}: ").strip()
        if valid is None or val in valid:
            return val
        print(f"  [!] Geçersiz seçim. Lütfen şunlardan birini girin: {sorted(valid)}")

# ─── Adım 1 — Hisse Senedi ───────────────────────────────────────────────────

def select_stock() -> str:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"'{DATA_DIR}' dizininde CSV dosyası bulunamadı.")

    stocks = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    _header("ADIM 1 | Hisse Senedi Seçimi")
    for i, s in enumerate(stocks, 1):
        print(f"  [{i}] {s}")

    choice = _ask("Numara girin", {str(i) for i in range(1, len(stocks) + 1)})
    selected = stocks[int(choice) - 1]
    print(f"  ✔ Seçildi: {selected}\n")
    return os.path.join(DATA_DIR, f"{selected}.csv")

# ─── Adım 2 — Validasyon Modu ────────────────────────────────────────────────

def select_validation_mode() -> str:
    _header("ADIM 2 | Validasyon Modu")
    print("  [1] single_split   — Tek seferlik eğit/test bölünmesi (hızlı)")
    print("  [2] walk_forward   — Kayan pencere çapraz doğrulama (kapsamlı)")
    choice = _ask("Numara girin", {"1", "2"})
    mode = "single_split" if choice == "1" else "walk_forward"
    print(f"  ✔ Seçildi: {mode}\n")
    return mode

# ─── Adım 3 — Model Seçimi ───────────────────────────────────────────────────

def _manual_model_select() -> list[str]:
    print()
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        print(f"  [{i}] {m}")
    print()
    raw = input("  » Virgülle ayırarak numara girin (örn: 1,3,4): ").strip()
    indices = {s.strip() for s in raw.split(",")}
    valid = {str(i) for i in range(1, len(AVAILABLE_MODELS) + 1)}
    chosen = []
    for idx in indices:
        if idx in valid:
            chosen.append(AVAILABLE_MODELS[int(idx) - 1])
        else:
            print(f"  [!] '{idx}' geçersiz, atlandı.")
    if not chosen:
        print("  [!] Hiç model seçilmedi, tüm modeller eğitilecek.")
        return AVAILABLE_MODELS[:]
    return chosen

def select_models() -> list[str]:
    _header("ADIM 3 | Model Seçimi")
    for key, (label, models) in PRESETS.items():
        model_str = ", ".join(models) if models else "—"
        print(f"  [{key}] {label:<28} ({model_str})")

    choice = _ask("Numara girin", set(PRESETS.keys()))
    label, models = PRESETS[choice]

    if models is None:
        selected = _manual_model_select()
    else:
        selected = models[:]

    print(f"  ✔ Seçildi: {', '.join(selected)}\n")
    return selected

# ─── Özet & Onay ─────────────────────────────────────────────────────────────

def confirm(data_file: str, mode: str, models: list[str]) -> bool:
    stock = os.path.splitext(os.path.basename(data_file))[0]
    _header("ÖZET — Başlamadan Önce Onayla")
    print(f"  Hisse        : {stock}")
    print(f"  Validasyon   : {mode}")
    print(f"  Modeller     : {', '.join(models)}")
    print(f"  Target Mode  : {DEFAULT_TARGET_MODE}")
    print(f"  Feature Mode : {DEFAULT_FEATURE_MODE}")
    print(f"  Scaling Mode : {DEFAULT_SCALING_MODE}")
    print(f"  Signal Mode  : {DEFAULT_SIGNAL_MODE}")
    print(f"  Macro Lag    : rate={DEFAULT_MACRO_RATE_LAG_DAYS}d, CPI={DEFAULT_MACRO_CPI_LAG_DAYS}d")
    print(
        f"  WF Config    : splits={DEFAULT_WF_N_SPLITS}, test={DEFAULT_WF_TEST_SIZE}, "
        f"max_train={DEFAULT_WF_MAX_TRAIN_SIZE}, type={DEFAULT_WF_WINDOW_TYPE}, holdout={DEFAULT_FINAL_HOLDOUT_SIZE}"
    )
    print(
        "  Model Config : "
        f"min_seq={DEFAULT_MODEL_CONFIG['deep_learning']['min_sequence_samples']}, "
        f"val_ratio={DEFAULT_MODEL_CONFIG['deep_learning']['validation_ratio']}, "
        f"arima_auto={DEFAULT_MODEL_CONFIG['arima']['auto_order']}"
    )
    print(
        "  Feature QC   : "
        f"corr_prune={DEFAULT_PRUNE_CORRELATED_FEATURES}, "
        f"corr_threshold={DEFAULT_CORRELATION_THRESHOLD}, "
        f"clip_warn={DEFAULT_CLIP_SHIFT_WARNING_THRESHOLD_PCT}%"
    )
    _divider()
    ans = _ask("Devam edilsin mi? [e/h]", {"e", "h", "E", "H"})
    return ans.lower() == "e"

# ─── Ana Akış ─────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    _header("TS FORECASTING LAB")

    data_file = select_stock()
    validation_mode = select_validation_mode()
    selected_models = select_models()

    if not confirm(data_file, validation_mode, selected_models):
        print("\n  İptal edildi.\n")
        return

    print()
    pipeline = ForecastingPipeline(
        data_file=data_file,
        test_ratio=0.20,
        time_steps=30,
        validation_mode=validation_mode,
        selected_models=selected_models,
        target_mode=DEFAULT_TARGET_MODE,
        feature_mode=DEFAULT_FEATURE_MODE,
        scaling_mode=DEFAULT_SCALING_MODE,
        signal_mode=DEFAULT_SIGNAL_MODE,
        signal_entry_cost_multiplier=DEFAULT_SIGNAL_ENTRY_COST_MULTIPLIER,
        signal_volatility_multiplier=DEFAULT_SIGNAL_VOLATILITY_MULTIPLIER,
        min_holding_bars=DEFAULT_MIN_HOLDING_BARS,
        max_holding_bars=DEFAULT_MAX_HOLDING_BARS,
        take_profit_vol_multiplier=DEFAULT_TAKE_PROFIT_VOL_MULTIPLIER,
        stop_loss_vol_multiplier=DEFAULT_STOP_LOSS_VOL_MULTIPLIER,
        min_directional_accuracy=DEFAULT_MIN_DIRECTIONAL_ACCURACY,
        max_rmse_vs_benchmark=DEFAULT_MAX_RMSE_VS_BENCHMARK,
        min_composite_score=DEFAULT_MIN_COMPOSITE_SCORE,
        emergency_stop_overrides_min_hold=DEFAULT_EMERGENCY_STOP_OVERRIDES_MIN_HOLD,
        macro_rate_lag_days=DEFAULT_MACRO_RATE_LAG_DAYS,
        macro_cpi_lag_days=DEFAULT_MACRO_CPI_LAG_DAYS,
        wf_n_splits=DEFAULT_WF_N_SPLITS,
        wf_min_train_size=DEFAULT_WF_MIN_TRAIN_SIZE,
        wf_test_size=DEFAULT_WF_TEST_SIZE,
        wf_max_train_size=DEFAULT_WF_MAX_TRAIN_SIZE,
        wf_window_type=DEFAULT_WF_WINDOW_TYPE,
        final_holdout_size=DEFAULT_FINAL_HOLDOUT_SIZE,
        model_config=DEFAULT_MODEL_CONFIG,
        prune_correlated_features=DEFAULT_PRUNE_CORRELATED_FEATURES,
        correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
        clip_shift_warning_threshold_pct=DEFAULT_CLIP_SHIFT_WARNING_THRESHOLD_PCT,
    )
    pipeline.run_all()


if __name__ == "__main__":
    main()
