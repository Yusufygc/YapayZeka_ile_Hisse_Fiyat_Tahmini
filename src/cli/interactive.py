# -*- coding: utf-8 -*-
"""
src.cli.interactive - Interactive orchestration entrypoint.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Başlamadan önce bir menü sunarak hisse, validasyon modu ve
eğitilecek modellerin seçilmesine olanak tanır.
"""

import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.orchestrator import ForecastingPipeline
from src.pipeline.config import (
    PipelineConfig, 
    DataConfig, 
    ValidationConfig, 
    ModelConfig, 
    ExecutionConfig
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
META_CSV_STEMS = {"bist_universe", "bist_calendar"}

AVAILABLE_MODELS = [
    "Prophet",
    "XGBoost",
    "Random Forest",
    "LightGBM Return",
    "LSTM",
    "LSTM Lite",
    "Ridge Return",
    "ElasticNet Return",
    "DLinear",
    "NLinear",
]

PRESETS = {
    "1": ("Tüm Modeller",            AVAILABLE_MODELS),
    "2": ("Agac Tabanli",            ["XGBoost", "Random Forest", "LightGBM Return"]),
    "3": ("Yalnızca Derin Öğrenme",  ["LSTM", "LSTM Lite"]),
    "4": ("Hızlı (XGBoost + LSTM)",  ["XGBoost", "LSTM"]),
    "5": ("Yalnızca Prophet",        ["Prophet"]),
    "6": ("Modern Baseline",         ["Ridge Return", "ElasticNet Return", "LightGBM Return", "DLinear", "NLinear"]),
    "7": ("Manuel Seçim",            None),
}

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

def _is_stock_csv(path: str) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem in META_CSV_STEMS:
        return False
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            header = handle.readline().strip().lower()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1254", errors="ignore") as handle:
            header = handle.readline().strip().lower()
    except OSError:
        return False
    columns = {
        col.strip()
        .replace("ı", "i")
        .replace("ş", "s")
        .replace("ü", "u")
        .replace("ğ", "g")
        .replace("ö", "o")
        .replace("ç", "c")
        for col in header.split(",")
    }
    has_date = bool({"date", "tarih"} & columns)
    has_close = bool({"close", "kapanis"} & columns)
    return has_date and has_close


def select_stock() -> str:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"'{DATA_DIR}' dizininde CSV dosyası bulunamadı.")

    # Sadece hisse dosyalarını listele, meta veri dosyalarını hariç tut
    stock_files = [f for f in csv_files if _is_stock_csv(f)]
    stocks = [os.path.splitext(os.path.basename(f))[0] for f in stock_files]
    if not stocks:
        raise FileNotFoundError(f"'{DATA_DIR}' dizininde OHLCV hisse CSV dosyasi bulunamadi.")

    _header("ADIM 1 | Hisse Senedi Seçimi")
    for i, s in enumerate(stocks, 1):
        print(f"  [{i}] {s}")

    choice = _ask("Numara girin", {str(i) for i in range(1, len(stocks) + 1)})
    selected = stocks[int(choice) - 1]
    print(f"  ✔ Seçildi: {selected}\n")
    return stock_files[int(choice) - 1]

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

def confirm(cfg: PipelineConfig) -> bool:
    stock = os.path.splitext(os.path.basename(cfg.data.data_file))[0]
    _header("ÖZET — Başlamadan Önce Onayla")
    print(f"  Hisse        : {stock}")
    print(f"  Validasyon   : {cfg.validation.validation_mode}")
    print(f"  Modeller     : {', '.join(cfg.models.selected_models or [])}")
    print(f"  Target Mode  : {cfg.data.target_mode}")
    print(f"  Feature Mode : {cfg.data.feature_mode}")
    print(f"  Scaling Mode : {cfg.data.scaling_mode}")
    print(f"  Signal Mode  : {cfg.execution.signal_mode}")
    print(f"  Quality Gate : {cfg.execution.signal_config.quality_gate_mode}")
    print(f"  Macro Lag    : rate={cfg.data.macro_rate_lag_days}d, CPI={cfg.data.macro_cpi_lag_days}d")
    print(
        f"  WF Config    : splits={cfg.validation.wf_n_splits}, test={cfg.validation.wf_test_size}, "
        f"max_train={cfg.validation.wf_max_train_size}, type={cfg.validation.wf_window_type}, holdout={cfg.validation.final_holdout_size}"
    )
    print(
        "  Data Window  : "
        f"training_window_years={cfg.data.training_window_years}, "
        f"min_history={cfg.data.min_history_days}, new_listing_min={cfg.data.new_listing_min_days}"
    )
    print(
        "  Model Config : "
        f"min_seq={cfg.models.model_settings['deep_learning']['min_sequence_samples']}, "
        f"lstm_lite_min_seq={cfg.models.model_settings['deep_learning'].get('lstm_lite_min_sequence_samples')}, "
        f"val_ratio={cfg.models.model_settings['deep_learning']['validation_ratio']}, "
        f"arima_auto={cfg.models.model_settings['arima']['auto_order']}"
    )
    print(
        "  Feature QC   : "
        f"corr_prune={cfg.data.prune_correlated_features}, "
        f"corr_threshold={cfg.data.correlation_threshold}, "
        f"clip_warn={cfg.data.clip_shift_warning_threshold_pct}%"
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

    # Rely on dataclass defaults for everything else
    pipeline_cfg = PipelineConfig(
        data=DataConfig(
            data_file=data_file,
            auto_update_data=True,
            auto_update_interactive=True,
        ),
        validation=ValidationConfig(validation_mode=validation_mode),
        models=ModelConfig(selected_models=selected_models),
        execution=ExecutionConfig()
    )

    if not confirm(pipeline_cfg):
        print("\n  İptal edildi.\n")
        return

    print()
    pipeline = ForecastingPipeline(cfg=pipeline_cfg)
    pipeline.run_all()


if __name__ == "__main__":
    main()
