# -*- coding: utf-8 -*-
"""
evaluation_manager.py - Evaluation and reporting orchestration.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting import plot_equity_curves, run_backtest, save_backtest_report, save_trade_logs, summarize_backtest
from src.database.stock_model_db import StockModelDB, compute_composite_score
from src.evaluator import compute_metrics, enrich_with_benchmark_metrics, plot_comparison, plot_prediction_interval, save_metrics_report
from src.experiments.experiment_tracker import ExperimentTracker
from src.model_registry.model_registry import ModelRegistry
from src.preprocessor import reconstruct_prices_from_logret, reconstruct_prices_from_return


class EvaluationManager:
    def __init__(
        self,
        stock_symbol: str,
        outputs_dir: str,
        models_dir: str,
        tracker: ExperimentTracker,
        registry: ModelRegistry,
        feature_names: list,
        dataset_hash: str,
        dataset_metadata: Dict[str, Any],
        selected_models: Optional[list] = None,
        registry_version: str = "v5",
        stock_db: Optional[StockModelDB] = None,
        backtest_enabled: bool = True,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 100000.0,
    ):
        self.stock_symbol = stock_symbol
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.tracker = tracker
        self.registry = registry
        self.feature_names = feature_names
        self.dataset_hash = dataset_hash
        self.dataset_metadata = dataset_metadata
        self.selected_models = set(selected_models) if selected_models else None
        self.registry_version = registry_version
        self.stock_db = stock_db
        self.backtest_enabled = backtest_enabled
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital

        self.predictions = {}
        self.prediction_targets = {}
        self.quantile_predictions = {}
        self.single_backtest_inputs = {}
        self.y_true_aligned = None

    @staticmethod
    def _attach_composite_scores(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        enriched = enrich_with_benchmark_metrics(metrics_dict)
        for _, model_metrics in enriched.items():
            model_metrics["Composite_Score"] = compute_composite_score(model_metrics)
        return enriched

    def _target_to_price(self, preds_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        preds_target = np.asarray(preds_target).ravel()
        prev_close = np.asarray(prev_close).ravel()

        if target_mode == "log_return":
            return reconstruct_prices_from_logret(preds_target, prev_close)
        if target_mode == "return":
            return reconstruct_prices_from_return(preds_target, prev_close)
        if target_mode == "price":
            return preds_target
        raise ValueError(f"Desteklenmeyen target_mode: {target_mode}")

    def _save_selected_models_plot(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], save_path: str, title: str) -> None:
        if not self.selected_models:
            return
        selected_predictions = {name: preds for name, preds in predictions.items() if name in self.selected_models}
        if not selected_predictions:
            return
        plot_comparison(y_true, selected_predictions, save_path=save_path, title=title)
        print(f"[OK] Secilen modeller grafigi kaydedildi -> {save_path}")

    def _run_backtests(self, backtest_inputs: Dict[str, Dict[str, Any]], suffix: str) -> None:
        if not self.backtest_enabled or not backtest_inputs:
            return

        results = {}
        metrics_by_model = {}
        trades_by_model = {}
        equity_curves = {}
        target_mode = self.dataset_metadata.get("target_mode", "log_return")

        for model_name, payload in backtest_inputs.items():
            try:
                result = run_backtest(
                    dates=payload.get("dates"),
                    y_true_price=payload["y_true_price"],
                    pred_price=payload["pred_price"],
                    prev_close=payload["prev_close"],
                    pred_target=payload.get("pred_target"),
                    model_name=model_name,
                    validation_mode=suffix,
                    target_mode=target_mode,
                    commission_bps=self.commission_bps,
                    slippage_bps=self.slippage_bps,
                )
                results[model_name] = result
                metrics_by_model[model_name] = summarize_backtest(
                    result,
                    initial_capital=self.initial_capital,
                )
                trades_by_model[model_name] = result["trades"]
                equity_curves[model_name] = result["equity_curve"]
            except Exception as exc:
                print(f"  [WARN] {model_name} backtest basarisiz, atlaniyor: {exc}")

        if not metrics_by_model:
            return

        report_path = os.path.join(self.outputs_dir, f"backtest_report_v1_{suffix}.csv")
        trades_path = os.path.join(self.outputs_dir, f"backtest_trades_v1_{suffix}.csv")
        equity_path = os.path.join(self.outputs_dir, f"backtest_equity_curve_v1_{suffix}.png")

        save_backtest_report(metrics_by_model, report_path)
        save_trade_logs(trades_by_model, trades_path)
        plot_equity_curves(
            equity_curves,
            save_path=equity_path,
            title=f"{self.stock_symbol} - Backtest Equity Curve [{suffix}]",
        )

        if self.selected_models:
            selected_equity_path = os.path.join(self.outputs_dir, f"backtest_equity_curve_v1_{suffix}_selected.png")
            plot_equity_curves(
                equity_curves,
                save_path=selected_equity_path,
                title=f"{self.stock_symbol} - Selected Backtest Equity Curve [{suffix}]",
                selected_models=self.selected_models,
            )

    def generate_predictions(self, trained_models: dict, tensors: dict):
        print("\n" + "=" * 60)
        print("  ADIM 5 | Tahmin Uretimi ve Inverse Transform (EvaluationManager)")
        print("=" * 60)

        seq_models = {"LSTM", "TFT", "AttentionLSTM"}
        tree_models = {"XGBoost", "Random Forest"}

        prev_close_test = np.asarray(tensors["prev_close_test"]).ravel()
        dates_test = np.asarray(tensors["dates_test"])
        y_test_price = np.asarray(tensors["original_y_test_aligned"]).ravel()

        raw_preds = {}
        raw_pred_targets = {}
        raw_quantiles = {}

        for name, model in trained_models.items():
            try:
                if name == "Prophet":
                    preds_target = model.predict(tensors["X_test"], dates_test=tensors["dates_test"])
                elif name in tree_models:
                    preds_scaled = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
                elif name in seq_models:
                    if hasattr(model, "predict_quantiles"):
                        quantile_scaled = model.predict_quantiles(tensors["X_test_seq"])
                        quantile_target = np.column_stack([
                            tensors["scaler_y"].inverse_transform(quantile_scaled[:, idx].reshape(-1, 1)).ravel()
                            for idx in range(quantile_scaled.shape[1])
                        ])
                        preds_target = quantile_target[:, quantile_scaled.shape[1] // 2]
                        raw_quantiles[name] = quantile_target
                    else:
                        preds_scaled = model.predict(tensors["X_test_seq"])
                        preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
                else:
                    try:
                        preds_scaled = model.predict(tensors["X_test_seq"])
                    except Exception:
                        preds_scaled = model.predict(tensors["X_test_s"])
                    preds_target = tensors["scaler_y"].inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()

                preds_target = np.asarray(preds_target).ravel()
                k = min(len(preds_target), len(prev_close_test), len(y_test_price), len(dates_test))
                preds_target = preds_target[-k:]
                prev_close_aligned = prev_close_test[-k:]
                y_true_price_aligned = y_test_price[-k:]
                dates_aligned = dates_test[-k:]

                raw_pred_targets[name] = preds_target
                raw_preds[name] = self._target_to_price(preds_target, prev_close_aligned)

                if name in raw_quantiles:
                    aligned_quantiles = raw_quantiles[name][-k:]
                    raw_quantiles[name] = np.column_stack([
                        self._target_to_price(aligned_quantiles[:, idx], prev_close_aligned)
                        for idx in range(aligned_quantiles.shape[1])
                    ])

                self.single_backtest_inputs[name] = {
                    "dates": dates_aligned,
                    "y_true_price": y_true_price_aligned,
                    "pred_price": raw_preds[name],
                    "prev_close": prev_close_aligned,
                    "pred_target": preds_target,
                }
                print(f"  [OK] {name} tahmini uretildi - {len(raw_preds[name])} adim")
            except Exception as exc:
                print(f"  [WARN] {name} tahmini basarisiz, atlaniyor: {exc}")

        if not raw_preds:
            raise RuntimeError("Hicbir model tahmin uretemedi. Egitim adimini kontrol edin.")

        min_len = min(len(v) for v in raw_preds.values())
        self.predictions = {name: preds[-min_len:] for name, preds in raw_preds.items()}
        self.prediction_targets = {name: preds[-min_len:] for name, preds in raw_pred_targets.items()}
        self.quantile_predictions = {name: preds[-min_len:] for name, preds in raw_quantiles.items()}
        self.y_true_aligned = y_test_price[-min_len:]

    def evaluate_single_split(self, trained_models: dict):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Degerlendirme ve Registry (EvaluationManager)")
        print("=" * 60)

        metrics = {name: compute_metrics(self.y_true_aligned, preds) for name, preds in self.predictions.items()}
        metrics = self._attach_composite_scores(metrics)

        for name, model_metrics in metrics.items():
            self.tracker.log_run(name, {"validation": "single"}, model_metrics, self.feature_names, self.dataset_hash, self.dataset_metadata)

            model_ext = ".pt" if name == "TFT" else ".keras" if name == "LSTM" else ".pkl"
            model_filename = f"{name.replace(' ', '_').lower()}_model{model_ext}"
            model_path = os.path.join(self.models_dir, model_filename)

            original_model = trained_models.get(name)
            if original_model is None:
                print(f"  [WARN] {name} icin kayitli model bulunamadi, dosya kaydi atlaniyor.")
                model_path = ""
            else:
                original_model.save(model_path)

            self.registry.register(name, self.registry_version, self.feature_names, model_metrics, model_path, self.dataset_hash, self.dataset_metadata)

            if self.stock_db is not None:
                self.stock_db.log_experiment(
                    stock_symbol=self.stock_symbol,
                    model_name=name,
                    metrics=model_metrics,
                    model_path=model_path,
                    features=self.feature_names,
                    dataset_hash=self.dataset_hash,
                    validation_mode="single_split",
                    dataset_metadata=self.dataset_metadata,
                )

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_latest.csv")
        save_metrics_report(metrics, report_latest)

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest.png")
        title_str = f"{self.stock_symbol} - Model Kiyaslama (Gercek vs Tahmin)"
        plot_comparison(self.y_true_aligned, self.predictions, save_path=plot_latest, title=title_str)

        selected_plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_latest_selected.png")
        selected_title_str = f"{self.stock_symbol} - Secilen Modeller (Gercek vs Tahmin)"
        self._save_selected_models_plot(self.y_true_aligned, self.predictions, save_path=selected_plot_latest, title=selected_title_str)

        self._run_backtests(self.single_backtest_inputs, suffix="latest")

        if "TFT" in self.quantile_predictions:
            tft_quantiles = self.quantile_predictions["TFT"]
            quantile_labels = [f"Q{idx + 1}" for idx in range(tft_quantiles.shape[1])]
            if tft_quantiles.shape[1] == 3:
                quantile_labels = ["P10", "P50", "P90"]
            quantile_df = pd.DataFrame(tft_quantiles, columns=quantile_labels)
            quantile_df.insert(0, "Actual", self.y_true_aligned[-len(quantile_df):])
            quantile_csv = os.path.join(self.outputs_dir, "tft_quantiles_v5_latest.csv")
            quantile_df.to_csv(quantile_csv, sep=";", index=False)
            print(f"[OK] TFT quantile raporu kaydedildi -> {quantile_csv}")
            if tft_quantiles.shape[1] >= 3:
                interval_plot = os.path.join(self.outputs_dir, "tft_prediction_interval_v5_latest.png")
                plot_prediction_interval(
                    self.y_true_aligned[-len(tft_quantiles):],
                    tft_quantiles[:, 1],
                    tft_quantiles[:, 0],
                    tft_quantiles[:, 2],
                    save_path=interval_plot,
                    title=f"{self.stock_symbol} - TFT Tahmin Araligi",
                )

    def evaluate_walk_forward(
        self,
        wf_results: dict,
        wf_predictions: dict,
        wf_y_true: Any,
        wf_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        print("\n" + "=" * 60)
        print("  ADIM 7 | Degerlendirme Gosterimi (Walk-Forward)")
        print("=" * 60)

        wf_results = self._attach_composite_scores(wf_results)
        df_wf = pd.DataFrame(wf_results).T
        if "Composite_Score" in df_wf.columns:
            df_wf = df_wf.sort_values(by=["Composite_Score", "RMSE"], ascending=[False, True])
        print("\nWalk-Forward Ortalama Metrikleri:")
        print(df_wf)

        if self.stock_db is not None:
            for model_name, avg_metrics in wf_results.items():
                self.stock_db.log_experiment(
                    stock_symbol=self.stock_symbol,
                    model_name=model_name,
                    metrics=avg_metrics,
                    model_path="",
                    features=self.feature_names,
                    dataset_hash=self.dataset_hash,
                    validation_mode="walk_forward",
                    dataset_metadata=self.dataset_metadata,
                )

        report_latest = os.path.join(self.outputs_dir, "metrics_report_v4_wf.csv")
        save_metrics_report(wf_results, report_latest)

        if wf_y_true is None or len(wf_predictions) == 0:
            print("  [WARN] Walk-forward sonucu yok - grafik olusturulamadi.")
            return

        plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_wf.png")
        title_str = f"{self.stock_symbol} - Model Kiyaslama (Gercek vs Tahmin) [Walk-Forward]"
        plot_comparison(wf_y_true, wf_predictions, save_path=plot_latest, title=title_str)
        print(f"[OK] Walk-Forward karsilastirma grafigi kaydedildi -> {plot_latest}")

        selected_plot_latest = os.path.join(self.outputs_dir, "benchmark_comparison_v4_wf_selected.png")
        selected_title_str = f"{self.stock_symbol} - Secilen Modeller (Gercek vs Tahmin) [Walk-Forward]"
        self._save_selected_models_plot(wf_y_true, wf_predictions, save_path=selected_plot_latest, title=selected_title_str)

        self._run_backtests(wf_backtest_inputs or {}, suffix="wf")
