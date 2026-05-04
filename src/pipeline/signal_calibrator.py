# -*- coding: utf-8 -*-
"""
signal_calibrator.py - Sinyal kalibrasyon ve esik optimizasyonu (Faz 2.1 Mixin).

Sorumluluklar:
  - _signal_threshold_metadata(): metaveri snapshot
  - _assert_wf_train_scope(): leakage guard — kalibrasyon yalnizca WF fold verisini kullanir
  - _calibrate_signal_quality_thresholds(): WF fold metriklerinden DIR/RMSE/Composite esik ayari
  - _calibrate_walk_forward_signal_parameters(): backtest grid search ile execution param tuning
  - _signal_calibration_grid(): parametre kombinasyon listesi
  - _summarize_signal_calibration_trial(): tek deneme ozeti
  - _signal_calibration_sort_key(): siralama anahtari
  - _get_signal_calibration_decision_md(): markdown raporu

Leakage Korumasi (Faz 2.5):
  calibration_scope = "wf_train"  (tek gecerli deger)
  - Kalibrasyon YALNIZCA walk-forward fold metrikleri + WF backtest girdilerini kullanir.
  - Final holdout verisi hicbir kalibrasyona girmez.
  - _assert_wf_train_scope() bu kurali her iki metoda giristeki RuntimeError ile zorunlu kilar.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from io import StringIO
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting import run_backtest, summarize_backtest
from src.backtesting.signals import SignalConfig

_VALID_CALIBRATION_SCOPES = ("wf_train",)


class _SignalCalibratorMixin:
    """Mixin: sinyal esik kalibrasyonu ve grid search optimizasyonu."""

    # ------------------------------------------------------------------ #
    #  Leakage guard                                                       #
    # ------------------------------------------------------------------ #

    def _assert_wf_train_scope(self) -> None:
        """
        calibration_scope'un guvensiz bir degere ayarlanmadigini dogrular.

        Gecerli tek deger: "wf_train"
          - Kalibrasyon yalnizca walk-forward fold verisi kullanir.
          - Final holdout verisi kalibrasyona GIREMEZ.

        Herhangi bir baska deger (ornegin "all" ya da "holdout")
        RuntimeError firlatir ve pipeline'i durdurur.
        """
        scope = getattr(self, "calibration_scope", "wf_train")
        if scope not in _VALID_CALIBRATION_SCOPES:
            raise RuntimeError(
                f"Gecersiz calibration_scope={scope!r}. "
                f"Izin verilen degerler: {_VALID_CALIBRATION_SCOPES}. "
                "Final holdout verisi kalibrasyona dahil edilemez."
            )

    # ------------------------------------------------------------------ #
    #  Metadata snapshot                                                   #
    # ------------------------------------------------------------------ #

    def _signal_threshold_metadata(self) -> Dict[str, Any]:
        cfg = asdict(self.signal_config)
        scope = getattr(self, "calibration_scope", "wf_train")
        return {
            "phase": "phase6_backtest_standard",
            "source": self.signal_threshold_source,
            "calibration_scope": scope,
            "selection_scope": (
                "walk_forward_calibration_folds"
                if self.signal_threshold_source != "default_config"
                else "configured_defaults"
            ),
            "active_from_stage": (
                "walk_forward_backtest_signal_filtering"
                if self.signal_threshold_source != "default_config"
                else "initial_signal_filtering"
            ),
            "final_holdout_optimized": False,
            "quality_thresholds": {
                "quality_gate_mode": self.signal_config.quality_gate_mode,
                "min_directional_accuracy": self.signal_config.min_directional_accuracy,
                "max_rmse_vs_benchmark": self.signal_config.max_rmse_vs_benchmark,
                "min_composite_score": self.signal_config.min_composite_score,
            },
            "default_quality_thresholds": {
                "quality_gate_mode": self.default_signal_config.quality_gate_mode,
                "min_directional_accuracy": self.default_signal_config.min_directional_accuracy,
                "max_rmse_vs_benchmark": self.default_signal_config.max_rmse_vs_benchmark,
                "min_composite_score": self.default_signal_config.min_composite_score,
            },
            "full_signal_config": cfg,
            "execution_policy": "decision_applies_to_next_bar_return",
            "cost_policy": {
                "commission_bps": self.commission_bps,
                "slippage_bps": self.slippage_bps,
                "entry_exit_accounted_separately": True,
            },
            "calibration_summary": self.signal_threshold_calibration_summary,
        }

    # ------------------------------------------------------------------ #
    #  Quality-threshold calibration (from WF fold metrics)              #
    # ------------------------------------------------------------------ #

    def _calibrate_signal_quality_thresholds(
        self, wf_fold_metrics: Dict[str, list[Dict[str, Any]]]
    ) -> None:
        # Leakage guard: yalnizca wf_train kapsaminda calisir
        self._assert_wf_train_scope()

        rows = []
        for model_name, model_rows in wf_fold_metrics.items():
            if model_name in self.signal_config.benchmark_only_models:
                continue
            rows.extend(model_rows)

        if len(rows) < 3:
            self.signal_threshold_source = "default_config"
            self.signal_threshold_calibration_summary = {
                "status": "skipped_insufficient_calibration_folds",
                "fold_metric_rows": len(rows),
                "calibration_fold_count": len({row.get("Fold") for row in rows}),
                "active_from_stage": "initial_signal_filtering",
                "calibration_scope": getattr(self, "calibration_scope", "wf_train"),
                "final_holdout_used": False,
            }
            self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()
            return

        calibration_df = pd.DataFrame(rows)
        dir_values = pd.to_numeric(calibration_df.get("Dir_Acc"), errors="coerce").dropna()
        rmse_values = pd.to_numeric(calibration_df.get("RMSE_vs_benchmark"), errors="coerce").dropna()
        composite_values = pd.to_numeric(calibration_df.get("Composite_Score"), errors="coerce").dropna()

        min_directional_accuracy = self.default_signal_config.min_directional_accuracy
        max_rmse_vs_benchmark = self.default_signal_config.max_rmse_vs_benchmark
        min_composite_score = self.default_signal_config.min_composite_score

        if not dir_values.empty:
            min_directional_accuracy = max(min_directional_accuracy, float(dir_values.quantile(0.25)))
        if not rmse_values.empty:
            max_rmse_vs_benchmark = min(max_rmse_vs_benchmark, float(rmse_values.quantile(0.75)))
        if not composite_values.empty:
            min_composite_score = max(min_composite_score, float(composite_values.quantile(0.25)))

        self.signal_config = replace(
            self.signal_config,
            min_directional_accuracy=round(min_directional_accuracy, 2),
            max_rmse_vs_benchmark=round(max_rmse_vs_benchmark, 4),
            min_composite_score=round(min_composite_score, 4),
        )
        self.signal_threshold_source = "walk_forward_calibration_folds"
        self.signal_threshold_calibration_summary = {
            "status": "applied",
            "fold_metric_rows": int(len(rows)),
            "calibration_fold_count": int(calibration_df["Fold"].nunique()) if "Fold" in calibration_df.columns else None,
            "dir_acc_q25": round(float(dir_values.quantile(0.25)), 4) if not dir_values.empty else None,
            "rmse_vs_benchmark_q75": round(float(rmse_values.quantile(0.75)), 4) if not rmse_values.empty else None,
            "composite_score_q25": round(float(composite_values.quantile(0.25)), 4) if not composite_values.empty else None,
            "calibration_scope": getattr(self, "calibration_scope", "wf_train"),
            "calibration_set": "walk_forward_folds_only",
            "active_from_stage": "walk_forward_backtest_signal_filtering",
            "final_holdout_used": False,
        }
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

    # ------------------------------------------------------------------ #
    #  Execution-parameter calibration (grid search over WF backtests)   #
    # ------------------------------------------------------------------ #

    def _calibrate_walk_forward_signal_parameters(
        self,
        *,
        wf_backtest_inputs: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Leakage guard: yalnizca wf_train kapsaminda calisir
        self._assert_wf_train_scope()

        target_mode = self.dataset_metadata.get("target_mode", "log_return")
        candidates = [
            name
            for name in wf_backtest_inputs
            if name not in self.signal_config.benchmark_only_models
        ]
        if not candidates:
            self.signal_threshold_calibration_summary.update({
                "execution_calibration_status": "skipped_no_non_benchmark_models",
                "calibration_scope": getattr(self, "calibration_scope", "wf_train"),
                "final_holdout_used": False,
            })
            self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()
            return {}

        base_cfg = self.signal_config
        min_trade_count = max(3, len(candidates))
        rows = []
        best_row = None
        best_config = base_cfg
        best_key = None

        grid = self._signal_calibration_grid(base_cfg)
        for trial_idx, params in enumerate(grid, start=1):
            cfg = replace(base_cfg, quality_gate_mode="soft", **params)
            summaries = []
            for model_name in candidates:
                payload = wf_backtest_inputs[model_name]
                try:
                    result = run_backtest(
                        dates=payload.get("dates"),
                        prediction_dates=payload.get("prediction_dates"),
                        y_true_price=payload["y_true_price"],
                        pred_price=payload["pred_price"],
                        prev_close=payload["prev_close"],
                        fold_ids=payload.get("fold_ids"),
                        market_regime=payload.get("market_regime"),
                        pred_target=payload.get("pred_target"),
                        model_name=model_name,
                        validation_mode="walk_forward_signal_calibration",
                        target_mode=target_mode,
                        commission_bps=self.commission_bps,
                        slippage_bps=self.slippage_bps,
                        signal_mode="professional",
                        signal_config=cfg,
                        model_metrics=model_metrics_by_model.get(model_name, {}),
                    )
                    summaries.append(summarize_backtest(
                        result,
                        initial_capital=self.initial_capital,
                        trial_count=max(1, len(candidates) * len(grid)),
                    ))
                except Exception as exc:
                    summaries.append({
                        "Model": model_name,
                        "Net_Return": np.nan,
                        "Max_Drawdown": np.nan,
                        "Trade_Count": 0,
                        "Sharpe": np.nan,
                        "Calibration_Error": str(exc),
                    })

            trial_row = self._summarize_signal_calibration_trial(
                trial_idx=trial_idx,
                params=params,
                summaries=summaries,
                min_trade_count=min_trade_count,
            )
            rows.append(trial_row)
            key = self._signal_calibration_sort_key(trial_row)
            if best_key is None or key > best_key:
                best_key = key
                best_row = trial_row
                best_config = cfg

        calibration_df = pd.DataFrame(rows)
        if not calibration_df.empty:
            calibration_df.sort_values(
                by=[
                    "Positive_Net_Return",
                    "Mean_Max_Drawdown",
                    "Meets_Min_Trade_Count",
                    "Mean_Sharpe",
                    "Mean_Net_Return",
                ],
                ascending=[False, False, False, False, False],
                inplace=True,
            )

        self.signal_config = best_config
        self.signal_threshold_source = "walk_forward_signal_calibration"
        self.signal_threshold_calibration_summary.update({
            "execution_calibration_status": "applied",
            "execution_calibration_trials": int(len(rows)),
            "execution_calibration_models": candidates,
            "execution_calibration_min_trade_count": int(min_trade_count),
            "execution_calibration_objective": "positive_net_return_then_max_drawdown_then_trade_count_then_sharpe",
            "execution_calibration_set": "walk_forward_backtest_inputs_only",
            "calibration_scope": getattr(self, "calibration_scope", "wf_train"),
            "final_holdout_used": False,
            "selected_execution_params": {
                "min_directional_accuracy": self.signal_config.min_directional_accuracy,
                "volatility_multiplier": self.signal_config.volatility_multiplier,
                "entry_cost_multiplier": self.signal_config.entry_cost_multiplier,
                "min_entry_threshold": self.signal_config.min_entry_threshold,
                "max_holding_bars": self.signal_config.max_holding_bars,
                "take_profit_vol_multiplier": self.signal_config.take_profit_vol_multiplier,
                "stop_loss_vol_multiplier": self.signal_config.stop_loss_vol_multiplier,
                "quality_gate_mode": self.signal_config.quality_gate_mode,
            },
            "selected_execution_result": best_row or {},
        })
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

        decision_md = self._get_signal_calibration_decision_md(best_row)

        return {
            "calibration_df": calibration_df,
            "decision_md": decision_md,
            "best_row": best_row,
        }

    # ------------------------------------------------------------------ #
    #  Grid / trial helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _signal_calibration_grid(base_cfg: SignalConfig) -> list[Dict[str, float | int]]:
        min_dir_values = sorted({48.0, 50.0, float(base_cfg.min_directional_accuracy)})
        vol_values = sorted({0.10, 0.15, 0.20, float(base_cfg.volatility_multiplier)})
        entry_cost_values = sorted({1.5, float(base_cfg.entry_cost_multiplier)})
        min_entry_values = sorted({0.0, 0.001, float(base_cfg.min_entry_threshold)})
        max_hold_values = sorted({10, int(base_cfg.max_holding_bars)})
        take_profit_values = sorted({1.0, float(base_cfg.take_profit_vol_multiplier)})
        stop_loss_values = sorted({0.75, float(base_cfg.stop_loss_vol_multiplier)})

        grid = []
        for min_dir in min_dir_values:
            for vol in vol_values:
                for entry_cost in entry_cost_values:
                    for min_entry in min_entry_values:
                        for max_hold in max_hold_values:
                            for take_profit in take_profit_values:
                                for stop_loss in stop_loss_values:
                                    grid.append({
                                        "min_directional_accuracy": round(float(min_dir), 2),
                                        "volatility_multiplier": round(float(vol), 4),
                                        "entry_cost_multiplier": round(float(entry_cost), 4),
                                        "min_entry_threshold": round(float(min_entry), 6),
                                        "max_holding_bars": int(max_hold),
                                        "take_profit_vol_multiplier": round(float(take_profit), 4),
                                        "stop_loss_vol_multiplier": round(float(stop_loss), 4),
                                    })
        return grid

    @staticmethod
    def _summarize_signal_calibration_trial(
        *,
        trial_idx: int,
        params: Dict[str, float | int],
        summaries: list[Dict[str, Any]],
        min_trade_count: int,
    ) -> Dict[str, Any]:
        valid = [row for row in summaries if np.isfinite(float(row.get("Net_Return", np.nan)))]
        if not valid:
            return {
                "Trial": trial_idx,
                **params,
                "Model_Count": 0,
                "Mean_Net_Return": np.nan,
                "Median_Net_Return": np.nan,
                "Mean_Max_Drawdown": np.nan,
                "Total_Trade_Count": 0,
                "Meets_Min_Trade_Count": False,
                "Mean_Sharpe": np.nan,
                "Positive_Net_Return": False,
                "Status": "failed_all_models",
            }

        net_returns = np.asarray([float(row.get("Net_Return", 0.0)) for row in valid], dtype=float)
        drawdowns = np.asarray([float(row.get("Max_Drawdown", 0.0)) for row in valid], dtype=float)
        sharpes = np.asarray([float(row.get("Sharpe", 0.0)) for row in valid], dtype=float)
        trade_count = int(sum(int(float(row.get("Trade_Count", 0) or 0)) for row in valid))
        mean_net = float(np.nanmean(net_returns))
        return {
            "Trial": trial_idx,
            **params,
            "Model_Count": int(len(valid)),
            "Mean_Net_Return": round(mean_net, 6),
            "Median_Net_Return": round(float(np.nanmedian(net_returns)), 6),
            "Mean_Max_Drawdown": round(float(np.nanmean(drawdowns)), 6),
            "Total_Trade_Count": trade_count,
            "Min_Trade_Count": int(min_trade_count),
            "Meets_Min_Trade_Count": bool(trade_count >= min_trade_count),
            "Mean_Sharpe": round(float(np.nanmean(sharpes)), 6),
            "Positive_Net_Return": bool(mean_net > 0.0),
            "Status": "ok",
        }

    @staticmethod
    def _signal_calibration_sort_key(row: Dict[str, Any]) -> tuple:
        mean_net = float(row.get("Mean_Net_Return", -1e9))
        mean_drawdown = float(row.get("Mean_Max_Drawdown", -1.0))
        mean_sharpe = float(row.get("Mean_Sharpe", -1e9))
        trade_count = int(row.get("Total_Trade_Count", 0) or 0)
        return (
            bool(row.get("Positive_Net_Return", False)),
            mean_drawdown,
            bool(row.get("Meets_Min_Trade_Count", False)),
            mean_sharpe,
            mean_net,
            trade_count,
        )

    def _get_signal_calibration_decision_md(self, best_row: Dict[str, Any] | None) -> str:
        scope = getattr(self, "calibration_scope", "wf_train")
        handle = StringIO()
        handle.write("# Signal Calibration Decision v2\n\n")
        handle.write(f"- Calibration scope: {scope}\n")
        handle.write("- Calibration set: walk-forward backtest inputs only\n")
        handle.write("- Final holdout used: False\n")
        handle.write("- Objective: positive net return, then lower max drawdown, then minimum trade count, then Sharpe\n\n")
        if not best_row:
            handle.write("No valid calibration trial was selected.\n")
            return handle.getvalue()

        handle.write("## Selected Parameters\n\n")
        for key in [
            "min_directional_accuracy",
            "volatility_multiplier",
            "entry_cost_multiplier",
            "min_entry_threshold",
            "max_holding_bars",
            "take_profit_vol_multiplier",
            "stop_loss_vol_multiplier",
        ]:
            handle.write(f"- `{key}`: `{best_row.get(key)}`\n")

        handle.write("\n## Selected Walk-Forward Result\n\n")
        for key in [
            "Mean_Net_Return",
            "Median_Net_Return",
            "Mean_Max_Drawdown",
            "Total_Trade_Count",
            "Min_Trade_Count",
            "Mean_Sharpe",
            "Positive_Net_Return",
            "Meets_Min_Trade_Count",
        ]:
            handle.write(f"- `{key}`: `{best_row.get(key)}`\n")
        return handle.getvalue()
