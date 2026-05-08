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
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting import run_backtest, summarize_backtest
from src.backtesting.signals import SignalConfig
from src.utils.reporting_utils import route_output_path, write_csv_and_aligned_view

_VALID_CALIBRATION_SCOPES = ("wf_train",)
SIGNAL_CALIBRATION_REPORT_COLUMNS = [
    "Trial",
    "min_directional_accuracy",
    "volatility_multiplier",
    "entry_cost_multiplier",
    "min_entry_threshold",
    "max_holding_bars",
    "take_profit_vol_multiplier",
    "stop_loss_vol_multiplier",
    "Sampler",
    "Seed",
    "Grid_Size",
    "Executed_Trials",
    "Adaptive_Expanded",
    "Coverage_Status",
    "Model_Count",
    "Total_Trade_Count",
    "Min_Trade_Count",
    "Mean_Net_Return",
    "Mean_BuyHold_Return",
    "Mean_Excess_Return",
    "Risk_Adjusted_Score",
    "Beats_BuyHold_Count",
    "Median_Net_Return",
    "Mean_Max_Drawdown",
    "Mean_Sharpe",
    "Mean_Calmar",
    "Eval_Net_Return",
    "Eval_BuyHold_Return",
    "Eval_Excess_Return",
    "Eval_Sharpe",
    "Eval_Max_Drawdown",
    "Eval_Trade_Count",
    "OOS_Constraint_Passed",
    "Reject_Reason",
    "Active_For_Execution",
    "Selection_Rank",
    "Positive_Net_Return",
    "Meets_Min_Trade_Count",
]


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _risk_adjusted_score(
    mean_net_return: float,
    mean_excess_return: float,
    mean_sharpe: float,
    mean_max_drawdown: float,
) -> float:
    sharpe_normalized = float(np.clip(mean_sharpe / 3.0, -1.0, 1.0))
    drawdown_score = 1.0 + mean_max_drawdown
    return float(
        0.35 * mean_net_return
        + 0.25 * mean_excess_return
        + 0.25 * sharpe_normalized
        + 0.15 * drawdown_score
    )


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
            "execution_policy": "decision_applies_to_aligned_next_bar_return",
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
        wf_evaluation_backtest_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str = "",
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
        configured_min_trades = int(getattr(self, "signal_calibration_min_trades", 6) or 6)
        min_trade_count = max(configured_min_trades, 3 * len(candidates))
        rows = []
        config_by_trial: Dict[int, SignalConfig] = {}

        full_grid = self._signal_calibration_grid(base_cfg)
        grid, grid_metadata = self._apply_signal_calibration_trial_policy(full_grid)
        trial_count_for_metrics = max(1, len(candidates) * int(grid_metadata["executed_trials"]))

        def run_trials(selected_grid: list[Dict[str, float | int]]) -> None:
            for params in selected_grid:
                trial_idx = len(rows) + 1
                cfg = replace(base_cfg, quality_gate_mode="soft", **params)
                config_by_trial[trial_idx] = cfg
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
                            trial_count=trial_count_for_metrics,
                        ))
                    except Exception as exc:
                        summaries.append({
                            "Model": model_name,
                            "Net_Return": np.nan,
                            "BuyHold_Return": np.nan,
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

        run_trials(grid)
        initial_best_row = self._select_signal_calibration_row(rows)
        initial_coverage = self._signal_calibration_coverage_status(full_grid, grid)
        adaptive_expanded = False
        if self._should_expand_signal_calibration(
            rows,
            initial_best_row,
            coverage_status=initial_coverage,
        ):
            used_keys = {self._grid_param_key(params) for params in grid}
            second_cap = min(len(full_grid), max(int(grid_metadata["executed_trials"]) * 2, 128))
            remaining_cap = max(0, second_cap - len(grid))
            if remaining_cap:
                extra_grid = self._sample_signal_calibration_grid(
                    full_grid,
                    cap=remaining_cap,
                    seed=int(grid_metadata["seed"]) + 1,
                    exclude_keys=used_keys,
                )
                if extra_grid:
                    adaptive_expanded = True
                    grid.extend(extra_grid)
                    grid_metadata["executed_trials"] = int(len(grid))
                    run_trials(extra_grid)

        grid_metadata["adaptive_expanded"] = bool(adaptive_expanded)
        grid_metadata["coverage_status"] = self._signal_calibration_coverage_status(full_grid, grid)
        require_oos = bool(getattr(self, "signal_calibration_require_oos_confirmation", True))
        oos_status = "not_required"
        if require_oos and wf_evaluation_backtest_inputs:
            oos_status = "applied"
            for row in rows:
                trial = int(row.get("Trial", 0) or 0)
                cfg = config_by_trial.get(trial, base_cfg)
                eval_summaries = self._evaluate_signal_config_on_backtest_inputs(
                    backtest_inputs=wf_evaluation_backtest_inputs,
                    model_names=candidates,
                    model_metrics_by_model=model_metrics_by_model,
                    target_mode=target_mode,
                    cfg=cfg,
                    trial_count=trial_count_for_metrics,
                    validation_mode="walk_forward_signal_oos_confirmation",
                )
                row.update(self._summarize_oos_confirmation(
                    summaries=eval_summaries,
                    min_trade_count=min_trade_count,
                ))
        elif require_oos:
            oos_status = "skipped_no_evaluation_inputs"
            for row in rows:
                row.update(self._empty_oos_confirmation("oos_skipped_no_evaluation_inputs"))
        else:
            for row in rows:
                row.update(self._empty_oos_confirmation("oos_not_required", passed=True, active=True))

        calibration_df = pd.DataFrame(rows)
        best_row = self._select_signal_calibration_row(rows)
        selected_row = self._select_confirmed_signal_calibration_row(rows) if require_oos and oos_status == "applied" else best_row
        rejection_active = bool(require_oos and oos_status == "applied" and selected_row is None)
        active_row = selected_row if selected_row is not None else best_row
        best_config = (
            config_by_trial.get(int(active_row["Trial"]), base_cfg)
            if active_row is not None and active_row.get("Trial") is not None
            else base_cfg
        )
        if not calibration_df.empty:
            calibration_df["Sampler"] = grid_metadata["sampler"]
            calibration_df["Seed"] = grid_metadata["seed"]
            calibration_df["Grid_Size"] = int(grid_metadata["grid_size"])
            calibration_df["Executed_Trials"] = int(grid_metadata["executed_trials"])
            calibration_df["Adaptive_Expanded"] = bool(grid_metadata["adaptive_expanded"])
            calibration_df["Coverage_Status"] = grid_metadata["coverage_status"]
            if rejection_active:
                calibration_df["Active_For_Execution"] = False
            elif active_row is not None and "Trial" in calibration_df.columns:
                calibration_df["Active_For_Execution"] = calibration_df["Trial"] == active_row.get("Trial")
            calibration_df["_sort_key"] = calibration_df.apply(
                lambda row: self._signal_calibration_sort_key(row.to_dict()),
                axis=1,
            )
            calibration_df.sort_values(
                by=[
                    "Meets_Min_Trade_Count",
                    "Risk_Adjusted_Score",
                    "Mean_Excess_Return",
                    "Mean_Net_Return",
                    "Mean_Max_Drawdown",
                    "Total_Trade_Count",
                ],
                ascending=[False, False, False, False, False, False],
                inplace=True,
            )
            calibration_df.drop(columns=["_sort_key"], inplace=True, errors="ignore")
            calibration_df["Selection_Rank"] = np.arange(1, len(calibration_df) + 1)
            if active_row is not None and "Trial" in calibration_df.columns:
                rank_match = calibration_df.loc[calibration_df["Trial"] == active_row.get("Trial"), "Selection_Rank"]
                if not rank_match.empty:
                    active_row = dict(active_row)
                    active_row["Selection_Rank"] = int(rank_match.iloc[0])
                    active_row["Sampler"] = grid_metadata["sampler"]
                    active_row["Seed"] = grid_metadata["seed"]
                    active_row["Grid_Size"] = int(grid_metadata["grid_size"])
                    active_row["Executed_Trials"] = int(grid_metadata["executed_trials"])
                    active_row["Adaptive_Expanded"] = bool(grid_metadata["adaptive_expanded"])
                    active_row["Coverage_Status"] = grid_metadata["coverage_status"]
                    active_row["Active_For_Execution"] = not rejection_active
                    if rejection_active:
                        active_row["Reject_Reason"] = "rejected_no_valid_oos_trial"

        self.signal_config = best_config
        self.signal_threshold_source = "walk_forward_signal_rejected" if rejection_active else "walk_forward_signal_calibration"
        no_trade_trials = int(sum(int(row.get("Total_Trade_Count", 0) or 0) == 0 for row in rows))
        self.signal_threshold_calibration_summary.update({
            "execution_calibration_status": "rejected_no_valid_oos_trial" if rejection_active else "applied",
            "execution_calibration_trials": int(len(rows)),
            "grid_size": int(grid_metadata["grid_size"]),
            "executed_trials": int(grid_metadata["executed_trials"]),
            "trial_cap": grid_metadata["trial_cap"],
            "calibration_profile": grid_metadata["calibration_profile"],
            "sampler": grid_metadata["sampler"],
            "seed": grid_metadata["seed"],
            "adaptive_expanded": bool(grid_metadata["adaptive_expanded"]),
            "coverage_status": grid_metadata["coverage_status"],
            "oos_confirmation_status": oos_status,
            "oos_confirmation_required": require_oos,
            "oos_min_eval_excess_return": float(getattr(self, "signal_calibration_min_eval_excess_return", 0.0)),
            "oos_min_eval_sharpe": float(getattr(self, "signal_calibration_min_eval_sharpe", 0.0)),
            "reject_behavior": getattr(self, "signal_calibration_reject_behavior", "no_trade"),
            "execution_calibration_models": candidates,
            "execution_calibration_min_trade_count": int(min_trade_count),
            "execution_calibration_objective": getattr(self, "signal_calibration_objective", "risk_adjusted"),
            "execution_calibration_set": "walk_forward_backtest_inputs_only",
            "calibration_scope": getattr(self, "calibration_scope", "wf_train"),
            "final_holdout_used": False,
            "no_trade_trials": no_trade_trials,
            "selected_meets_min_trade_count": bool(active_row.get("Meets_Min_Trade_Count", False)) if active_row else False,
            "selected_total_trade_count": int(active_row.get("Total_Trade_Count", 0) or 0) if active_row else 0,
            "active_for_execution": bool(active_row and not rejection_active),
            "reject_reason": "rejected_no_valid_oos_trial" if rejection_active else "",
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
            "selected_execution_result": active_row or {},
        })
        self.dataset_metadata["signal_threshold_config"] = self._signal_threshold_metadata()

        decision_md = self._get_signal_calibration_decision_md(active_row)
        self._write_signal_calibration_reports(calibration_df, decision_md, suffix=suffix)

        return {
            "calibration_df": calibration_df,
            "decision_md": decision_md,
            "best_row": active_row,
            "rejected": rejection_active,
        }

    # ------------------------------------------------------------------ #
    #  Grid / trial helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _signal_calibration_grid(base_cfg: SignalConfig) -> list[Dict[str, float | int]]:
        min_dir_values = sorted({48.0, 50.0, float(base_cfg.min_directional_accuracy)})
        vol_values = sorted({0.10, 0.15, 0.20, 0.25, 0.30, float(base_cfg.volatility_multiplier)})
        entry_cost_values = sorted({1.5, 2.0, 2.5, float(base_cfg.entry_cost_multiplier)})
        min_entry_values = sorted({0.0, 0.001, 0.002, float(base_cfg.min_entry_threshold)})
        max_hold_values = sorted({10, 15, 20, int(base_cfg.max_holding_bars)})
        take_profit_values = sorted({1.0, 1.5, 2.0, float(base_cfg.take_profit_vol_multiplier)})
        stop_loss_values = sorted({0.75, 1.0, 1.25, float(base_cfg.stop_loss_vol_multiplier)})

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

    def _apply_signal_calibration_trial_policy(
        self,
        grid: list[Dict[str, float | int]],
    ) -> tuple[list[Dict[str, float | int]], Dict[str, Any]]:
        profile = str(getattr(self, "signal_calibration_profile", "production") or "production").lower()
        if profile not in {"production", "research"}:
            profile = "production"
        sampler = str(getattr(self, "signal_calibration_sampler", "adaptive_stratified") or "adaptive_stratified").lower()
        seed = int(getattr(self, "signal_calibration_seed", 42) or 42)
        trial_cap = getattr(self, "signal_calibration_max_trials", 64)
        trial_cap = None if trial_cap is None else max(1, int(trial_cap))
        if profile == "research":
            selected_grid = list(grid)
            effective_cap = None
            sampler = "full_grid"
        elif sampler in {"adaptive_stratified", "stratified"}:
            effective_cap = trial_cap
            selected_grid = self._sample_signal_calibration_grid(
                grid,
                cap=effective_cap if effective_cap is not None else len(grid),
                seed=seed,
            )
        else:
            effective_cap = trial_cap
            selected_grid = list(grid[:effective_cap]) if effective_cap is not None else list(grid)
            sampler = "prefix"
        return selected_grid, {
            "grid_size": int(len(grid)),
            "executed_trials": int(len(selected_grid)),
            "trial_cap": effective_cap,
            "calibration_profile": profile,
            "sampler": sampler,
            "seed": seed,
            "adaptive_expanded": False,
            "coverage_status": self._signal_calibration_coverage_status(grid, selected_grid),
        }

    @staticmethod
    def _grid_param_key(params: Dict[str, float | int]) -> tuple:
        return tuple(sorted(params.items()))

    @staticmethod
    def _signal_calibration_param_values(grid: list[Dict[str, float | int]]) -> Dict[str, set]:
        values: Dict[str, set] = {}
        for params in grid:
            for key, value in params.items():
                values.setdefault(key, set()).add(value)
        return values

    @classmethod
    def _signal_calibration_coverage_status(
        cls,
        full_grid: list[Dict[str, float | int]],
        selected_grid: list[Dict[str, float | int]],
    ) -> str:
        if not full_grid:
            return "empty_grid"
        full_values = cls._signal_calibration_param_values(full_grid)
        selected_values = cls._signal_calibration_param_values(selected_grid)
        missing = []
        for key, values in sorted(full_values.items()):
            uncovered = sorted(values - selected_values.get(key, set()))
            if uncovered:
                missing.append(f"{key}={uncovered}")
        return "complete" if not missing else "missing:" + "; ".join(missing)

    @classmethod
    def _sample_signal_calibration_grid(
        cls,
        grid: list[Dict[str, float | int]],
        *,
        cap: int,
        seed: int,
        exclude_keys: Optional[set[tuple]] = None,
    ) -> list[Dict[str, float | int]]:
        if cap <= 0 or not grid:
            return []
        exclude_keys = exclude_keys or set()
        available = [
            (idx, params)
            for idx, params in enumerate(grid)
            if cls._grid_param_key(params) not in exclude_keys
        ]
        if not available:
            return []
        if cap >= len(available):
            return [dict(params) for _, params in available]

        rng = np.random.default_rng(seed)
        full_values = cls._signal_calibration_param_values([params for _, params in available])
        required = {(key, value) for key, values in full_values.items() for value in values}
        selected_indices: list[int] = []
        selected_keys: set[tuple] = set()
        covered: set[tuple] = set()

        while required - covered and len(selected_indices) < cap:
            best = None
            for idx, params in available:
                key = cls._grid_param_key(params)
                if key in selected_keys:
                    continue
                gained = {(name, value) for name, value in params.items()} - covered
                score = (len(gained), float(rng.random()))
                if best is None or score > best[0]:
                    best = (score, idx, params)
            if best is None or best[0][0] == 0:
                break
            _, idx, params = best
            selected_indices.append(idx)
            selected_keys.add(cls._grid_param_key(params))
            covered.update((name, value) for name, value in params.items())

        remaining = [
            (idx, params)
            for idx, params in available
            if cls._grid_param_key(params) not in selected_keys
        ]
        if len(selected_indices) < cap and remaining:
            order = rng.permutation(len(remaining))
            for pos in order[: cap - len(selected_indices)]:
                idx, params = remaining[int(pos)]
                selected_indices.append(idx)
                selected_keys.add(cls._grid_param_key(params))

        index_to_params = {idx: params for idx, params in available}
        return [dict(index_to_params[idx]) for idx in selected_indices[:cap]]

    @staticmethod
    def _should_expand_signal_calibration(
        rows: list[Dict[str, Any]],
        best_row: Dict[str, Any] | None,
        *,
        coverage_status: str,
    ) -> bool:
        if not rows or best_row is None:
            return True
        if coverage_status != "complete":
            return True
        if not bool(best_row.get("Meets_Min_Trade_Count", False)):
            return True
        if _safe_float(best_row.get("Mean_Excess_Return"), -1.0) <= 0.0:
            return True
        if _safe_float(best_row.get("Mean_Sharpe"), -1.0) <= 0.0:
            return True
        valid_count = sum(bool(row.get("Meets_Min_Trade_Count", False)) for row in rows)
        return valid_count < max(1, int(np.ceil(len(rows) * 0.15)))

    def _evaluate_signal_config_on_backtest_inputs(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        model_names: list[str],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        target_mode: str,
        cfg: SignalConfig,
        trial_count: int,
        validation_mode: str,
    ) -> list[Dict[str, Any]]:
        summaries = []
        for model_name in model_names:
            payload = backtest_inputs.get(model_name)
            if payload is None:
                continue
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
                    validation_mode=validation_mode,
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
                    trial_count=trial_count,
                ))
            except Exception as exc:
                summaries.append({
                    "Model": model_name,
                    "Net_Return": np.nan,
                    "BuyHold_Return": np.nan,
                    "Max_Drawdown": np.nan,
                    "Trade_Count": 0,
                    "Sharpe": np.nan,
                    "Calibration_Error": str(exc),
                })
        return summaries

    @staticmethod
    def _empty_oos_confirmation(
        reason: str,
        *,
        passed: bool = False,
        active: bool = False,
    ) -> Dict[str, Any]:
        return {
            "Eval_Net_Return": np.nan,
            "Eval_BuyHold_Return": np.nan,
            "Eval_Excess_Return": np.nan,
            "Eval_Sharpe": np.nan,
            "Eval_Max_Drawdown": np.nan,
            "Eval_Trade_Count": 0,
            "OOS_Constraint_Passed": bool(passed),
            "Reject_Reason": "" if passed else reason,
            "Active_For_Execution": bool(active),
        }

    def _summarize_oos_confirmation(
        self,
        *,
        summaries: list[Dict[str, Any]],
        min_trade_count: int,
    ) -> Dict[str, Any]:
        valid = [row for row in summaries if np.isfinite(float(row.get("Net_Return", np.nan)))]
        if not valid:
            return self._empty_oos_confirmation("oos_failed_all_models")

        net_returns = np.asarray([float(row.get("Net_Return", 0.0)) for row in valid], dtype=float)
        buy_hold_returns = np.asarray([float(row.get("BuyHold_Return", 0.0)) for row in valid], dtype=float)
        drawdowns = np.asarray([float(row.get("Max_Drawdown", 0.0)) for row in valid], dtype=float)
        sharpes = np.asarray([float(row.get("Sharpe", 0.0)) for row in valid], dtype=float)
        trade_count = int(sum(int(float(row.get("Trade_Count", 0) or 0)) for row in valid))
        eval_net = float(np.nanmean(net_returns))
        eval_buy_hold = float(np.nanmean(buy_hold_returns))
        eval_excess = eval_net - eval_buy_hold
        eval_sharpe = float(np.nanmean(sharpes))
        eval_drawdown = float(np.nanmean(drawdowns))

        min_excess = float(getattr(self, "signal_calibration_min_eval_excess_return", 0.0))
        min_sharpe = float(getattr(self, "signal_calibration_min_eval_sharpe", 0.0))
        reasons = []
        if eval_excess <= min_excess:
            reasons.append("eval_excess_return_below_min")
        if eval_sharpe <= min_sharpe:
            reasons.append("eval_sharpe_below_min")
        if trade_count < int(min_trade_count):
            reasons.append("eval_trade_count_below_min")
        passed = not reasons
        return {
            "Eval_Net_Return": round(eval_net, 6),
            "Eval_BuyHold_Return": round(eval_buy_hold, 6),
            "Eval_Excess_Return": round(eval_excess, 6),
            "Eval_Sharpe": round(eval_sharpe, 6),
            "Eval_Max_Drawdown": round(eval_drawdown, 6),
            "Eval_Trade_Count": trade_count,
            "OOS_Constraint_Passed": bool(passed),
            "Reject_Reason": "" if passed else ",".join(reasons),
            "Active_For_Execution": False,
        }

    @staticmethod
    def _calibration_constraint_passed(row: Dict[str, Any]) -> bool:
        return (
            bool(row.get("Meets_Min_Trade_Count", False))
            and _safe_float(row.get("Mean_Excess_Return"), -1.0) > 0.0
        )

    @classmethod
    def _select_confirmed_signal_calibration_row(
        cls,
        rows: list[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        candidates = [
            row
            for row in rows
            if cls._calibration_constraint_passed(row)
            and bool(row.get("OOS_Constraint_Passed", False))
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                _safe_float(row.get("Risk_Adjusted_Score"), -1e9),
                _safe_float(row.get("Eval_Excess_Return"), -1e9),
                _safe_float(row.get("Eval_Sharpe"), -1e9),
                _safe_float(row.get("Mean_Max_Drawdown"), -1.0),
            ),
        )

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
                "Mean_BuyHold_Return": np.nan,
                "Mean_Excess_Return": np.nan,
                "Risk_Adjusted_Score": np.nan,
                "Beats_BuyHold_Count": 0,
                "Median_Net_Return": np.nan,
                "Mean_Max_Drawdown": np.nan,
                "Total_Trade_Count": 0,
                "Meets_Min_Trade_Count": False,
                "Mean_Sharpe": np.nan,
                "Mean_Calmar": np.nan,
                "Selection_Rank": None,
                "Positive_Net_Return": False,
                "Status": "failed_all_models",
            }

        net_returns = np.asarray([float(row.get("Net_Return", 0.0)) for row in valid], dtype=float)
        buy_hold_returns = np.asarray([float(row.get("BuyHold_Return", 0.0)) for row in valid], dtype=float)
        excess_returns = net_returns - buy_hold_returns
        drawdowns = np.asarray([float(row.get("Max_Drawdown", 0.0)) for row in valid], dtype=float)
        sharpes = np.asarray([float(row.get("Sharpe", 0.0)) for row in valid], dtype=float)
        calmars = np.asarray([_safe_float(row.get("Calmar"), 0.0) for row in valid], dtype=float)
        trade_count = int(sum(int(float(row.get("Trade_Count", 0) or 0)) for row in valid))
        beats_buy_hold_count = int(sum(bool(row.get("Beats_BuyHold_NetReturn", False)) for row in valid))
        mean_net = float(np.nanmean(net_returns))
        mean_buy_hold = float(np.nanmean(buy_hold_returns))
        mean_excess = float(np.nanmean(excess_returns))
        mean_drawdown = float(np.nanmean(drawdowns))
        mean_sharpe = float(np.nanmean(sharpes))
        risk_score = _risk_adjusted_score(
            mean_net_return=mean_net,
            mean_excess_return=mean_excess,
            mean_sharpe=mean_sharpe,
            mean_max_drawdown=mean_drawdown,
        )
        return {
            "Trial": trial_idx,
            **params,
            "Model_Count": int(len(valid)),
            "Mean_Net_Return": round(mean_net, 6),
            "Mean_BuyHold_Return": round(mean_buy_hold, 6),
            "Mean_Excess_Return": round(mean_excess, 6),
            "Risk_Adjusted_Score": round(risk_score, 6),
            "Beats_BuyHold_Count": beats_buy_hold_count,
            "Median_Net_Return": round(float(np.nanmedian(net_returns)), 6),
            "Mean_Max_Drawdown": round(mean_drawdown, 6),
            "Total_Trade_Count": trade_count,
            "Min_Trade_Count": int(min_trade_count),
            "Meets_Min_Trade_Count": bool(trade_count >= min_trade_count),
            "Mean_Sharpe": round(mean_sharpe, 6),
            "Mean_Calmar": round(float(np.nanmean(calmars)), 6),
            "Selection_Rank": None,
            "Positive_Net_Return": bool(mean_net > 0.0),
            "Status": "ok",
        }

    @staticmethod
    def _signal_calibration_sort_key(row: Dict[str, Any]) -> tuple:
        mean_net = _safe_float(row.get("Mean_Net_Return"), -1e9)
        mean_excess = _safe_float(row.get("Mean_Excess_Return"), mean_net)
        risk_score = _safe_float(
            row.get("Risk_Adjusted_Score"),
            _risk_adjusted_score(
                mean_net_return=mean_net,
                mean_excess_return=mean_excess,
                mean_sharpe=_safe_float(row.get("Mean_Sharpe"), 0.0),
                mean_max_drawdown=_safe_float(row.get("Mean_Max_Drawdown"), -1.0),
            ),
        )
        mean_drawdown = _safe_float(row.get("Mean_Max_Drawdown"), -1.0)
        trade_count = int(row.get("Total_Trade_Count", 0) or 0)
        return (
            bool(row.get("Meets_Min_Trade_Count", False)),
            risk_score,
            mean_excess,
            mean_net,
            mean_drawdown,
            trade_count,
        )

    @classmethod
    def _select_signal_calibration_row(cls, rows: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not rows:
            return None
        meeting = [row for row in rows if bool(row.get("Meets_Min_Trade_Count", False))]
        if meeting:
            return max(meeting, key=cls._signal_calibration_sort_key)

        traded = [row for row in rows if int(row.get("Total_Trade_Count", 0) or 0) > 0]
        if traded:
            return max(
                traded,
                key=lambda row: (
                    int(row.get("Total_Trade_Count", 0) or 0),
                    _safe_float(row.get("Risk_Adjusted_Score"), -1e9),
                    _safe_float(row.get("Mean_Excess_Return"), -1e9),
                    _safe_float(row.get("Mean_Net_Return"), -1e9),
                    _safe_float(row.get("Mean_Max_Drawdown"), -1.0),
                ),
            )
        return max(rows, key=cls._signal_calibration_sort_key)

    def _write_signal_calibration_reports(
        self,
        calibration_df: pd.DataFrame,
        decision_md: str,
        *,
        suffix: str = "",
    ) -> None:
        outputs_dir = getattr(self, "outputs_dir", "")
        if not outputs_dir:
            return
        try:
            os.makedirs(outputs_dir, exist_ok=True)
            suffix_part = f"_{suffix}" if suffix else ""
            csv_path = os.path.join(outputs_dir, f"signal_calibration_v1{suffix_part}.csv")
            md_path = route_output_path(os.path.join(outputs_dir, f"signal_calibration_decision_v1{suffix_part}.md"))
            output_paths = write_csv_and_aligned_view(
                calibration_df,
                csv_path,
                columns=SIGNAL_CALIBRATION_REPORT_COLUMNS,
            )
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as handle:
                handle.write(decision_md)
            print(f"  [OK] Signal calibration raporu kaydedildi -> {output_paths['csv']}")
            print(f"  [OK] Signal calibration karar raporu kaydedildi -> {md_path}")
        except Exception as exc:
            print(f"  [WARN] Signal calibration raporu kaydedilemedi: {exc}")

    def _get_signal_calibration_decision_md(self, best_row: Dict[str, Any] | None) -> str:
        scope = getattr(self, "calibration_scope", "wf_train")
        handle = StringIO()
        handle.write("# Signal Calibration Decision v2\n\n")
        handle.write(f"- Calibration scope: {scope}\n")
        handle.write("- Calibration set: walk-forward calibration backtest inputs only\n")
        handle.write("- Final holdout used: False\n")
        handle.write("- Objective: valid trade count, then risk-adjusted score, excess return, net return, drawdown\n\n")
        if not best_row:
            handle.write("No valid calibration trial was selected.\n")
            return handle.getvalue()

        active = bool(best_row.get("Active_For_Execution", True))
        if active:
            handle.write("## Selected Parameters\n\n")
        else:
            handle.write("## Rejected Candidate Parameters\n\n")
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
            "Sampler",
            "Seed",
            "Grid_Size",
            "Executed_Trials",
            "Adaptive_Expanded",
            "Coverage_Status",
            "OOS_Constraint_Passed",
            "Reject_Reason",
            "Active_For_Execution",
            "Mean_Net_Return",
            "Mean_BuyHold_Return",
            "Mean_Excess_Return",
            "Risk_Adjusted_Score",
            "Median_Net_Return",
            "Mean_Max_Drawdown",
            "Total_Trade_Count",
            "Min_Trade_Count",
            "Mean_Sharpe",
            "Mean_Calmar",
            "Eval_Net_Return",
            "Eval_BuyHold_Return",
            "Eval_Excess_Return",
            "Eval_Sharpe",
            "Eval_Max_Drawdown",
            "Eval_Trade_Count",
            "Beats_BuyHold_Count",
            "Selection_Rank",
            "Positive_Net_Return",
            "Meets_Min_Trade_Count",
        ]:
            handle.write(f"- `{key}`: `{best_row.get(key)}`\n")
        return handle.getvalue()
