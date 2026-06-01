# -*- coding: utf-8 -*-
"""
backtest_runner.py - Backtest yurutme ve sinyal gate diagnostikleri (Faz 2.1 Mixin).

Sorumluluklar:
  - _run_backtests(): tum modeller icin backtest yurutme
  - _get_shadow_backtests(): farkli sinyal senaryolari karsilastirmasi
  - _get_signal_gate_diagnostics(): kalite kapi diagnostikleri
  - Dusuk seviye yardimcilar: _diagnostic_numeric, _diagnostic_float, vb.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting import run_backtest, summarize_backtest
from src.backtesting.reporting import (
    plot_equity_curves,
    save_backtest_report,
    save_order_report,
    save_trade_logs,
)


class _BacktestRunnerMixin:
    """Mixin: backtest yurutme ve diagnostik raporlama.

    Faz 3.2 (E1 owner-forward epigi): owner-forward kaldirildi. READ-ONLY config
    `self.ctx.X` (commission_bps, slippage_bps, initial_capital, backtest_enabled,
    signal_mode, dataset_metadata, outputs_dir, stock_symbol, write_trade_logs,
    signal_calibration_min_trades, signal_calibration_reject_behavior,
    auto_signal_diagnostics, enable_gate_diagnostics, enable_shadow_backtests),
    mutable runtime `self.state.X` (signal_config, signal_threshold_source,
    latest_backtest_results, latest_backtest_metrics).
    """

    # ------------------------------------------------------------------ #
    #  Low-level helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _diagnostic_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)

    @staticmethod
    def _diagnostic_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _count_decision(frame: pd.DataFrame, decision: str) -> int:
        if not isinstance(frame, pd.DataFrame) or "Decision" not in frame.columns:
            return 0
        return int((frame["Decision"].astype(str) == decision).sum())

    @staticmethod
    def _payload_expected_return(payload: Dict[str, Any], target_mode: str) -> np.ndarray:
        pred_target = payload.get("pred_target")
        if pred_target is not None and target_mode in {"log_return", "return"}:
            return np.asarray(pred_target, dtype=float).ravel()
        pred_price = np.asarray(payload.get("pred_price", []), dtype=float).ravel()
        prev_close = np.asarray(payload.get("prev_close", []), dtype=float).ravel()
        k = min(len(pred_price), len(prev_close))
        if k == 0:
            return np.array([], dtype=float)
        return (pred_price[-k:] / np.maximum(prev_close[-k:], 1e-12)) - 1.0

    # ------------------------------------------------------------------ #
    #  Shadow backtests (scenario comparison)                             #
    # ------------------------------------------------------------------ #

    def _get_shadow_backtests(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> Dict[str, Any]:
        if not backtest_inputs:
            return {}

        scenarios = []
        if str(self.ctx.signal_mode).lower() == "simple":
            scenarios.append(("simple_current", "simple", self.state.signal_config, False))
        scenarios.extend([
            ("professional_current", "professional", replace(self.state.signal_config, quality_gate_mode="hard"), True),
            ("professional_soft_gate", "professional", replace(self.state.signal_config, quality_gate_mode="soft"), True),
            ("legacy_directional", "legacy", self.state.signal_config, False),
        ])

        rows = []
        trade_frames = []
        for model_name, payload in backtest_inputs.items():
            for shadow_mode, signal_mode, scenario_config, use_model_metrics in scenarios:
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
                        validation_mode=f"{suffix}_{shadow_mode}",
                        target_mode=target_mode,
                        commission_bps=self.ctx.commission_bps,
                        slippage_bps=self.ctx.slippage_bps,
                        signal_mode=signal_mode,
                        signal_config=scenario_config,
                        model_metrics=model_metrics_by_model.get(model_name, {}) if use_model_metrics else {},
                    )
                    summary = summarize_backtest(
                        result,
                        initial_capital=self.ctx.initial_capital,
                        trial_count=max(1, len(backtest_inputs)),
                    )
                    curve = result.get("equity_curve", pd.DataFrame())
                    decisions = curve["Decision"].astype(str) if isinstance(curve, pd.DataFrame) and "Decision" in curve.columns else pd.Series(dtype=str)
                    risk_states = curve["Risk_State"].astype(str) if isinstance(curve, pd.DataFrame) and "Risk_State" in curve.columns else pd.Series(dtype=str)
                    multipliers = self._diagnostic_numeric(curve, "Quality_Threshold_Multiplier")
                    regime_multipliers = self._diagnostic_numeric(curve, "Regime_Threshold_Multiplier")
                    vol_multipliers = self._diagnostic_numeric(curve, "Volatility_Threshold_Multiplier")
                    final_multipliers = self._diagnostic_numeric(curve, "Final_Threshold_Multiplier")
                    rows.append({
                        "Model": model_name,
                        "Shadow_Mode": shadow_mode,
                        "Signal_Mode": signal_mode,
                        "Quality_Gate_Mode": scenario_config.quality_gate_mode,
                        "Trade_Count": summary.get("Trade_Count"),
                        "Exposure": summary.get("Exposure"),
                        "Net_Return": summary.get("Net_Return"),
                        "BuyHold_Return": summary.get("BuyHold_Return"),
                        "Sharpe": summary.get("Sharpe"),
                        "Max_Drawdown": summary.get("Max_Drawdown"),
                        "Would_Buy_Count": int((decisions == "BUY").sum()),
                        "Blocked_By_DirAcc": int((risk_states == "quality_dir_acc").sum()),
                        "Blocked_By_RMSE": int((risk_states == "quality_rmse").sum()),
                        "Blocked_By_Composite": int((risk_states == "quality_composite").sum()),
                        "Below_Entry_Threshold": int((risk_states == "below_threshold").sum()),
                        "Mean_Quality_Threshold_Multiplier": float(np.nanmean(multipliers)) if multipliers.size else np.nan,
                        "Mean_Regime_Threshold_Multiplier": float(np.nanmean(regime_multipliers)) if regime_multipliers.size else np.nan,
                        "Mean_Volatility_Threshold_Multiplier": float(np.nanmean(vol_multipliers)) if vol_multipliers.size else np.nan,
                        "Mean_Final_Threshold_Multiplier": float(np.nanmean(final_multipliers)) if final_multipliers.size else np.nan,
                        "Status": "ok",
                    })

                    trades = result.get("trades", pd.DataFrame())
                    if isinstance(trades, pd.DataFrame) and not trades.empty:
                        frame = trades.copy()
                        frame.insert(0, "Shadow_Mode", shadow_mode)
                        frame.insert(1, "Signal_Mode", signal_mode)
                        frame.insert(2, "Quality_Gate_Mode", scenario_config.quality_gate_mode)
                        trade_frames.append(frame)
                except Exception as exc:
                    rows.append({
                        "Model": model_name,
                        "Shadow_Mode": shadow_mode,
                        "Signal_Mode": signal_mode,
                        "Quality_Gate_Mode": scenario_config.quality_gate_mode,
                        "Status": f"failed: {exc}",
                    })

        comparison_df = pd.DataFrame(rows)
        trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

        return {
            "comparison_df": comparison_df,
            "trades_df": trades_df,
        }

    def _run_shadow_backtests(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> Dict[str, Any]:
        shadow_results = self._get_shadow_backtests(
            backtest_inputs=backtest_inputs,
            model_metrics_by_model=model_metrics_by_model,
            suffix=suffix,
            target_mode=target_mode,
        )
        self._write_shadow_backtest_reports(shadow_results, suffix)
        return shadow_results

    # ------------------------------------------------------------------ #
    #  Signal gate diagnostics                                            #
    # ------------------------------------------------------------------ #

    def _get_signal_gate_diagnostics(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        backtest_results: Dict[str, Dict[str, Any]],
        backtest_metrics: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> pd.DataFrame:
        rows = []
        for model_name, payload in backtest_inputs.items():
            current_result = backtest_results.get(model_name, {})
            current_curve = current_result.get("equity_curve", pd.DataFrame())
            current_states = (
                current_curve["Risk_State"].astype(str)
                if isinstance(current_curve, pd.DataFrame) and "Risk_State" in current_curve.columns
                else pd.Series(dtype=str)
            )
            model_metrics = model_metrics_by_model.get(model_name, {})
            bt_metrics = backtest_metrics.get(model_name, {})
            n_bars = int(len(current_curve)) if isinstance(current_curve, pd.DataFrame) else 0
            dir_acc = self._diagnostic_float(model_metrics.get("Dir_Acc"))
            rmse_vs_benchmark = self._diagnostic_float(model_metrics.get("RMSE_vs_benchmark"))
            composite_score = self._diagnostic_float(model_metrics.get("Composite_Score"))

            probe_signal_mode = str(self.ctx.signal_mode).lower()
            if probe_signal_mode not in {"simple", "legacy", "professional"}:
                probe_signal_mode = "professional"
            probe_status = "skipped_benchmark_only"
            probe_curve = pd.DataFrame()
            if probe_signal_mode != "professional" or model_name not in self.state.signal_config.benchmark_only_models:
                try:
                    probe_result = run_backtest(
                        dates=payload.get("dates"),
                        prediction_dates=payload.get("prediction_dates"),
                        y_true_price=payload["y_true_price"],
                        pred_price=payload["pred_price"],
                        prev_close=payload["prev_close"],
                        fold_ids=payload.get("fold_ids"),
                        market_regime=payload.get("market_regime"),
                        pred_target=payload.get("pred_target"),
                        model_name=model_name,
                        validation_mode=f"{suffix}_gate_probe",
                        target_mode=target_mode,
                        commission_bps=self.ctx.commission_bps,
                        slippage_bps=self.ctx.slippage_bps,
                        signal_mode=probe_signal_mode,
                        signal_config=self.state.signal_config,
                        model_metrics={},
                    )
                    probe_curve = probe_result.get("equity_curve", pd.DataFrame())
                    probe_status = "ok"
                except Exception as exc:
                    probe_status = f"failed: {exc}"

            expected_return = self._diagnostic_numeric(probe_curve, "Expected_Return")
            entry_threshold = self._diagnostic_numeric(probe_curve, "Entry_Threshold")
            if expected_return.size == 0:
                expected_return = self._payload_expected_return(payload, target_mode)

            above_entry = np.array([], dtype=bool)
            if expected_return.size and entry_threshold.size:
                k = min(expected_return.size, entry_threshold.size)
                above_entry = expected_return[-k:] > entry_threshold[-k:]

            rows.append({
                "Model": model_name,
                "Validation_Suffix": suffix,
                "Gate_Mode": f"{self.ctx.signal_mode}_current",
                "Probe_Status": probe_status,
                "Dir_Acc": dir_acc,
                "RMSE_vs_benchmark": rmse_vs_benchmark,
                "Composite_Score": composite_score,
                "Would_Buy_Count": self._count_decision(probe_curve, "BUY"),
                "Blocked_By_DirAcc": n_bars if np.isfinite(dir_acc) and dir_acc < self.state.signal_config.min_directional_accuracy else 0,
                "Blocked_By_RMSE": n_bars if np.isfinite(rmse_vs_benchmark) and rmse_vs_benchmark > self.state.signal_config.max_rmse_vs_benchmark else 0,
                "Blocked_By_Composite": n_bars if np.isfinite(composite_score) and composite_score < self.state.signal_config.min_composite_score else 0,
                "Primary_Blocked_By_DirAcc": int((current_states == "quality_dir_acc").sum()),
                "Primary_Blocked_By_RMSE": int((current_states == "quality_rmse").sum()),
                "Primary_Blocked_By_Composite": int((current_states == "quality_composite").sum()),
                "Blocked_By_BenchmarkOnly": int((current_states == "benchmark_only").sum()),
                "Below_Entry_Threshold": int(
                    probe_curve.get("Risk_State", pd.Series(dtype=str)).astype(str).isin(
                        ["below_threshold", "simple_flat"]
                    ).sum()
                ) if isinstance(probe_curve, pd.DataFrame) else 0,
                "Trade_Count": self._diagnostic_float(bt_metrics.get("Trade_Count")),
                "Exposure": self._diagnostic_float(bt_metrics.get("Exposure")),
                "Net_Return": self._diagnostic_float(bt_metrics.get("Net_Return")),
                "BuyHold_Return": self._diagnostic_float(bt_metrics.get("BuyHold_Return")),
                "Mean_Abs_Predicted_Return": float(np.nanmean(np.abs(expected_return))) if expected_return.size else np.nan,
                "Median_Entry_Threshold": float(np.nanmedian(entry_threshold)) if entry_threshold.size else np.nan,
                "Pct_Pred_Above_Threshold": float(np.nanmean(above_entry) * 100.0) if above_entry.size else np.nan,
                "Min_Directional_Accuracy_Config": self.state.signal_config.min_directional_accuracy,
                "Max_RMSE_vs_Benchmark_Config": self.state.signal_config.max_rmse_vs_benchmark,
                "Min_Composite_Score_Config": self.state.signal_config.min_composite_score,
                "Entry_Cost_Multiplier": self.state.signal_config.entry_cost_multiplier,
                "Volatility_Multiplier": self.state.signal_config.volatility_multiplier,
            })

        return pd.DataFrame(rows)

    def _write_signal_gate_diagnostics(self, diagnostics: pd.DataFrame, suffix: str) -> None:
        outputs_dir = self.ctx.outputs_dir
        if not outputs_dir or not isinstance(diagnostics, pd.DataFrame):
            return
        try:
            os.makedirs(outputs_dir, exist_ok=True)
            path = os.path.join(outputs_dir, f"signal_gate_diagnostics_v1_{suffix}.csv")
            diagnostics.to_csv(path, index=False)
            print(f"  [OK] Signal gate diagnostik raporu kaydedildi -> {path}")
        except Exception as exc:
            print(f"  [WARN] Signal gate diagnostik raporu kaydedilemedi: {exc}")

    def _write_shadow_backtest_reports(self, shadow_results: Dict[str, Any], suffix: str) -> None:
        outputs_dir = self.ctx.outputs_dir
        if not outputs_dir or not shadow_results:
            return
        try:
            os.makedirs(outputs_dir, exist_ok=True)
            comparison_df = shadow_results.get("comparison_df", pd.DataFrame())
            trades_df = shadow_results.get("trades_df", pd.DataFrame())
            comparison_path = os.path.join(outputs_dir, f"shadow_backtest_comparison_v1_{suffix}.csv")
            trades_path = os.path.join(outputs_dir, f"shadow_backtest_trades_v1_{suffix}.csv")
            if isinstance(comparison_df, pd.DataFrame):
                comparison_df.to_csv(comparison_path, index=False)
                print(f"  [OK] Shadow backtest karsilastirma raporu kaydedildi -> {comparison_path}")
            if isinstance(trades_df, pd.DataFrame) and bool(self.ctx.write_trade_logs):
                trades_df.to_csv(trades_path, index=False)
                print(f"  [OK] Shadow backtest islem raporu kaydedildi -> {trades_path}")
        except Exception as exc:
            print(f"  [WARN] Shadow backtest raporu kaydedilemedi: {exc}")

    def _attach_signal_diagnosis(
        self,
        *,
        metrics_by_model: Dict[str, Dict[str, Any]],
        shadow_results: Dict[str, Any],
    ) -> None:
        comparison_df = shadow_results.get("comparison_df") if isinstance(shadow_results, dict) else None
        min_trades = int(self.ctx.signal_calibration_min_trades or 6)
        for model_name, metrics in metrics_by_model.items():
            labels = []
            net_return = self._diagnostic_float(metrics.get("Net_Return"))
            buy_hold_return = self._diagnostic_float(metrics.get("BuyHold_Return"))
            trade_count = int(self._diagnostic_float(metrics.get("Trade_Count")) or 0)
            sharpe = self._diagnostic_float(metrics.get("Sharpe"))

            if np.isfinite(net_return) and np.isfinite(buy_hold_return) and net_return <= buy_hold_return:
                labels.append("underperform_buyhold")
            if trade_count < min_trades:
                labels.append("insufficient_trades")
            if np.isfinite(sharpe) and sharpe <= 0.0:
                labels.append("model_signal_weak")
            if self.state.signal_threshold_source == "walk_forward_signal_rejected":
                labels.append("rejected_no_trade")

            if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                model_shadow = comparison_df[comparison_df["Model"].astype(str) == str(model_name)]
                current = model_shadow[model_shadow["Shadow_Mode"].astype(str) == "professional_current"]
                alternatives = model_shadow[
                    model_shadow["Shadow_Mode"].astype(str).isin(["professional_soft_gate", "legacy_directional"])
                ]
                current_trades = trade_count
                current_net = net_return
                if not current.empty:
                    current_trades = int(self._diagnostic_float(current.iloc[0].get("Trade_Count")) or current_trades)
                    current_net = self._diagnostic_float(current.iloc[0].get("Net_Return"))
                if not alternatives.empty:
                    alt_trades = pd.to_numeric(alternatives.get("Trade_Count"), errors="coerce").fillna(0.0)
                    alt_net = pd.to_numeric(alternatives.get("Net_Return"), errors="coerce")
                    if float(alt_trades.max()) > float(current_trades) and (
                        not np.isfinite(current_net) or float(alt_net.max()) > float(current_net)
                    ):
                        labels.append("gate_too_strict")

            metrics["Signal_Diagnosis"] = ",".join(dict.fromkeys(labels)) if labels else "ok"

    def _execute_backtest_batch(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
        model_metrics_by_model: Dict[str, Dict[str, Any]],
    ) -> tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, pd.DataFrame],
        Dict[str, pd.DataFrame],
    ]:
        results: Dict[str, Dict[str, Any]] = {}
        metrics_by_model: Dict[str, Dict[str, Any]] = {}
        trades_by_model: Dict[str, pd.DataFrame] = {}
        equity_curves: Dict[str, pd.DataFrame] = {}

        for model_name, payload in backtest_inputs.items():
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
                    validation_mode=suffix,
                    target_mode=target_mode,
                    commission_bps=self.ctx.commission_bps,
                    slippage_bps=self.ctx.slippage_bps,
                    signal_mode=self._effective_signal_mode(),
                    signal_config=self.state.signal_config,
                    model_metrics=model_metrics_by_model.get(model_name, {}),
                )
            except Exception as exc:
                print(f"  [WARN] {model_name} backtest basarisiz, atlaniyor: {exc}")
                continue

            results[model_name] = result
            metrics_by_model[model_name] = self._summarize_backtest_result(result, len(backtest_inputs))
            trades_by_model[model_name] = result["trades"]
            equity_curves[model_name] = result["equity_curve"]

        return results, metrics_by_model, trades_by_model, equity_curves

    def _effective_signal_mode(self) -> str:
        if (
            self.state.signal_threshold_source == "walk_forward_signal_rejected"
            and str(self.ctx.signal_calibration_reject_behavior).lower() == "no_trade"
        ):
            return "rejected_no_trade"
        return self.ctx.signal_mode

    def _summarize_backtest_result(
        self,
        result: Dict[str, Any],
        backtest_count: int,
    ) -> Dict[str, Any]:
        summary = summarize_backtest(
            result,
            initial_capital=self.ctx.initial_capital,
            trial_count=max(1, backtest_count),
        )
        summary.update({
            "Target_Semantics": self.ctx.dataset_metadata.get("target_semantics", ""),
            "Execution_Lag": self.ctx.dataset_metadata.get("execution_lag", ""),
            "Macro_Release_Lag": str(self.ctx.dataset_metadata.get("macro_release_lag", {})),
            "Transaction_Costs": f"commission_bps={self.ctx.commission_bps}; slippage_bps={self.ctx.slippage_bps}",
            "Validation_Protocol": str(self.ctx.dataset_metadata.get("validation_config", {})),
            "Threshold_Config": str(self.ctx.dataset_metadata.get("signal_threshold_config", {})),
        })
        return summary

    def _run_backtest_diagnostics(
        self,
        *,
        backtest_inputs: Dict[str, Dict[str, Any]],
        results: Dict[str, Dict[str, Any]],
        metrics_by_model: Dict[str, Dict[str, Any]],
        model_metrics_by_model: Dict[str, Dict[str, Any]],
        suffix: str,
        target_mode: str,
    ) -> tuple[pd.DataFrame | Dict[str, str], Dict[str, Any]]:
        auto_diagnostics = bool(self.ctx.auto_signal_diagnostics) and suffix in {"wf", "final_holdout"}

        if bool(self.ctx.enable_gate_diagnostics) or auto_diagnostics:
            gate_diagnostics = self._get_signal_gate_diagnostics(
                backtest_inputs=backtest_inputs,
                backtest_results=results,
                backtest_metrics=metrics_by_model,
                model_metrics_by_model=model_metrics_by_model,
                suffix=suffix,
                target_mode=target_mode,
            )
            self._write_signal_gate_diagnostics(gate_diagnostics, suffix)
        else:
            gate_diagnostics = {"status": "disabled"}

        if bool(self.ctx.enable_shadow_backtests) or auto_diagnostics:
            shadow_results = self._get_shadow_backtests(
                backtest_inputs=backtest_inputs,
                model_metrics_by_model=model_metrics_by_model,
                suffix=suffix,
                target_mode=target_mode,
            )
            self._write_shadow_backtest_reports(shadow_results, suffix)
        else:
            shadow_results = {"status": "disabled"}

        return gate_diagnostics, shadow_results

    def _write_backtest_tables(
        self,
        *,
        outputs_dir: str,
        suffix: str,
        metrics_by_model: Dict[str, Dict[str, Any]],
        equity_curves: Dict[str, pd.DataFrame],
        trades_by_model: Dict[str, pd.DataFrame],
    ) -> None:
        if metrics_by_model:
            save_backtest_report(
                metrics_by_model,
                save_path=os.path.join(outputs_dir, f"backtest_report_{suffix}.csv"),
            )
        if equity_curves:
            save_order_report(
                equity_curves,
                save_path=os.path.join(outputs_dir, f"backtest_orders_{suffix}.csv"),
            )
        if trades_by_model and bool(self.ctx.write_trade_logs):
            save_trade_logs(
                trades_by_model,
                save_path=os.path.join(outputs_dir, f"backtest_trades_{suffix}.csv"),
            )

    # ------------------------------------------------------------------ #
    #  Main backtest runner                                               #
    # ------------------------------------------------------------------ #

    def _run_backtests(
        self,
        backtest_inputs: Dict[str, Dict[str, Any]],
        suffix: str,
        model_metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.ctx.backtest_enabled or not backtest_inputs:
            return {}

        target_mode = self.ctx.dataset_metadata.get("target_mode", "log_return")
        results, metrics_by_model, trades_by_model, equity_curves = self._execute_backtest_batch(
            backtest_inputs=backtest_inputs,
            suffix=suffix,
            target_mode=target_mode,
            model_metrics_by_model=model_metrics_by_model or {},
        )

        if not metrics_by_model:
            return {}

        self.state.latest_backtest_results[suffix] = results
        self.state.latest_backtest_metrics[suffix] = metrics_by_model
        gate_diagnostics, shadow_results = self._run_backtest_diagnostics(
            backtest_inputs=backtest_inputs,
            results=results,
            metrics_by_model=metrics_by_model,
            model_metrics_by_model=model_metrics_by_model or {},
            suffix=suffix,
            target_mode=target_mode,
        )
        self._attach_signal_diagnosis(metrics_by_model=metrics_by_model, shadow_results=shadow_results)

        # ── Grafik ve rapor kayıtları ──────────────────────────────
        try:
            _out = self.ctx.outputs_dir
            if _out and equity_curves:
                try:
                    plot_equity_curves(
                        equity_curves,
                        save_path=os.path.join(_out, f'backtest_equity_{suffix}.png'),
                        title=f'{self.ctx.stock_symbol} Equity Curves ({suffix})',
                        selected_models=set(metrics_by_model),
                    )
                except Exception as _plot_exc:
                    print(f"  [WARN] Backtest equity grafigi kaydedilemedi: {_plot_exc}")
            if _out:
                self._write_backtest_tables(
                    outputs_dir=_out,
                    suffix=suffix,
                    metrics_by_model=metrics_by_model,
                    equity_curves=equity_curves,
                    trades_by_model=trades_by_model,
                )
        except Exception as _save_exc:
            print(f'  [WARN] Backtest grafik/rapor kaydı başarısız: {_save_exc}')

        return {
            "metrics": metrics_by_model,
            "trades": trades_by_model,
            "equity_curves": equity_curves,
            "gate_diagnostics": gate_diagnostics,
            "shadow_results": shadow_results,
            "suffix": suffix,
            "results": results,
        }
