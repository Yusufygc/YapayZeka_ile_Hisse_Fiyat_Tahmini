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
    save_trade_logs,
)


class _BacktestRunnerMixin:
    """Mixin: backtest yurutme ve diagnostik raporlama."""

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

        scenarios = [
            ("professional_current", "professional", replace(self.signal_config, quality_gate_mode="hard"), True),
            ("professional_soft_gate", "professional", replace(self.signal_config, quality_gate_mode="soft"), True),
            ("legacy_directional", "legacy", self.signal_config, False),
        ]

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
                        commission_bps=self.commission_bps,
                        slippage_bps=self.slippage_bps,
                        signal_mode=signal_mode,
                        signal_config=scenario_config,
                        model_metrics=model_metrics_by_model.get(model_name, {}) if use_model_metrics else {},
                    )
                    summary = summarize_backtest(
                        result,
                        initial_capital=self.initial_capital,
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

            probe_status = "skipped_benchmark_only"
            probe_curve = pd.DataFrame()
            if model_name not in self.signal_config.benchmark_only_models:
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
                        commission_bps=self.commission_bps,
                        slippage_bps=self.slippage_bps,
                        signal_mode="professional",
                        signal_config=self.signal_config,
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
                "Gate_Mode": f"{self.signal_mode}_current",
                "Probe_Status": probe_status,
                "Dir_Acc": dir_acc,
                "RMSE_vs_benchmark": rmse_vs_benchmark,
                "Composite_Score": composite_score,
                "Would_Buy_Count": self._count_decision(probe_curve, "BUY"),
                "Blocked_By_DirAcc": n_bars if np.isfinite(dir_acc) and dir_acc < self.signal_config.min_directional_accuracy else 0,
                "Blocked_By_RMSE": n_bars if np.isfinite(rmse_vs_benchmark) and rmse_vs_benchmark > self.signal_config.max_rmse_vs_benchmark else 0,
                "Blocked_By_Composite": n_bars if np.isfinite(composite_score) and composite_score < self.signal_config.min_composite_score else 0,
                "Primary_Blocked_By_DirAcc": int((current_states == "quality_dir_acc").sum()),
                "Primary_Blocked_By_RMSE": int((current_states == "quality_rmse").sum()),
                "Primary_Blocked_By_Composite": int((current_states == "quality_composite").sum()),
                "Blocked_By_BenchmarkOnly": int((current_states == "benchmark_only").sum()),
                "Below_Entry_Threshold": int((probe_curve.get("Risk_State", pd.Series(dtype=str)).astype(str) == "below_threshold").sum()) if isinstance(probe_curve, pd.DataFrame) else 0,
                "Trade_Count": self._diagnostic_float(bt_metrics.get("Trade_Count")),
                "Exposure": self._diagnostic_float(bt_metrics.get("Exposure")),
                "Net_Return": self._diagnostic_float(bt_metrics.get("Net_Return")),
                "BuyHold_Return": self._diagnostic_float(bt_metrics.get("BuyHold_Return")),
                "Mean_Abs_Predicted_Return": float(np.nanmean(np.abs(expected_return))) if expected_return.size else np.nan,
                "Median_Entry_Threshold": float(np.nanmedian(entry_threshold)) if entry_threshold.size else np.nan,
                "Pct_Pred_Above_Threshold": float(np.nanmean(above_entry) * 100.0) if above_entry.size else np.nan,
                "Min_Directional_Accuracy_Config": self.signal_config.min_directional_accuracy,
                "Max_RMSE_vs_Benchmark_Config": self.signal_config.max_rmse_vs_benchmark,
                "Min_Composite_Score_Config": self.signal_config.min_composite_score,
                "Entry_Cost_Multiplier": self.signal_config.entry_cost_multiplier,
                "Volatility_Multiplier": self.signal_config.volatility_multiplier,
            })

        return pd.DataFrame(rows)

    def _write_signal_gate_diagnostics(self, diagnostics: pd.DataFrame, suffix: str) -> None:
        outputs_dir = getattr(self, "outputs_dir", "")
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
        outputs_dir = getattr(self, "outputs_dir", "")
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
            if isinstance(trades_df, pd.DataFrame) and bool(getattr(self, "write_trade_logs", False)):
                trades_df.to_csv(trades_path, index=False)
                print(f"  [OK] Shadow backtest islem raporu kaydedildi -> {trades_path}")
        except Exception as exc:
            print(f"  [WARN] Shadow backtest raporu kaydedilemedi: {exc}")

    # ------------------------------------------------------------------ #
    #  Main backtest runner                                               #
    # ------------------------------------------------------------------ #

    def _run_backtests(
        self,
        backtest_inputs: Dict[str, Dict[str, Any]],
        suffix: str,
        model_metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.backtest_enabled or not backtest_inputs:
            return {}

        results = {}
        metrics_by_model = {}
        trades_by_model = {}
        equity_curves = {}
        target_mode = self.dataset_metadata.get("target_mode", "log_return")

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
                    commission_bps=self.commission_bps,
                    slippage_bps=self.slippage_bps,
                    signal_mode=self.signal_mode,
                    signal_config=self.signal_config,
                    model_metrics=(model_metrics_by_model or {}).get(model_name, {}),
                )
                results[model_name] = result
                metrics_by_model[model_name] = summarize_backtest(
                    result,
                    initial_capital=self.initial_capital,
                    trial_count=max(1, len(backtest_inputs)),
                )
                metrics_by_model[model_name].update({
                    "Target_Semantics": self.dataset_metadata.get("target_semantics", ""),
                    "Execution_Lag": self.dataset_metadata.get("execution_lag", ""),
                    "Macro_Release_Lag": str(self.dataset_metadata.get("macro_release_lag", {})),
                    "Transaction_Costs": f"commission_bps={self.commission_bps}; slippage_bps={self.slippage_bps}",
                    "Validation_Protocol": str(self.dataset_metadata.get("validation_config", {})),
                    "Threshold_Config": str(self.dataset_metadata.get("signal_threshold_config", {})),
                })
                trades_by_model[model_name] = result["trades"]
                equity_curves[model_name] = result["equity_curve"]
            except Exception as exc:
                print(f"  [WARN] {model_name} backtest basarisiz, atlaniyor: {exc}")

        if not metrics_by_model:
            return {}

        self.latest_backtest_results[suffix] = results
        self.latest_backtest_metrics[suffix] = metrics_by_model

        if bool(getattr(self, "enable_gate_diagnostics", False)):
            gate_diagnostics = self._get_signal_gate_diagnostics(
                backtest_inputs=backtest_inputs,
                backtest_results=results,
                backtest_metrics=metrics_by_model,
                model_metrics_by_model=model_metrics_by_model or {},
                suffix=suffix,
                target_mode=target_mode,
            )
            self._write_signal_gate_diagnostics(gate_diagnostics, suffix)
        else:
            gate_diagnostics = {"status": "disabled"}

        if bool(getattr(self, "enable_shadow_backtests", False)):
            shadow_results = self._get_shadow_backtests(
                backtest_inputs=backtest_inputs,
                model_metrics_by_model=model_metrics_by_model or {},
                suffix=suffix,
                target_mode=target_mode,
            )
            self._write_shadow_backtest_reports(shadow_results, suffix)
        else:
            shadow_results = {"status": "disabled"}

        # ── Grafik ve rapor kayıtları ──────────────────────────────
        try:
            import os as _os
            _out = getattr(self, 'outputs_dir', '')
            if _out and equity_curves:
                plot_equity_curves(
                    equity_curves,
                    save_path=_os.path.join(_out, f'backtest_equity_{suffix}.png'),
                    title=f'{getattr(self, "stock_symbol", "")} Equity Curves ({suffix})',
                    selected_models=set(metrics_by_model),
                )
            if _out and metrics_by_model:
                save_backtest_report(
                    metrics_by_model,
                    save_path=_os.path.join(_out, f'backtest_report_{suffix}.csv'),
                )
            if _out and trades_by_model and bool(getattr(self, "write_trade_logs", False)):
                save_trade_logs(
                    trades_by_model,
                    save_path=_os.path.join(_out, f'backtest_trades_{suffix}.csv'),
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
