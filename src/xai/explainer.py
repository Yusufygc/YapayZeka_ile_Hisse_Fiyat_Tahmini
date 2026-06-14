# -*- coding: utf-8 -*-
"""
explainer.py - Model specific XAI generation for forecasting outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.xai.background import XAIBackgroundProvider
from src.xai.feature_dictionary import describe_feature, feature_group
from src.xai.narrative import contribution_sentence, model_summary_sentence
from src.xai.strategies import (
    SequenceContributionStrategy,
    TabularContributionStrategy,
    compute_feature_stability_scores,
)


TREE_MODELS = {"XGBoost", "Random Forest", "LightGBM Return"}
LINEAR_MODELS = {"Ridge Return", "ElasticNet Return"}
SEQ_MODELS = {"LSTM", "LSTM Lite", "AttentionLSTM", "AttentionLSTM v2", "DLinear", "NLinear"}
BASELINE_MODELS = {"Naive Last Value", "Naive Zero Return", "Naive Drift", "ARIMA", "Prophet"}


class XAIExplainer:
    def __init__(
        self,
        stock_symbol: str,
        feature_names: List[str],
        dataset_metadata: Dict[str, Any],
        max_rows: int = 80,
        top_k: int = 5,
    ):
        self.stock_symbol = stock_symbol
        self.feature_names = feature_names
        self.dataset_metadata = dataset_metadata
        self.max_rows = max_rows
        self.top_k = top_k
        self.tabular_strategy = TabularContributionStrategy(feature_names, max_rows=max_rows)
        self.sequence_strategy = SequenceContributionStrategy(feature_names, max_rows=max_rows)

    def _set_background(self, background: XAIBackgroundProvider) -> None:
        self.tabular_strategy = TabularContributionStrategy(
            self.feature_names,
            max_rows=self.max_rows,
            background_provider=background,
        )
        self.sequence_strategy = SequenceContributionStrategy(
            self.feature_names,
            max_rows=self.max_rows,
            background_provider=background,
        )

    def explain_single_split(
        self,
        trained_models: Dict[str, Any],
        tensors: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        prediction_targets: Dict[str, np.ndarray],
        y_true_aligned: np.ndarray,
        quantile_predictions: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, pd.DataFrame | str]:
        top_rows: List[Dict[str, Any]] = []
        daily_rows: List[Dict[str, Any]] = []
        heatmap_rows: List[Dict[str, Any]] = []
        summary_blocks: List[str] = [self._summary_header("latest")]

        quantile_predictions = quantile_predictions or {}
        background = XAIBackgroundProvider.from_arrays(
            X_train=tensors.get("X_train_s"),
            X_train_seq=tensors.get("X_train_seq"),
        )
        self._set_background(background)

        for model_name, model in trained_models.items():
            if model_name not in predictions:
                continue
            try:
                if model_name in TREE_MODELS:
                    model_top, model_daily = self._explain_tree_model(
                        model_name, model, tensors, predictions, prediction_targets, y_true_aligned
                    )
                elif model_name in LINEAR_MODELS:
                    model_top, model_daily = self._explain_linear_model(
                        model_name, model, tensors, predictions, prediction_targets, y_true_aligned
                    )
                elif model_name in {"LSTM", "LSTM Lite", "AttentionLSTM", "AttentionLSTM v2"}:
                    model_top, model_daily, model_heatmap = self._explain_sequence_permutation(
                        model_name, model, tensors, predictions, prediction_targets, y_true_aligned
                    )
                    heatmap_rows.extend(model_heatmap)
                else:
                    model_top, model_daily = self._explain_rule_based(
                        model_name, model, tensors, predictions, prediction_targets, y_true_aligned
                    )

                top_rows.extend(model_top)
                daily_rows.extend(model_daily)
                summary_blocks.append(self._model_markdown(model_name, model_top, model_daily))
            except Exception as exc:
                summary_blocks.append(f"## {model_name}\n\nXAI açıklaması üretilemedi: {exc}\n")

        payload: Dict[str, Any] = {
            "top_reasons":  pd.DataFrame(top_rows),
            "daily_reasons": pd.DataFrame(daily_rows),
            "sequence_heatmap": pd.DataFrame(heatmap_rows),
            "summary_md":   "\n\n".join(summary_blocks),
            "background_scope": background.scope,
            "run_id": self.dataset_metadata.get("run_id"),
            "validation_mode": self.dataset_metadata.get("validation_mode", "single_split"),
        }
        return payload

    def explain_walk_forward(
        self,
        wf_predictions: Dict[str, np.ndarray],
        wf_y_true: np.ndarray,
        wf_backtest_inputs: Dict[str, Dict[str, Any]] | None = None,
        backtest_results: Dict[str, Dict[str, Any]] | None = None,
        wf_xai_records: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, pd.DataFrame | str]:
        rows: List[Dict[str, Any]] = []
        daily_rows: List[Dict[str, Any]] = []
        signal_rows: List[Dict[str, Any]] = []
        trade_rows: List[Dict[str, Any]] = []
        heatmap_rows: List[Dict[str, Any]] = []
        summary = [self._summary_header("wf")]
        wf_backtest_inputs = wf_backtest_inputs or {}
        backtest_results = backtest_results or {}
        wf_xai_records = wf_xai_records or {}
        fold_importances_by_model: Dict[str, List[Dict[str, float]]] = {}

        for model_name, preds in wf_predictions.items():
            payload = wf_backtest_inputs.get(model_name, {})
            pred_target = np.asarray(payload.get("pred_target", []), dtype=float)
            latest_target = float(pred_target[-1]) if len(pred_target) else None
            latest_pred = float(np.asarray(preds).ravel()[-1]) if len(preds) else None
            latest_true = float(np.asarray(wf_y_true).ravel()[-1]) if wf_y_true is not None and len(wf_y_true) else None
            sentence = model_summary_sentence(model_name, latest_target, latest_pred, latest_true)
            records = wf_xai_records.get(model_name, [])
            if records:
                model_top, model_daily, model_heatmap, fold_importances = self._rows_from_wf_records(
                    model_name=model_name,
                    records=records,
                    fallback_dates=payload.get("dates"),
                    fallback_pred_target=pred_target,
                    fallback_pred_price=np.asarray(preds).ravel(),
                    fallback_true=wf_y_true,
                )
                if model_top or model_daily:
                    rows.extend(model_top)
                    daily_rows.extend(model_daily)
                    heatmap_rows.extend(model_heatmap)
                    fold_importances_by_model[model_name] = fold_importances
                    summary.append(
                        f"## {model_name}\n\n"
                        f"{sentence}\n\n"
                        "Walk-forward XAI fold bazli out-of-sample katkilar ve sinyal gerekceleriyle raporlanir."
                    )
                    continue
            rows.append(self._row(model_name, "WalkForward_Summary", 1.0, latest_target, sentence, "rule_based", False))
            daily_rows.append({
                "Model": model_name,
                "Date": self._latest_date(payload.get("dates")),
                "Predicted_Direction": self._direction(latest_target),
                "Reason_Rank": 1,
                "Feature": "WalkForward_Summary",
                "Readable_Feature": "walk-forward pencerelerindeki genel model davranışı",
                "Reason": sentence,
                "Method": "rule_based",
            })
            summary.append(
                f"## {model_name}\n\n"
                f"{sentence}\n\n"
                "Walk-forward karar ayrintilari ozet XAI PNG/MD ciktisinda tutulur; tablo dokumu yalnizca research profilinde yazilir."
            )

        for model_name, result in backtest_results.items():
            signal_rows.extend(self._signal_reason_rows(model_name, result))
            trade_rows.extend(self._trade_explanation_rows(model_name, result))
            rows.extend(self._signal_summary_rows(model_name, result))

        return {
            "top_reasons": pd.DataFrame(rows),
            "daily_reasons": pd.DataFrame(daily_rows),
            "signal_reasons": pd.DataFrame(signal_rows),
            "trade_explanations": pd.DataFrame(trade_rows),
            "sequence_heatmap": pd.DataFrame(heatmap_rows),
            "summary_md": "\n\n".join(summary),
            "feature_stability_top": self._merge_stability(fold_importances_by_model),
            "background_scope": self._background_scope_from_records(wf_xai_records),
            "run_id": self.dataset_metadata.get("run_id"),
            "validation_mode": "walk_forward",
        }

    def _signal_reason_rows(self, model_name: str, backtest_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        curve = backtest_result.get("equity_curve")
        if curve is None or getattr(curve, "empty", True):
            return []

        rows: List[Dict[str, Any]] = []
        for _, row in curve.iterrows():
            decision = str(row.get("Decision", ""))
            expected_return = self._safe_float(row.get("Expected_Return"))
            entry_threshold = self._safe_float(row.get("Entry_Threshold"))
            exit_threshold = self._safe_float(row.get("Exit_Threshold"))
            risk_state = str(row.get("Risk_State", ""))
            signal_reason = str(row.get("Signal_Reason", ""))
            rows.append({
                "Model": model_name,
                "Date": row.get("Date"),
                "Decision": decision,
                "Position": self._safe_float(row.get("Position")),
                "Expected_Return": expected_return,
                "Entry_Threshold": entry_threshold,
                "Exit_Threshold": exit_threshold,
                "Signal_Strength": self._safe_float(row.get("Signal_Strength")),
                "Rolling_Volatility": self._safe_float(row.get("Rolling_Volatility")),
                "Risk_State": risk_state,
                "Reason": self._decision_explanation(
                    decision,
                    expected_return,
                    entry_threshold,
                    exit_threshold,
                    risk_state,
                    signal_reason,
                ),
            })
        return rows

    def _trade_explanation_rows(self, model_name: str, backtest_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        trades = backtest_result.get("trades")
        if trades is None or getattr(trades, "empty", True):
            return []

        rows: List[Dict[str, Any]] = []
        for _, trade in trades.iterrows():
            net_return = self._safe_float(trade.get("Net_Return"))
            outcome = "kazanc" if net_return > 0 else "zarar" if net_return < 0 else "basabas"
            entry_reason = str(trade.get("Entry_Reason", ""))
            exit_reason = str(trade.get("Exit_Reason", ""))
            rows.append({
                "Model": model_name,
                "Entry_Date": trade.get("Entry_Date"),
                "Exit_Date": trade.get("Exit_Date"),
                "Entry_Price": self._safe_float(trade.get("Entry_Price")),
                "Exit_Price": self._safe_float(trade.get("Exit_Price")),
                "Net_Return": net_return,
                "Holding_Period": trade.get("Holding_Period"),
                "Outcome": outcome,
                "Entry_Reason": entry_reason,
                "Exit_Reason": exit_reason,
                "Explanation": f"{model_name} islemi {outcome} ile kapandi. Giris nedeni: {entry_reason} Cikis nedeni: {exit_reason}",
            })
        return rows

    def _signal_summary_rows(self, model_name: str, backtest_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        curve = backtest_result.get("equity_curve")
        if curve is None or getattr(curve, "empty", True) or "Risk_State" not in curve.columns:
            return []

        total = max(len(curve), 1)
        grouped = (
            curve.groupby(["Decision", "Risk_State"], dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(self.top_k)
        )

        rows: List[Dict[str, Any]] = []
        for _, item in grouped.iterrows():
            decision = str(item.get("Decision", ""))
            risk_state = str(item.get("Risk_State", ""))
            count = int(item.get("Count", 0))
            importance = count / total
            sample = curve[curve["Risk_State"].astype(str) == risk_state].iloc[0]
            reason = self._decision_explanation(
                decision,
                self._safe_float(sample.get("Expected_Return")),
                self._safe_float(sample.get("Entry_Threshold")),
                self._safe_float(sample.get("Exit_Threshold")),
                risk_state,
                str(sample.get("Signal_Reason", "")),
            )
            rows.append({
                "Model": model_name,
                "Feature": f"Signal_{risk_state}",
                "Readable_Feature": self._risk_state_readable(risk_state),
                "Feature_Group": "Sinyal karari",
                "Importance": importance,
                "Contribution": importance if decision in {"BUY", "HOLD"} else -importance,
                "Direction": decision,
                "Reason": f"{count} gun: {reason}",
                "Method": "signal_rules",
                "Approximate": False,
            })
        return rows

    def _decision_explanation(
        self,
        decision: str,
        expected_return: float,
        entry_threshold: float,
        exit_threshold: float,
        risk_state: str,
        fallback_reason: str,
    ) -> str:
        if decision == "BUY":
            return "Modelin beklenen getirisi maliyet ve volatilite esigini astigi icin AL karari uretildi."
        if decision == "HOLD":
            if risk_state == "min_hold":
                return "Pozisyon minimum elde tutma suresi dolmadigi icin korunuyor."
            return "Pozisyon korunuyor; beklenen getiri cikis esiginin uzerinde kaldi."
        if decision == "EXIT":
            if risk_state == "take_profit":
                return "Kar-al bariyeri tetiklendigi icin pozisyondan cikildi."
            if risk_state == "stop_loss":
                return "Zarar-kes bariyeri tetiklendigi icin pozisyondan cikildi."
            if risk_state == "max_hold":
                return "Maksimum elde tutma suresi doldugu icin pozisyondan cikildi."
            return "Beklenen getiri cikis esiginin altina indigi icin pozisyondan cikildi."
        if decision == "NO_TRADE":
            if risk_state == "benchmark_only":
                return fallback_reason or "Bu model sadece benchmark olarak kullanildigi icin islem acilmadi."
            if risk_state == "quality_dir_acc":
                return fallback_reason or "Modelin yon dogrulugu profesyonel islem esigini gecemedigi icin islem acilmadi."
            if risk_state == "quality_rmse":
                return fallback_reason or "Modelin hata seviyesi benchmark'a gore yuksek kaldigi icin islem acilmadi."
            if risk_state == "quality_composite":
                return fallback_reason or "Modelin genel kalite skoru islem icin yetersiz kaldigi icin islem acilmadi."
            if risk_state == "cooldown":
                return "Son cikistan sonra bekleme suresi devam ettigi icin yeni islem acilmadi."
            if np.isfinite(expected_return) and np.isfinite(entry_threshold):
                return f"Model sinyali islem acmak icin yetersiz: beklenen getiri {expected_return:.6f}, giris esigi {entry_threshold:.6f}."
            return "Model sinyali islem acmak icin yeterli bulunmadi."
        return fallback_reason

    def _risk_state_readable(self, risk_state: str) -> str:
        mapping = {
            "benchmark_only": "model benchmark oldugu icin islem disi",
            "quality_dir_acc": "yon dogrulugu islem esiginin altinda",
            "quality_rmse": "tahmin hatasi benchmark'a gore yuksek",
            "quality_composite": "genel model kalite skoru dusuk",
            "below_threshold": "beklenen getiri alis esigini gecemedi",
            "cooldown": "cikis sonrasi bekleme suresi",
            "min_hold": "minimum elde tutma suresi",
            "take_profit": "kar-al bariyeri",
            "stop_loss": "zarar-kes bariyeri",
            "max_hold": "maksimum elde tutma suresi",
            "weak_signal": "zayiflayan tahmin sinyali",
            "entry_ok": "alis esigi gecildi",
            "in_position": "pozisyon korunuyor",
            "simple_entry": "basit mod al sinyali",
            "simple_exit": "basit mod sat sinyali",
            "simple_hold": "basit mod pozisyonu koruma",
            "simple_flat": "basit mod nakitte bekleme",
        }
        return mapping.get(risk_state, risk_state or "sinyal karari")

    def _explain_tree_model(
        self,
        model_name: str,
        wrapper: Any,
        tensors: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        prediction_targets: Dict[str, np.ndarray],
        y_true_aligned: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        X = self._align_rows(np.asarray(tensors["X_test_s"], dtype=float), len(predictions[model_name]))
        contribs, method, approximate = self.tabular_strategy.tree_contributions(wrapper.model, X)
        dates = self._align_rows(np.asarray(tensors.get("dates_test", [])), len(predictions[model_name]))
        return self._rows_from_contributions(
            model_name, contribs, predictions[model_name], prediction_targets.get(model_name), y_true_aligned, method, approximate=approximate, dates=dates
        )

    def _tree_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str]:
        contribs, method, _approximate = self.tabular_strategy.tree_contributions(estimator, X)
        return contribs, method

    def _explain_linear_model(
        self,
        model_name: str,
        wrapper: Any,
        tensors: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        prediction_targets: Dict[str, np.ndarray],
        y_true_aligned: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        X = self._align_rows(np.asarray(tensors["X_test_s"], dtype=float), len(predictions[model_name]))
        estimator = getattr(wrapper, "model", wrapper)
        contribs, method, approximate = self.tabular_strategy.linear_contributions(estimator, X)
        dates = self._align_rows(np.asarray(tensors.get("dates_test", [])), len(predictions[model_name]))
        return self._rows_from_contributions(
            model_name, contribs, predictions[model_name], prediction_targets.get(model_name), y_true_aligned, method, approximate=approximate, dates=dates
        )

    def _linear_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str]:
        contribs, method, _approximate = self.tabular_strategy.linear_contributions(estimator, X)
        return contribs, method

    def _lime_tabular_contributions(self, estimator: Any, X: np.ndarray) -> Tuple[np.ndarray, str]:
        contribs, method, _approximate = self.tabular_strategy.lime_tabular_contributions(estimator, X)
        return contribs, method

    def _lime_sequence_local_contributions(self, model: Any, X_seq: np.ndarray) -> Tuple[np.ndarray, str]:
        contribs, method, _approximate = self.sequence_strategy.lime_sequence_local_contributions(model, X_seq)
        return contribs, method

    def _explain_sequence_permutation(
        self,
        model_name: str,
        model: Any,
        tensors: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        prediction_targets: Dict[str, np.ndarray],
        y_true_aligned: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        X_seq = self._align_rows(np.asarray(tensors["X_test_seq"], dtype=float), len(predictions[model_name]))
        contribs, heatmap, method, approximate = self.sequence_strategy.feature_lag_contributions(model, X_seq)
        top_rows, daily_rows = self._rows_from_contributions(
            model_name,
            contribs,
            np.asarray(predictions[model_name])[-1:],
            np.asarray(prediction_targets.get(model_name, []))[-1:],
            np.asarray(y_true_aligned)[-1:],
            method,
            approximate=approximate,
            dates=self._align_rows(np.asarray(tensors.get("dates_test", [])), len(predictions[model_name]))[-1:],
        )
        for row in heatmap:
            row["Model"] = model_name
            row["Readable_Feature"] = describe_feature(str(row.get("Feature", "")))
            row["Feature_Group"] = feature_group(str(row.get("Feature", "")))
        return top_rows, daily_rows, heatmap

    def _explain_rule_based(
        self,
        model_name: str,
        model: Any,
        tensors: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        prediction_targets: Dict[str, np.ndarray],
        y_true_aligned: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pred_target = self._latest_array_value(prediction_targets.get(model_name))
        pred_price = self._latest_array_value(predictions.get(model_name))
        actual = self._latest_array_value(y_true_aligned)
        if model_name == "Naive Zero Return":
            feature = "Zero_Return_Assumption"
            reason = "Bu baseline fiyat değişmeyecek varsayımını kullanır; model davranışı bir referans çizgisidir."
        elif model_name == "Naive Last Value":
            feature = "Last_Return_Assumption"
            reason = "Bu baseline son gözlenen getirinin devam edeceğini varsayar."
        elif model_name == "Naive Drift":
            feature = "Historical_Drift"
            reason = "Bu baseline eğitim dönemindeki ortalama getirinin devam edeceğini varsayar."
        elif model_name == "ARIMA":
            feature = "Historical_Autocorrelation"
            reason = "ARIMA son dönem tarihsel hareket yapısını kullanarak tahmin üretir."
        else:
            feature = "Prophet_Trend_Seasonality"
            reason = "Prophet tarihsel trend ve takvimsel tekrar eden hareketleri kullanarak tahmin üretir."

        summary = model_summary_sentence(model_name, pred_target, pred_price, actual)
        full_reason = f"{summary} {reason}"
        rows = [self._row(model_name, feature, 1.0, pred_target, full_reason, "rule_based", False)]
        daily = [{
            "Model": model_name,
            "Date": self._latest_date(tensors.get("dates_test")),
            "Predicted_Direction": self._direction(pred_target),
            "Reason_Rank": 1,
            "Feature": feature,
            "Readable_Feature": describe_feature(feature),
            "Reason": full_reason,
            "Method": "rule_based",
        }]
        return rows, daily

    def _rows_from_contributions(
        self,
        model_name: str,
        contribs: np.ndarray,
        pred_prices: Iterable[float],
        pred_targets: Iterable[float] | None,
        y_true: Iterable[float],
        method: str,
        approximate: bool,
        dates: Iterable[Any] | None = None,
        background_scope: str | None = None,
        fold_id: Any | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        contribs = np.asarray(contribs, dtype=float)
        if contribs.ndim == 1:
            contribs = contribs.reshape(1, -1)

        global_importance = np.mean(np.abs(contribs), axis=0)
        global_signed = np.mean(contribs, axis=0)
        ranked = np.argsort(global_importance)[::-1][: self.top_k]
        pred_targets_arr = self._to_float_array(pred_targets)
        pred_prices_arr = self._to_float_array(pred_prices)
        y_true_arr = self._to_float_array(y_true)
        dates_arr = np.asarray(list(dates)) if dates is not None else np.asarray([])
        rows = []
        daily = []
        for rank, feature_idx in enumerate(ranked, start=1):
            feature = self.feature_names[int(feature_idx)]
            contribution = float(global_signed[int(feature_idx)])
            reason = contribution_sentence(feature, contribution, approximate=approximate)
            row = self._row(model_name, feature, float(global_importance[int(feature_idx)]), contribution, reason, method, approximate)
            if background_scope:
                row["Background_Scope"] = background_scope
            if fold_id is not None:
                row["Fold"] = fold_id
            rows.append(row)

        start = max(0, len(contribs) - self.max_rows)
        for local_idx, contrib_row in enumerate(contribs[start:], start=start):
            daily_ranked = np.argsort(np.abs(contrib_row))[::-1][: self.top_k]
            pred_target = self._value_at(pred_targets_arr, local_idx)
            pred_price = self._value_at(pred_prices_arr, local_idx)
            actual = self._value_at(y_true_arr, local_idx)
            date_value = str(dates_arr[local_idx]) if len(dates_arr) > local_idx else None
            for rank, feature_idx in enumerate(daily_ranked, start=1):
                feature = self.feature_names[int(feature_idx)]
                contribution = float(contrib_row[int(feature_idx)])
                daily_row = {
                    "Model": model_name,
                    "Date": date_value,
                    "Predicted_Direction": self._direction(pred_target),
                    "Predicted_Price": pred_price,
                    "Actual_Price": actual,
                    "Reason_Rank": rank,
                    "Feature": feature,
                    "Readable_Feature": describe_feature(feature),
                    "Feature_Group": feature_group(feature),
                    "Contribution": contribution,
                    "Reason": contribution_sentence(feature, contribution, approximate=approximate),
                    "Method": method,
                    "Approximate": approximate,
                }
                if background_scope:
                    daily_row["Background_Scope"] = background_scope
                if fold_id is not None:
                    daily_row["Fold"] = fold_id
                daily.append(daily_row)
        return rows, daily

    def _rows_from_wf_records(
        self,
        *,
        model_name: str,
        records: List[Dict[str, Any]],
        fallback_dates: Any,
        fallback_pred_target: np.ndarray,
        fallback_pred_price: np.ndarray,
        fallback_true: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, float]]]:
        all_contribs: List[np.ndarray] = []
        all_dates: List[Any] = []
        all_pred_target: List[float] = []
        all_pred_price: List[float] = []
        all_true: List[float] = []
        heatmap_rows: List[Dict[str, Any]] = []
        fold_importances: List[Dict[str, float]] = []
        method = "walk_forward_attribution"
        approximate = False
        background_scope = "unavailable"
        for record in records:
            contribs = np.asarray(record.get("contributions", []), dtype=float)
            if contribs.ndim != 2 or contribs.size == 0:
                continue
            all_contribs.append(contribs)
            all_dates.extend(record.get("dates", []))
            all_pred_target.extend(record.get("pred_target", []))
            all_pred_price.extend(record.get("pred_price", []))
            all_true.extend(record.get("y_true_price", []))
            method = str(record.get("method") or method)
            approximate = approximate or bool(record.get("approximate", False))
            background_scope = str(record.get("background_scope") or background_scope)
            fold_imp = record.get("fold_importance")
            if isinstance(fold_imp, dict):
                fold_importances.append({str(k): float(v) for k, v in fold_imp.items()})
            for row in record.get("sequence_heatmap", []) or []:
                if isinstance(row, dict):
                    heatmap_row = dict(row)
                    heatmap_row["Model"] = model_name
                    heatmap_row.setdefault("Fold", record.get("fold"))
                    heatmap_row.setdefault("Readable_Feature", describe_feature(str(row.get("Feature", ""))))
                    heatmap_row.setdefault("Feature_Group", feature_group(str(row.get("Feature", ""))))
                    heatmap_rows.append(heatmap_row)
        if not all_contribs:
            return [], [], heatmap_rows, fold_importances
        contribs_all = np.vstack(all_contribs)
        dates = all_dates or list(np.asarray(fallback_dates).ravel()[-len(contribs_all):])
        pred_target = all_pred_target or list(np.asarray(fallback_pred_target).ravel()[-len(contribs_all):])
        pred_price = all_pred_price or list(np.asarray(fallback_pred_price).ravel()[-len(contribs_all):])
        y_true = all_true or list(np.asarray(fallback_true).ravel()[-len(contribs_all):])
        top_rows, daily_rows = self._rows_from_contributions(
            model_name,
            contribs_all,
            pred_price,
            pred_target,
            y_true,
            method,
            approximate,
            dates=dates,
            background_scope=background_scope,
        )
        return top_rows, daily_rows, heatmap_rows, fold_importances

    def _merge_stability(self, fold_importances_by_model: Dict[str, List[Dict[str, float]]]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for model_name, fold_importances in fold_importances_by_model.items():
            scores = compute_feature_stability_scores(fold_importances, top_k=self.top_k)
            for feature, score in scores.items():
                merged[f"{model_name}:{feature}"] = score
        return dict(sorted(merged.items(), key=lambda item: item[1], reverse=True)[: self.top_k * 3])

    @staticmethod
    def _background_scope_from_records(records_by_model: Dict[str, List[Dict[str, Any]]]) -> str:
        scopes = {
            str(record.get("background_scope"))
            for records in records_by_model.values()
            for record in records
            if record.get("background_scope")
        }
        return ",".join(sorted(scopes)) if scopes else "unavailable"

    def _row(
        self,
        model_name: str,
        feature: str,
        importance: float,
        contribution: float | None,
        reason: str,
        method: str,
        approximate: bool,
    ) -> Dict[str, Any]:
        return {
            "Model": model_name,
            "Feature": feature,
            "Readable_Feature": describe_feature(feature),
            "Feature_Group": feature_group(feature),
            "Importance": float(importance) if importance is not None else np.nan,
            "Contribution": contribution,
            "Direction": self._direction(contribution),
            "Reason": reason,
            "Method": method,
            "Approximate": approximate,
        }

    def _model_markdown(self, model_name: str, top_rows: List[Dict[str, Any]], daily_rows: List[Dict[str, Any]]) -> str:
        latest_reason = ""
        if daily_rows:
            latest_reason = "\n".join(f"- {row['Reason']}" for row in daily_rows[-self.top_k :])
        elif top_rows:
            latest_reason = "\n".join(f"- {row['Reason']}" for row in top_rows[: self.top_k])
        else:
            latest_reason = "- Bu model için açıklanabilir sinyal bulunamadı."
        return (
            f"## {model_name}\n\n"
            "Model tahmininde en etkili sinyaller şunlar oldu:\n\n"
            f"{latest_reason}\n\n"
            "Not: Bu rapor modelin hangi sinyallere dayandığını açıklar; yatırım tavsiyesi değildir."
        )

    def _summary_header(self, suffix: str) -> str:
        target_mode = self.dataset_metadata.get("target_mode", "N/A")
        validation_mode = self.dataset_metadata.get("validation_mode", suffix)
        return (
            f"# {self.stock_symbol} XAI Açıklanabilirlik Raporu\n\n"
            f"- Validation modu: {validation_mode}\n"
            f"- Hedef tipi: {target_mode}\n"
            "- Dil: Model tahmininde etkili olan sinyaller sade Türkçeye çevrilmiştir.\n"
            "- Uyarı: Bu rapor model davranışını açıklar; yatırım tavsiyesi değildir."
        )

    def _align_rows(self, values: np.ndarray, target_len: int) -> np.ndarray:
        values = np.asarray(values)
        if target_len <= 0:
            return values
        return values[-target_len:]

    def _latest_array_value(self, values: Iterable[float] | None) -> float | None:
        arr = self._to_float_array(values)
        if len(arr) == 0:
            return None
        value = float(arr[-1])
        return value if np.isfinite(value) else None

    def _safe_float(self, value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    def _to_float_array(self, values: Iterable[float] | None) -> np.ndarray:
        if values is None:
            return np.asarray([], dtype=float)
        return np.asarray(values, dtype=float).ravel()

    def _value_at(self, values: np.ndarray, idx: int) -> float | None:
        if len(values) == 0:
            return None
        safe_idx = min(idx, len(values) - 1)
        value = float(values[safe_idx])
        return value if np.isfinite(value) else None

    def _latest_date(self, dates: Any) -> str | None:
        if dates is None:
            return None
        arr = np.asarray(dates)
        if len(arr) == 0:
            return None
        try:
            return str(pd.to_datetime(arr[-1]).date())
        except Exception:
            return str(arr[-1])

    def _direction(self, value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "nötr"
        if value > 0:
            return "yukarı"
        if value < 0:
            return "aşağı"
        return "nötr"
