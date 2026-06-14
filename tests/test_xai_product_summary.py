# -*- coding: utf-8 -*-
"""xai/product_summary.py birim testleri."""
import os
import json
import tempfile
import textwrap

import pytest

from src.xai.product_summary import XaiProductSummary, build_xai_product_summary


def _write_csv(dir_path: str, filename: str, content: str) -> str:
    path = os.path.join(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).strip())
    return path


class TestBuildXaiProductSummary:
    def _make_outputs(self, symbol: str, model_name: str, content: str):
        tmpdir = tempfile.mkdtemp()
        safe = model_name.replace(" ", "_").replace("/", "_")
        xai_dir = os.path.join(tmpdir, symbol, "latest", "xai")
        os.makedirs(xai_dir)
        _write_csv(xai_dir, f"feature_importance_{safe}_wf.csv", content)
        return tmpdir

    def test_available_when_csv_exists(self):
        outputs = self._make_outputs(
            "TUPRS", "XGBoost",
            """
            Feature,Mean_Importance_WF
            Return,0.08
            RSI_14,0.06
            SMA_50_rel,-0.04
            Relative_Strength,0.03
            RollStd_20_norm,-0.02
            """
        )
        result = build_xai_product_summary("TUPRS", "XGBoost", outputs_base=outputs)
        assert result.available is True
        assert result.method == "SHAP TreeExplainer"
        assert len(result.top_positive_reasons) > 0

    def test_positive_negative_split(self):
        outputs = self._make_outputs(
            "ASELS", "XGBoost",
            """
            Feature,Mean_Importance_WF
            Return,0.09
            RSI_14,0.05
            SMA_50_rel,-0.04
            Relative_Strength,-0.03
            """
        )
        result = build_xai_product_summary("ASELS", "XGBoost", outputs_base=outputs)
        assert result.available is True
        pos_names = {f.feature_name for f in result.top_positive_reasons}
        neg_names = {f.feature_name for f in result.top_negative_reasons}
        assert "Return" in pos_names
        assert "SMA_50_rel" in neg_names or "Relative_Strength" in neg_names

    def test_caveat_always_present(self):
        outputs = self._make_outputs("X", "Ridge Return", "Feature,Mean_Importance_WF\nReturn,0.5")
        result = build_xai_product_summary("X", "Ridge Return", outputs_base=outputs)
        assert "nedensellik" in result.caveat.lower() or "xai" in result.caveat.lower()

    def test_unavailable_when_no_directory(self):
        result = build_xai_product_summary("ZZZ", "XGBoost", outputs_base="/nonexistent/path")
        assert result.available is False

    def test_unavailable_when_no_csv(self):
        tmpdir = tempfile.mkdtemp()
        xai_dir = os.path.join(tmpdir, "ZZZ", "latest", "xai")
        os.makedirs(xai_dir)
        result = build_xai_product_summary("ZZZ", "XGBoost", outputs_base=tmpdir)
        assert result.available is False

    def test_model_family_caveat_tree(self):
        outputs = self._make_outputs("X", "XGBoost", "Feature,Mean_Importance_WF\nReturn,0.5")
        result = build_xai_product_summary("X", "XGBoost", outputs_base=outputs)
        assert "TreeExplainer" in result.model_family_caveat or "tree" in result.model_family_caveat.lower()

    def test_model_family_caveat_linear(self):
        outputs = self._make_outputs("X", "Ridge Return", "Feature,Mean_Importance_WF\nReturn,0.5")
        result = build_xai_product_summary("X", "Ridge Return", outputs_base=outputs)
        assert "linear" in result.model_family_caveat.lower() or "katsay" in result.model_family_caveat.lower()

    def test_human_label_populated(self):
        outputs = self._make_outputs(
            "X", "XGBoost",
            "Feature,Mean_Importance_WF\nRSI_14,0.07"
        )
        result = build_xai_product_summary("X", "XGBoost", outputs_base=outputs)
        assert result.available is True
        if result.top_positive_reasons:
            assert result.top_positive_reasons[0].human_label != ""

    def test_unsigned_importance_splits_top_bottom(self):
        """CSV'de tüm değerler pozitifse top-N ve bottom-N ayrımı yapılmalı."""
        outputs = self._make_outputs(
            "X", "XGBoost",
            """
            Feature,Mean_Importance_WF
            A,0.9
            B,0.8
            C,0.7
            D,0.6
            E,0.5
            F,0.4
            G,0.3
            H,0.2
            I,0.1
            J,0.05
            """
        )
        result = build_xai_product_summary("X", "XGBoost", outputs_base=outputs, top_k=3)
        assert result.available is True
        assert len(result.top_positive_reasons) == 3
        assert result.top_positive_reasons[0].feature_name == "A"
        # ascending sort → en düşük önem değeri (J) ilk sıradadır
        assert result.top_negative_reasons[0].feature_name == "J"

    def test_standard_top_reasons_reads_semicolon_csv(self):
        tmpdir = tempfile.mkdtemp()
        xai_csv = os.path.join(tmpdir, "ASELS", "latest", "xai", "csv")
        os.makedirs(xai_csv)
        _write_csv(
            xai_csv,
            "xai_top_reasons_wf.csv",
            """
            Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate
            LSTM;Return;Gunluk getiri;technical;0.12;0.03;positive;reason;sequence;True
            LSTM;RSI_14;RSI;technical;-0.05;-0.01;negative;reason;sequence;True
            """,
        )

        result = build_xai_product_summary("ASELS", "LSTM", outputs_base=tmpdir)

        assert result.available is True
        assert result.top_positive_reasons[0].feature_name == "Return"
        assert result.top_negative_reasons[0].feature_name == "RSI_14"
        assert result.top_positive_reasons[0].feature_group == "technical"
        assert result.top_positive_reasons[0].reason == "reason"
        assert result.top_positive_reasons[0].method == "sequence"
        assert result.top_positive_reasons[0].contribution == pytest.approx(0.03)
        assert result.top_positive_reasons[0].approximate is True
        assert result.group_summaries
        technical = result.group_summaries[0]
        assert technical.feature_group == "technical"
        assert technical.total_importance == pytest.approx(0.17)
        assert technical.net_contribution == pytest.approx(0.02)
        assert technical.direction == "yukari"
        assert technical.approximate_ratio == 1.0

    def test_group_summaries_include_macro_and_market_rows_beyond_top_k(self):
        tmpdir = tempfile.mkdtemp()
        xai_csv = os.path.join(tmpdir, "ASELS", "latest", "xai", "csv")
        os.makedirs(xai_csv)
        _write_csv(
            xai_csv,
            "xai_top_reasons_wf.csv",
            """
            Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate
            LSTM;RSI_14;RSI;technical;0.50;0.03;positive;technical reason;sequence;False
            LSTM;USDTRY_Return;USDTRY return;macro;0.30;-0.02;negative;macro reason;sequence;True
            LSTM;BIST100_Return;BIST100 return;market_relative;0.20;0.01;positive;market reason;sequence;False
            """,
        )

        result = build_xai_product_summary("ASELS", "LSTM", outputs_base=tmpdir, top_k=1)
        groups = {item.feature_group: item for item in result.group_summaries}

        assert set(groups) >= {"technical", "macro", "market_relative"}
        assert groups["macro"].direction == "asagi"
        assert groups["macro"].net_contribution == pytest.approx(-0.02)
        assert groups["macro"].top_features == ["USDTRY_Return"]
        assert "model tahminini asagi" in groups["macro"].reason

    def test_legacy_importance_group_summary_without_contribution_is_attention(self):
        outputs = self._make_outputs(
            "X",
            "XGBoost",
            """
            Feature,Mean_Importance_WF
            USDTRY_Return,0.40
            RSI_14,0.20
            """
        )
        result = build_xai_product_summary("X", "XGBoost", outputs_base=outputs, top_k=1)
        groups = {item.feature_group: item for item in result.group_summaries}
        assert groups["macro"].direction == "dikkat"
        assert groups["macro"].total_importance == pytest.approx(0.40)

    def test_walk_forward_summary_has_specific_non_generic_label(self):
        tmpdir = tempfile.mkdtemp()
        xai_csv = os.path.join(tmpdir, "ASELS", "latest", "xai", "csv")
        os.makedirs(xai_csv)
        _write_csv(
            xai_csv,
            "xai_top_reasons_wf.csv",
            """
            Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate
            LSTM;WalkForward_Summary;;model_summary;1.0;-0.02;negative;Son pencerede tahmin zayıfladı.;rule_based;False
            """,
        )

        result = build_xai_product_summary("ASELS", "LSTM", outputs_base=tmpdir)

        assert result.available is True
        factor = result.top_negative_reasons[0]
        assert factor.feature_name == "WalkForward_Summary"
        assert "walk-forward" in factor.human_label.lower()
        assert "teknik veya makro" not in factor.human_label.lower()
        assert factor.reason == "Son pencerede tahmin zayıfladı."

    def test_run_id_xai_lookup_preferred_over_stale_latest(self):
        tmpdir = tempfile.mkdtemp()
        latest_csv = os.path.join(tmpdir, "ASELS", "latest", "xai", "csv")
        run_csv = os.path.join(tmpdir, "ASELS", "runs", "lstm-run", "xai", "csv")
        os.makedirs(latest_csv)
        os.makedirs(run_csv)
        _write_csv(
            latest_csv,
            "xai_top_reasons_wf.csv",
            """
            Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate
            NLinear;NFeature;N feature;technical;0.20;0.02;positive;reason;sequence;True
            """,
        )
        _write_csv(
            run_csv,
            "xai_top_reasons_wf.csv",
            """
            Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate
            LSTM;LFeature;L feature;technical;0.30;0.03;positive;reason;sequence;True
            """,
        )

        result = build_xai_product_summary(
            "ASELS",
            "LSTM",
            outputs_base=tmpdir,
            run_id="lstm-run",
        )

        assert result.available is True
        assert result.top_positive_reasons[0].feature_name == "LFeature"

    def test_manifest_metadata_is_exposed(self):
        tmpdir = tempfile.mkdtemp()
        xai_csv = os.path.join(tmpdir, "ASELS", "runs", "run-x", "xai", "csv")
        xai_dir = os.path.dirname(xai_csv)
        os.makedirs(xai_csv)
        _write_csv(
            xai_csv,
            "xai_top_reasons_wf.csv",
            """
            Model;Feature;Readable_Feature;Feature_Group;Importance;Contribution;Direction;Reason;Method;Approximate
            LSTM;Return;Gunluk getiri;technical;0.12;0.03;positive;reason;sequence_feature_lag_permutation;True
            """,
        )
        with open(os.path.join(xai_dir, "xai_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": "run-x",
                    "created_at": "2026-06-14T10:00:00+00:00",
                    "method_detail": "sequence_feature_lag_permutation",
                    "approximate_ratio": 1.0,
                    "background_scope": "train_slice_sequence",
                    "top_feature_stability": {"LSTM:Return": 1.0},
                    "dictionary_coverage": {"total": 1, "covered": 1, "ratio": 1.0, "missing": []},
                },
                handle,
            )

        result = build_xai_product_summary("ASELS", "LSTM", outputs_base=tmpdir, run_id="run-x")

        assert result.status == "fallback"
        assert result.method == "sequence_feature_lag_permutation"
        assert result.method_detail == "sequence_feature_lag_permutation"
        assert result.approximate_ratio == 1.0
        assert result.feature_stability_top == {"LSTM:Return": 1.0}
        assert result.background_scope == "train_slice_sequence"
