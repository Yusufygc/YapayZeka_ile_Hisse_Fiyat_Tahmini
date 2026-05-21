# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
import hashlib
import subprocess
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.reporting_utils import write_csv_and_aligned_view

logger = logging.getLogger(__name__)


def write_window_selection_decision(comparison_df: Any, save_path: str) -> str:
    import pandas as _pd

    if not isinstance(comparison_df, _pd.DataFrame):
        comparison_df = _pd.DataFrame(comparison_df)
    decision_path = os.path.splitext(save_path)[0] + "_decision.md"
    os.makedirs(os.path.dirname(decision_path), exist_ok=True)
    best = comparison_df.iloc[0].to_dict() if not comparison_df.empty else {}
    with open(decision_path, "w", encoding="utf-8") as handle:
        handle.write("# Training Window Selection Decision\n\n")
        handle.write("Final holdout used for selection: `False`\n\n")
        if best:
            handle.write("## Selected Row\n\n")
            for key, value in best.items():
                handle.write(f"- `{key}`: `{value}`\n")
    return decision_path


def write_validation_and_quality_reports(pipeline: Any) -> None:
    try:
        vp_df = pipeline.data_manager.get_validation_protocol_data()
        if vp_df is not None and not vp_df.empty:
            vp_path = os.path.join(pipeline.outputs_dir, "validation_protocol_report.csv")
            write_csv_and_aligned_view(
                vp_df,
                vp_path,
                columns=[
                    "Split",
                    "Protocol",
                    "Window_Type",
                    "Train_Rows",
                    "Test_Rows",
                    "Train_Date_Start",
                    "Train_Date_End",
                    "Test_Date_Start",
                    "Test_Date_End",
                    "Scaler_Fit_Start",
                    "Scaler_Fit_End",
                    "Features_Count",
                    "Selection_Set",
                    "Evaluation_Set",
                    "Final_Holdout_Used_For_Selection",
                ],
            )
            print(f"  [OK] Validation protocol raporu kaydedildi -> {vp_path}")

        dq_reports = pipeline.data_manager.get_data_quality_reports()
        summary_rows = []
        for report_name, report_data in dq_reports.items():
            if not report_data:
                continue
            summary_rows.append({
                "Report": report_name,
                "Row_Count": len(report_data) if hasattr(report_data, "__len__") else 1,
                "Status": "available",
            })
            if pipeline.report_detail_level == "research":
                import pandas as _pd

                dq_df = _pd.DataFrame(report_data)
                dq_path = os.path.join(pipeline.outputs_dir, f"data_quality_{report_name}.csv")
                write_csv_and_aligned_view(dq_df, dq_path)

        if summary_rows:
            import pandas as _pd

            summary_path = os.path.join(pipeline.outputs_dir, "data_quality_summary.csv")
            write_csv_and_aligned_view(_pd.DataFrame(summary_rows), summary_path)
            print(f"  [OK] Data quality ozet raporu kaydedildi -> {summary_path}")
    except Exception as exc:
        print(f"  [WARN] Validation/data quality raporlari kaydedilemedi: {exc}")


def write_run_manifest(pipeline: Any) -> None:
    def _md5_file(path: str) -> str:
        h = hashlib.md5()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return "unavailable"

    def _dict_hash(d: dict) -> str:
        raw = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _git_commit() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=pipeline.project_root,
            )
            return result.stdout.strip() if result.returncode == 0 else "unavailable"
        except Exception:
            return "unavailable"

    def _lib_versions() -> Dict[str, str]:
        libs = ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "torch"]
        versions: Dict[str, str] = {}
        for lib in libs:
            try:
                import importlib
                mod = importlib.import_module(lib if lib != "sklearn" else "sklearn")
                versions[lib] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[lib] = "not_installed"
        return versions

    signal_cfg_dict = {}
    try:
        sc = pipeline._cfg.execution.signal_config
        signal_cfg_dict = {
            "quality_gate_mode": pipeline.quality_gate_mode,
            "min_directional_accuracy": pipeline.min_directional_accuracy,
            "max_rmse_vs_benchmark": pipeline.max_rmse_vs_benchmark,
            "min_composite_score": pipeline.min_composite_score,
        }
    except Exception as exc:
        logger.warning(f"Error parsing signal config for manifest: {exc}")

    manifest = {
        "run_id": pipeline.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stock_symbol": pipeline.stock_symbol,
        "data_hash": _md5_file(pipeline.data_file),
        "feature_pipeline_version": getattr(pipeline._cfg.data, "feature_mode", "unknown"),
        "model_config_hash": _dict_hash(pipeline.model_config),
        "signal_config_hash": _dict_hash(signal_cfg_dict),
        "random_seed": 42,
        "model_list": sorted(pipeline.candidate_models),
        "validation_protocol": pipeline.validation_mode,
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "lib_versions": _lib_versions(),
    }

    manifest_path = os.path.join(pipeline.outputs_dir, "run_manifest.json")
    os.makedirs(pipeline.outputs_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"  [OK] Run manifest yazildi -> {manifest_path}")
