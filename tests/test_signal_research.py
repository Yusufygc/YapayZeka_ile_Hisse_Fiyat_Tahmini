# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.signal_research import (
    build_decision_report,
    check_universe,
    ensure_symbol_data,
    plan_runs,
    run_research_matrix,
)


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    lines = [";".join(headers)]
    lines.extend(";".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_check_universe_reports_history_classes_and_missing_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "LONG.csv",
        ["Date", "Close"],
        [["2010-01-01", 1], ["2026-01-01", 2]],
    )
    _write_csv(
        data_dir / "SHORT.csv",
        ["Date", "Close"],
        [["2022-01-01", 1], ["2026-01-01", 2]],
    )
    _write_csv(
        data_dir / "MID.csv",
        ["Date", "Close"],
        [["2020-01-01", 1], ["2026-01-01", 2]],
    )

    frame = check_universe(
        symbols=["LONG", "MID", "SHORT", "MISS"],
        data_dir=data_dir,
        min_history_years=10,
    )
    rows = {row["symbol"]: row for row in frame.to_dict(orient="records")}

    assert rows["LONG"]["history_class"] == "long_history"
    assert rows["LONG"]["history_bucket"] == "long_history"
    assert rows["MID"]["history_bucket"] == "mid_history"
    assert rows["SHORT"]["history_class"] == "short_history"
    assert rows["MISS"]["history_class"] == "missing_data"
    assert rows["MISS"]["history_bucket"] == "missing_data"
    assert rows["LONG"]["eligible_10y"] is True
    assert rows["MID"]["eligible_10y"] is False
    assert rows["MID"]["meets_10y_reference"] is False


def test_check_universe_does_not_shift_iso_dates(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "ARDYZ.csv",
        ["Date", "Close"],
        [["2020-02-06", 1], ["2026-05-22", 2]],
    )

    frame = check_universe(symbols=["ARDYZ"], data_dir=data_dir)
    row = frame.iloc[0].to_dict()

    assert row["first_date"] == "2020-02-06"
    assert row["last_date"] == "2026-05-22"
    assert row["history_bucket"] == "mid_history"
    assert row["history_years"] == 6.2888


def test_plan_runs_builds_dry_run_policy_matrix_without_final_holdout_selection():
    frame = plan_runs(
        symbols=["ARDYZ"],
        models=["DLinear", "LightGBM Return"],
        policies=["V1", "V3"],
    )

    assert len(frame) == 4
    assert set(frame["policy"]) == {"V1", "V3"}
    assert frame["dry_run_only"].all()
    assert not frame["uses_final_holdout_for_selection"].any()
    v3 = frame[frame["policy"] == "V3"].iloc[0]
    assert v3["trade_count_min"] == 8
    assert v3["trade_count_max"] == 20
    assert v3["exposure_min"] == 25
    assert v3["exposure_max"] == 70


def test_plan_runs_does_not_filter_mid_history_symbols(tmp_path):
    frame = plan_runs(
        symbols=["ARDYZ"],
        models=["DLinear"],
        policies=["V0", "V1"],
    )

    assert len(frame) == 2
    assert set(frame["symbol"]) == {"ARDYZ"}


def test_check_universe_uses_sector_index_when_sector_label_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "TECH.csv",
        ["Date", "Close"],
        [["2020-01-01", 1], ["2026-01-01", 2]],
    )
    universe = tmp_path / "universe.csv"
    _write_csv(
        universe,
        ["Symbol", "Sector_Index"],
        [["TECH", "XTEK"]],
    )

    frame = check_universe(
        symbols=["TECH"],
        data_dir=data_dir,
        universe_file=universe,
    )

    assert frame.iloc[0]["sector"] == "XTEK"


def test_decision_report_includes_history_bucket_context():
    leaderboard = pd.DataFrame(
        [
            {
                "symbol": "ARDYZ",
                "model": "DLinear",
                "leader_reliability_class": "defensive",
                "history_bucket": "mid_history",
                "holdout_complete_flag": True,
            }
        ]
    )
    history_summary = pd.DataFrame(
        [{"history_bucket": "mid_history", "run_row_count": 1, "defensive_count": 1}]
    )
    sector_summary = pd.DataFrame([{"sector": "Technology", "run_row_count": 1}])

    report = build_decision_report(
        leaderboard=leaderboard,
        sector_summary=sector_summary,
        history_summary=history_summary,
    )

    assert "mid_history" in report
    assert "ARDYZ / DLinear / defensive / mid_history" in report


def test_ensure_symbol_data_restores_from_old_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    _write_csv(old_dir / "KCHOL.csv", ["Date", "Close"], [["2020-01-01", 1]])

    result = ensure_symbol_data(symbol="KCHOL", data_dir=data_dir, old_data_dir=old_dir)

    assert result["status"] == "restored_from_old"
    assert (data_dir / "KCHOL.csv").exists()


def test_run_research_matrix_dry_run_reports_mid_history_and_policy_metadata(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "ARDYZ.csv",
        ["Date", "Close"],
        [["2020-02-06", 1], ["2026-05-22", 2]],
    )

    frame = run_research_matrix(
        symbols=["ARDYZ"],
        data_dir=data_dir,
        outputs_base=tmp_path / "outputs",
        models=["DLinear"],
        policies=["V3"],
        resume=True,
        dry_run=True,
    )
    row = frame.iloc[0].to_dict()

    assert row["status"] == "planned"
    assert row["history_bucket"] == "mid_history"
    assert row["policy"] == "V3"
    assert row["trade_count_min"] == 8
    assert row["uses_final_holdout_for_selection"] is False


def test_run_research_matrix_resume_skips_complete_policy_run(tmp_path):
    data_dir = tmp_path / "data"
    outputs = tmp_path / "outputs"
    data_dir.mkdir()
    run_dir = outputs / "ARDYZ" / "runs" / "run_v1"
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True)
    _write_csv(
        data_dir / "ARDYZ.csv",
        ["Date", "Close"],
        [["2020-02-06", 1], ["2026-05-22", 2]],
    )
    (run_dir / "run_manifest.json").write_text(
        (
            '{"run_id": "run_v1", "research_policy": "V1", '
            '"model_list": ["DLinear"], "final_holdout_status": {"status": "success"}}'
        ),
        encoding="utf-8",
    )
    _write_csv(csv_dir / "backtest_report_final_holdout.csv", ["Model"], [["DLinear"]])

    frame = run_research_matrix(
        symbols=["ARDYZ"],
        data_dir=data_dir,
        outputs_base=outputs,
        models=["DLinear"],
        policies=["V1"],
        resume=True,
        dry_run=False,
    )

    assert frame.iloc[0]["status"] == "skipped_resume"
