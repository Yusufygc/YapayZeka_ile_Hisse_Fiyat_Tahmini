# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.run_leaderboard import (
    build_multi_symbol_leaderboard,
    build_history_effect_summary,
    build_run_leaderboard,
    build_sector_summary,
    classify_history_bucket,
    leaderboard_to_records,
    list_symbols_with_runs,
)


def _write_run(
    outputs: Path,
    *,
    symbol: str = "ARDYZ",
    run_id: str,
    model: str,
    wf_net: float = 0.10,
    final_net: float | None = 0.12,
    final_buyhold: float | None = 0.05,
    final_trade_count: int | None = 10,
    final_sharpe: float | None = 1.2,
    final_diagnosis: str = "ok",
    include_final: bool = True,
    include_protocol: bool = True,
    bom: bool = False,
) -> Path:
    run_dir = outputs / symbol / "runs" / run_id
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run_id, "stock_symbol": symbol, "model_list": [model]}),
        encoding="utf-8",
    )
    _write_csv(
        csv_dir / "backtest_report_wf.csv",
        ["Model", "Net_Return", "BuyHold_Return", "Sharpe", "Trade_Count", "Signal_Diagnosis"],
        [[model, wf_net, 0.03, 1.0, 9, "ok"]],
        bom=bom,
    )
    if include_final:
        _write_csv(
            csv_dir / "backtest_report_final_holdout.csv",
            ["Model", "Net_Return", "BuyHold_Return", "Sharpe", "Trade_Count", "Signal_Diagnosis"],
            [[model, final_net, final_buyhold, final_sharpe, final_trade_count, final_diagnosis]],
            bom=bom,
        )
    if include_protocol:
        _write_csv(
            csv_dir / "validation_protocol_report.csv",
            ["Split", "Protocol", "Final_Holdout_Used_For_Selection"],
            [["final_holdout", "final_holdout", "False"]],
            bom=bom,
        )
    return run_dir


def _write_csv(
    path: Path, headers: list[str], rows: list[list[object]], *, bom: bool = False
) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    lines = [";".join(headers)]
    lines.extend(";".join("" if value is None else str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines), encoding=encoding)


def _single_row(outputs: Path, symbol: str = "ARDYZ") -> dict:
    frame = build_run_leaderboard(outputs_base=outputs, symbol=symbol)
    records = leaderboard_to_records(frame)
    assert len(records) == 1
    return records[0]


def test_complete_run_computes_gap_and_stable_class(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(
        outputs,
        run_id="run_stable",
        model="DLinear",
        wf_net=0.20,
        final_net=0.15,
        final_buyhold=0.05,
        final_trade_count=10,
        final_sharpe=1.5,
    )

    row = _single_row(outputs)

    assert row["symbol"] == "ARDYZ"
    assert row["model"] == "DLinear"
    assert row["wf_final_net_gap"] == pytest.approx(0.05)
    assert row["final_excess_vs_buyhold"] == pytest.approx(0.10)
    assert row["holdout_complete_flag"] is True
    assert row["trade_sufficiency_flag"] is True
    assert row["leader_reliability_class"] == "stable"


def test_missing_final_holdout_report_marks_incomplete(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(outputs, run_id="run_incomplete", model="Prophet-ML/DL Hybrid", include_final=False)

    row = _single_row(outputs)

    assert row["holdout_complete_flag"] is False
    assert row["leader_reliability_class"] == "incomplete"
    assert "csv/backtest_report_final_holdout.csv" in row["missing_required_files"]


def test_single_trade_buyhold_clone_is_invalid(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(
        outputs,
        run_id="run_clone",
        model="ARIMA",
        wf_net=0.10,
        final_net=0.408158,
        final_buyhold=0.408158,
        final_trade_count=1,
        final_sharpe=2.2,
    )

    row = _single_row(outputs)

    assert row["benchmark_clone_flag"] is True
    assert row["leader_reliability_class"] == "invalid"


def test_positive_wf_negative_final_is_unstable(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(
        outputs,
        run_id="run_unstable",
        model="Random Forest",
        wf_net=0.94,
        final_net=-0.02,
        final_buyhold=0.40,
        final_trade_count=12,
        final_sharpe=-1.7,
    )

    row = _single_row(outputs)

    assert row["trade_sufficiency_flag"] is True
    assert row["leader_reliability_class"] == "unstable"


def test_positive_final_below_buyhold_with_good_trades_is_defensive(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(
        outputs,
        run_id="run_defensive",
        model="DLinear",
        wf_net=0.30,
        final_net=0.20,
        final_buyhold=0.30,
        final_trade_count=11,
        final_sharpe=2.0,
    )

    row = _single_row(outputs)

    assert row["final_excess_vs_buyhold"] == pytest.approx(-0.10)
    assert row["leader_reliability_class"] == "defensive"


def test_semicolon_bom_csv_is_read(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(
        outputs,
        run_id="run_bom",
        model="LightGBM Return",
        wf_net=0.80,
        final_net=0.09,
        final_buyhold=0.40,
        final_trade_count=11,
        final_sharpe=0.38,
        bom=True,
    )

    row = _single_row(outputs)

    assert row["model"] == "LightGBM Return"
    assert row["leader_reliability_class"] == "unstable"


def test_multi_symbol_history_sector_and_ranks_are_reported(tmp_path):
    outputs = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_run(
        outputs,
        symbol="ARDYZ",
        run_id="run_defensive",
        model="DLinear",
        wf_net=0.30,
        final_net=0.20,
        final_buyhold=0.30,
        final_trade_count=11,
        final_sharpe=2.0,
    )
    _write_run(
        outputs,
        symbol="ASELS",
        run_id="run_stable",
        model="LightGBM Return",
        wf_net=0.25,
        final_net=0.35,
        final_buyhold=0.20,
        final_trade_count=12,
        final_sharpe=1.7,
    )
    _write_csv(
        data_dir / "ARDYZ.csv",
        ["Date", "Close"],
        [["2020-01-01", 10], ["2026-01-01", 20]],
    )
    _write_csv(
        data_dir / "ASELS.csv",
        ["Date", "Close"],
        [["2010-01-01", 10], ["2026-01-01", 20]],
    )
    sector_file = data_dir / "universe.csv"
    _write_csv(
        sector_file,
        ["Symbol", "Sector"],
        [["ARDYZ", "Technology"], ["ASELS", "Defense"]],
    )

    frame = build_multi_symbol_leaderboard(
        outputs_base=outputs,
        symbols=["ARDYZ", "ASELS"],
        data_dir=data_dir,
        sector_file=sector_file,
        min_history_years=10,
    )
    records = {row["symbol"]: row for row in leaderboard_to_records(frame)}

    assert records["ARDYZ"]["history_class"] == "mid_history"
    assert records["ARDYZ"]["history_bucket"] == "mid_history"
    assert records["ARDYZ"]["meets_10y_reference"] is False
    assert records["ASELS"]["history_class"] == "long_history"
    assert records["ASELS"]["history_bucket"] == "long_history"
    assert records["ASELS"]["meets_10y_reference"] is True
    assert records["ARDYZ"]["sector"] == "Technology"
    assert records["ASELS"]["prediction_leader_rank"] == 1
    assert records["ASELS"]["trading_leader_rank"] == 1


def test_leaderboard_history_parsing_preserves_iso_dates(tmp_path):
    outputs = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_run(outputs, symbol="ARDYZ", run_id="run_mid", model="DLinear")
    _write_csv(
        data_dir / "ARDYZ.csv",
        ["Date", "Close"],
        [["2020-02-06", 10], ["2026-05-22", 20]],
    )

    frame = build_run_leaderboard(outputs_base=outputs, symbol="ARDYZ", data_dir=data_dir)
    row = leaderboard_to_records(frame)[0]

    assert row["history_bucket"] == "mid_history"
    assert row["history_years"] == pytest.approx(6.2888)


def test_sector_summary_counts_reliability_and_repeated_families(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(outputs, symbol="AAA", run_id="run_a", model="DLinear")
    _write_run(outputs, symbol="BBB", run_id="run_b", model="DLinear")
    sector_file = tmp_path / "sector.csv"
    _write_csv(
        sector_file,
        ["Symbol", "Sector"],
        [["AAA", "Technology"], ["BBB", "Technology"]],
    )

    frame = build_multi_symbol_leaderboard(
        outputs_base=outputs,
        symbols=["AAA", "BBB"],
        sector_file=sector_file,
    )
    summary = build_sector_summary(frame)

    row = summary.iloc[0].to_dict()
    assert row["sector"] == "Technology"
    assert row["symbol_count"] == 2
    assert row["stable_count"] == 2
    assert row["history_bucket_breakdown"] == "unknown:2"
    assert row["repeated_model_families"] == "DLinear:2"


def test_history_effect_summary_groups_reliability_and_gap(tmp_path):
    outputs = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_run(
        outputs, symbol="LONG", run_id="run_long", model="DLinear", wf_net=0.3, final_net=0.2
    )
    _write_run(
        outputs,
        symbol="MID",
        run_id="run_mid",
        model="Random Forest",
        wf_net=0.8,
        final_net=-0.1,
        final_buyhold=0.2,
        final_trade_count=10,
        final_sharpe=-1.0,
    )
    _write_csv(data_dir / "LONG.csv", ["Date", "Close"], [["2010-01-01", 1], ["2026-01-01", 2]])
    _write_csv(data_dir / "MID.csv", ["Date", "Close"], [["2020-01-01", 1], ["2026-01-01", 2]])

    frame = build_multi_symbol_leaderboard(
        outputs_base=outputs,
        symbols=["LONG", "MID"],
        data_dir=data_dir,
    )
    summary = build_history_effect_summary(frame)
    rows = {row["history_bucket"]: row for row in summary.to_dict(orient="records")}

    assert rows["long_history"]["stable_count"] == 1
    assert rows["mid_history"]["unstable_count"] == 1
    assert rows["mid_history"]["avg_wf_final_net_gap"] == pytest.approx(0.9)


def test_history_bucket_boundaries_are_diagnostic_not_filters():
    assert classify_history_bucket(4.9) == "short_history"
    assert classify_history_bucket(6.3) == "mid_history"
    assert classify_history_bucket(10.0) == "long_history"
    assert (
        classify_history_bucket(None, data_dir_provided=True, missing_history=True)
        == "missing_data"
    )
    assert (
        classify_history_bucket(None, data_dir_provided=False, missing_history=False) == "unknown"
    )


def test_sector_lookup_uses_sector_index_when_sector_label_missing(tmp_path):
    outputs = tmp_path / "outputs"
    _write_run(outputs, symbol="AAA", run_id="run_a", model="DLinear")
    sector_file = tmp_path / "sector.csv"
    _write_csv(
        sector_file,
        ["Symbol", "Sector_Index"],
        [["AAA", "XTEK"]],
    )

    frame = build_multi_symbol_leaderboard(
        outputs_base=outputs,
        symbols=["AAA"],
        sector_file=sector_file,
    )
    row = leaderboard_to_records(frame)[0]

    assert row["sector"] == "XTEK"


def test_list_symbols_with_runs_ignores_latest(tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "AAA" / "runs").mkdir(parents=True)
    (outputs / "BBB" / "latest").mkdir(parents=True)

    assert list_symbols_with_runs(outputs) == ["AAA"]
