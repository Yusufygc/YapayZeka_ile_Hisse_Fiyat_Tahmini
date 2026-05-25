# -*- coding: utf-8 -*-
"""
Sprint 2 (2026-05-25) Plan A2.2 — Corporate action audit script testleri.

tools/audit_corporate_actions.py |log_return| >= threshold satirlarini
dogru tespit etmeli, severity (high/extreme) atamasini yapmali, CSV
raporu uretmeli.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# tools dizinini path'e ekle (audit_corporate_actions.py orada)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

import audit_corporate_actions as audit  # noqa: E402


def _write_csv(path: str, dates, closes) -> None:
    pd.DataFrame({"Date": dates, "Close": closes, "Volume": [100] * len(dates)}).to_csv(
        path, index=False, encoding="utf-8"
    )


def test_audit_detects_split_anomaly():
    """1:2 split benzeri ani %50 dusus -> log_return ≈ -0.69 -> tespit."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        dates = pd.date_range("2024-01-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        closes = [100.0] * 10 + [50.0] * 10  # 11. gunde split
        _write_csv(os.path.join(data_dir, "TEST.csv"), dates, closes)

        report = audit.scan_corporate_actions(data_dir=data_dir, threshold=0.30)
        assert not report.empty
        assert (report["Symbol"] == "TEST").any()
        assert (report["log_return"] < -0.5).any()
        assert (report["severity"] == "extreme").any()


def test_audit_no_false_positive_smooth_series():
    """Yumusak gunluk %1 hareketler -> anomali yok."""
    with tempfile.TemporaryDirectory() as data_dir:
        rng = np.random.default_rng(seed=42)
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        # Gunluk %1 normal sapma -> log_return ≈ 0.01, |log_return| < 0.05
        rets = rng.normal(0.0005, 0.01, size=n)
        closes = 100.0 * np.cumprod(1.0 + rets)
        _write_csv(os.path.join(data_dir, "CLEAN.csv"), dates, closes.tolist())

        report = audit.scan_corporate_actions(data_dir=data_dir, threshold=0.30)
        assert report.empty


def test_audit_severity_split():
    """Threshold 0.30, extreme threshold 0.50 -> high/extreme dogru ayrim."""
    with tempfile.TemporaryDirectory() as data_dir:
        dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        # day 5: 35% dusus -> log_return ≈ -0.43 (high)
        # day 8: 60% dusus -> log_return ≈ -0.92 (extreme)
        closes = [100.0, 100.0, 100.0, 100.0, 100.0, 65.0, 65.0, 65.0, 26.0, 26.0]
        _write_csv(os.path.join(data_dir, "MIX.csv"), dates, closes)

        report = audit.scan_corporate_actions(data_dir=data_dir, threshold=0.30)
        assert "high" in report["severity"].values
        assert "extreme" in report["severity"].values


def test_audit_writes_csv_to_disk():
    """write_report() CSV ve latest snapshot uretir."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
        closes = [100.0, 100.0, 40.0, 40.0, 40.0]
        _write_csv(os.path.join(data_dir, "WRITE.csv"), dates, closes)

        report = audit.scan_corporate_actions(data_dir=data_dir, threshold=0.30)
        out_path = audit.write_report(report, out_dir=out_dir)
        assert out_path is not None
        assert os.path.exists(out_path)
        assert os.path.exists(os.path.join(out_dir, "corporate_action_audit_latest.csv"))


def test_audit_universe_filter():
    """--universe ile sadece allowlist'teki semboller taranir."""
    with tempfile.TemporaryDirectory() as data_dir:
        for sym in ["A", "B", "C"]:
            dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
            closes = [100.0, 100.0, 40.0, 40.0, 40.0]
            _write_csv(os.path.join(data_dir, f"{sym}.csv"), dates, closes)

        uni_path = os.path.join(data_dir, "_universe.csv")
        pd.DataFrame({"symbol": ["A", "C"]}).to_csv(uni_path, index=False)

        report = audit.scan_corporate_actions(
            data_dir=data_dir, threshold=0.30, universe=uni_path,
        )
        symbols = set(report["Symbol"].unique())
        assert "A" in symbols and "C" in symbols
        assert "B" not in symbols
