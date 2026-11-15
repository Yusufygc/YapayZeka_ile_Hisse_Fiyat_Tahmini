# -*- coding: utf-8 -*-
"""bist_calendar.ensure_bist_calendar: sabit tatiller + manuel override korunumu."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.forecasting.bist_calendar import ensure_bist_calendar


def test_generated_calendar_closes_weekends_and_fixed_holidays(tmp_path):
    path = str(tmp_path / "cal" / "bist_calendar.csv")
    ensure_bist_calendar(path, today=date(2026, 5, 4), years_back=1, years_forward=1)
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    assert bool(df.loc["2026-05-01", "Is_Trading_Day"]) is False  # Emek ve Dayanisma Gunu
    assert bool(df.loc["2026-05-02", "Is_Trading_Day"]) is False  # cumartesi
    assert bool(df.loc["2026-05-04", "Is_Trading_Day"]) is True


def test_manual_override_is_preserved_on_extension(tmp_path):
    path = tmp_path / "bist_calendar.csv"
    path.write_text(
        "Date,Is_Trading_Day,Session_Type,Note\n2026-05-05,False,closed,dini tatil\n",
        encoding="utf-8",
    )
    ensure_bist_calendar(str(path), today=date(2026, 5, 4), years_back=1, years_forward=1)
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    assert bool(df.loc["2026-05-05", "Is_Trading_Day"]) is False
    assert df.loc["2026-05-05", "Note"] == "dini tatil"
