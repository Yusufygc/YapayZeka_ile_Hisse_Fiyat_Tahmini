"""BIST pay piyasası fiyat kuralları.

Sorumluluklar:
  - Fiyat adımı (tick size) ve günlük fiyat bandı (PriceBand) hesabı.
  - Tahmin fiyatlarını geçerli tick/band'a yuvarlar ve clip eder.

RULES_VERSION ile kural sürümü izlenir.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Iterable

import numpy as np
import pandas as pd


RULES_VERSION = "bist_equity_rules_2026_01"


@dataclass(frozen=True)
class PriceBand:
    base_price: float
    lower_band: float
    upper_band: float
    price_tick: float


class BistMarketRules:
    """
    BIST equity-market rule helper for forecast outputs.

    The implementation intentionally covers common listed equities. Segment-
    specific margins can be added later; active default is 10%.
    """

    STOCK_TICKS: tuple[tuple[float, float], ...] = (
        (20.0, 0.01),
        (50.0, 0.02),
        (100.0, 0.05),
        (250.0, 0.10),
        (500.0, 0.25),
        (1000.0, 0.50),
        (2500.0, 1.00),
        (float("inf"), 2.50),
    )

    def __init__(
        self,
        calendar_path: str | None = None,
        *,
        default_price_margin: float = 0.10,
    ) -> None:
        self.calendar_path = calendar_path
        self.default_price_margin = float(default_price_margin)
        self.calendar = self._load_calendar(calendar_path)

    def next_trading_days(self, start_date: str | pd.Timestamp, count: int) -> list[pd.Timestamp]:
        if count <= 0:
            return []
        start = pd.to_datetime(start_date).normalize()
        days: list[pd.Timestamp] = []
        current = start
        calendar_status: dict[pd.Timestamp, bool] = {}
        if not self.calendar.empty:
            calendar_status = {
                pd.Timestamp(row.Date).normalize(): bool(row.Is_Trading_Day)
                for row in self.calendar.itertuples(index=False)
            }
        while len(days) < count:
            current = current + pd.Timedelta(days=1)
            if current in calendar_status:
                if calendar_status[current]:
                    days.append(current.normalize())
                continue
            if current.weekday() >= 5:
                continue
            days.append(current.normalize())
        return days

    def price_tick(self, price: float) -> float:
        price = abs(float(price))
        for upper_bound, tick in self.STOCK_TICKS:
            if price < upper_bound or math.isinf(upper_bound):
                return tick
        return 2.50

    def round_to_tick(self, price: float, *, direction: str = "nearest", tick: float | None = None) -> float:
        tick = self.price_tick(price) if tick is None else float(tick)
        price_d = Decimal(str(float(price)))
        tick_d = Decimal(str(tick))
        units = price_d / tick_d
        if direction == "floor":
            rounded_units = units.to_integral_value(rounding=ROUND_FLOOR)
        elif direction == "ceil":
            rounded_units = units.to_integral_value(rounding=ROUND_CEILING)
        elif direction == "nearest":
            rounded_units = units.to_integral_value(rounding=ROUND_HALF_UP)
        else:
            raise ValueError("direction must be one of: nearest, floor, ceil")
        return float(rounded_units * tick_d)

    def price_band(self, previous_close: float, *, margin: float | None = None) -> PriceBand:
        margin = self.default_price_margin if margin is None else float(margin)
        base_tick = self.price_tick(previous_close)
        base_price = self.round_to_tick(previous_close, direction="nearest", tick=base_tick)
        raw_lower = base_price * (1.0 - margin)
        raw_upper = base_price * (1.0 + margin)
        lower_tick = self.price_tick(raw_lower)
        upper_tick = self.price_tick(raw_upper)
        return PriceBand(
            base_price=base_price,
            lower_band=self.round_to_tick(raw_lower, direction="ceil", tick=lower_tick),
            upper_band=self.round_to_tick(raw_upper, direction="floor", tick=upper_tick),
            price_tick=self.price_tick(base_price),
        )

    def bound_forecast_price(self, predicted_close: float, previous_close: float) -> tuple[float, PriceBand]:
        band = self.price_band(previous_close)
        clipped = min(max(float(predicted_close), band.lower_band), band.upper_band)
        bounded = self.round_to_tick(clipped, direction="nearest")
        bounded = min(max(bounded, band.lower_band), band.upper_band)
        return bounded, band

    @staticmethod
    def trend_threshold(
        close_prices: Iterable[float],
        *,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        horizon_days: int = 5,
    ) -> float:
        close = pd.Series(list(close_prices), dtype=float).dropna()
        if len(close) >= 21:
            returns = np.log(close / close.shift(1)).dropna().tail(20)
            realized_vol = float(returns.std(ddof=0)) if len(returns) else 0.0
        else:
            realized_vol = 0.0
        round_trip_cost = 2.0 * (float(commission_bps) + float(slippage_bps)) / 10000.0
        volatility_floor = 0.25 * realized_vol * math.sqrt(max(1, int(horizon_days)))
        return float(max(0.005, round_trip_cost, volatility_floor))

    @staticmethod
    def trend_label(weekly_expected_return: float, threshold: float) -> str:
        if weekly_expected_return > threshold:
            return "UP"
        if weekly_expected_return < -threshold:
            return "DOWN"
        return "FLAT"

    @staticmethod
    def _load_calendar(calendar_path: str | None) -> pd.DataFrame:
        if not calendar_path or not os.path.exists(calendar_path):
            return pd.DataFrame(columns=["Date", "Is_Trading_Day", "Session_Type", "Note"])
        calendar = pd.read_csv(calendar_path)
        required = {"Date", "Is_Trading_Day", "Session_Type", "Note"}
        missing = required - set(calendar.columns)
        if missing:
            raise ValueError(f"BIST calendar missing columns: {sorted(missing)}")
        calendar = calendar.copy()
        calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.normalize()
        calendar["Is_Trading_Day"] = calendar["Is_Trading_Day"].map(_as_bool)
        calendar.sort_values("Date", inplace=True)
        calendar.reset_index(drop=True, inplace=True)
        return calendar


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "open", "trading"}
