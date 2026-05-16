# -*- coding: utf-8 -*-
"""Grid construction and deterministic sampling for signal calibration."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.backtesting.signals import SignalConfig

GridParams = dict[str, float | int]


def signal_calibration_grid(base_cfg: SignalConfig) -> list[GridParams]:
    min_dir_values = sorted({48.0, 50.0, float(base_cfg.min_directional_accuracy)})
    vol_values = sorted({0.10, 0.15, 0.20, 0.25, 0.30, float(base_cfg.volatility_multiplier)})
    entry_cost_values = sorted({1.5, 2.0, 2.5, float(base_cfg.entry_cost_multiplier)})
    min_entry_values = sorted({0.0, 0.001, 0.002, float(base_cfg.min_entry_threshold)})
    max_hold_values = sorted({10, 15, 20, int(base_cfg.max_holding_bars)})
    take_profit_values = sorted({1.0, 1.5, 2.0, float(base_cfg.take_profit_vol_multiplier)})
    stop_loss_values = sorted({0.75, 1.0, 1.25, float(base_cfg.stop_loss_vol_multiplier)})

    grid = []
    for min_dir in min_dir_values:
        for vol in vol_values:
            for entry_cost in entry_cost_values:
                for min_entry in min_entry_values:
                    for max_hold in max_hold_values:
                        for take_profit in take_profit_values:
                            for stop_loss in stop_loss_values:
                                grid.append({
                                    "min_directional_accuracy": round(float(min_dir), 2),
                                    "volatility_multiplier": round(float(vol), 4),
                                    "entry_cost_multiplier": round(float(entry_cost), 4),
                                    "min_entry_threshold": round(float(min_entry), 6),
                                    "max_holding_bars": int(max_hold),
                                    "take_profit_vol_multiplier": round(float(take_profit), 4),
                                    "stop_loss_vol_multiplier": round(float(stop_loss), 4),
                                })
    return grid


def apply_trial_policy(owner: Any, grid: list[GridParams]) -> tuple[list[GridParams], dict[str, Any]]:
    profile = str(getattr(owner, "signal_calibration_profile", "production") or "production").lower()
    if profile not in {"production", "research"}:
        profile = "production"
    sampler = str(getattr(owner, "signal_calibration_sampler", "adaptive_stratified") or "adaptive_stratified").lower()
    seed = int(getattr(owner, "signal_calibration_seed", 42) or 42)
    trial_cap = getattr(owner, "signal_calibration_max_trials", 64)
    trial_cap = None if trial_cap is None else max(1, int(trial_cap))

    if profile == "research":
        selected_grid = list(grid)
        effective_cap = None
        sampler = "full_grid"
    elif sampler in {"adaptive_stratified", "stratified"}:
        effective_cap = trial_cap
        selected_grid = sample_signal_calibration_grid(
            grid,
            cap=effective_cap if effective_cap is not None else len(grid),
            seed=seed,
        )
    else:
        effective_cap = trial_cap
        selected_grid = list(grid[:effective_cap]) if effective_cap is not None else list(grid)
        sampler = "prefix"
    return selected_grid, {
        "grid_size": int(len(grid)),
        "executed_trials": int(len(selected_grid)),
        "trial_cap": effective_cap,
        "calibration_profile": profile,
        "sampler": sampler,
        "seed": seed,
        "adaptive_expanded": False,
        "coverage_status": coverage_status(grid, selected_grid),
    }


def grid_param_key(params: GridParams) -> tuple:
    return tuple(sorted(params.items()))


def param_values(grid: list[GridParams]) -> dict[str, set]:
    values: dict[str, set] = {}
    for params in grid:
        for key, value in params.items():
            values.setdefault(key, set()).add(value)
    return values


def coverage_status(full_grid: list[GridParams], selected_grid: list[GridParams]) -> str:
    if not full_grid:
        return "empty_grid"
    full_values = param_values(full_grid)
    selected_values = param_values(selected_grid)
    missing = []
    for key, values in sorted(full_values.items()):
        uncovered = sorted(values - selected_values.get(key, set()))
        if uncovered:
            missing.append(f"{key}={uncovered}")
    return "complete" if not missing else "missing:" + "; ".join(missing)


def expansion_grid(
    *,
    full_grid: list[GridParams],
    selected_grid: list[GridParams],
    executed_trials: int,
    seed: int,
) -> list[GridParams]:
    used_keys = {grid_param_key(params) for params in selected_grid}
    second_cap = min(len(full_grid), max(executed_trials * 2, 128))
    remaining_cap = max(0, second_cap - len(selected_grid))
    if not remaining_cap:
        return []
    return sample_signal_calibration_grid(
        full_grid,
        cap=remaining_cap,
        seed=seed + 1,
        exclude_keys=used_keys,
    )


def sample_signal_calibration_grid(
    grid: list[GridParams],
    *,
    cap: int,
    seed: int,
    exclude_keys: Optional[set[tuple]] = None,
) -> list[GridParams]:
    if cap <= 0 or not grid:
        return []
    exclude_keys = exclude_keys or set()
    available = [
        (idx, params)
        for idx, params in enumerate(grid)
        if grid_param_key(params) not in exclude_keys
    ]
    if not available:
        return []
    if cap >= len(available):
        return [dict(params) for _, params in available]

    rng = np.random.default_rng(seed)
    selected_indices, selected_keys = select_coverage_grid_indices(
        available=available,
        cap=cap,
        rng=rng,
    )
    append_random_grid_indices(
        available=available,
        cap=cap,
        rng=rng,
        selected_indices=selected_indices,
        selected_keys=selected_keys,
    )

    index_to_params = {idx: params for idx, params in available}
    return [dict(index_to_params[idx]) for idx in selected_indices[:cap]]


def select_coverage_grid_indices(
    *,
    available: list[tuple[int, GridParams]],
    cap: int,
    rng: np.random.Generator,
) -> tuple[list[int], set[tuple]]:
    full_values = param_values([params for _, params in available])
    required = {(key, value) for key, values in full_values.items() for value in values}
    selected_indices: list[int] = []
    selected_keys: set[tuple] = set()
    covered: set[tuple] = set()

    while required - covered and len(selected_indices) < cap:
        best = best_grid_coverage_candidate(
            available=available,
            selected_keys=selected_keys,
            covered=covered,
            rng=rng,
        )
        if best is None or best[0][0] == 0:
            break
        _, idx, params = best
        selected_indices.append(idx)
        selected_keys.add(grid_param_key(params))
        covered.update((name, value) for name, value in params.items())
    return selected_indices, selected_keys


def best_grid_coverage_candidate(
    *,
    available: list[tuple[int, GridParams]],
    selected_keys: set[tuple],
    covered: set[tuple],
    rng: np.random.Generator,
) -> tuple[tuple[int, float], int, GridParams] | None:
    best = None
    for idx, params in available:
        key = grid_param_key(params)
        if key in selected_keys:
            continue
        gained = {(name, value) for name, value in params.items()} - covered
        score = (len(gained), float(rng.random()))
        if best is None or score > best[0]:
            best = (score, idx, params)
    return best


def append_random_grid_indices(
    *,
    available: list[tuple[int, GridParams]],
    cap: int,
    rng: np.random.Generator,
    selected_indices: list[int],
    selected_keys: set[tuple],
) -> None:
    remaining = [
        (idx, params)
        for idx, params in available
        if grid_param_key(params) not in selected_keys
    ]
    if len(selected_indices) >= cap or not remaining:
        return
    order = rng.permutation(len(remaining))
    for pos in order[: cap - len(selected_indices)]:
        idx, params = remaining[int(pos)]
        selected_indices.append(idx)
        selected_keys.add(grid_param_key(params))
