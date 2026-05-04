# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is skipped in minimal runtimes
    plt = None

from src.backtesting.metrics import summarize_backtest
from src.utils.reporting_utils import (
    bullet_list,
    prepare_csv_dataframe,
    route_output_path,
    section_table,
    with_output_extension,
    write_csv_and_aligned_view,
)


def save_backtest_report(metrics_by_model: Dict[str, Dict[str, object]], save_path: str) -> pd.DataFrame:
    df = pd.DataFrame(metrics_by_model).T
    df.index.name = "Model"
    if "Model" in df.columns:
        df.drop(columns=["Model"], inplace=True)
    if not df.empty and "Net_Return" in df.columns:
        df.sort_values(by=["Net_Return", "Sharpe"], ascending=[False, False], inplace=True)

    csv_df = df.reset_index()
    output_paths = write_csv_and_aligned_view(csv_df, save_path)

    md_path = with_output_extension(save_path, ".md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    display_df = prepare_csv_dataframe(csv_df)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Backtest Raporu\n\n")
        if not df.empty:
            best_model = df.index[0]
            best_end_capital = float(df.iloc[0].get("End_Capital", 0.0))
            best_profit_tl = float(df.iloc[0].get("Profit_TL", 0.0))
            buy_hold_end_capital = float(df.iloc[0].get("BuyHold_End_Capital", 0.0))
            handle.write("## Ozet\n\n")
            handle.write(
                bullet_list(
                    [
                        f"Lider strateji: `{best_model}`",
                        f"Donem sonu sermaye: `{best_end_capital:,.2f} TL`",
                        f"Net kar/zarar: `{best_profit_tl:,.2f} TL`",
                        f"Buy & Hold donem sonu: `{buy_hold_end_capital:,.2f} TL`",
                        f"Islem kalitesi: Profit Factor `{float(df.iloc[0].get('Profit_Factor', 0.0)):.4f}`, "
                        f"Expectancy `{float(df.iloc[0].get('Expectancy', 0.0)):.6f}`",
                    ]
                )
            )
            if len(df) >= 8:
                handle.write(
                    "\n\n"
                    + bullet_list(
                        [
                            "Multiple testing risk: cok sayida model/parametre denemesi raporlandigi icin lider strateji performansi temkinli yorumlanmalidir.",
                        ]
                    )
                )

        handle.write("\n\n## Getiri ve Risk Ozeti\n\n")
        handle.write(
            section_table(
                display_df,
                [
                    "Model", "Net_Return", "CAGR", "Annualized_Return", "Volatility",
                    "Sharpe", "Sortino", "Max_Drawdown", "Calmar",
                    "VaR_95", "CVaR_95", "Deflated_Sharpe", "Sharpe_Probabilistic_Score",
                    "Beats_BuyHold_NetReturn",
                ],
            )
        )

        handle.write("\n\n## Pozisyon ve Islem Ozeti\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Exposure", "Active_Bars", "Signal_Count", "Days_In_Market", "Trade_Count", "Turnover", "Avg_Holding_Period", "Win_Rate", "Avg_Trade_Return"],
            )
        )

        handle.write("\n\n## Zamanlama Varsayimlari\n\n")
        handle.write(
            bullet_list(
                [
                    "Prediction_Date: tahminin uretildigi karar tarihi.",
                    "Desired_Position karar barinda uretilir; Position bir sonraki bar getirisinde uygulanan yurutme pozisyonudur.",
                    "Execution_Date / Realized_Return_Date: bir onceki karar pozisyonunun getirisinin olculdugu bar.",
                    "Professional sinyal modu karar aninda gerceklesen fiyat bilgisini kullanmaz.",
                ]
            )
        )

        handle.write("\n\n## Islem Kalitesi ve Maliyet\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Profit_Factor", "Avg_Win", "Avg_Loss", "Expectancy", "Cost_Drag", "Commission_Drag", "Slippage_Drag", "Entry_Cost_Drag", "Exit_Cost_Drag", "Trade_Efficiency"],
            )
        )

        handle.write("\n\n## Leakage Guard\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Target_Semantics", "Execution_Lag", "Macro_Release_Lag", "Transaction_Costs", "Threshold_Config", "Validation_Protocol"],
            )
        )

        handle.write("\n\n## Sermaye Karsilastirmasi\n\n")
        handle.write(
            section_table(
                display_df,
                [
                    "Model", "Initial_Capital", "End_Capital", "Profit_TL",
                    "BuyHold_End_Capital", "BuyHold_Profit_TL", "BuyHold_Return",
                    "BuyHold_Sharpe", "BuyHold_VaR_95", "BuyHold_CVaR_95",
                ],
            )
        )

    print(f"[OK] Backtest raporu kaydedildi -> {output_paths['csv']}")
    print(f"[OK] Backtest markdown raporu kaydedildi -> {md_path}")
    return df


def save_fold_backtest_report(
    backtest_results: Dict[str, Dict[str, object]],
    save_path: str,
    initial_capital: float = 100000.0,
    trial_count: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    trial_count = len(backtest_results) if trial_count is None else max(1, int(trial_count))
    for model_name, result in backtest_results.items():
        curve = result.get("equity_curve")
        if curve is None or curve.empty or "Fold" not in curve.columns:
            continue
        trades = result.get("trades", pd.DataFrame())
        for fold_id, fold_curve in curve.groupby("Fold", sort=False):
            fold_curve = fold_curve.copy()
            fold_curve["Equity"] = (1.0 + fold_curve["Net_Return"].to_numpy(dtype=float)).cumprod()
            fold_curve["BuyHold_Equity"] = (1.0 + fold_curve["Realized_Return"].to_numpy(dtype=float)).cumprod()
            fold_trades = trades[trades["Fold"] == fold_id].copy() if isinstance(trades, pd.DataFrame) and "Fold" in trades.columns else pd.DataFrame()
            summary = summarize_backtest(
                {
                    "model_name": model_name,
                    "equity_curve": fold_curve,
                    "trades": fold_trades,
                },
                initial_capital=initial_capital,
                trial_count=trial_count,
            )
            summary["Fold"] = fold_id
            rows.append(summary)

    fold_df = pd.DataFrame(rows)
    if not fold_df.empty:
        fold_df.sort_values(by=["Model", "Fold"], inplace=True)

    fold_paths = write_csv_and_aligned_view(fold_df, save_path)

    worst_rows = []
    if not fold_df.empty:
        for model_name, model_df in fold_df.groupby("Model", sort=False):
            worst = model_df.sort_values(
                by=["Net_Return", "Sharpe", "Max_Drawdown"],
                ascending=[True, True, True],
            ).iloc[0].copy()
            worst["Worst_Fold_Rule"] = "min_net_return_then_min_sharpe"
            worst_rows.append(worst)
    worst_df = pd.DataFrame(worst_rows)

    worst_path = with_output_extension(save_path.replace(".csv", "_worst.csv"), ".csv")
    worst_paths = write_csv_and_aligned_view(worst_df, worst_path)

    md_path = with_output_extension(save_path, ".md")
    display_fold = prepare_csv_dataframe(fold_df)
    display_worst = prepare_csv_dataframe(worst_df)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Walk-Forward Fold Backtest Raporu\n\n")
        handle.write("## Fold Dagilimi\n\n")
        handle.write(section_table(display_fold, ["Model", "Fold", "Net_Return", "Sharpe", "Deflated_Sharpe", "VaR_95", "CVaR_95", "Max_Drawdown", "Exposure", "Turnover", "Avg_Holding_Period", "Trade_Count"]))
        handle.write("\n\n## Worst-Fold Ozeti\n\n")
        handle.write(section_table(display_worst, ["Model", "Fold", "Net_Return", "Sharpe", "Deflated_Sharpe", "VaR_95", "CVaR_95", "Max_Drawdown", "Exposure", "Turnover", "Avg_Holding_Period", "Worst_Fold_Rule"]))
        if len(backtest_results) >= 8:
            handle.write(
                "\n\n## Overfitting Kontrolu\n\n"
                + bullet_list(
                    [
                        "Multiple testing risk: walk-forward raporda cok sayida model/strateji karsilastiriliyor; worst-fold ve fold dagilimi lider performansi ile birlikte okunmalidir.",
                    ]
                )
            )

    print(f"[OK] Fold backtest dagilim raporu kaydedildi -> {fold_paths['csv']}")
    print(f"[OK] Worst-fold backtest raporu kaydedildi -> {worst_paths['csv']}")
    print(f"[OK] Fold backtest markdown raporu kaydedildi -> {md_path}")
    return fold_df, worst_df


def save_trade_logs(trades_by_model: Dict[str, pd.DataFrame], save_path: str) -> pd.DataFrame:
    frames = []
    for model_name, trades in trades_by_model.items():
        if trades is None or trades.empty:
            continue
        frame = trades.copy()
        if "Model" not in frame.columns:
            frame.insert(0, "Model", model_name)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_paths = write_csv_and_aligned_view(combined, save_path)
    print(f"[OK] Backtest trade log kaydedildi -> {output_paths['csv']}")
    return combined


def plot_equity_curves(
    equity_curves: Dict[str, pd.DataFrame],
    save_path: str,
    title: str,
    selected_models: set[str] | None = None,
) -> None:
    if plt is None:
        print("[WARN] matplotlib yok; backtest equity curve grafigi atlandi.")
        return
    save_path = route_output_path(save_path)
    curves = equity_curves
    if selected_models is not None:
        curves = {name: curve for name, curve in equity_curves.items() if name in selected_models}
    if not curves:
        return

    plt.figure(figsize=(16, 7))
    buy_hold_plotted = False
    for name, curve in curves.items():
        if curve.empty:
            continue
        if not buy_hold_plotted and "BuyHold_Equity" in curve.columns:
            plt.plot(curve["BuyHold_Equity"].to_numpy(), label="Buy & Hold", color="black", linewidth=2.2, alpha=0.85)
            buy_hold_plotted = True
        plt.plot(curve["Equity"].to_numpy(), label=name, linewidth=1.5, alpha=0.85)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Bar Index")
    plt.ylabel("Equity")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Backtest equity curve kaydedildi -> {save_path}")
