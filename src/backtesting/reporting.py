# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.reporting_utils import bullet_list, prepare_csv_dataframe, section_table, write_csv_and_aligned_view


def save_backtest_report(metrics_by_model: Dict[str, Dict[str, object]], save_path: str) -> pd.DataFrame:
    df = pd.DataFrame(metrics_by_model).T
    df.index.name = "Model"
    if "Model" in df.columns:
        df.drop(columns=["Model"], inplace=True)
    if not df.empty and "Net_Return" in df.columns:
        df.sort_values(by=["Net_Return", "Sharpe"], ascending=[False, False], inplace=True)

    csv_df = df.reset_index()
    write_csv_and_aligned_view(csv_df, save_path)

    md_path = save_path.replace(".csv", ".md")
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
                    ]
                )
            )

        handle.write("\n\n## Getiri ve Risk Ozeti\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Net_Return", "Annualized_Return", "Volatility", "Sharpe", "Max_Drawdown", "Calmar", "Beats_BuyHold_NetReturn"],
            )
        )

        handle.write("\n\n## Pozisyon ve Islem Ozeti\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Exposure", "Active_Bars", "Signal_Count", "Days_In_Market", "Trade_Count", "Turnover", "Win_Rate", "Avg_Trade_Return"],
            )
        )

        handle.write("\n\n## Sermaye Karsilastirmasi\n\n")
        handle.write(
            section_table(
                display_df,
                ["Model", "Initial_Capital", "End_Capital", "Profit_TL", "BuyHold_End_Capital", "BuyHold_Profit_TL", "BuyHold_Return", "BuyHold_Sharpe"],
            )
        )

    print(f"[OK] Backtest raporu kaydedildi -> {save_path}")
    print(f"[OK] Backtest markdown raporu kaydedildi -> {md_path}")
    return df


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
    write_csv_and_aligned_view(combined, save_path)
    print(f"[OK] Backtest trade log kaydedildi -> {save_path}")
    return combined


def plot_equity_curves(
    equity_curves: Dict[str, pd.DataFrame],
    save_path: str,
    title: str,
    selected_models: set[str] | None = None,
) -> None:
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
