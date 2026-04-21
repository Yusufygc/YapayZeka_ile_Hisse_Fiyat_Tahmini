from __future__ import annotations

import os
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_FLOAT_DIGITS = 4
INTEGER_COLUMNS = {
    "Sira",
    "Sıra",
    "Active_Bars",
    "Signal_Count",
    "Days_In_Market",
    "Trade_Count",
    "Holding_Period",
}
BOOLEAN_COLUMNS = {
    "Beats_Benchmark_RMSE",
    "Beats_BuyHold_NetReturn",
}
FLOAT_DIGITS_BY_COLUMN = {
    "MAE": 4,
    "RMSE": 4,
    "MAPE": 6,
    "Dir_Acc": 2,
    "Sharpe": 4,
    "Hit_Rate": 2,
    "Neutral_Rate": 2,
    "BuyHold_Sharpe": 4,
    "RMSE_vs_benchmark": 4,
    "DirAcc_vs_benchmark": 2,
    "Sharpe_excess_vs_buy_hold": 4,
    "Composite_Score": 4,
    "Net_Return": 6,
    "Annualized_Return": 6,
    "Volatility": 6,
    "Max_Drawdown": 6,
    "Calmar": 4,
    "Exposure": 2,
    "Turnover": 2,
    "Win_Rate": 2,
    "Avg_Trade_Return": 6,
    "BuyHold_Return": 6,
    "Initial_Capital": 2,
    "End_Capital": 2,
    "Profit_TL": 2,
    "BuyHold_End_Capital": 2,
    "BuyHold_Profit_TL": 2,
    "Entry_Price": 4,
    "Exit_Price": 4,
    "Gross_Return": 6,
}


def prepare_csv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    csv_df = df.copy()
    for column in csv_df.columns:
        if column in BOOLEAN_COLUMNS:
            csv_df[column] = csv_df[column].map(_format_bool)
            continue
        if column in INTEGER_COLUMNS:
            csv_df[column] = csv_df[column].apply(_format_int)
            continue
        if pd.api.types.is_numeric_dtype(csv_df[column]):
            digits = FLOAT_DIGITS_BY_COLUMN.get(column, DEFAULT_FLOAT_DIGITS)
            csv_df[column] = csv_df[column].round(digits)
    return csv_df


def prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].apply(lambda value: format_cell(value, column))
    return display_df.fillna("-")


def write_csv_and_aligned_view(df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_df = prepare_csv_dataframe(df)
    csv_df.to_csv(save_path, sep=";", index=False, encoding="utf-8-sig")

    txt_path = save_path.replace(".csv", ".txt")
    aligned_df = prepare_display_dataframe(csv_df)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(aligned_df.to_string(index=False))
        handle.write("\n")


def section_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if not available:
        return ""
    return _to_markdown_table(prepare_display_dataframe(df.loc[:, available]))


def format_cell(value: object, column: str | None = None) -> str:
    if pd.isna(value):
        return "-"
    if column in BOOLEAN_COLUMNS:
        return _format_bool(value)
    if column in INTEGER_COLUMNS:
        return _format_int(value)
    if isinstance(value, bool):
        return _format_bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        digits = FLOAT_DIGITS_BY_COLUMN.get(column or "", DEFAULT_FLOAT_DIGITS)
        if column and (column.endswith("_Capital") or column.endswith("_TL")):
            return f"{value:,.2f}"
        return f"{value:.{digits}f}"
    return str(value)


def bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _format_bool(value: object) -> str:
    if isinstance(value, str):
        return value
    return "Yes" if bool(value) else "No"


def _format_int(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(int(round(float(value))))


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    separator = ["---"] * len(headers)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
