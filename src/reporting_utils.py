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
    "Fold",
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
    "CAGR": 6,
    "Volatility": 6,
    "Max_Drawdown": 6,
    "Calmar": 4,
    "Sortino": 4,
    "Exposure": 2,
    "Turnover": 2,
    "Win_Rate": 2,
    "Avg_Trade_Return": 6,
    "Avg_Holding_Period": 2,
    "Profit_Factor": 4,
    "Avg_Win": 6,
    "Avg_Loss": 6,
    "Expectancy": 6,
    "Cost_Drag": 6,
    "Commission_Drag": 6,
    "Slippage_Drag": 6,
    "Entry_Cost_Drag": 6,
    "Exit_Cost_Drag": 6,
    "Trade_Efficiency": 4,
    "BuyHold_Return": 6,
    "Initial_Capital": 2,
    "End_Capital": 2,
    "Profit_TL": 2,
    "BuyHold_End_Capital": 2,
    "BuyHold_Profit_TL": 2,
    "Entry_Price": 4,
    "Exit_Price": 4,
    "Gross_Return": 6,
    "Return_RMSE": 6,
    "Return_MAE": 6,
    "Pinball_Loss": 6,
    "Median_Pinball_Loss": 6,
    "Interval_Coverage": 2,
    "P10_P90_Coverage": 2,
    "Avg_Interval_Width": 4,
    "Winkler_Score": 4,
    "RMSE_vs_zero_return": 4,
    "Mandatory_Zero_Return_RMSE": 4,
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


def route_output_path(save_path: str) -> str:
    directory, filename = os.path.split(save_path)
    _, extension = os.path.splitext(filename)
    if not extension:
        return save_path

    extension_dir = extension.lstrip(".").lower()
    if os.path.basename(directory).lower() == extension_dir:
        return save_path
    return os.path.join(directory, extension_dir, filename)


def with_output_extension(save_path: str, extension: str) -> str:
    if not extension.startswith("."):
        extension = f".{extension}"

    directory, filename = os.path.split(save_path)
    stem, _ = os.path.splitext(filename)
    return route_output_path(os.path.join(directory, f"{stem}{extension}"))


def write_csv_and_aligned_view(df: pd.DataFrame, save_path: str) -> dict[str, str]:
    csv_path = route_output_path(save_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_df = prepare_csv_dataframe(df)
    csv_df.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")

    txt_path = with_output_extension(save_path, ".txt")
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    aligned_df = prepare_display_dataframe(csv_df)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(aligned_df.to_string(index=False))
        handle.write("\n")
    return {"csv": csv_path, "txt": txt_path}


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
