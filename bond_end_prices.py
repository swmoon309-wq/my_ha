#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    # 한글 주석: 명령행 인자를 정의하고 파싱한다.
    parser = argparse.ArgumentParser(
        description=(
            "Download daily closing prices for a ticker, list the four-month window "
            "around a reference date, compute the average, and store the results in Excel."
        )
    )
    parser.add_argument(
        "ticker",
        help="Instrument ticker symbol understood by Yahoo Finance (e.g., 'IEF').",
    )
    parser.add_argument(
        "as_of",
        help="Reference date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Excel file path. Defaults to '{ticker}_{as_of}_daily_closes.xlsx'.",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    # 한글 주석: YYYY-MM-DD 형식의 문자열을 날짜 객체로 변환한다.
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{value}': {exc}") from exc


def compute_window_bounds(target: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    # 한글 주석: 기준일 기준 앞뒤 두 달의 기간을 계산한다.
    target_ts = pd.Timestamp(target)
    start_ts = target_ts - pd.DateOffset(months=2)
    end_ts = target_ts + pd.DateOffset(months=2)
    return start_ts.normalize(), end_ts.normalize()


def download_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # 한글 주석: 야후 파이낸스에서 지정된 기간의 시세를 내려받는다.
    # Add one day to the end so the range is inclusive when passed to yfinance.
    yf_end = (end + pd.Timedelta(days=1)).date()
    data = yf.download(
        ticker,
        start=start.date(),
        end=yf_end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if data.empty:
        raise SystemExit(
            f"No price data was returned for ticker '{ticker}' between {start.date()} and {end.date()}."
        )
    return data


def extract_close_series(history: pd.DataFrame, ticker: str) -> pd.Series:
    # 한글 주석: 멀티 인덱스 형태를 고려하여 종가 시리즈만 추출한다.
    # Yahoo Finance may return a MultiIndex with ('Price', 'Ticker') levels.
    if isinstance(history.columns, pd.MultiIndex):
        if ("Close", ticker) in history.columns:
            close = history[("Close", ticker)]
        else:
            try:
                close_df = history.xs("Close", axis=1, level=0)
            except KeyError as exc:
                raise SystemExit("Downloaded data does not include 'Close' prices.") from exc
            if isinstance(close_df, pd.Series):
                close = close_df
            elif ticker in close_df.columns:
                close = close_df[ticker]
            elif close_df.shape[1] == 1:
                close = close_df.squeeze(axis=1)
            else:
                available = ", ".join(map(str, close_df.columns))
                raise SystemExit(
                    f"Unable to find close prices for ticker '{ticker}'. Available columns: {available}"
                )
    else:
        if "Close" not in history.columns:
            raise SystemExit("Downloaded data does not include 'Close' prices.")
        close = history["Close"]

    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.to_datetime(history.index)
    close = close.loc[~close.index.duplicated(keep="last")].dropna()
    if close.empty:
        raise SystemExit("Close price series contains no numeric data.")
    return close.sort_index()


def build_daily_frame(
    close: pd.Series, target: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    # 한글 주석: 4개월 기간 내 일별 종가 데이터프레임을 구축하고 구간 라벨을 붙인다.
    windowed = close.loc[(close.index >= start) & (close.index <= end)]
    if windowed.empty:
        raise SystemExit(
            f"No daily close prices available between {start.date()} and {end.date()}."
        )
    df = windowed.to_frame(name="Close").reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    before_label = f"Before {target.date().isoformat()}"
    after_label = f"On/After {target.date().isoformat()}"
    df["Segment"] = df["Date"].apply(
        lambda d: before_label if d < target.date() else after_label
    )
    return df


def write_excel(daily_df: pd.DataFrame, average: float, output_path: Path) -> None:
    # 한글 주석: 일별 종가와 평균을 엑셀 파일로 저장한다.
    summary = pd.DataFrame(
        {
            "Metric": ["Four-month average close"],
            "Value": [average],
        }
    )
    output_path = output_path.with_suffix(".xlsx")
    with pd.ExcelWriter(output_path) as writer:
        daily_df.to_excel(writer, sheet_name="Daily Closes", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)


def main() -> None:
    # 한글 주석: 전체 실행 흐름을 제어하는 진입점.
    args = parse_args()
    target = parse_date(args.as_of)
    start_ts, end_ts = compute_window_bounds(target)
    history = download_history(args.ticker, start_ts, end_ts)
    history.index = pd.to_datetime(history.index)
    close = extract_close_series(history, args.ticker)
    daily_df = build_daily_frame(close, pd.Timestamp(target), start_ts, end_ts)
    four_month_average = daily_df["Close"].mean()

    output_path = (
        args.output
        if args.output
        else Path(f"{args.ticker}_{target.isoformat()}_daily_closes.xlsx")
    )
    write_excel(daily_df, four_month_average, output_path)

    print(f"Ticker: {args.ticker}")
    print(f"Reference date: {target.isoformat()}")
    print(f"Window: {start_ts.date().isoformat()} to {end_ts.date().isoformat()}")
    print(f"Excel output: {output_path.resolve()}")
    print("\nDaily closing prices:")
    for _, row in daily_df.iterrows():
        print(f"{row['Date'].isoformat()} ({row['Segment']}): {row['Close']:,.4f}")
    print(f"\nFour-month average close: {four_month_average:,.4f}")


if __name__ == "__main__":
    main()
