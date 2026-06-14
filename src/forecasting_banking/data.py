from __future__ import annotations

from pathlib import Path
import pandas as pd


def clean_amount(series: pd.Series) -> pd.Series:
    """
    Clean currency amount strings (e.g. '$77.00', '($77.00)') to floats.
    """
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = cleaned.replace("", "0")
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def compute_daily_totals_streaming(
    csv_path: Path,
    *,
    chunksize: int = 250000,
) -> pd.DataFrame:
    """
    Aggregates transaction amounts by date using streaming to handle large CSV files.
    Returns a DataFrame sorted chronologically by date with columns ['date_only', 'amount'].
    """
    daily_totals: dict[str, float] = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if "amount" not in chunk.columns or "date" not in chunk.columns:
            raise ValueError("CSV must contain 'amount' and 'date' columns.")

        cleaned_amounts = clean_amount(chunk["amount"])
        # Extract date-only component (ignores time)
        dates = pd.to_datetime(chunk["date"], errors="coerce").dt.date

        temp_df = pd.DataFrame({"date_only": dates, "amount": cleaned_amounts})
        temp_df = temp_df.dropna(subset=["date_only"])

        # Aggregate within the chunk
        grouped = temp_df.groupby("date_only")["amount"].sum()
        for dt, val in grouped.items():
            # Keep string representation of date for mapping consistency or use date objects
            dt_str = str(dt)
            daily_totals[dt_str] = daily_totals.get(dt_str, 0.0) + val

    # Convert to DataFrame
    df_daily = pd.DataFrame(list(daily_totals.items()), columns=["date_only", "amount"])
    df_daily["date_only"] = pd.to_datetime(df_daily["date_only"]).dt.date
    # Sort chronologically by date
    df_daily = df_daily.sort_values("date_only").reset_index(drop=True)
    return df_daily


def generate_day_index_timeline(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a chronological 'day_index' (0, 1, 2...) for the daily totals.
    """
    df_out = df_daily.copy()
    df_out["day_index"] = range(len(df_out))
    return df_out
