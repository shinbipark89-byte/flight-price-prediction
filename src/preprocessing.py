import pandas as pd
import numpy as np

INR_TO_USD = 0.012

OUTLIER_AIRLINES = [
    "Multiple carriers Premium economy",
    "Jet Airways Business",
    "Trujet",
    "Vistara Premium economy",
]

DROP_COLS_BASE = [
    "Route",
    "Additional_Info",
    "Duration",
    "Date_of_Journey",
    "Dep_Time",
    "Arrival_Time",
]

def parse_duration_to_min(x: str) -> int:
    x = str(x).lower()
    h = int(x.split("h")[0]) if "h" in x else 0
    m = int(x.split("h")[-1].split("m")[0]) if "m" in x else 0
    return h * 60 + m

def base_preprocess(df: pd.DataFrame, inr_to_usd: float = INR_TO_USD) -> pd.DataFrame:
    df = df.copy()

    # Price -> USD
    if "Price" in df.columns:
        df["Price_USD"] = df["Price"] * inr_to_usd

    # Total_Stops cleanup
    df["Total_Stops"] = df["Total_Stops"].fillna("0 stops")
    df["Total_Stops"] = df["Total_Stops"].replace({"non-stop": "0 stops"})
    df["Total_Stops"] = (
        df["Total_Stops"]
        .astype(str)
        .str.split()
        .str[0]
        .astype(int)
    )

    # Duration -> minutes
    df["Duration_min"] = df["Duration"].apply(parse_duration_to_min)

    # Date_of_Journey -> day/month
    journey_dt = pd.to_datetime(
        df["Date_of_Journey"].astype(str),
        format="%d/%m/%Y",
        errors="raise",
    )
    df["Journey_day"] = journey_dt.dt.day
    df["Journey_month"] = journey_dt.dt.month

    # Dep/Arrival time -> hour/min
    dep_time_str = df["Dep_Time"].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
    arr_time_str = df["Arrival_Time"].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]

    dep_time = pd.to_datetime(dep_time_str, format="%H:%M", errors="coerce")
    arr_time = pd.to_datetime(arr_time_str, format="%H:%M", errors="coerce")

    df["Dep_hour"] = dep_time.dt.hour
    df["Dep_min"] = dep_time.dt.minute
    df["Arr_hour"] = arr_time.dt.hour
    df["Arr_min"] = arr_time.dt.minute

    # Drop raw text/time columns
    df = df.drop(columns=[c for c in DROP_COLS_BASE if c in df.columns])

    # Remove outlier airlines
    if "Airline" in df.columns:
        df = df[~df["Airline"].isin(OUTLIER_AIRLINES)].copy()

    return df