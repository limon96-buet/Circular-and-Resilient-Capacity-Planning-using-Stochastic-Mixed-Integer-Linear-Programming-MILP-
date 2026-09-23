# =============================================================================
# 06_feature_engineering.py — Machine Learning Feature Engineering
# =============================================================================
# WHAT IT CREATES (from the raw collected data):
#
#   TEMPORAL FEATURES (calendar patterns):
#     - Month, Quarter, Year, IsRamadan, IsEidMonth
#     - Seasonal dummy variables (Q1/Q2/Q3/Q4)
#
#   LAG FEATURES (past values as predictors):
#     - RMG export t-1, t-2, t-3, t-6, t-12 (autoregressive features)
#     - Exchange rate lags, cotton price lags
#
#   ROLLING WINDOW STATISTICS (moving averages, trends):
#     - 3-month, 6-month, 12-month rolling mean and std dev
#     - Month-over-month (MoM) growth rates
#     - Year-over-year (YoY) growth rates
#
#   INTERACTION FEATURES (product of related variables):
#     - ExchangeRate × Oil_Price (cost-push interaction)
#     - Cotton_Price × ExchangeRate (raw material import cost in BDT)
#     - EU_Imports_total = HS61 + HS62
#
#   TECHNICAL INDICATORS (used in time-series ML):
#     - Exponential Weighted Moving Averages (EWMA)
#     - Bollinger Bands (deviation from rolling mean)
#     - Trend indicator (linear trend coefficient over 12M window)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df: pd.DataFrame,
                      target_col: str = "RMG_Export_USDmn") -> pd.DataFrame:
    """
    Add machine learning features to the merged dataset.

    Parameters
    ----------
    df         : pd.DataFrame — merged raw data indexed by 'YYYY-MM' Date strings
    target_col : str — name of the primary target variable

    Returns
    -------
    pd.DataFrame — enriched dataset with ML-ready features
    """
    df = df.copy()

    # Ensure datetime index for time operations
    df.index = pd.to_datetime(df.index + "-01")
    df.index.name = "Date"

    # ── 1. TEMPORAL FEATURES ─────────────────────────────────────────────────
    df["Month"]     = df.index.month
    df["Quarter"]   = df.index.quarter
    df["Year"]      = df.index.year
    df["MonthSin"]  = np.sin(2 * np.pi * df["Month"] / 12)  # circular encoding
    df["MonthCos"]  = np.cos(2 * np.pi * df["Month"] / 12)
    df["QuarterSin"]= np.sin(2 * np.pi * df["Quarter"] / 4)
    df["QuarterCos"]= np.cos(2 * np.pi * df["Quarter"] / 4)

    # Seasonal dummies (Q1-Q3 as dummies; Q4 is baseline)
    for q in range(1, 4):
        df[f"Q{q}_dummy"] = (df["Quarter"] == q).astype(int)

    # Peak export season: Aug-Oct (pre-winter orders for Northern Hemisphere)
    df["PeakSeason_dummy"] = df["Month"].isin([8, 9, 10]).astype(int)

    # Slow season: Jun-Jul (mid-year factory transitions)
    df["SlowSeason_dummy"] = df["Month"].isin([6, 7]).astype(int)

    # ── 2. LAG FEATURES (autoregressive) ─────────────────────────────────────
    # These are the MOST IMPORTANT features for RMG export forecasting
    # because export orders are placed 3-6 months in advance
    if target_col in df.columns:
        for lag in [1, 2, 3, 6, 12]:
            df[f"{target_col}_Lag{lag}"] = df[target_col].shift(lag)

        # Lag of log-transformed target (stabilizes variance)
        df[f"Log_{target_col}"]       = np.log1p(df[target_col])
        for lag in [1, 3, 6, 12]:
            df[f"Log_{target_col}_Lag{lag}"] = df[f"Log_{target_col}"].shift(lag)

    # Lags for key predictors
    for col in ["ExchangeRate_BDT_USD", "Cotton_Price_USDkg",
                "Oil_Price_USD_bbl", "Inflation_pct"]:
        if col in df.columns:
            for lag in [1, 3, 6]:
                df[f"{col}_Lag{lag}"] = df[col].shift(lag)

    # ── 3. ROLLING WINDOW STATISTICS ─────────────────────────────────────────
    if target_col in df.columns:
        for window in [3, 6, 12]:
            df[f"{target_col}_MA{window}"]      = df[target_col].rolling(window).mean()
            df[f"{target_col}_Std{window}"]     = df[target_col].rolling(window).std()
            df[f"{target_col}_EWMA{window}"]    = df[target_col].ewm(span=window).mean()

        # Bollinger Band: current value relative to 12M rolling mean
        ma12  = df[target_col].rolling(12).mean()
        std12 = df[target_col].rolling(12).std()
        df[f"{target_col}_BollingerPos"] = (df[target_col] - ma12) / (std12 + 1e-9)

    # Rolling stats for predictors
    for col in ["ExchangeRate_BDT_USD", "Cotton_Price_USDkg", "Remittances_USDmn"]:
        if col in df.columns:
            df[f"{col}_MA3"]  = df[col].rolling(3).mean()
            df[f"{col}_MA12"] = df[col].rolling(12).mean()

    # ── 4. GROWTH RATE FEATURES ───────────────────────────────────────────────
    if target_col in df.columns:
        df[f"{target_col}_MoM_Growth"] = df[target_col].pct_change(1) * 100
        df[f"{target_col}_YoY_Growth"] = df[target_col].pct_change(12) * 100
        df[f"{target_col}_QoQ_Growth"] = df[target_col].pct_change(3) * 100

    # Exchange rate growth (depreciation pressure)
    if "ExchangeRate_BDT_USD" in df.columns:
        df["FX_MoM_Change_pct"] = df["ExchangeRate_BDT_USD"].pct_change(1) * 100
        df["FX_YoY_Change_pct"] = df["ExchangeRate_BDT_USD"].pct_change(12) * 100

    # Inflation acceleration
    if "Inflation_pct" in df.columns:
        df["Inflation_Change_MoM"] = df["Inflation_pct"].diff(1)

    # ── 5. INTERACTION & DERIVED FEATURES ─────────────────────────────────────
    # Real exchange rate pressure: FX × Inflation gap
    if "ExchangeRate_BDT_USD" in df.columns and "Inflation_pct" in df.columns:
        df["RealFX_Pressure"] = df["ExchangeRate_BDT_USD"] * (1 + df["Inflation_pct"] / 100)

    # Cotton cost in BDT (key input cost for RMG)
    if "Cotton_Price_USDkg" in df.columns and "ExchangeRate_BDT_USD" in df.columns:
        df["Cotton_Price_BDTkg"] = df["Cotton_Price_USDkg"] * df["ExchangeRate_BDT_USD"]

    # Combined oil + cotton raw material index
    if "Cotton_Price_USDkg" in df.columns and "Oil_Price_USD_bbl" in df.columns:
        cotton_norm = df["Cotton_Price_USDkg"] / df["Cotton_Price_USDkg"].mean()
        oil_norm    = df["Oil_Price_USD_bbl"] / df["Oil_Price_USD_bbl"].mean()
        df["RawMaterial_CostIndex"] = (cotton_norm + oil_norm) / 2

    # EU total RMG imports (HS61 + HS62)
    if "EU_Imports_HS61" in df.columns and "EU_Imports_HS62" in df.columns:
        df["EU_Imports_Total"] = df["EU_Imports_HS61"] + df["EU_Imports_HS62"]

    # EU + US total demand
    if "EU_Imports_Total" in df.columns and "US_Imports" in df.columns:
        df["EUUS_Demand_Total"] = df["EU_Imports_Total"] + df["US_Imports"]

    # Remittances / Export ratio (measures economic balance)
    if "Remittances_USDmn" in df.columns and target_col in df.columns:
        df["Rem_to_Export_Ratio"] = df["Remittances_USDmn"] / (df[target_col] + 1e-9)

    # Composite disruption score (sum of all disruption indices)
    disruption_cols = []
    for c in ["COVID_dummy", "LaborUnrest_idx", "Flood_severity",
              "PowerOutage_idx", "PoliticalInstability_dummy"]:
        if c in df.columns:
            disruption_cols.append(c)
    if disruption_cols:
        df["Disruption_CompositeScore"] = df[disruption_cols].sum(axis=1)

    # ── 6. TREND INDICATOR (12-month linear trend coefficient) ───────────────
    def rolling_trend(series, window=12):
        """Compute slope of linear regression over rolling window."""
        results = []
        x = np.arange(window)
        for i in range(len(series)):
            if i < window - 1:
                results.append(np.nan)
            else:
                y = series.iloc[i - window + 1: i + 1].values
                if np.isnan(y).any():
                    results.append(np.nan)
                else:
                    slope = np.polyfit(x, y, 1)[0]
                    results.append(slope)
        return pd.Series(results, index=series.index)

    if target_col in df.columns:
        df[f"{target_col}_Trend12M"] = rolling_trend(df[target_col], 12)

    # ── 7. FORWARD TARGETS (for supervised learning) ──────────────────────────
    # These create the Y labels for different forecasting horizons
    if target_col in df.columns:
        for horizon in [1, 3, 6]:
            df[f"{target_col}_Lead{horizon}"] = df[target_col].shift(-horizon)
            df[f"{target_col}_Lead{horizon}_Growth"] = (
                df[target_col].pct_change(periods=-horizon) * -100
            )

    # ── 8. RESTORE DATE INDEX AS STRING ──────────────────────────────────────
    df.index = df.index.strftime("%Y-%m")
    df.index.name = "Date"

    print(f"  ✔ Feature engineering complete: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Summary of features by category ──────────────────────────────────────
    raw_cols  = [c for c in df.columns if not any(
        k in c for k in ["Lag", "MA", "Growth", "Lead", "Std", "EWMA",
                          "dummy", "Trend", "Score", "Index", "Ratio",
                          "Sin", "Cos", "Quarter", "Month", "Year", "Peak",
                          "Slow", "Pressure", "BDT", "Real", "Total",
                          "BollingerPos"])]
    lag_cols  = [c for c in df.columns if "Lag" in c]
    roll_cols = [c for c in df.columns if any(k in c for k in ["MA", "Std", "EWMA", "Bollinger"])]
    grwt_cols = [c for c in df.columns if "Growth" in c or "Change" in c or "Trend" in c]
    intx_cols = [c for c in df.columns if any(k in c for k in
                 ["Pressure", "BDT", "Real", "Total", "Ratio", "Score", "Index"])]
    lead_cols = [c for c in df.columns if "Lead" in c]
    temp_cols = [c for c in df.columns if any(k in c for k in
                 ["Month", "Quarter", "Year", "Sin", "Cos", "Season", "dummy"])]

    print(f"\n  Feature categories:")
    print(f"    Raw variables     : {len(raw_cols)}")
    print(f"    Temporal/Calendar : {len(temp_cols)}")
    print(f"    Lag features      : {len(lag_cols)}")
    print(f"    Rolling stats     : {len(roll_cols)}")
    print(f"    Growth rates      : {len(grwt_cols)}")
    print(f"    Interaction terms : {len(intx_cols)}")
    print(f"    Forward targets   : {len(lead_cols)}")

    return df


if __name__ == "__main__":
    # Demo: test feature engineering on synthetic data
    print("Feature engineering module — demo run\n")

    idx = pd.date_range("2013-01", "2024-12", freq="MS")
    demo = pd.DataFrame({
        "RMG_Export_USDmn"    : np.random.normal(2000, 300, len(idx)),
        "ExchangeRate_BDT_USD": np.linspace(78, 116, len(idx)),
        "Cotton_Price_USDkg"  : np.random.normal(1.8, 0.4, len(idx)),
        "Oil_Price_USD_bbl"   : np.random.normal(70, 20, len(idx)),
        "Inflation_pct"       : np.linspace(6, 10, len(idx)),
        "Remittances_USDmn"   : np.random.normal(1500, 200, len(idx)),
        "EU_Imports_HS61"     : np.random.normal(600, 100, len(idx)),
        "EU_Imports_HS62"     : np.random.normal(450, 80, len(idx)),
        "US_Imports"          : np.random.normal(350, 60, len(idx)),
        "COVID_dummy"         : [0]*84 + [0.8]*21 + [0.3]*12 + [0]*27,
        "LaborUnrest_idx"     : np.random.randint(0, 3, len(idx)),
        "Flood_severity"      : np.random.randint(0, 3, len(idx)),
        "PowerOutage_idx"     : np.random.randint(0, 2, len(idx)),
        "PoliticalInstability_dummy": np.random.randint(0, 1, len(idx)),
    }, index=idx.strftime("%Y-%m"))
    demo.index.name = "Date"

    enriched = engineer_features(demo, target_col="RMG_Export_USDmn")
    print(f"\nFinal dataset: {enriched.shape[0]} rows × {enriched.shape[1]} columns")
    print(enriched.columns.tolist())
