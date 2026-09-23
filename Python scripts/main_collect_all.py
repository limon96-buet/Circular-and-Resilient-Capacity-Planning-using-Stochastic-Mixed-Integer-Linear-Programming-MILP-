# =============================================================================
# main_collect_all.py — Bangladesh RMG ML Dataset: Master Data Collection Script
# =============================================================================
#
# USAGE:
#   python main_collect_all.py
#
# PREREQUISITES:
#   1. Install dependencies:
#      pip install -r requirements.txt
#
#   2. (Optional) Add API keys to config.py:
#      - FRED_API_KEY     → free from https://fred.stlouisfed.org/docs/api/api_key.html
#      - COMTRADE_API_KEY → free from https://comtradedeveloper.un.org/
#      Without keys: historically documented fallback data is used automatically.
#
# WHAT IT PRODUCES:
#   Bangladesh_RMG_ML_Dataset.xlsx  with 9 sheets:
#
#   Sheet 1: Raw_Variables          — 17 core variables (your template + 2 new)
#   Sheet 2: ML_Dataset             — full feature matrix (100+ ML features)
#   Sheet 3: Dummy_Variables        — all disruption + policy proxies with sources
#   Sheet 4: Engineered_Features    — lag, rolling, growth, interaction features
#   Sheet 5: Trade_Data             — RMG exports to World, EU, USA
#   Sheet 6: Commodities            — cotton & oil (World Bank Pink Sheet)
#   Sheet 7: WorldBank_Data         — CPI, GDP, lending rate, reserves
#   Sheet 8: FRED_Data              — exchange rate, oil, US indicators
#   Sheet 9: DataDictionary         — all columns: units, descriptions, sources
#
# PERIOD: 2005-01 to 2025-12  (252 months, 21 years)
# Rationale for extension to 2005:
#   - Captures 2006 wage crisis (Tk 1,662), 2010 wage crisis (Tk 3,000)
#   - Captures 2007-08 commodity super-cycle (oil $147/bbl)
#   - Captures 2011 cotton price crisis ($4.50/kg — worst for RMG since 1990s)
#   - Captures 2008-09 GFC and Bangladesh's resilience (exports held up)
#   - 252 months gives ML models 4× more training data than the original 144
#
# NEW VARIABLES (v2, aligned with IVPF DEMATEL causal analysis):
#
#   GovernmentPolicy_idx  (DEMATEL F16 — #2 net cause, D−R = 1.478)
#   ─────────────────────────────────────────────────────────────────
#   Scale 0-3. Captures REGULATORY/POLICY shifts DISTINCT from election
#   disruption (PoliticalInstability_dummy):
#     - Minimum wage board decisions (2006, 2010, 2013, 2018, 2023)
#     - EU trade policy (CBAM Oct 2023, CSDDD Jun 2024, Textile Strategy 2022)
#     - Bangladesh Bank financial policy (9% cap Apr 2020, SMART rate Jun 2023)
#     - Factory safety compliance deadlines (Accord 2013, RSC 2021)
#     - Bangladesh Labour Act amendments
#
#   LendingRate_pct  (DEMATEL F14 — #3 net cause, Finance driver)
#   ─────────────────────────────────────────────────────────────────
#   Monthly weighted average bank lending rate (%).
#   Source: Bangladesh Bank Annual Reports / World Bank FR.INR.LEND
#   Key policy events encoded at monthly precision:
#     - 14.5% (2005) → 9.5% (2019): market-determined decline
#     - 9.0% (Apr 2020 – May 2023): BB imposed cap (BRPD Circular No. 03/2020)
#     - 11.0% → 14.0% (Jun 2023 – Dec 2024): SMART rate (T-bill + 3.5% spread)
# =============================================================================

import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Import configuration ──────────────────────────────────────────────────────
try:
    from config import (FRED_API_KEY, COMTRADE_API_KEY,
                        START_DATE, END_DATE, START_YEAR, END_YEAR,
                        BANGLADESH_ISO3, OUTPUT_FILE)
except ImportError:
    print("⚠  config.py not found. Using default settings.")
    FRED_API_KEY       = "6454d699f35a71d34510d8ac14fb32e4"
    COMTRADE_API_KEY   = "YOUR_COMTRADE_API_KEY_HERE"
    START_DATE         = "2005-01-01"
    END_DATE           = "2025-12-31"
    START_YEAR         = 2005
    END_YEAR           = 2025
    BANGLADESH_ISO3    = "BGD"
    OUTPUT_FILE        = "Bangladesh_RMG_ML_Dataset.xlsx"

# ── Import collector modules ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s01_fred_data          import collect_fred_data, get_fred_fallback_data
from s02_worldbank_data     import collect_worldbank_data, get_worldbank_fallback_data
from s03_pinksheet_data     import collect_pinksheet_data, get_pinksheet_fallback
from s04_comtrade_data      import collect_comtrade_data, get_rmg_fallback_data
from s05_dummy_variables    import build_dummy_variables
from s06_feature_engineering import engineer_features


# =============================================================================
# DATA DICTIONARY
# =============================================================================
DATA_DICTIONARY = [
    # ── Core template variables ───────────────────────────────────────────────
    ("Date",                        "YYYY-MM",
     "Month identifier (2005-01 to 2025-12)",
     "Generated"),

    ("RMG_Export_USDmn",            "USD million",
     "Bangladesh total RMG exports (HS61 knitted + HS62 woven), monthly",
     "EPB monthly reports (http://epb.gov.bd/site/view/report) / "
     "UN Comtrade reporter BGD HS 61+62 exports"),

    ("ExchangeRate_BDT_USD",        "BDT per 1 USD",
     "Monthly average official exchange rate (BDT depreciated from 64 in 2005 to 122 in 2025)",
     "FRED series EXBNUS / Bangladesh Bank (https://www.bb.org.bd)"),

    ("CPI",                         "Index (2010=100)",
     "Consumer Price Index, general (Bangladesh BBS)",
     "Bangladesh Bureau of Statistics / World Bank indicator FP.CPI.TOTL"),

    ("Inflation_pct",               "%",
     "Year-on-year CPI inflation rate",
     "World Bank FP.CPI.TOTL.ZG / IMF World Economic Outlook"),

    ("Cotton_Price_USDkg",          "USD per kg",
     "Cotton A Index, monthly average — CRITICAL: 2011 spike to $4.50/kg (highest since 1990s)",
     "World Bank Pink Sheet (CMO-Historical-Data-Monthly.xlsx)"),

    ("Cotton_Price_centsLb",        "US cents per pound",
     "Cotton A Index converted (1 USD/kg = 45.36 cents/lb)",
     "World Bank Pink Sheet (derived)"),

    ("Oil_Price_USD_bbl",           "USD per barrel",
     "Crude oil simple average (Brent/WTI/Dubai); key 2008 peak at $133/bbl, 2020 crash to $20/bbl",
     "World Bank Pink Sheet / FRED DCOILBRENTEU"),

    ("Remittances_USDmn",           "USD million",
     "Monthly inward remittances (grew from $540mn/mo in 2005 to $2,300mn in 2025)",
     "Bangladesh Bank monthly data / World Bank BX.TRF.PWKR.CD.DT"),

    ("EU_Imports_HS61",             "USD million",
     "EU27 imports from Bangladesh — HS Chapter 61 (knitted/crocheted apparel)",
     "UN Comtrade / Eurostat COMEXT database"),

    ("EU_Imports_HS62",             "USD million",
     "EU27 imports from Bangladesh — HS Chapter 62 (woven apparel)",
     "UN Comtrade / Eurostat COMEXT database"),

    ("US_Imports",                  "USD million",
     "US apparel imports from Bangladesh (HS61 + HS62)",
     "UN Comtrade / OTEXA (https://otexa.trade.gov/msrpoint.htm)"),

    ("COVID_dummy",                 "0–1 graded scale",
     "COVID-19 disruption severity for RMG sector (0=none, 1=peak Apr2020/Apr2021)",
     "WHO Bangladesh Situation Reports / BGMEA factory-closure announcements"),

    ("PowerOutage_idx",             "0=none, 1=mild, 2=severe",
     "Monthly load-shedding severity (2022–23 energy crisis = 2; regular summer = 1)",
     "BPDB Annual Reports / Daily Star load-shedding archives"),

    ("LaborUnrest_idx",             "0–3 scale",
     "RMG sector labor disputes (0=none, 3=sector-wide; key: Oct2013, Nov2018, Oct-Nov2023)",
     "ILO, Human Rights Watch, BGMEA press releases, news archives"),

    ("Flood_severity",              "0–3 scale",
     "Flood severity (0=none, 3=national catastrophe; key: Jul-Aug2020 = 37% inundated)",
     "BWDB, FFWC Flood Reports, ReliefWeb, EM-DAT disaster database"),

    ("PoliticalInstability_dummy",  "0 or 1",
     "Hartal/blockade/election month (general elections 2008,2014,2018,2024; blockades 2015)",
     "Bangladesh Election Commission, Daily Star, Human Rights Watch"),

    # ── NEW: DEMATEL F16 Government Policy (D−R = 1.478, #2 net cause) ───────
    ("GovernmentPolicy_idx",        "0–3 scale",
     "Regulatory/policy shifts DISTINCT from election instability. "
     "0=none; 1=minor review; 2=significant enacted policy; "
     "3=structural shift (wage increase, EU CBAM, BB 9% cap+SMART rate). "
     "Level-3 months: Oct2006(wage Tk1662), Nov2010(wage Tk3000), "
     "Dec2013(wage Tk5300), Dec2018(wage Tk8000), Apr2020(9%cap+COVID stimulus), "
     "Oct2023(EU CBAM), Nov2023(wage Tk12500).",
     "BGMEA Annual Reports; EU Official Journal (CBAM Reg.2023/956, CSDDD 2024/1760); "
     "Bangladesh Bank BRPD Circular No.03/2020; ILO Sustainability Compact "
     "Progress Reports; Bangladesh Gazette (wage board notifications)"),

    # ── NEW: DEMATEL F14 Finance (#3 net cause) ───────────────────────────────
    ("LendingRate_pct",             "%",
     "Weighted average bank lending rate (%). Key policy regimes: "
     "14.5% (2005, market rate) → 9.5% (2019) → 9.0% cap (Apr2020–May2023, "
     "BB BRPD No.03/2020) → SMART rate 11–14% (Jun2023+, 6M T-bill+3.5% spread). "
     "The 9% cap was a major RMG sector cost relief during COVID.",
     "Bangladesh Bank Annual Reports; World Bank FR.INR.LEND; "
     "BB Monetary Policy Statements 2020–2024; "
     "https://www.bb.org.bd/en/index.php/econdata/ir"),

    # ── Macro context variables ───────────────────────────────────────────────
    ("GDP_Growth_pct",              "%",
     "Real GDP growth rate (annual, interpolated monthly)",
     "World Bank NY.GDP.MKTP.KD.ZG"),

    ("ForeignReserves_USDb",        "USD billion",
     "Official foreign exchange reserves (peaked $46bn in 2021, fell to $21bn by 2023)",
     "Bangladesh Bank / World Bank FI.RES.TOTL.CD"),

    ("GDP_CurrentUSDbn",            "USD billion",
     "GDP in current USD (grew from $60bn in 2005 to $460bn in 2024)",
     "World Bank (interpolated annual to monthly)"),

    # ── Additional dummies ───────────────────────────────────────────────────
    ("RanaPlaza_dummy",             "0 or 1",
     "Rana Plaza disaster aftermath (Apr–Jun 2013; 1,134 deaths; sector-transforming event)",
     "ILO, BGMEA, international news archives"),

    ("GSP_Suspension_dummy",        "0 or 1",
     "US GSP suspension for Bangladesh (Jun–Dec 2013; mainly non-RMG, but confidence effect)",
     "USTR Federal Register, June 2013"),

    ("Eid_month_dummy",             "0 or 1",
     "Islamic Eid holiday month (Eid ul-Fitr and Eid ul-Adha; remittance demand spikes)",
     "Islamic calendar (approximate Gregorian months)"),

    # ── Engineered ML features ────────────────────────────────────────────────
    ("RMG_Export_USDmn_Lag1",       "USD million",
     "RMG export 1 month ago (strongest autoregressive predictor)",
     "Derived from RMG_Export_USDmn"),

    ("RMG_Export_USDmn_Lag3",       "USD million",
     "RMG export 3 months ago (order lead-time signal)",
     "Derived"),

    ("RMG_Export_USDmn_Lag12",      "USD million",
     "RMG export 12 months ago (year-over-year baseline)",
     "Derived"),

    ("RMG_Export_USDmn_MA12",       "USD million",
     "12-month rolling mean (removes seasonality, shows trend)",
     "Derived"),

    ("RMG_Export_USDmn_YoY_Growth", "%",
     "Year-on-year RMG export growth rate",
     "Derived: pct_change(12)"),

    ("Cotton_Price_BDTkg",          "BDT per kg",
     "Cotton A Index in local currency (key input cost in BDT for RMG sector)",
     "Derived: Cotton_Price_USDkg × ExchangeRate_BDT_USD"),

    ("LendingRate_RealPct",         "%",
     "Real lending rate = LendingRate_pct − Inflation_pct (financial tightness indicator)",
     "Derived: LendingRate_pct − Inflation_pct"),

    ("GovernmentPolicy_LaborUnrest_Interaction", "0–12",
     "Interaction of policy and labor unrest (captures combined regulatory-labor shock)",
     "Derived: GovernmentPolicy_idx × LaborUnrest_idx"),

    ("Disruption_CompositeScore",   "0–12",
     "Sum of all disruption indices (COVID + Labor + Flood + Power + Political + Policy)",
     "Derived"),

    ("RMG_Export_USDmn_Lead1",      "USD million",
     "FORWARD TARGET: next month RMG export (1-step-ahead supervised learning label)",
     "Derived: shift(-1)"),

    ("RMG_Export_USDmn_Lead3",      "USD million",
     "FORWARD TARGET: 3-month-ahead RMG export (quarterly forecasting label)",
     "Derived: shift(-3)"),
]


# =============================================================================
# STEP 1: COLLECT ALL DATA
# =============================================================================
def collect_all_data() -> dict:
    """
    Run all data collection scripts. Uses live API when key is available,
    otherwise falls back to historically documented values automatically.
    """
    print("=" * 70)
    print("  BANGLADESH RMG ML DATASET  (v2 — IVPF DEMATEL aligned)")
    print(f"  Period  : {START_DATE} to {END_DATE}  (252 months)")
    print(f"  New vars: GovernmentPolicy_idx (F16) + LendingRate_pct (F14)")
    print(f"  Output  : {OUTPUT_FILE}")
    print("=" * 70)

    collected = {}

    # [1] FRED
    print("\n[1/5] FRED Macro Data (Exchange Rate, Oil, US Demand)")
    print("-" * 50)
    if FRED_API_KEY and FRED_API_KEY != "YOUR_FRED_API_KEY_HERE":
        collected["fred"] = collect_fred_data(FRED_API_KEY, START_DATE, END_DATE)
    else:
        print("  ℹ  No FRED key — using documented fallback data.")
        collected["fred"] = get_fred_fallback_data(START_DATE, END_DATE)

    # [2] World Bank  (includes LendingRate_pct — DEMATEL F14)
    print("\n[2/5] World Bank (CPI, Inflation, GDP, LendingRate [F14-Finance])")
    print("-" * 50)
    try:
        collected["worldbank"] = collect_worldbank_data(START_YEAR, END_YEAR, BANGLADESH_ISO3)
    except Exception as e:
        print(f"  !! World Bank API error: {e} → using fallback.")
        collected["worldbank"] = get_worldbank_fallback_data(START_YEAR, END_YEAR)

    # [3] Pink Sheet
    print("\n[3/5] World Bank Pink Sheet (Cotton, Oil — incl. 2011 cotton crisis)")
    print("-" * 50)
    try:
        collected["pinksheet"] = collect_pinksheet_data(START_DATE, END_DATE)
    except Exception as e:
        print(f"  !! Pink Sheet error: {e} → using fallback.")
        collected["pinksheet"] = get_pinksheet_fallback(START_DATE, END_DATE)

    # [4] Trade data
    print("\n[4/5] Trade Data — RMG Exports (UN Comtrade / EPB/BGMEA)")
    print("-" * 50)
    if COMTRADE_API_KEY and COMTRADE_API_KEY != "YOUR_COMTRADE_API_KEY_HERE":
        collected["trade"] = collect_comtrade_data(COMTRADE_API_KEY, START_YEAR, END_YEAR)
    else:
        print("  ℹ  No Comtrade key — using EPB/BGMEA documented data.")
        collected["trade"] = get_rmg_fallback_data(START_YEAR, END_YEAR)

    # [5] Dummy variables (includes GovernmentPolicy_idx — DEMATEL F16)
    print("\n[5/5] Disruption + Policy Proxies (incl. GovernmentPolicy_idx [F16])")
    print("-" * 50)
    collected["dummies"] = build_dummy_variables(START_DATE, END_DATE)

    return collected


# =============================================================================
# STEP 2: MERGE
# =============================================================================
def merge_all_data(collected: dict) -> pd.DataFrame:
    """
    Merge all collected DataFrames into a single ML-ready DataFrame.
    Left join on the 252-month backbone (2005-01 to 2025-12).
    Forward-fill gaps ≤ 3 months for slow-moving macro variables.
    """
    print("\n[Merging] Combining all data sources...")

    monthly_index = pd.date_range(START_DATE, END_DATE, freq="MS")
    base = pd.DataFrame(index=monthly_index.strftime("%Y-%m"))
    base.index.name = "Date"

    merge_order = ["trade", "fred", "worldbank", "pinksheet", "dummies"]
    df = base.copy()

    for key in merge_order:
        if key in collected:
            src      = collected[key].copy()
            new_cols = [c for c in src.columns if c not in df.columns]
            df       = df.join(src[new_cols], how="left")
            print(f"  ✔ Merged '{key}': +{len(new_cols)} cols  "
                  f"(total = {df.shape[1]})")

    # ── Column resolution ─────────────────────────────────────────────────────
    if "Oil_Price_USD_bbl" not in df.columns and "Oil_Brent_USD_bbl_PS" in df.columns:
        df["Oil_Price_USD_bbl"] = df["Oil_Brent_USD_bbl_PS"]

    if "Cotton_Price_USDkg" not in df.columns and "Cotton_Price_centsLb" in df.columns:
        df["Cotton_Price_USDkg"] = df["Cotton_Price_centsLb"] * 2.2046 / 100

    # ── Forward-fill short gaps in slow-moving macro variables ────────────────
    for col in ["CPI", "Inflation_pct", "GDP_Growth_pct",
                "ForeignReserves_USDb", "ExchangeRate_BDT_USD", "LendingRate_pct"]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=3).bfill(limit=3)

    print(f"\n  ✔ Merged dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# STEP 3: FEATURE ENGINEERING  (adds real lending rate + policy interaction)
# =============================================================================
def add_extra_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features specific to the DEMATEL F14/F16 new variables.
    Called AFTER engineer_features() from s06.
    """
    df = df.copy()

    # Real lending rate = nominal rate − inflation (financial tightness)
    if "LendingRate_pct" in df.columns and "Inflation_pct" in df.columns:
        df["LendingRate_RealPct"] = df["LendingRate_pct"] - df["Inflation_pct"]

    # Lag features for the two new variables
    if "LendingRate_pct" in df.columns:
        for lag in [1, 3, 6, 12]:
            df[f"LendingRate_pct_Lag{lag}"] = df["LendingRate_pct"].shift(lag)

    if "GovernmentPolicy_idx" in df.columns:
        for lag in [1, 3, 6, 12]:
            df[f"GovernmentPolicy_idx_Lag{lag}"] = df["GovernmentPolicy_idx"].shift(lag)
        # Rolling policy intensity (captures sustained regulatory pressure)
        df["GovernmentPolicy_MA6"] = df["GovernmentPolicy_idx"].rolling(6).mean()

    # Interaction: GovernmentPolicy × LaborUnrest (co-occurring shocks)
    if "GovernmentPolicy_idx" in df.columns and "LaborUnrest_idx" in df.columns:
        df["GovernmentPolicy_LaborUnrest_Interaction"] = (
            df["GovernmentPolicy_idx"] * df["LaborUnrest_idx"]
        )

    # Extended composite disruption score  (now includes GovernmentPolicy)
    disruption_cols = [c for c in ["COVID_dummy", "LaborUnrest_idx", "Flood_severity",
                                    "PowerOutage_idx", "PoliticalInstability_dummy",
                                    "GovernmentPolicy_idx"] if c in df.columns]
    if disruption_cols:
        df["Disruption_CompositeScore"] = df[disruption_cols].sum(axis=1)

    return df


# =============================================================================
# STEP 4: SAVE TO EXCEL
# =============================================================================
def save_to_excel(df_raw: pd.DataFrame,
                  df_ml:  pd.DataFrame,
                  collected: dict,
                  output_path: str) -> None:
    """
    Save to a professionally formatted Excel workbook with 9 sheets,
    color-coded by variable category.
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    print(f"\n[Saving] Writing to '{output_path}'...")

    # ── Color scheme by DEMATEL/variable category ─────────────────────────────
    COL_COLORS = {
        'target'    : ('1A7A3B', 'D9F2E0'),   # green  — RMG exports
        'macro'     : ('1F5FA6', 'DDEAF7'),   # blue   — macro (CPI, FX, etc.)
        'finance'   : ('0D5C8C', 'CCE5F5'),   # dark blue — LendingRate (F14)
        'demand'    : ('B5880F', 'FFF5CC'),   # yellow — EU/US imports
        'disruption': ('C85A00', 'FFE8D0'),   # orange — operational disruptions
        'policy'    : ('7B1FA2', 'F3E5F5'),   # purple — GovernmentPolicy (F16)
        'engineered': ('5C5C5C', 'F5F5F5'),   # gray   — lag/rolling/interaction
        'other'     : ('333333', 'FFFFFF'),
    }

    def get_cat(col):
        if 'RMG_Export' in col and not any(
                k in col for k in ['Lag','MA','Growth','Lead','Std','EWMA','Trend']):
            return 'target'
        if 'GovernmentPolicy' in col:         return 'policy'
        if 'LendingRate' in col:              return 'finance'
        if any(k in col for k in ['Exchange','CPI','Inflation','Cotton','Oil',
                                   'Remit','GDP','Reserve','Crop','Population',
                                   'Unemployment','FDI','Trade','Manufacturing']):
            return 'macro'
        if any(k in col for k in ['EU_','US_Import']):
            return 'demand'
        if any(k in col for k in ['COVID','Labor','Flood','Power','Political',
                                   'Rana','GSP','Eid','Disruption']):
            return 'disruption'
        if any(k in col for k in ['Lag','MA','Growth','EWMA','Lead','Trend',
                                   'Bollinger','Score','Pressure','Ratio',
                                   'Total','Sin','Cos','BDT','Real','Interaction',
                                   'Season','dummy','Quarter','Month','Year']):
            return 'engineered'
        return 'other'

    def style_ws(ws):
        for col_idx, hcell in enumerate(ws[1], 1):
            cat      = get_cat(str(hcell.value or ''))
            hfg, hbg = COL_COLORS[cat]
            hcell.font      = Font(bold=True, color='FFFFFF', name='Arial', size=9)
            hcell.fill      = PatternFill('solid', fgColor=hfg)
            hcell.alignment = Alignment(horizontal='center', vertical='center',
                                        wrap_text=True)
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    cell.fill      = PatternFill('solid', fgColor=hbg)
                    cell.font      = Font(name='Arial', size=9)
                    cell.alignment = Alignment(horizontal='right')

        ws['A1'].font = Font(bold=True, color='FFFFFF', name='Arial', size=9)
        ws['A1'].fill = PatternFill('solid', fgColor='1C2E4A')
        for row in ws.iter_rows(min_row=2, min_col=1, max_col=1):
            for cell in row:
                cell.font = Font(bold=True, name='Arial', size=9, color='1C2E4A')
                cell.fill = PatternFill('solid', fgColor='EEF2F7')

        ws.freeze_panes = 'B2'
        for col in ws.columns:
            max_len = max((len(str(c.value or '')) for c in col), default=8)
            ws.column_dimensions[
                get_column_letter(col[0].column)].width = min(max_len + 2, 32)
        ws.row_dimensions[1].height = 45

    # ── Template column order (17 core variables) ─────────────────────────────
    template_cols = [
        "RMG_Export_USDmn", "ExchangeRate_BDT_USD", "CPI", "Inflation_pct",
        "Cotton_Price_USDkg", "Oil_Price_USD_bbl", "Remittances_USDmn",
        "EU_Imports_HS61", "EU_Imports_HS62", "US_Imports",
        "COVID_dummy", "PowerOutage_idx", "LaborUnrest_idx",
        "Flood_severity", "PoliticalInstability_dummy",
        "GovernmentPolicy_idx",   # NEW — DEMATEL F16
        "LendingRate_pct",        # NEW — DEMATEL F14
    ]
    available_tmpl = [c for c in template_cols if c in df_raw.columns]

    # ── Write sheets ──────────────────────────────────────────────────────────
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_raw[available_tmpl].round(3).to_excel(writer, sheet_name="Raw_Variables")
        df_ml.round(4).to_excel(writer, sheet_name="ML_Dataset")

        if collected.get("dummies") is not None:
            collected["dummies"].to_excel(writer, sheet_name="Dummy_Variables")

        eng_kw = ["Lag", "MA", "Growth", "EWMA", "Trend", "Lead",
                  "Bollinger", "Score", "Pressure", "Ratio", "Total",
                  "Real", "Interaction"]
        eng_cols = [c for c in df_ml.columns if any(k in c for k in eng_kw)]
        if eng_cols:
            df_ml[eng_cols].round(4).to_excel(
                writer, sheet_name="Engineered_Features")

        if "trade"     in collected: collected["trade"].round(2).to_excel(
            writer, sheet_name="Trade_Data")
        if "pinksheet" in collected: collected["pinksheet"].round(4).to_excel(
            writer, sheet_name="Commodities")
        if "worldbank" in collected: collected["worldbank"].round(3).to_excel(
            writer, sheet_name="WorldBank_Data")
        if "fred"      in collected: collected["fred"].round(3).to_excel(
            writer, sheet_name="FRED_Data")

        pd.DataFrame(
            DATA_DICTIONARY,
            columns=["Column_Name", "Unit", "Description", "Source"]
        ).to_excel(writer, sheet_name="DataDictionary", index=False)

    # ── Apply formatting ──────────────────────────────────────────────────────
    wb = openpyxl.load_workbook(output_path)
    for sname in wb.sheetnames:
        ws = wb[sname]
        if ws.max_row >= 2:
            style_ws(ws)

    # ── Summary sheet ──────────────────────────────────────────────────────────
    ws0 = wb.create_sheet("📋 Summary", 0)
    rows = [
        ["🇧🇩  BANGLADESH RMG ML DATASET  v2 — IVPF DEMATEL Aligned", ""],
        ["", ""],
        ["Generated on",     datetime.now().strftime("%Y-%m-%d")],
        ["Period",           f"{START_DATE} to {END_DATE}  (252 months, 21 years)"],
        ["Core variables",   f"{len(available_tmpl)} (original 15 + 2 DEMATEL-driven additions)"],
        ["Total ML features",str(df_ml.shape[1])],
        ["Total rows",       "252 monthly observations"],
        ["", ""],
        ["NEW VARIABLES (v2)", "DEMATEL RATIONALE"],
        ["GovernmentPolicy_idx (0–3)",
         "F16 = #2 net cause, D−R=1.478. Wage board decisions, EU CBAM/CSDDD, "
         "BB 9% lending cap, Accord/RSC deadlines. 7 level-3 structural events."],
        ["LendingRate_pct (%)",
         "F14 Finance = #3 net cause. Market rate→9% cap (Apr2020)→SMART rate (Jun2023). "
         "Source: Bangladesh Bank BRPD Circulars / World Bank FR.INR.LEND."],
        ["", ""],
        ["DATE RANGE EXTENSION", "RATIONALE"],
        ["2005–2025 (was 2013–2024)",
         "252 months vs 144: captures 2006 wage crisis (Tk1,662), 2007-08 "
         "commodity super-cycle, 2011 cotton crisis ($4.50/kg — highest since 1990s), "
         "2008-09 GFC resilience. 4× more ML training data."],
        ["", ""],
        ["SHEETS", "DESCRIPTION"],
        ["Raw_Variables",      "17 core variables (original 15 + GovernmentPolicy + LendingRate)"],
        ["ML_Dataset",         f"Full {df_ml.shape[1]}-feature ML matrix"],
        ["Dummy_Variables",    "9 disruption/policy proxies with source documentation"],
        ["Engineered_Features","Lag (1,2,3,6,12), rolling (3M,6M,12M), growth, interactions"],
        ["Trade_Data",         "RMG exports by partner (World/EU/USA) and HS chapter"],
        ["Commodities",        "Cotton A Index + Crude Oil (World Bank Pink Sheet, 2005-2025)"],
        ["WorldBank_Data",     "CPI, Inflation, GDP, Reserves, LendingRate (annual→monthly)"],
        ["FRED_Data",          "Exchange rate (EXBNUS), Oil (DCOILBRENTEU), US CPI"],
        ["DataDictionary",     "Every column: unit, description, and official source"],
        ["", ""],
        ["DATA SOURCES", "URL"],
        ["EPB (Export Promotion Bureau)",         "http://epb.gov.bd/site/view/report"],
        ["BGMEA Annual Reports",                  "https://www.bgmea.com.bd/feature/exportdata"],
        ["Bangladesh Bank",                       "https://www.bb.org.bd/en/index.php"],
        ["World Bank Open Data",                  "https://data.worldbank.org/country/BGD"],
        ["World Bank Pink Sheet",                 "https://www.worldbank.org/en/research/commodity-markets"],
        ["UN Comtrade API v3 (free key)",         "https://comtradedeveloper.un.org"],
        ["FRED — Exchange Rate EXBNUS",           "https://fred.stlouisfed.org/series/EXBNUS"],
        ["OTEXA — US Textile Imports",            "https://otexa.trade.gov/msrpoint.htm"],
        ["EU CBAM Regulation 2023/956",           "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R0956"],
        ["EU CSDDD 2024/1760",                    "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401760"],
        ["BB BRPD Circular No.03/2020 (9% cap)",  "https://www.bb.org.bd/mediaroom/circulars/brpd/apr072020brpd03e.pdf"],
        ["BWDB Flood Reports",                    "http://www.bwdb.gov.bd"],
        ["ILO Sustainability Compact",            "https://www.ilo.org/dhaka/Areasofwork/sustainability-compact"],
    ]
    for row in rows:
        ws0.append(row)

    ws0["A1"].font  = Font(bold=True, size=15, color="1C2E4A", name="Arial")
    for r in [9, 13, 16, 22, 31]:
        try:
            ws0.cell(r, 1).fill = PatternFill('solid', fgColor='DDEAF7')
            ws0.cell(r, 2).fill = PatternFill('solid', fgColor='DDEAF7')
            ws0.cell(r, 1).font = Font(bold=True, size=10, color='1C2E4A', name='Arial')
            ws0.cell(r, 2).font = Font(bold=True, size=10, color='1C2E4A', name='Arial')
        except:
            pass
    ws0.column_dimensions["A"].width = 44
    ws0.column_dimensions["B"].width = 70
    ws0.freeze_panes = "A1"

    wb.save(output_path)
    print(f"  ✔ Saved: '{output_path}'")
    print(f"  ✔ Sheets: {wb.sheetnames}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  STEP 1: Collecting data")
    print("=" * 70)
    collected = collect_all_data()

    print("\n" + "=" * 70)
    print("  STEP 2: Merging datasets")
    print("=" * 70)
    df_merged = merge_all_data(collected)

    print("\n" + "=" * 70)
    print("  STEP 3: Feature engineering")
    print("=" * 70)
    df_ml = engineer_features(df_merged, target_col="RMG_Export_USDmn")
    df_ml = add_extra_derived_features(df_ml)
    print(f"  ✔ Extra DEMATEL features added (LendingRate lags, Policy lags, interaction)")

    print("\n" + "=" * 70)
    print("  STEP 4: Saving to Excel")
    print("=" * 70)
    save_to_excel(df_merged, df_ml, collected, OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("  COMPLETE — Dataset Summary:")
    print("=" * 70)
    print(f"  Rows       : {df_ml.shape[0]} months (2005-01 to 2025-12)")
    print(f"  Features   : {df_ml.shape[1]} total ML features")
    print(f"  Core vars  : 17 (15 original + GovernmentPolicy_idx + LendingRate_pct)")
    print(f"  File       : {OUTPUT_FILE}")
    print(f"  Size       : {os.path.getsize(OUTPUT_FILE)/1024:.1f} KB")
    print()
