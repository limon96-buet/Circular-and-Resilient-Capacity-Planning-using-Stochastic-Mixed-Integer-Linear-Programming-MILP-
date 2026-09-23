# =============================================================================
# config.py — Bangladesh ML Data Collection: API Keys & Settings
# =============================================================================
# HOW TO GET FREE API KEYS:
#
# 1. FRED API KEY (Federal Reserve Economic Data)
#    → Go to: https://fred.stlouisfed.org/docs/api/api_key.html
#    → Click "Request API Key" and register (free, instant)
#
# 2. UN COMTRADE API KEY (International Trade Data)
#    → Go to: https://comtradedeveloper.un.org/
#    → Click "Sign Up" (free, limited 100 calls/day)
#
# 3. World Bank API — NO KEY NEEDED (public, free)
# 4. World Bank Pink Sheet — NO KEY NEEDED (direct download)
# 5. Eurostat API — NO KEY NEEDED (EU open data)
# =============================================================================

# ── API Keys ─────────────────────────────────────────────────────────────────
FRED_API_KEY     = "6454d699f35a71d34510d8ac14fb32e4"
COMTRADE_API_KEY = "4b3ba9b5dbb94e3eb8ad78a71b6e14c4"

# ── Date Range ────────────────────────────────────────────────────────────────
# Extended from 2005 for richer ML training set (252 months ≈ 21 years).
# 2005 is the earliest year with reliable monthly RMG + Bangladesh Bank data.
# 2025-12 captures post-interim-government recovery and EU CBAM transition.
START_DATE = "2005-01-01"
END_DATE   = "2025-12-31"
START_YEAR = 2005
END_YEAR   = 2025

# ── Country Identifiers ───────────────────────────────────────────────────────
BANGLADESH_ISO3     = "BGD"
BANGLADESH_COMTRADE = "50"
EU_COMTRADE         = "97"
USA_COMTRADE        = "842"

# ── HS Codes for RMG Garments ─────────────────────────────────────────────────
HS_KNITTED = "61"
HS_WOVEN   = "62"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "Bangladesh_RMG_ML_Dataset.xlsx"
OUTPUT_DIR  = r"C:\Users\mdlim\Downloads\Research Projects\IVTF DEMAETAL + ML THESIS"
