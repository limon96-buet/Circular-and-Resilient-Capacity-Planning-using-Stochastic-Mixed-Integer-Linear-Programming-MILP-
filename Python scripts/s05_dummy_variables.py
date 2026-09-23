# =============================================================================
# s05_dummy_variables.py — Bangladesh Disruption Proxy Variables
# =============================================================================
# VARIABLES (all based on documented, publicly verifiable events):
#
#   COVID_dummy              — 0–1 graded by RMG sector disruption severity
#   PoliticalInstability_dummy — 0/1 hartals, blockades, election months
#   LaborUnrest_idx          — 0–3 RMG sector labor disputes/strikes
#   Flood_severity           — 0–3 annual flood events
#   PowerOutage_idx          — 0–2 load-shedding crisis
#   RanaPlaza_dummy          — 0/1 Rana Plaza aftermath (Apr–Jun 2013)
#   GSP_Suspension_dummy     — 0/1 US GSP suspension (Jun–Dec 2013)
#   Eid_month_dummy          — 0/1 Islamic Eid holiday months
#
#   GovernmentPolicy_idx  ← NEW (DEMATEL F16 — strongest overall cause)
#   ─────────────────────────────────────────────────────────────────────
#   Scale: 0 = no notable policy change
#          1 = minor regulatory announcement or review
#          2 = significant enacted policy (major compliance deadline, GSP review)
#          3 = structural policy shift (minimum wage increase, GSP suspension,
#              EU CBAM/CSDDD entry into force, lending rate cap/removal)
#
#   DISTINCT from PoliticalInstability_dummy, which captures short-run
#   hartals and election disruptions. GovernmentPolicy_idx captures
#   medium-to-long-run regulatory drivers identified in your DEMATEL:
#     - Wage board decisions and minimum wage changes
#     - EU trade policy (CBAM, CSDDD, GSP+, Textile Strategy)
#     - Bangladesh Bank lending rate policy (9% cap, SMART rate)
#     - Fire/building safety compliance deadlines (Accord, Alliance, RSC)
#     - Bangladesh Labour Act amendments
#   These have 3–12 month lagged effects on RMG exports — precisely the
#   reason DEMATEL placed F16 as the second-strongest net cause (D−R = 1.478).
#
# ALL SOURCES:
#   COVID:     WHO Bangladesh Situation Reports
#   Political: Bangladesh Election Commission, news archives (Prothom Alo, DS)
#   Labor:     ILO, BGMEA press releases, Human Rights Watch reports
#   Floods:    BWDB, FFWC daily reports, ReliefWeb, EM-DAT
#   Power:     BPDB Annual Reports, Daily Star archives
#   Policy:    BGMEA Annual Reports, EU Official Journal, Bangladesh Bank
#              circulars (BRPD), ILO Sustainability Compact progress reports
# =============================================================================

import pandas as pd
import numpy as np


def build_dummy_variables(start_date: str = "2005-01-01",
                          end_date: str   = "2025-12-31") -> pd.DataFrame:
    """
    Build monthly disruption proxy variables for Bangladesh (2005–2025).
    Returns pd.DataFrame indexed by 'YYYY-MM' strings.
    """
    monthly_index = pd.date_range(start_date, end_date, freq="MS")
    date_strings  = [dt.strftime("%Y-%m") for dt in monthly_index]

    df = pd.DataFrame({"Date": date_strings}).set_index("Date")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. COVID_dummy
    # ─────────────────────────────────────────────────────────────────────────
    covid_map = {
        "2020-03": 0.5,  "2020-04": 1.0,  "2020-05": 0.8,
        "2020-06": 0.7,  "2020-07": 0.6,  "2020-08": 0.5,
        "2020-09": 0.4,  "2020-10": 0.3,  "2020-11": 0.3,
        "2020-12": 0.4,  "2021-01": 0.5,  "2021-02": 0.5,
        "2021-03": 0.7,  "2021-04": 1.0,  "2021-05": 0.8,
        "2021-06": 0.7,  "2021-07": 0.8,  "2021-08": 0.6,
        "2021-09": 0.5,  "2021-10": 0.3,  "2021-11": 0.2,
        "2021-12": 0.2,
    }
    df["COVID_dummy"] = pd.Series(covid_map).reindex(df.index, fill_value=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PoliticalInstability_dummy
    # ─────────────────────────────────────────────────────────────────────────
    political_events = {
        # 2006-07: Emergency Rule (Caretaker Government Jan 2007)
        "2006-10": 1, "2006-11": 1,          # Pre-election violence
        "2007-01": 1,                          # State of Emergency declared Jan 11
        "2007-02": 1,                          # Continued emergency rule
        # 9th General Election Dec 2008
        "2008-11": 1, "2008-12": 1,
        # 10th General Election — opposition boycott, hartals 2013-14
        "2013-10": 1, "2013-11": 1, "2013-12": 1, "2014-01": 1,
        # 2015 blockades (90-day BNP blockade, petrol bomb attacks)
        "2015-01": 1, "2015-02": 1, "2015-03": 1,
        # 11th General Election Dec 2018
        "2018-11": 1, "2018-12": 1,
        # Pre-12th election
        "2023-10": 1, "2023-11": 1, "2023-12": 1, "2024-01": 1,
        # Student protests → government change Aug 2024
        "2024-07": 1, "2024-08": 1, "2024-09": 1,
    }
    df["PoliticalInstability_dummy"] = pd.Series(political_events).reindex(
        df.index, fill_value=0).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. LaborUnrest_idx  (0–3)
    # ─────────────────────────────────────────────────────────────────────────
    labor_events = {
        # 2006: Minimum wage protests (new wage Tk 1,662 — workers demand more)
        "2006-04": 2, "2006-05": 2, "2006-06": 1,
        # 2010: Wage protests ahead of Tk 3,000 announcement
        "2010-07": 2, "2010-08": 2, "2010-09": 2,
        "2010-10": 3, "2010-11": 3, "2010-12": 2, "2011-01": 2,
        # 2012: Tazreen factory fire (Nov 24) → immediate protests
        "2012-11": 2, "2012-12": 2,
        # 2013: Rana Plaza (Apr 24) — sector-wide shutdown
        "2013-04": 3, "2013-05": 3, "2013-06": 2,
        "2013-09": 2, "2013-10": 3, "2013-11": 3, "2013-12": 2,
        "2014-01": 2,
        # 2016-2017 scattered events
        "2016-01": 1, "2016-12": 1, "2017-01": 2,
        # 2018: Major minimum wage dispute
        "2018-09": 1, "2018-10": 2, "2018-11": 3, "2018-12": 3,
        "2019-01": 2,
        # 2020-2021: Mostly COVID-suppressed
        "2020-04": 1, "2021-07": 1,
        # 2022-2023: Inflation-driven unrest
        "2022-10": 1, "2022-11": 1,
        "2023-07": 1, "2023-08": 1, "2023-09": 2,
        "2023-10": 3, "2023-11": 3, "2023-12": 2,
        "2024-01": 2, "2024-02": 1,
        "2024-07": 2, "2024-08": 2, "2024-09": 1,
    }
    df["LaborUnrest_idx"] = pd.Series(labor_events).reindex(
        df.index, fill_value=0).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Flood_severity  (0–3)
    # ─────────────────────────────────────────────────────────────────────────
    flood_events = {
        # 2005-2009: annual monsoon flooding
        "2005-07": 1, "2005-08": 1,
        "2006-07": 1, "2006-08": 1, "2006-09": 1,
        "2007-07": 2, "2007-08": 2, "2007-09": 2,  # Severe — major 2007 floods
        "2007-11": 1,                                  # Cyclone Sidr (Nov 15) — coastal
        "2008-07": 1, "2008-08": 1,
        "2009-07": 1, "2009-09": 1,
        "2009-05": 1,  # Cyclone Aila (May 25) — coastal flooding
        # 2010-2016
        "2010-07": 1, "2010-08": 1,
        "2011-07": 1, "2011-08": 1,
        "2012-06": 1, "2012-07": 1, "2012-08": 1,
        "2013-07": 1, "2013-08": 1, "2013-09": 1,
        "2014-07": 1, "2014-08": 1, "2014-09": 1,
        "2015-07": 1, "2015-08": 1,
        "2016-07": 1, "2016-08": 1, "2016-09": 1,
        # 2017: SEVERE — 32 districts affected
        "2017-07": 1, "2017-08": 2, "2017-09": 2,
        "2018-07": 1, "2018-08": 1,
        # 2019: Severe Sylhet/Sunamganj
        "2019-06": 1, "2019-07": 2, "2019-08": 2, "2019-09": 1,
        # 2020: CATASTROPHIC — 37% of country inundated
        "2020-06": 2, "2020-07": 3, "2020-08": 3, "2020-09": 2, "2020-10": 1,
        "2021-07": 1, "2021-08": 1, "2021-09": 1,
        # 2022: Sylhet catastrophic flash floods (122-year event)
        "2022-05": 1, "2022-06": 2, "2022-07": 2, "2022-08": 1, "2022-09": 1,
        "2023-07": 1, "2023-08": 1, "2023-09": 1,
        # 2024: Feni/Cumilla major floods
        "2024-05": 1, "2024-06": 1, "2024-07": 1, "2024-08": 2, "2024-09": 1,
        # 2025: Normal monsoon expected
        "2025-07": 1, "2025-08": 1,
    }
    df["Flood_severity"] = pd.Series(flood_events).reindex(
        df.index, fill_value=0).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. PowerOutage_idx  (0–2)
    # ─────────────────────────────────────────────────────────────────────────
    def power_outage_level(date_str):
        yr  = int(date_str[:4])
        mon = int(date_str[5:7])
        # 2022-2023 energy crisis
        if yr == 2022:
            return 1 if mon <= 3 else 2
        elif yr == 2023:
            return 2 if mon <= 9 else 1
        elif yr == 2024:
            return 1
        elif yr >= 2005:
            # Regular summer load-shedding
            return 1 if 4 <= mon <= 9 else 0
        return 0

    df["PowerOutage_idx"] = pd.Series(
        {d: power_outage_level(d) for d in date_strings})

    # ─────────────────────────────────────────────────────────────────────────
    # 6. GovernmentPolicy_idx  ← NEW (DEMATEL F16 — #2 net cause, D−R = 1.478)
    # ─────────────────────────────────────────────────────────────────────────
    # Captures REGULATORY/POLICY shifts distinct from election-driven instability.
    # Scale: 0=none, 1=minor, 2=significant, 3=structural/transformative
    #
    # COVERED POLICY CATEGORIES:
    #   (A) Minimum wage board decisions — most direct RMG cost driver
    #   (B) EU trade policy (CBAM, CSDDD, GSP/GSP+, Textile Strategy)
    #   (C) Bangladesh Bank financial policy (lending rate cap, SMART rate)
    #   (D) Factory safety compliance (Accord, Alliance, RSC deadlines)
    #   (E) Bangladesh Labour Act amendments
    #   (F) Export incentive/duty-drawback policy changes
    clean_policy = {
        # 2006 wage
        "2006-07": 1, "2006-08": 1, "2006-09": 1, "2006-10": 3, "2006-11": 2,
        # 2009 stimulus
        "2009-03": 1,
        # 2010 wage
        "2010-06": 1, "2010-07": 1, "2010-08": 2,
        "2010-11": 3, "2010-12": 2, "2011-01": 1,
        # 2012 duty drawback
        "2012-07": 1,
        # 2013 Accord, Sustainability Compact, Labour Act, wage
        "2013-05": 2, "2013-06": 1, "2013-07": 2, "2013-08": 2,
        "2013-09": 2, "2013-10": 2, "2013-11": 2, "2013-12": 3,
        # 2014 compliance inspections begin
        "2014-01": 2,
        # 2015 GSP review
        "2015-01": 1, "2015-06": 1,
        # 2016 Labour Act / incentives
        "2016-07": 1, "2016-10": 1,
        # 2018 wage
        "2018-08": 1, "2018-09": 2, "2018-10": 2, "2018-11": 2,
        "2018-12": 3, "2018-10": 2,
        # 2019 implementation + green incentive
        "2019-01": 2, "2019-07": 1,
        # 2020 9%-cap + COVID stimulus package
        "2020-04": 3,  # Both financial policy AND export stimulus → 3
        # 2021 Labour Rules + RSC + International Accord
        "2021-01": 1, "2021-06": 1, "2021-09": 1,
        # 2022 EU policy cascade
        "2022-03": 2, "2022-04": 1, "2022-07": 1,
        "2022-11": 2, "2022-12": 1,
        # 2023 SMART rate + CBAM + wage
        "2023-05": 2, "2023-06": 1, "2023-07": 1,
        "2023-08": 2, "2023-09": 2, "2023-10": 3, "2023-11": 3,
        "2023-12": 2,
        # 2024 implementation + CSDDD
        "2024-01": 2, "2024-05": 2, "2024-06": 2, "2024-07": 1,
        # 2025 EU compliance phase-in
        "2025-01": 2, "2025-06": 1,
    }

    df["GovernmentPolicy_idx"] = pd.Series(clean_policy).reindex(
        df.index, fill_value=0).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. RanaPlaza_dummy
    # ─────────────────────────────────────────────────────────────────────────
    rana_plaza = {"2013-04": 1, "2013-05": 1, "2013-06": 1}
    df["RanaPlaza_dummy"] = pd.Series(rana_plaza).reindex(
        df.index, fill_value=0).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. GSP_Suspension_dummy
    # ─────────────────────────────────────────────────────────────────────────
    gsp_map = {d: 1 for d in [
        "2013-06","2013-07","2013-08","2013-09","2013-10","2013-11","2013-12"]}
    df["GSP_Suspension_dummy"] = pd.Series(gsp_map).reindex(
        df.index, fill_value=0).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Eid_month_dummy
    # ─────────────────────────────────────────────────────────────────────────
    eid_months = [
        "2005-11","2005-01",  # Eid ul-Adha Jan, Eid ul-Fitr Nov (approx)
        "2006-01","2006-10",
        "2007-01","2007-10",
        "2008-01","2008-09",
        "2009-09","2009-11",
        "2010-08","2010-11",
        "2011-08","2011-11",
        "2012-08","2012-10",
        "2013-08","2013-10",
        "2014-07","2014-10",
        "2015-07","2015-09",
        "2016-07","2016-09",
        "2017-06","2017-09",
        "2018-06","2018-08",
        "2019-06","2019-08",
        "2020-05","2020-07",
        "2021-05","2021-07",
        "2022-05","2022-07",
        "2023-04","2023-06",
        "2024-04","2024-06",
        "2025-03","2025-06",
    ]
    df["Eid_month_dummy"] = df.index.isin(eid_months).astype(int)

    print(f"  ✔ Dummy variables created: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    from config import START_DATE, END_DATE
    print("Building dummy variables...\n")
    df = build_dummy_variables(START_DATE, END_DATE)
    print("\nGovernmentPolicy_idx value distribution:")
    print(df["GovernmentPolicy_idx"].value_counts().sort_index())
    print("\nKey policy months (level 3):")
    print(df[df["GovernmentPolicy_idx"] == 3][["GovernmentPolicy_idx",
                                                "LaborUnrest_idx" if "LaborUnrest_idx" in df.columns
                                                else "GovernmentPolicy_idx"]].to_string())
    print("\nLending rate policy months comparison:")
    print(df[["GovernmentPolicy_idx"]]["2020-01":"2023-12"].to_string())
