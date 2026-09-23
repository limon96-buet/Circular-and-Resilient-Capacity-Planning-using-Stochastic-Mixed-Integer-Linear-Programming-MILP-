import pandas as pd

# ── 1. Load the file ──────────────────────────────────────────────────────────
FILE_IN  = r"C:\Users\mdlim\Downloads\Research Projects\IVTF DEMAETAL + ML THESIS\files (1)\Bangladesh_ML_Scripts_v2\Bangladesh_RMG_ML_Dataset_v2.xlsx"
FILE_OUT = "merged_dataset.xlsx"

SKIP_SHEETS = ["📋 Summary", "DataDictionary"]

xl = pd.ExcelFile(FILE_IN)

# ── 2. Read & merge all sheets on 'Date' ─────────────────────────────────────
merged = None

for sheet in xl.sheet_names:
    if sheet in SKIP_SHEETS:
        print(f"  Skipping : {sheet}")
        continue

    df = pd.read_excel(xl, sheet_name=sheet)
    print(f"  Reading  : {sheet}  →  {df.shape[0]} rows × {df.shape[1]} cols")

    if merged is None:
        merged = df
    else:
        # Merge on Date; only bring in NEW columns from the right sheet
        new_cols = [c for c in df.columns if c not in merged.columns]
        merged = merged.merge(df[["Date"] + new_cols], on="Date", how="outer")

# ── 3. Sort by Date & reset index ────────────────────────────────────────────
merged["Date"] = pd.to_datetime(merged["Date"])
merged = merged.sort_values("Date").reset_index(drop=True)

print(f"\n✅ Merged shape : {merged.shape[0]} rows × {merged.shape[1]} cols")

# ── 4. Save to Excel ─────────────────────────────────────────────────────────
merged.to_excel(FILE_OUT, index=False, sheet_name="Merged_Data")
print(f"✅ Saved to     : {FILE_OUT}")