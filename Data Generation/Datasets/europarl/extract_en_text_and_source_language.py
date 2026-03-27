import pandas as pd
from pathlib import Path

root_folder = Path("./europarl/data/")
output_file = "combined_output.csv"

header_written = False

for file in root_folder.rglob("*"):
    if not file.is_file():
        continue

    print(f"Processing file: {file}")

    try:
        # Try CSV first
        try:
            df = pd.read_csv(file)
        except:
            # If CSV fails → try Excel
            df = pd.read_excel(file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Rename variations
        df = df.rename(columns={
            "source language": "source_language",
            "english text": "english_text"
        })

        if "source_language" in df.columns and "english_text" in df.columns:
            extracted = df[["source_language", "english_text"]].dropna()

            if not extracted.empty:
                extracted.to_csv(
                    output_file,
                    mode="a",
                    header=not header_written,
                    index=False,
                    encoding="utf-8"
                )
                header_written = True
        else:
            print(f"⚠️ Columns not found: {df.columns.tolist()}")

    except Exception as e:
        print(f"Failed to process {file}: {e}")

print("Done")