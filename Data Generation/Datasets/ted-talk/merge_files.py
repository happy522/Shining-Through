import os
import pandas as pd

base_path = ""
languages = ["fr", "es", "it"]
splits = ["train", "test", "valid"]

data = []

for lang in languages:
    lang_path = os.path.join(base_path, lang)
    
    for split in splits:
        src_file = os.path.join(lang_path, f"{split}.{lang}")
        en_file = os.path.join(lang_path, f"{split}.en")
        
        # Read files
        with open(src_file, "r", encoding="utf-8") as f:
            src_lines = f.readlines()
        
        with open(en_file, "r", encoding="utf-8") as f:
            en_lines = f.readlines()
        
        # Safety check
        if len(src_lines) != len(en_lines):
            print(f"Warning: mismatch in {lang} {split}")
        
        # Combine line by line
        for src, en in zip(src_lines, en_lines):
            data.append({
                "split": split,
                "source_lang": lang.upper(),
                "source_text": src.strip(),
                "english_text": en.strip()
            })

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel("ted_talk_combined.xlsx", index=False)

print("Excel file created: ted_talk_combined.xlsx")