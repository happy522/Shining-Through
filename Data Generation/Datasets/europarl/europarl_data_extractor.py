import os
import re
import pandas as pd

en_folder = "en"
source_folder = input("Enter language: ")

output_folder = source_folder + "_output_recovered"
os.makedirs(output_folder, exist_ok=True)

# REGEX
en_speaker_pattern = re.compile(
    r'<SPEAKER\s+ID=(\d+)\s+LANGUAGE="([^"]+)"\s+NAME="([^"]+)">(.*?)(?=<SPEAKER|$)',
    re.DOTALL
)

src_speaker_pattern = re.compile(
    r'<SPEAKER\s+ID=(\d+)\s+NAME="([^"]+)">(.*?)(?=<SPEAKER|$)',
    re.DOTALL
)

# FUNCTIONS
def extract_en_blocks(text):
    return [
        {
            "speaker_id": m.group(1),
            "language": m.group(2),
            "speaker_name": m.group(3),
            "speech": m.group(4).strip()
        }
        for m in en_speaker_pattern.finditer(text)
    ]


def extract_src_blocks(text):
    return [
        {
            "speaker_id": m.group(1),
            "speaker_name": m.group(2),
            "speech": m.group(3).strip()
        }
        for m in src_speaker_pattern.finditer(text)
    ]


def build_lookup(blocks):
    by_id_name = {}
    by_id = {}

    for b in blocks:
        sid = b["speaker_id"]
        by_id_name[(sid, b["speaker_name"])] = b
        by_id[sid] = b   # fallback map

    return by_id_name, by_id


files = sorted(os.listdir(en_folder))

for filename in files:

    en_path = os.path.join(en_folder, filename)
    src_path = os.path.join(source_folder, filename)

    if not os.path.exists(src_path):
        continue

    print("Processing:", filename)

    with open(en_path, "r", encoding="utf-8") as f:
        en_text = f.read()

    with open(src_path, "r", encoding="utf-8") as f:
        src_text = f.read()

    en_blocks = extract_en_blocks(en_text)
    src_blocks = extract_src_blocks(src_text)

    lookup_name, lookup_id = build_lookup(src_blocks)

    data = []

    for en in en_blocks:

        # language filter
        if en["language"].upper() != source_folder.upper():
            continue

        sid = en["speaker_id"]
        key_name = (sid, en["speaker_name"])

        source_text = ""

        # 1. Try exact match (ID + NAME)
        if key_name in lookup_name:
            source_text = lookup_name[key_name]["speech"]

        # 2. fallback → ONLY ID match
        elif sid in lookup_id:
            source_text = lookup_id[sid]["speech"]

        # else stays ""

        data.append({
            "speaker_id": sid,
            "speaker_name": en["speaker_name"],
            "source_language": source_folder,
            "english_text": en["speech"],
            "source_text": source_text
        })

    if data:
        df = pd.DataFrame(data)

        out_file = os.path.join(
            output_folder,
            filename.replace(".txt", ".xlsx")
        )

        df.to_excel(out_file, index=False)

        print(f"Saved {len(data)} rows → {out_file}")

    else:
        print("No matching blocks found.")