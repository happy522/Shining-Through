
import os
import re
import pandas as pd

# folders
en_folder ="en"
source_folder = input("Enter language: ")

output_folder = source_folder + "_output"
os.makedirs(output_folder, exist_ok=True)

# regex for SPEAKER block
speaker_pattern = re.compile(
    r'<SPEAKER ID=(\d+) LANGUAGE="([^"]+)" NAME="([^"]+)">(.*?)((?=<SPEAKER)|$)',
    re.DOTALL
)


def extract_blocks(text):
    matches = speaker_pattern.findall(text)
    blocks = []

    for m in matches:
        blocks.append({
            "speaker_id": m[0],
            "language": m[1],
            "name": m[2],
            "speech": m[3].strip()
        })

    return blocks


# get first 5 files only
files = sorted(os.listdir(en_folder))

for filename in files:

    en_path = os.path.join(en_folder, filename)
    de_path = os.path.join(source_folder, filename)

    if not os.path.exists(de_path):
        continue

    print("Processing:", filename)

    with open(en_path, "r", encoding="utf-8") as f:
        en_text = f.read()

    with open(de_path, "r", encoding="utf-8") as f:
        de_text = f.read()

    en_blocks = extract_blocks(en_text)
    de_blocks = extract_blocks(de_text)

    data = []

    for i in range(min(len(en_blocks), len(de_blocks))):

        if en_blocks[i]["language"] == source_folder:
            data.append({
                "speaker_id": en_blocks[i]["speaker_id"],
                "speaker_name": en_blocks[i]["name"],
                "source_language": source_folder,
                "english_text": en_blocks[i]["speech"],
                "source_text": de_blocks[i]["speech"]
            })

    if data:
        df = pd.DataFrame(data)

        out_file = os.path.join(
            output_folder,
            filename.replace(".txt", ".xlsx")
        )

        df.to_csv(out_file, index=False)
