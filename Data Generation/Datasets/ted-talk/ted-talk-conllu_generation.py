from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL
import pandas as pd
import torch

# ---------- CONFIG ----------
INPUT_FILE = Path("ted_talk_combined.xlsx")
OUTPUT_FOLDER = Path("conllu_by_lang")

LANGUAGE = "en"
USE_GPU = True

BATCH_SIZE = 32   # 🔥 tune this (128–512 depending on GPU memory)
# ----------------------------


def process_language(nlp, texts, out_path, lang):
    total = len(texts)
    processed = 0

    print(f"Writing to {out_path} ({total} sentences)")

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]

            # Create lightweight documents
            docs = [stanza.Document([], text=t) for t in batch]

            # Run NLP
            docs = nlp(docs)

            # Write immediately (streaming, no memory buildup)
            for d in docs:
                CoNLL.write_doc2conll(d, f)

            processed += len(batch)

            # Progress log (lightweight)
            if processed % (BATCH_SIZE * 10) == 0 or processed == total:
                print(f"{lang}: {processed}/{total} processed")
            import torch

            del docs
            torch.cuda.empty_cache()

            print(f"Finished {lang} ✅")


def main():
    print("CUDA available:", torch.cuda.is_available())
    if USE_GPU and torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading data...")
    df = pd.read_excel(INPUT_FILE)

    required_cols = {"source_lang", "english_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    # Drop empty early (saves compute)
    df = df.dropna(subset=["english_text"])
    df["english_text"] = df["english_text"].astype(str)

    print(f"Total usable rows: {len(df)}")

    # Load Stanza ONCE (important for GPU efficiency)
    print("Loading Stanza pipeline...")
    nlp = stanza.Pipeline(
        lang=LANGUAGE,
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=USE_GPU,
        tokenize_batch_size=2000  # helps throughput
    )

    # Process each language sequentially (1 GPU)
    for lang in ["FR", "ES", "IT"]:
        lang_df = df[df["source_lang"] == lang]

        if lang_df.empty:
            print(f"No data for {lang}, skipping...")
            continue

        texts = lang_df["english_text"].tolist()

        out_path = OUTPUT_FOLDER / f"{lang}.conllu"

        try:
            process_language(nlp, texts, out_path, lang)
        except Exception as e:
            print(f"Error in {lang}: {e}")


if __name__ == "__main__":
    main()