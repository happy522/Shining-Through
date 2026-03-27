from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL
import pandas as pd
import os
import openpyxl
# ---------- CONFIG ----------
INPUT_FOLDER = Path(r"/home/translation/shining_through/data/DE_output").resolve()
OUTPUT_FOLDER = Path(r"/home/translation/shining_through/data/DE_output_en_conllu").resolve()

LANGUAGE = "en"   # change depending on source language
USE_GPU = True

DOC_BATCH_SIZE = 128
# ----------------------------
def main():
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    excel_files = sorted(INPUT_FOLDER.glob("*.csv"))

    gpu_count = torch.cuda.device_count()
    gpu_id = int(os.environ.get("GPU_ID", 0))
    excel_files = sorted(INPUT_FOLDER.glob("*.csv"))
    excel_files = excel_files[gpu_id::1]
    total_files = len(excel_files)
    print(f"GPU {gpu_id} processing {len(excel_files)} files")
    if total_files == 0:
        return

    print(f"Loading Stanza pipeline (GPU={USE_GPU})...")

    nlp = stanza.Pipeline(
        lang=LANGUAGE,
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=USE_GPU,
        tokenize_batch_size=2000,
        batch_size=DOC_BATCH_SIZE
    )

    processed_count = 0
    failed_count = 0

    for batch_start in range(0, total_files, DOC_BATCH_SIZE):

        batch_end = min(batch_start + DOC_BATCH_SIZE, total_files)
        batch_paths_raw = excel_files[batch_start:batch_end]

        batch_paths = []

        for src_path in batch_paths_raw:
            out_path = OUTPUT_FOLDER / src_path.with_suffix(".conllu").name
            if not out_path.exists():
                batch_paths.append(src_path)
            else:
                processed_count += 1

        print(f"\nBatch {batch_start // DOC_BATCH_SIZE + 1}"
              f" ({batch_start + 1}–{batch_end} of {total_files})")

        texts = []
        keep_paths = []

        for src_path in batch_paths:

            try:
                df = pd.read_csv(src_path)

                if "english_text" not in df.columns:
                    print(f"source_text column missing: {src_path}")
                    failed_count += 1
                    continue

                text = "\n".join(df["english_text"].dropna().astype(str))

                if not text.strip():
                    processed_count += 1
                    continue

                texts.append(text)
                keep_paths.append(src_path)

            except Exception as e:
                print(f"Error reading {src_path}: {e}")
                failed_count += 1

        if not texts:
            continue

        try:
            in_docs = [stanza.Document([], text=t) for t in texts]
            out_docs = nlp(in_docs)

            if not isinstance(out_docs, list):
                out_docs = [out_docs]

            for src_path, doc in zip(keep_paths, out_docs):

                try:
                    out_path = OUTPUT_FOLDER / src_path.with_suffix(".conllu").name
                    CoNLL.write_doc2conll(doc, str(out_path))
                    processed_count += 1

                except Exception as e:
                    print(f"Error writing {src_path}: {e}")
                    failed_count += 1

            print(f"Processed: {processed_count}/{total_files} "
                  f"(failed so far: {failed_count})")

        except Exception as e:
            print(f"NLP batch error {batch_start + 1}–{batch_end}: {e}")
            failed_count += len(keep_paths)

    print("\nAll batches completed.")
    print(f"Total successful: {processed_count}, failed: {failed_count}")


if __name__ == "__main__":
    main()