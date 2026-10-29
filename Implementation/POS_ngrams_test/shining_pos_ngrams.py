'''
POS n-gram experiment for shining-through.

Compares UPOS n-grams (taken from the existing Stanza CoNLL-U chunk files) with the
45 handcrafted features from mega_collector.py, for L1 identification (de/es/fr/it):
  - within each domain  : stratified 5-fold cross-validation
  - across domains      : train on all of one domain, test on another (every direction)

Domains are registered in DOMAINS below. Each needs a folder of <n>.conllu files per L1
and the matching *_chunk_conllu_out.csv feature table (afile = <n>). To add L2 / TR1
later, chunk them the same way and add an entry.

Run:  python shining_pos_ngrams.py               (defaults below)
      python shining_pos_ngrams.py --per-class own --min-df 5
'''
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------- CONFIG ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = SCRIPT_DIR.parent.parent / "Data Generation" / "Datasets"

# domain -> (folder holding <lang>_chunk_conllu/ and <lang>_chunk_conllu_out.csv)
DOMAINS = {
    "NN": DATASETS / "Non-native_Europarl",        # non-native English (ACL2016 + EUPDCorp)
    "EP": DATASETS / "europarl" / "europarl" / "data",  # Europarl translations (TR2)
}

LANGUAGES = ["de", "es", "fr", "it"]
SENTENCES_PER_CHUNK = 5        # chunks with fewer sentences (e.g. the last fr file) are dropped
META_COLS = ["afile", "alang", "akorp", "astatus"]
RANDOM_SEED = 42
N_FOLDS = 5
# ----------------------------


# ============================================================
# LOADING
# ============================================================

def read_conllu_upos(path):
    """Return one list of UPOS tags per sentence.
    Multiword range rows (1-2) and empty nodes (1.1) are skipped."""
    sentences, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        current.append(cols[3])
    if current:
        sentences.append(current)
    return sentences


def load_domain(name, folder):
    rows = []
    for lang in LANGUAGES:
        feats = pd.read_csv(folder / f"{lang}_chunk_conllu_out.csv", sep="\t")
        feats = feats.set_index(feats["afile"].astype(str))
        for path in sorted((folder / f"{lang}_chunk_conllu").glob("*.conllu"), key=lambda p: int(p.stem)):
            sents = read_conllu_upos(path)
            if len(sents) != SENTENCES_PER_CHUNK:
                print(f"  skipping {name}/{lang}/{path.name}: {len(sents)} sentences")
                continue
            row = {"doc_id": f"{name}_{lang}_{path.stem}", "domain": name, "L1": lang, "pos": sents}
            row.update(feats.loc[path.stem].drop(META_COLS, errors="ignore").to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def balance(df, n, seed):
    return df.groupby("L1", group_keys=False).sample(n=n, random_state=seed).reset_index(drop=True)


# ============================================================
# FEATURES AND MODELS
# ============================================================

def pos_analyzer(n_min, n_max, include_punct):
    """Callable for CountVectorizer: n-grams never cross sentence boundaries."""
    def analyze(sentences):
        grams = []
        for tags in sentences:
            if not include_punct:
                tags = [t for t in tags if t != "PUNCT"]
            for n in range(n_min, n_max + 1):
                grams.extend("_".join(tags[i:i + n]) for i in range(len(tags) - n + 1))
        return grams
    return analyze


def pos_block(n_min, n_max, min_df, include_punct=False):
    # vocabulary, min_df and idf are fitted on the training part only (inside the pipeline)
    return Pipeline([
        ("count", CountVectorizer(analyzer=pos_analyzer(n_min, n_max, include_punct),
                                  min_df=min_df, lowercase=False)),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
    ])


def num_block():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def svm():
    return LinearSVC(C=1.0, max_iter=10000, random_state=RANDOM_SEED)


def build_models(feature_cols, min_df):
    def pos_only(n_min, n_max, punct=False):
        return Pipeline([("cols", ColumnTransformer([("pos", pos_block(n_min, n_max, min_df, punct), "pos")])),
                         ("clf", svm())])

    def num_only(clf):
        return Pipeline([("cols", ColumnTransformer([("num", num_block(), feature_cols)])), ("clf", clf)])

    return {
        "POS1":           pos_only(1, 1),
        "POS2-3":         pos_only(2, 3),
        "POS2-3+punct":   pos_only(2, 3, punct=True),
        "POS1-3":         pos_only(1, 3),
        "POS2-4":         pos_only(2, 4),
        "45_SVM":         num_only(svm()),
        # same MLP as in the earlier notebooks, as the reference baseline
        "45_MLP":         num_only(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True,
                                                 validation_fraction=0.1, random_state=RANDOM_SEED)),
        "45+POS2-3_SVM":  Pipeline([("cols", ColumnTransformer([
                                        ("pos", pos_block(2, 3, min_df), "pos"),
                                        ("num", num_block(), feature_cols)])),
                                    ("clf", svm())]),
    }


# ============================================================
# EVALUATION
# ============================================================

def labels_of(df):
    # integer labels: MLPClassifier with early_stopping fails on string labels in sklearn 1.8
    return df["L1"].map(LANGUAGES.index).to_numpy()


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def within_domain(df, models):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y = labels_of(df)
    results, cms = [], {}
    for name, model in models.items():
        y_pred = np.empty_like(y)
        fold_scores = []
        for train, test in skf.split(df, y):
            model.fit(df.iloc[train], y[train])
            y_pred[test] = model.predict(df.iloc[test])
            fold_scores.append(scores(y[test], y_pred[test]))
        mean = pd.DataFrame(fold_scores).mean().to_dict()
        std = pd.DataFrame(fold_scores).std().add_suffix("_std").to_dict()
        results.append({"model": name, **mean, **std})
        cms[name] = confusion_matrix(y, y_pred, labels=range(len(LANGUAGES)))
    return results, cms


def cross_domain(src, tgt, models):
    y_tgt = labels_of(tgt)
    results, cms = [], {}
    for name, model in models.items():
        model.fit(src, labels_of(src))
        pred = model.predict(tgt)
        results.append({"model": name, **scores(y_tgt, pred)})
        cms[name] = confusion_matrix(y_tgt, pred, labels=range(len(LANGUAGES)))
    return results, cms


def top_ngrams(df, min_df, k=15):
    """Fit POS2-3 on a whole domain and list the n-grams with the largest weight per L1.
    Associations of this model only - not proof of transfer by themselves."""
    vec = pos_block(2, 3, min_df)
    X = vec.fit_transform(df["pos"])
    clf = svm().fit(X, df["L1"])
    vocab = vec.named_steps["count"].get_feature_names_out()
    rows = []
    for cls, coefs in zip(clf.classes_, clf.coef_):
        for rank, i in enumerate(np.argsort(coefs)[::-1][:k], 1):
            rows.append({"L1": cls, "rank": rank, "ngram": vocab[i], "weight": coefs[i]})
    return pd.DataFrame(rows)


def save_cms(cms, path):
    frames = [pd.DataFrame(cm, index=LANGUAGES, columns=LANGUAGES).rename_axis("true").reset_index()
              .assign(model=name) for name, cm in cms.items()]
    pd.concat(frames)[["model", "true"] + LANGUAGES].to_csv(path, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", default="min",
                        help="'min' = same size for every L1 in every domain (smallest class overall), "
                             "'own' = balance each domain to its own smallest class, or a number")
    parser.add_argument("--min-df", type=int, default=2, help="minimum document frequency of an n-gram")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results"))
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading domains...")
    data = {name: load_domain(name, folder) for name, folder in DOMAINS.items()}
    for name, df in data.items():
        print(f"  {name}: {len(df)} chunks  {df['L1'].value_counts().sort_index().to_dict()}")

    smallest = min(df["L1"].value_counts().min() for df in data.values())
    for name, df in data.items():
        n = (smallest if args.per_class == "min" else
             df["L1"].value_counts().min() if args.per_class == "own" else int(args.per_class))
        data[name] = balance(df, n, RANDOM_SEED)
        print(f"  {name}: balanced to {n} per L1 -> {len(data[name])} chunks")

    feature_cols = [c for c in next(iter(data.values())).columns if c not in ["doc_id", "domain", "L1", "pos"]]
    print(f"  {len(feature_cols)} handcrafted features")
    models = build_models(feature_cols, args.min_df)

    all_results = []
    for name, df in data.items():
        print(f"\n=== within {name} ({N_FOLDS}-fold CV) ===")
        res, cms = within_domain(df, models)
        all_results += [{"setting": "within", "train": name, "test": name, **r} for r in res]
        save_cms(cms, out / f"cm_within_{name}.csv")
        top_ngrams(df, args.min_df).to_csv(out / f"top_ngrams_POS2-3_{name}.csv", index=False)
        print(pd.DataFrame(res)[["model", "accuracy", "accuracy_std", "macro_f1"]].round(3).to_string(index=False))

    for src in data:
        for tgt in data:
            if src == tgt:
                continue
            print(f"\n=== {src} -> {tgt} ===")
            res, cms = cross_domain(data[src], data[tgt], models)
            all_results += [{"setting": "cross", "train": src, "test": tgt, **r} for r in res]
            save_cms(cms, out / f"cm_{src}_to_{tgt}.csv")
            print(pd.DataFrame(res)[["model", "accuracy", "macro_f1"]].round(3).to_string(index=False))

    results = pd.DataFrame(all_results)
    results.to_csv(out / "results.csv", index=False)

    summary = results.assign(run=results["train"] + "->" + results["test"]) \
                     .pivot(index="model", columns="run", values="macro_f1") \
                     .reindex(models.keys())
    summary.to_csv(out / "summary_macro_f1.csv")
    print("\n=== macro-F1 summary (chance = 0.25) ===")
    print(summary.round(3).to_string())
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
