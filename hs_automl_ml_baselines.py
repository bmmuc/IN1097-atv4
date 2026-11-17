"""Atividade 4 (AutoML vs meta-aprendizado): baselines de ML classico,
custos de codificacao, meta-features (so treino) e recomendacoes LODO.

Protocolo comum: teste = fold 0 de StratifiedKFold(5, shuffle, random_state=42) sobre
o dataset inteiro (labels como no parquet); treino = o resto. Os splits sao gravados em
results_automl/splits/<id>.json para os demais scripts.

Saidas em results_automl/raw/: algos_test.csv, encode_time.csv, metafeatures_train.csv,
mf_time.csv, recommendations.csv. Caches por dataset em results_automl/cache/ (retomavel).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import hs_algorithms as algos
import hs_meta_features as mf
from approaches import AverageRank, RegressorOnP, ranks_from_P
from hs_build_metadataset import build_meta_features, quick_word_tfidf_n_features

SPLIT_SEED = 42
ENCODERS = ("minilm", "hatebert", "mbert")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, help="subconjunto de dataset_ids")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=ENCODERS)
    ap.add_argument("--encode-threads", type=int, default=12)
    ap.add_argument("--data-dir", default="data_hs")
    ap.add_argument("--out-dir", default="results_automl")
    return ap.parse_args()


def load_dataset(data_dir, d):
    df = pd.read_parquet(Path(data_dir) / "datasets" / f"{d}.parquet", columns=["text", "label"])
    return df["text"].astype(str).tolist(), df["label"].to_numpy()


def make_split(y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED)
    tr, te = next(iter(skf.split(np.zeros(len(y)), y)))
    return tr, te


def write_split(out, data_dir, d):
    p = out / "splits" / f"{d}.json"
    if p.exists():
        return
    _, y = load_dataset(data_dir, d)
    tr, te = make_split(y)
    p.write_text(json.dumps({"train_idx": tr.tolist(), "test_idx": te.tolist()}))


def read_split(out, d):
    s = json.loads((out / "splits" / f"{d}.json").read_text())
    return np.array(s["train_idx"]), np.array(s["test_idx"])


# Passo 2: algoritmos default

def run_algos(out, data_dir, d):
    cache = out / "cache" / "algos" / f"{d}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    texts, y = load_dataset(data_dir, d)
    tr, te = read_split(out, d)
    ytr, yte = y[tr], y[te]
    Ttr, Tte = [texts[i] for i in tr], [texts[i] for i in te]
    rows = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algo in algos.ALL_ALGORITHMS:
            if algo in algos.TEXT_ALGORITHMS:
                if algo.startswith("svd_"):
                    nc = max(2, min(200, quick_word_tfidf_n_features(Ttr) - 1))
                    mk = lambda n=nc: algos.make_text_pipeline(algo, svd_n_components=n)
                else:
                    mk = lambda: algos.make_text_pipeline(algo)

                def fit_pred(pipe):
                    # n_jobs=1 nos estimadores (default do modulo e -1) para tempo single-thread
                    if "clf" in pipe.named_steps and "n_jobs" in pipe.named_steps["clf"].get_params():
                        pipe.set_params(clf__n_jobs=1)
                    t0 = time.perf_counter()
                    pipe.fit(Ttr, ytr)
                    t1 = time.perf_counter()
                    p = pipe.predict(Tte)
                    return p, t1 - t0, time.perf_counter() - t1

                try:
                    preds, fs, ps = fit_pred(mk())
                except ValueError as e:
                    if algo.startswith("svd_") and "n_components" in str(e):
                        preds, fs, ps = fit_pred(algos.make_text_pipeline(algo, svd_n_components=2))
                    else:
                        raise
            else:
                emb = np.load(Path(data_dir) / "embeddings" / f"{d}__{algos.EMBED_MODEL_FOR_ALGO[algo]}.npy")
                clf = algos.make_embedding_classifier(algo)
                t0 = time.perf_counter()
                clf.fit(emb[tr], ytr)
                t1 = time.perf_counter()
                preds = clf.predict(emb[te])
                fs, ps = t1 - t0, time.perf_counter() - t1
            rows.append({"dataset_id": d, "algorithm": algo,
                         "f1_test": f1_score(yte, preds, average="macro"), "fit_s": fs, "predict_s": ps})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# Passo 4: meta-features so do treino

def run_metafeatures(out, data_dir, d, manifest, cols):
    cache = out / "cache" / "mf" / f"{d}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    texts, y = load_dataset(data_dir, d)
    tr, _ = read_split(out, d)
    emb = np.load(Path(data_dir) / "embeddings" / f"{d}__minilm.npy")[tr]
    Ttr, ytr = [texts[i] for i in tr], y[tr]
    row = manifest.loc[d]
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        feats = build_meta_features(d, Ttr, ytr, emb, row, lambda *a, **k: None)
        mf_s = time.perf_counter() - t0
    r = {"dataset_id": d, **{c: feats.get(c, np.nan) for c in cols}, "mf_s": mf_s}
    df = pd.DataFrame([r])
    df.to_csv(cache, index=False)
    return df


# Passo 5: recomendacoes LODO

def run_recommend(out, d, Xall, P, R, xq):
    cache = out / "cache" / "rec" / f"{d}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    others = [i for i in P.index if i != d]
    with threadpool_limits(limits=1):
        ar = AverageRank().fit(None, P.loc[others], R.loc[others])
        best_avg = ar.recommend()[0][0]
        t0 = time.perf_counter()
        reg = RegressorOnP().fit(Xall.loc[others], P.loc[others], R.loc[others])
        order, _ = reg.recommend(xq)
        meta_s = time.perf_counter() - t0
    df = pd.DataFrame([{"dataset_id": d, "best_avg_algo": best_avg, "ml_algo": order[0],
                        "ml_predicted_ranking": ";".join(order), "meta_s": meta_s}])
    df.to_csv(cache, index=False)
    return df


def worker(out, data_dir, d, manifest, Xall, P, R):
    t0 = time.perf_counter()
    a = run_algos(out, data_dir, d)
    m = run_metafeatures(out, data_dir, d, manifest, list(Xall.columns))
    xq = m[list(Xall.columns)].iloc[0].astype(float)
    xq = xq.fillna(Xall.median()).to_numpy()  # NaN -> mediana das outras (como em X.csv)
    r = run_recommend(out, d, Xall, P, R, xq)
    return d, time.perf_counter() - t0


# Passo 3: custo de codificacao (sequencial, torch com N threads)

def run_encode(out, data_dir, ids, n_threads, encoders=ENCODERS):
    import torch
    from hs_embed_worker import BATCH_SIZE, MAX_LENGTH, MODEL_IDS, _mean_pool
    torch.set_num_threads(n_threads)
    cache_dir = out / "cache" / "encode"
    for enc in encoders:
        todo = [d for d in ids if not (cache_dir / f"{d}__{enc}.json").exists()]
        if not todo:
            continue
        try:
            if enc == "minilm":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_IDS[enc], device="cpu")
                enc_fn = lambda tx: model.encode(tx, batch_size=BATCH_SIZE, show_progress_bar=False,
                                                 convert_to_numpy=True, normalize_embeddings=False)
            else:
                from transformers import AutoModel, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(MODEL_IDS[enc])
                model = AutoModel.from_pretrained(MODEL_IDS[enc]).eval()

                def enc_fn(tx):
                    ch = []
                    with torch.no_grad():
                        for i in range(0, len(tx), BATCH_SIZE):
                            e = tok(tx[i:i + BATCH_SIZE], padding=True, truncation=True,
                                    max_length=MAX_LENGTH, return_tensors="pt")
                            o = model(**e)
                            ch.append(_mean_pool(o.last_hidden_state, e["attention_mask"]).float().numpy())
                    return np.concatenate(ch)
        except Exception as ex:  # download/carga falhou
            print(f"[encode] FALHA ao carregar {enc}: {ex!r}", flush=True)
            continue
        enc_fn(["warmup text"])
        for d in todo:
            texts, _ = load_dataset(data_dir, d)
            t0 = time.perf_counter()
            enc_fn(texts)
            dt = time.perf_counter() - t0
            (cache_dir / f"{d}__{enc}.json").write_text(
                json.dumps({"dataset_id": d, "encoder": enc, "encode_s": dt, "n_threads": n_threads}))
            print(f"[encode] {d} {enc} n={len(texts)} {dt:.1f}s", flush=True)


def main():
    args = parse_args()
    out, data_dir = Path(args.out_dir), Path(args.data_dir)
    raw = out / "raw"
    for sub in ("splits", "raw", "cache/algos", "cache/mf", "cache/rec", "cache/encode"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Xall = pd.read_csv(data_dir / "X.csv", index_col="dataset_id")
    P = pd.read_csv(data_dir / "P.csv", index_col="dataset_id").loc[Xall.index]
    R = ranks_from_P(P)
    manifest = pd.read_csv(data_dir / "datasets_manifest.csv", dtype={"dataset_id": str}).set_index("dataset_id")
    ids = list(Xall.index)
    sel = args.datasets or ids
    for d in sel:
        write_split(out, data_dir, d)
    print(f"[A] {len(sel)} datasets; splits ok", flush=True)

    t0 = time.perf_counter()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(worker)(out, data_dir, d, manifest, Xall, P, R) for d in sel)
    for d, t in res:
        print(f"[A] {d}: {t:.1f}s", flush=True)
    print(f"[A] fase paralela: {time.perf_counter() - t0:.1f}s", flush=True)

    def collect(sub):
        fs = sorted((out / "cache" / sub).glob("*.csv"))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

    collect("algos").to_csv(raw / "algos_test.csv", index=False)
    m = collect("mf")
    m.drop(columns="mf_s").to_csv(raw / "metafeatures_train.csv", index=False)
    m[["dataset_id", "mf_s"]].to_csv(raw / "mf_time.csv", index=False)
    collect("rec").to_csv(raw / "recommendations.csv", index=False)

    if not args.skip_encode:
        run_encode(out, data_dir, sel, args.encode_threads, args.encoders)
    fs = sorted((out / "cache" / "encode").glob("*.json"))
    if fs:
        pd.DataFrame([json.loads(f.read_text()) for f in fs]).to_csv(raw / "encode_time.csv", index=False)
    print("[A] pronto", flush=True)


if __name__ == "__main__":
    main()
