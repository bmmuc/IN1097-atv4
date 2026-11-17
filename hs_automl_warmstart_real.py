"""Atividade 4: SMBO com warm-start real para minilm_mlp
no split de treino do protocolo comum (definido em hs_automl_ml_baselines.py).

Por alvo: matriz de perdas L sem a coluna do alvo; vizinhos = k datasets mais proximos
(L1 em X base z-score, linha do alvo = meta-features so do treino); init = interleave
dos rankings de configs dos vizinhos; depois GP(Matern 2.5 + White)+EI sobre as demais
configs ate T avaliacoes. Cada avaliacao = media de (1 - macro-F1) em StratifiedKFold(3)
interno no treino com embeddings MiniLM. Em t in {n_init, 15, T}: refit do incumbente
no treino completo e F1 no teste.

Saidas: results_automl/raw/ws_evals.csv, ws_test.csv (caches em results_automl/cache/ws/).
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
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from hs_hpo_build_matrix import make_model
from hs_hpo_warmstart import U_COLS, interleave, prep_meta

SEED = 0
INNER_SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--data-dir", default="data_hs")
    ap.add_argument("--out-dir", default="results_automl")
    return ap.parse_args()


def next_config(U, loss_by_cfg, K):
    """GP + EI sobre as configs ainda nao avaliadas (mesmo kernel/EI de hs_hpo_warmstart.smbo)."""
    ev_idx = list(loss_by_cfg)
    y = np.array([loss_by_cfg[c] for c in ev_idx])
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(0.05, 10.0), nu=2.5)
              + WhiteKernel(1e-2, (1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(U[ev_idx], y)
        cand = np.setdiff1d(np.arange(K), ev_idx)
        mu, sd = gp.predict(U[cand], return_std=True)
    imp = y.min() - mu - 0.01
    z = imp / np.maximum(sd, 1e-12)
    ei = np.where(sd > 1e-12, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
    return int(cand[int(np.argmax(ei))])


def run_target(args, d, L, cfgs, U, Xbase, mf_train):
    out = Path(args.out_dir)
    c_ev, c_te = out / "cache" / "ws" / f"{d}_evals.csv", out / "cache" / "ws" / f"{d}_test.csv"
    if c_ev.exists() and c_te.exists():
        return d, 0.0
    t_start = time.perf_counter()
    K, T = len(L), min(args.T, len(L))
    # vizinhos (so outros datasets; alvo com meta-features so do treino)
    Xt = Xbase.copy()
    Xt.loc[d] = mf_train.loc[d, Xbase.columns].astype(float).fillna(Xbase.drop(index=d).median())
    Z, z, ids = prep_meta(Xt, d)
    dist = np.abs(Z - z).sum(axis=1)
    order = np.argsort(dist, kind="stable")[:args.k]
    nn_ids, nn_d = [ids[i] for i in order], dist[order]
    Ltr = L.drop(columns=d)
    init = interleave([np.argsort(Ltr[n].to_numpy(), kind="stable") for n in nn_ids], args.n_init)

    df = pd.read_parquet(Path(args.data_dir) / "datasets" / f"{d}.parquet", columns=["label"])
    y = df["label"].to_numpy()
    emb = np.load(Path(args.data_dir) / "embeddings" / f"{d}__minilm.npy")
    sp = json.loads((out / "splits" / f"{d}.json").read_text())
    tr, te = np.array(sp["train_idx"]), np.array(sp["test_idx"])
    Etr, ytr = emb[tr], y[tr]
    inner = list(StratifiedKFold(args.inner_folds, shuffle=True, random_state=INNER_SEED).split(Etr, ytr))
    budgets = sorted({args.n_init, min(15, T), T})

    ev_rows, test_rows = [], []
    loss_by_cfg, cum_eval, cum_gp = {}, 0.0, 0.0
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*batch_size.*clipped.*")
        for t in range(1, T + 1):
            gp_s = 0.0
            if t <= len(init):
                c = init[t - 1]
            else:
                g0 = time.perf_counter()
                c = next_config(U, loss_by_cfg, K)
                gp_s = time.perf_counter() - g0
            e0 = time.perf_counter()
            ls = []
            for itr, ite in inner:
                m = make_model(cfgs.loc[c], auto_batch=(c == 0)).fit(Etr[itr], ytr[itr])
                ls.append(1.0 - f1_score(ytr[ite], m.predict(Etr[ite]), average="macro"))
            eval_s = time.perf_counter() - e0
            loss_by_cfg[c] = float(np.mean(ls))
            cum_eval += eval_s
            cum_gp += gp_s
            ev_rows.append({"dataset_id": d, "t": t, "config_id": c, "inner_loss": loss_by_cfg[c],
                            "eval_s": eval_s, "gp_s": gp_s})
            if t in budgets:
                inc = min(loss_by_cfg, key=lambda k_: (loss_by_cfg[k_], k_))
                r0 = time.perf_counter()
                m = make_model(cfgs.loc[inc], auto_batch=(inc == 0)).fit(Etr, ytr)
                f1 = f1_score(y[te], m.predict(emb[te]), average="macro")
                refit_s = time.perf_counter() - r0
                test_rows.append({"dataset_id": d, "t": t, "incumbent_config": inc,
                                  "inner_loss": loss_by_cfg[inc], "f1_test": f1,
                                  "cum_search_s": cum_eval + cum_gp, "cum_eval_s": cum_eval,
                                  "cum_gp_s": cum_gp, "refit_s": refit_s,
                                  "nn_ids": ";".join(nn_ids), "nn1_dist": float(nn_d[0])})
    pd.DataFrame(test_rows).to_csv(c_te, index=False)
    pd.DataFrame(ev_rows).to_csv(c_ev, index=False)  # evals por ultimo = marca de conclusao
    return d, time.perf_counter() - t_start


def main():
    args = parse_args()
    out, data_dir = Path(args.out_dir), Path(args.data_dir)
    (out / "cache" / "ws").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(data_dir / "hpo" / "loss_matrix.csv", index_col="config_id")
    cfgs = pd.read_csv(data_dir / "hpo" / "configs.csv", index_col="config_id").loc[L.index]
    Xbase = pd.read_csv(data_dir / "X.csv", index_col="dataset_id")
    mf_train = pd.read_csv(out / "raw" / "metafeatures_train.csv", index_col="dataset_id")
    U = cfgs[U_COLS].to_numpy()
    sel = args.datasets or list(L.columns)
    missing = [d for d in sel if d not in mf_train.index]
    assert not missing, f"sem meta-features de treino (rode hs_automl_ml_baselines.py): {missing}"
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_target)(args, d, L, cfgs, U, Xbase, mf_train) for d in sel)
    for d, t in res:
        print(f"[B] {d}: {t:.1f}s", flush=True)
    for name, suf in (("ws_evals", "evals"), ("ws_test", "test")):
        fs = sorted((out / "cache" / "ws").glob(f"*_{suf}.csv"))
        pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(out / "raw" / f"{name}.csv", index=False)
    print("[B] pronto", flush=True)


if __name__ == "__main__":
    main()
