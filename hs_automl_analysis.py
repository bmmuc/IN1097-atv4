#!/usr/bin/env python
"""Analise "AutoML vs meta-learning" (hate speech, 50 datasets, 1 split de teste por dataset).

Le results_automl/raw/* (+ autogluon/{variant}_{budget}/{id}.json) e escreve
results_automl/analysis/: per_dataset.csv, summary_perf.{csv,md}, friedman.txt, cd_*.png,
wilcoxon.csv, perf_cost.png, offline_cost.csv, ag_choices.{csv,md}, gain_correlates.csv,
summary.md, tex/*.tex.  Funciona com resultados parciais (usa a intersecao de datasets
disponiveis e reporta a cobertura).  Somente leitura nos dados brutos.

Uso: python hs_automl_analysis.py [--raw results_automl/raw] [--out results_automl/analysis]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import evaluate as ev

ROOT = Path(__file__).resolve().parent
CD_KW = dict(figsize=(6.4, 3.6), dpi=220, tick_fontsize=13, label_fontsize=13,
             cd_fontsize=13, title_fontsize=14, tight=True)
TOL = 1e-9
AG_RUNSETS = [("main", 60), ("main", 240), ("main", 900), ("nometa", 240), ("emb", 240)]
AG_MAIN_BUDGETS = [60, 240, 900]
WS_T = [5, 15, 30]
ENC_OF = {"minilm_lr": "minilm", "minilm_mlp": "minilm", "hatebert_lr": "hatebert", "mbert_lr": "mbert"}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "results_automl/raw"))
ap.add_argument("--out", default=str(ROOT / "results_automl/analysis"))
ap.add_argument("--data", default=str(ROOT / "data_hs"))
ap.add_argument("--p-cores", type=int, default=12, help="fator do limite superior (CPU-equivalente) do build de P")
args = ap.parse_args()
RAW, OUT, DATA = Path(args.raw), Path(args.out), Path(args.data)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tex").mkdir(exist_ok=True)
WARN: list[str] = []


def warn(msg):
    print("[warn]", msg)
    WARN.append(msg)


def label(m):
    if m == "ws_t30":
        return "ML+WS (T=30)"
    if m.startswith("ws_t"):
        return f"ML+WS (T={m[4:]})"
    if m == "metalearn":
        return "Meta-learning"
    if m == "best_avg":
        return "Melhor media"
    mm = re.match(r"AG_(\w+)_(\d+)$", m)
    if mm and mm.group(1) == "emb":
        return f"AG+MiniLM {mm.group(2)}s"
    if mm:
        return f"AG {mm.group(1)} {mm.group(2)}s"
    return m


def md_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tex_table(df, caption, lab, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{lab}}}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("--" if np.isnan(v) else (fmt.format(v) if abs(v) < 1e5 else f"{v:.3g}"))
            else:
                cells.append(tex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, run = len(idx), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def paired(a, b):
    """a,b Series f1 (alto=melhor) -> dict n, mean_diff, wins/ties/losses (de a), p wilcoxon."""
    d = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = d.a - d.b
    res = dict(n=len(d), mean_diff=float(diff.mean()) if len(d) else np.nan,
               wins=int((diff > TOL).sum()), ties=int((diff.abs() <= TOL).sum()), losses=int((diff < -TOL).sum()))
    nz = diff[diff.abs() > TOL]
    if len(nz) >= 5:
        try:
            res["p"] = float(wilcoxon(d.a, d.b, zero_method="zsplit").pvalue)
        except Exception:
            res["p"] = np.nan
    else:
        res["p"] = np.nan
    return res


def friedman_block(df, name, txt, png, title):
    if df.shape[0] < 3 or df.shape[1] < 3:
        txt.append(f"== {name} == pulado (N={df.shape[0]}, k={df.shape[1]}; precisa N>=3, k>=3)\n")
        return None
    r = ev.friedman_nemenyi(df, higher_is_better=False)
    ev.plot_cd_diagram(r["mean_ranks"].rename(index=label), r["cd"], title, str(png), **CD_KW)
    txt.append(f"== {name} (N={r['N']}, k={r['k']}) ==\nFriedman chi2={r['statistic']:.3f} p={r['pvalue']:.4g}\n"
               f"CD(Nemenyi, 0.05)={r['cd']:.3f}\nranks medios (1=melhor):\n{r['mean_ranks'].sort_values().round(3).to_string()}\n"
               f"pares significativos: {[(label(a), label(b), round(d, 2)) for a, b, d in r['significant_pairs']] or 'nenhum'}\n")
    return r


# ------------------------------------------------------------------ carga
def rd(path, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            raise SystemExit(f"arquivo obrigatorio ausente: {p}")
        warn(f"arquivo ausente: {p}")
        return None
    return pd.read_csv(p)


algos = rd(RAW / "algos_test.csv")
rec = rd(RAW / "recommendations.csv").set_index("dataset_id")
mf_t = rd(RAW / "mf_time.csv").set_index("mf_s".replace("mf_s", "dataset_id"))["mf_s"]
mf_base = rd(RAW / "metafeatures_train.csv").set_index("dataset_id")
enc = rd(RAW / "encode_time.csv", required=False)
wst = rd(RAW / "ws_test.csv", required=False)
P = pd.read_csv(DATA / "P.csv", index_col="dataset_id")
manifest = pd.read_csv(DATA / "datasets_manifest.csv").set_index("dataset_id")
ids = list(P.index)

F = algos.pivot_table(index="dataset_id", columns="algorithm", values="f1_test").reindex(ids)
FIT = algos.pivot_table(index="dataset_id", columns="algorithm", values="fit_s").reindex(ids)
PRED = algos.pivot_table(index="dataset_id", columns="algorithm", values="predict_s").reindex(ids)
ALG = list(F.columns)

# encode: wall (encode_s) e core-seconds (encode_s * n_threads)
enc_wall, enc_core = {}, {}
if enc is not None and len(enc):
    for _, r in enc.iterrows():
        enc_wall[(r.dataset_id, r.encoder)] = float(r.encode_s)
        enc_core[(r.dataset_id, r.encoder)] = float(r.encode_s) * float(r.n_threads)
else:
    warn("encode_time.csv ausente/vazio: custo de encoding tratado como 0 (NaN->0)")
_missing_enc = set()


def enc_c(d, e, core=True):
    k = (d, e)
    src = enc_core if core else enc_wall
    if k not in src:
        if enc is not None and len(enc):
            _missing_enc.add(k)
        return 0.0
    return src[k]


# ------------------------------------------------------------------ metodos
f1 = pd.DataFrame(index=ids)
cc = pd.DataFrame(index=ids)  # custo online core-s
cw = pd.DataFrame(index=ids)  # custo online wall-s
cov_notes = {}
ccraw = pd.DataFrame(index=ids)  # custo AG bruto (sem descontar pausas)

f1["best_avg"] = [F.at[d, rec.at[d, "best_avg_algo"]] if d in rec.index else np.nan for d in ids]
f1["metalearn"] = [F.at[d, rec.at[d, "ml_algo"]] if d in rec.index else np.nan for d in ids]


def algo_cost(d, a, core):
    c = FIT.at[d, a] + PRED.at[d, a]
    e = ENC_OF.get(a)
    return c + (enc_c(d, e, core) if e else 0.0)


cc["best_avg"] = [algo_cost(d, rec.at[d, "best_avg_algo"], True) if d in rec.index else np.nan for d in ids]
cw["best_avg"] = [algo_cost(d, rec.at[d, "best_avg_algo"], False) if d in rec.index else np.nan for d in ids]


def ml_cost(d, core):
    a = rec.at[d, "ml_algo"]
    c = mf_t[d] + enc_c(d, "minilm", core) + rec.at[d, "meta_s"] + FIT.at[d, a] + PRED.at[d, a]
    e = ENC_OF.get(a)
    if e and e != "minilm":
        c += enc_c(d, e, core)
    return c


cc["metalearn"] = [ml_cost(d, True) if d in rec.index else np.nan for d in ids]
cw["metalearn"] = [ml_cost(d, False) if d in rec.index else np.nan for d in ids]

ws_nn1 = pd.Series(np.nan, index=ids)
if wst is not None and len(wst):
    wst = wst.drop_duplicates(["dataset_id", "t"], keep="last")
    for t in WS_T:
        s = wst[wst.t == t].set_index("dataset_id").reindex(ids)
        f1[f"ws_t{t}"] = s["f1_test"]
        base = mf_t.reindex(ids) + pd.Series({d: enc_c(d, "minilm", True) for d in ids})
        basew = mf_t.reindex(ids) + pd.Series({d: enc_c(d, "minilm", False) for d in ids})
        cc[f"ws_t{t}"] = base + s["cum_search_s"] + s["refit_s"]
        cw[f"ws_t{t}"] = basew + s["cum_search_s"] + s["refit_s"]
    ws_nn1 = wst[wst.t == max(WS_T)].set_index("dataset_id")["nn1_dist"].reindex(ids)
else:
    warn("ws_test.csv ausente/vazio: metodos ws_* omitidos")

f1["oracle"] = F.max(axis=1)

# ------------------------------------------------------------------ AutoGluon
FAM_ORDER = [("fastai", "FASTAI"), ("torch", "NN_TORCH"), ("catboost", "CAT"), ("xgb", "XGB"), ("lightgbm", "GBM"),
             ("gbm", "GBM"), ("lgb", "GBM"), ("randomforest", "RF"), ("extratrees", "XT"), ("kneighbors", "KNN"),
             ("linearmodel", "LR"), ("linear", "LR")]
ABBR = {"RF": "RF", "XT": "XT", "KNN": "KNN", "LR": "LR", "GBM": "GBM", "CAT": "CAT", "XGB": "XGB", "NN": "NN_TORCH"}
FAM_GROUP = {"RF": "RF", "XT": "RF", "GBM": "GBM", "XGB": "GBM", "CAT": "GBM", "KNN": "KNN", "LR": "LR",
             "FASTAI": "NN", "NN_TORCH": "NN"}
FAM_ANALOG = {"RF": "svd_rf", "XT": "svd_rf", "GBM": "svd_histgb", "XGB": "svd_histgb", "CAT": "svd_histgb",
              "KNN": "svd_knn", "LR": "tfidf_word_lr", "FASTAI": "minilm_mlp", "NN_TORCH": "minilm_mlp"}
OUR_GROUP = {"svd_rf": "RF", "svd_histgb": "GBM", "svd_knn": "KNN", "tfidf_word_lr": "LR", "minilm_lr": "LR",
             "hatebert_lr": "LR", "mbert_lr": "LR", "tfidf_char_svm": "LR", "minilm_mlp": "NN", "tfidf_nb": "NB"}


def family(name):
    if name is None:
        return None
    s = str(name)
    if s.startswith("WeightedEnsemble"):
        return None
    tok = s.split("_")[0]
    if tok in ABBR:
        return ABBR[tok]
    low = s.lower()
    for key, fam in FAM_ORDER:
        if key in low:
            return fam
    return "OTHER"


def parse_ensemble(j):
    """-> list[(family, weight)] dos modelos base do melhor modelo (peso 1/n se nao ha pesos)."""
    best = j.get("best_model")
    ens = j.get("ensemble") or {}
    weights = ens.get("weights") if isinstance(ens, dict) else None
    base = ens.get("base_models") if isinstance(ens, dict) else None
    items = []
    if isinstance(weights, dict) and weights:
        items = [(k, float(v)) for k, v in weights.items()]
    elif base:
        names = []
        for b in base:
            if isinstance(b, dict):
                names.append(b.get("model") or b.get("name") or b.get("type"))
            elif isinstance(b, (list, tuple)):
                names.append(b[0])
            else:
                names.append(b)
        if isinstance(base, dict):
            names = list(base.keys())
        items = [(n, 1.0 / max(len(names), 1)) for n in names]
    if not items and best and not str(best).startswith("WeightedEnsemble"):
        items = [(best, 1.0)]
    out = []
    for n, w in items:
        fam = family(n)
        if fam:
            out.append((fam, w))
    return out


def ft_counts(j):
    g = (j.get("features") or {}).get("by_group")
    out = {}
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, (list, tuple, dict)):
                out[k] = len(v)
            elif isinstance(v, (int, float)):
                out[k] = v
    return out


def parse_thermal(path):
    """-> list[(start, end)] de intervalos PAUSE->RESUME (datetime local naive)."""
    iv, t0 = [], None
    try:
        lines = open(path).read().splitlines()
    except Exception as e:
        warn(f"thermal.log: {e}")
        return iv
    last = None
    for ln in lines:
        m_ = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\s+(PAUSE|RESUME)\b", ln)
        if not m_:
            m2 = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", ln)
            if m2:
                last = dt.datetime.fromisoformat(m2.group(1))
            continue
        t = dt.datetime.fromisoformat(m_.group(1))
        last = t
        if m_.group(2) == "PAUSE":
            if t0 is None:
                t0 = t
        elif t0 is not None:
            iv.append((t0, t))
            t0 = None
    if t0 is not None:  # pausa ainda aberta
        iv.append((t0, max(last or t0, t0)))
    return iv


PAUSES = parse_thermal(ROOT / "results_automl/logs/thermal.log")
PARALLEL_CUT = dt.datetime(2026, 11, 16, 8, 30)
PARALLEL_CUT2 = dt.datetime(2026, 11, 16, 12, 10)
thermal_rows = []
AG_VAL = {}  # runset -> {did: score_val do melhor modelo}
ag_rows = []  # linhas p/ ag_choices
AG_COV = {}
for variant, b in AG_RUNSETS:
    name = f"AG_{variant}_{b}"
    d_ = RAW / "autogluon" / f"{variant}_{b}"
    fs = sorted(d_.glob("*.json")) if d_.exists() else []
    n_err = 0
    f1c, ccc, cwc, ccr = {}, {}, {}, {}
    valc = {}
    for fp in fs:
        try:
            j = json.loads(fp.read_text())
        except Exception as e:
            warn(f"{fp}: json invalido ({e})")
            continue
        did = fp.stem
        if did not in f1.index:
            continue
        if "error" in j or j.get("f1_test") is None:
            n_err += 1
            continue
        tw = j.get("total_wall_s")
        if tw is None:
            tw = (j.get("fit_wall_s") or 0) + (j.get("predict_wall_s") or 0)
        ncpu = j.get("num_cpus") or 4
        mt = dt.datetime.fromtimestamp(os.path.getmtime(fp))
        t_start = mt - dt.timedelta(seconds=float(tw))
        paused = 0.0
        npz = 0
        for a_, b_ in PAUSES:
            ov = (min(mt, b_) - max(t_start, a_)).total_seconds()
            if ov > 0:
                paused += ov
                npz += 1
        paused = min(paused, float(tw))
        thermal_rows.append(dict(run_set=name, dataset_id=did, start=t_start.isoformat(timespec="seconds"),
                                 end=mt.isoformat(timespec="seconds"), total_wall_s=float(tw), paused_s=paused,
                                 paused_frac=paused / float(tw) if tw else np.nan, n_pauses=npz,
                                 n_parallel=3 if mt < PARALLEL_CUT else (2 if mt < PARALLEL_CUT2 else 1)))
        lb = j.get("leaderboard")
        if isinstance(lb, list):
            for row_ in lb:
                if row_.get("model") == j.get("best_model") and row_.get("score_val") is not None:
                    valc[did] = float(row_["score_val"])
                    break
        tw_raw = float(tw)
        tw = tw_raw - paused  # tempo efetivo (sem pausas termicas)
        f1c[did] = float(j["f1_test"])
        extra_c = enc_c(did, "minilm", True) if variant == "emb" else 0.0
        extra_w = enc_c(did, "minilm", False) if variant == "emb" else 0.0
        ccc[did] = float(tw) * ncpu + extra_c
        ccr[did] = tw_raw * ncpu + extra_c
        cwc[did] = float(tw) + extra_w
        ens = parse_ensemble(j)
        ft = ft_counts(j)
        ag_rows.append(dict(dataset_id=did, runset=name, variant=variant, budget=b, best_model=j.get("best_model"),
                            n_base=len(ens), is_ens=len(ens) > 1,
                            fam_weights=";".join(f"{f}:{w:.3f}" for f, w in ens),
                            n_stack=j.get("n_stack_levels"), n_models=j.get("n_models"),
                            dyn_stack=j.get("dynamic_stacking"),
                            **{f"feat_{k}": v for k, v in ft.items()}, _ens=ens))
    AG_COV[name] = dict(n_json=len(fs), n_ok=len(f1c), n_error=n_err)
    if f1c:
        f1[name] = pd.Series(f1c)
        cc[name] = pd.Series(ccc)
        cw[name] = pd.Series(cwc)
        ccraw[name] = pd.Series(ccr)
        AG_VAL[name] = valc
    else:
        warn(f"{name}: nenhum resultado valido ({len(fs)} json)")
if _missing_enc:
    warn(f"{len(_missing_enc)} pares (dataset, encoder) sem encode_time -> custo 0")

AG_NAMES = [f"AG_{v}_{b}" for v, b in AG_RUNSETS if f"AG_{v}_{b}" in f1]
METHODS = [m for m in f1.columns if m != "oracle"]
ML_NAMES = [m for m in ["best_avg", "metalearn", "ws_t5", "ws_t15", "ws_t30"] if m in f1]

thdf = pd.DataFrame(thermal_rows, columns=["run_set", "dataset_id", "start", "end", "total_wall_s", "paused_s",
                                           "paused_frac", "n_pauses", "n_parallel"])
thdf.to_csv(OUT / "ag_thermal.csv", index=False)
ccraw.dropna(how="all").to_csv(OUT / "ag_cost_raw_core_s.csv")

# ------------------------------------------------------------------ 1. desempenho
f1 = f1.reindex(columns=METHODS + ["oracle"])
best_obs = f1.max(axis=1)
loss = f1.rsub(best_obs, axis=0)
per_ds = f1.copy()
per_ds.columns = [f"f1_{c}" for c in per_ds.columns]
per_ds["best_observed"] = best_obs
for c in METHODS + ["oracle"]:
    per_ds[f"loss_{c}"] = loss[c]
per_ds.to_csv(OUT / "per_dataset.csv")

common = f1[METHODS].dropna().index.tolist()
cover = pd.DataFrame({"n_datasets": f1[METHODS + ["oracle"]].notna().sum()})
print(f"[info] metodos={METHODS}; N comum={len(common)}")
cov_df = cover.copy()
for nm, c in AG_COV.items():
    if nm in cov_df.index:
        cov_df.loc[nm, "n_error"] = c["n_error"]

fc = f1.loc[common] if common else f1.iloc[0:0]
lc = loss.loc[common] if common else loss.iloc[0:0]
rows = []
if common:
    rk = ev.mean_ranks(lc[METHODS], higher_is_better=False)
    mn = lc[METHODS].min(axis=1)
    for m in METHODS + ["oracle"]:
        r = dict(metodo=m, N=len(common), f1_medio=fc[m].mean(), perda_media=lc[m].mean(), perda_mediana=lc[m].median())
        r["vitorias"] = int((lc[m] <= mn + TOL).sum()) if m != "oracle" else np.nan
        r["rank_medio"] = rk[m] if m != "oracle" else np.nan
        r["n_proprio"] = int(f1[m].notna().sum())
        r["f1_medio_proprio"] = f1[m].mean()
        r["custo_med_core_s"] = cc[m].loc[common].median() if m in cc else 0.0
        r["custo_med_wall_s"] = cw[m].loc[common].median() if m in cw else 0.0
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "summary_perf.csv", index=False)
(OUT / "summary_perf.md").write_text(md_table(summ) + "\n")

# Friedman + CD
txt = []
fr_res = {}
for b in AG_MAIN_BUDGETS:
    nm = f"AG_main_{b}"
    cols = [c for c in ["best_avg", "metalearn", "ws_t30", nm] if c in f1]
    if nm not in f1 or "ws_t30" not in f1:
        txt.append(f"== AG main {b}s == pulado (metodos ausentes)\n")
        continue
    sub = loss[cols].dropna()
    fr_res[b] = friedman_block(sub, f"AG main {b}s", txt, OUT / f"cd_ag{b}.png", f"Datasets: N={len(sub)}, AG {b}s")
if "AG_emb_240" in f1 and "ws_t30" in f1:
    cols = [c for c in ["best_avg", "metalearn", "ws_t30", "AG_emb_240"] if c in f1]
    sub = loss[cols].dropna()
    fr_res["emb240"] = friedman_block(sub, "AG+MiniLM 240s", txt, OUT / "cd_agemb240.png", f"Datasets: N={len(sub)}, AG+MiniLM 240s")
else:
    txt.append("== AG+MiniLM 240s == pulado (metodos ausentes)\n")
allm = loss[METHODS].dropna()
if len(allm) >= 3 and len(METHODS) >= 3:
    friedman_block(allm, "todos os metodos (sem oracle)", txt, OUT / "cd_all.png", f"Todos os metodos (N={len(allm)})")
txt.append("cobertura (datasets por metodo):\n" + cov_df.to_string() + "\n")
(OUT / "friedman.txt").write_text("\n".join(txt))

# Wilcoxon
wrows = []
for a in [m for m in ["ws_t30", "metalearn"] if m in f1]:
    for g in AG_NAMES:
        wrows.append(dict(group="vs_AG", a=a, b=g, **paired(f1[a], f1[g])))
if "AG_main_240" in f1 and "AG_nometa_240" in f1:
    wrows.append(dict(group="ablation_meta", a="AG_main_240", b="AG_nometa_240", **paired(f1["AG_main_240"], f1["AG_nometa_240"])))
if "AG_emb_240" in f1 and "AG_main_240" in f1:
    wrows.append(dict(group="ablation_repr", a="AG_emb_240", b="AG_main_240", **paired(f1["AG_emb_240"], f1["AG_main_240"])))
wil = pd.DataFrame(wrows)
if len(wil):
    wil["p_holm"] = holm(wil["p"].values)
    wil.to_csv(OUT / "wilcoxon.csv", index=False)
else:
    pd.DataFrame(columns=["group", "a", "b", "n", "mean_diff", "wins", "ties", "losses", "p", "p_holm"]).to_csv(OUT / "wilcoxon.csv", index=False)

# ------------------------------------------------------------------ 2. custo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pcost = None
if len(common):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pts = {}
    for m in METHODS:
        c = cc[m].loc[common]
        pts[m] = (c.median(), c.quantile(.25), c.quantile(.75), lc[m].mean())
    colors = {"best_avg": "tab:gray", "metalearn": "tab:blue"}
    for m, (med, q1, q3, y) in pts.items():
        col = colors.get(m, "tab:green" if m.startswith("ws_") else "tab:red")
        if "nometa" in m:
            col = "tab:orange"
        if m == "AG_emb_240":
            col = "tab:purple"
        ax.errorbar(med, y, xerr=[[med - q1], [q3 - med]], fmt="D" if m == "AG_emb_240" else "o", color=col,
                    ms=7 if m == "AG_emb_240" else 6, capsize=2, lw=1)
        ax.annotate(label(m), (med, y), textcoords="offset points", xytext=(4, 5), fontsize=7)
    for grp, col in [([f"AG_main_{b}" for b in AG_MAIN_BUDGETS], "tab:red"), ([f"ws_t{t}" for t in WS_T], "tab:green")]:
        g = [m for m in grp if m in pts]
        if len(g) > 1:
            ax.plot([pts[m][0] for m in g], [pts[m][3] for m in g], "-", color=col, alpha=.6)
    ax.axhline(lc["oracle"].mean(), ls="--", color="k", lw=1, label="oracle (10 algoritmos)")
    ax.set_xscale("log")
    ax.set_xlabel("custo online por dataset (nucleo-s, mediana e IQR)")
    ax.set_ylabel("perda media vs melhor observado (menor = melhor)")
    ax.set_title(f"Desempenho vs custo (N={len(common)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "perf_cost.png", dpi=180)
    plt.close(fig)

# offline
p_wall = np.nan
try:
    m_ = re.search(r"Total wall-clock:\s*([\d.]+)s", (DATA / "build.log").read_text())
    p_wall = float(m_.group(1))
except Exception as e:
    warn(f"build.log: {e}")
par_hint = ""
try:
    lg = (DATA / "build.log").read_text()
    par_hint = "log menciona paralelismo" if re.search(r"n_jobs|workers|parallel", lg, re.I) else "log nao informa n de cores"
except Exception:
    pass
ec = pd.read_csv(DATA / "extraction_cost.csv")
x_base_s = float(ec[ec.group.astype(str).str.startswith("base_")].seconds.sum())
x_all_s = float(ec.seconds.sum())
hpo_s = float(pd.read_csv(DATA / "hpo" / "build_cost.csv").total_fit_seconds.sum())
off = pd.DataFrame([
    dict(item="P (CV macro-F1 50x10), wall", segundos=p_wall, nota=par_hint),
    dict(item=f"P, limite superior CPU-equiv (x{args.p_cores})", segundos=p_wall * args.p_cores, nota="assume build com n_cores workers"),
    dict(item="X base (grupos base_* de extraction_cost.csv)", segundos=x_base_s, nota="soma 50 datasets"),
    dict(item="X (todos os grupos, extendido)", segundos=x_all_s, nota="nao usado no break-even"),
    dict(item="matriz de perdas HPO (CPU-s)", segundos=hpo_s, nota="soma build_cost.csv"),
])
off.to_csv(OUT / "offline_cost.csv", index=False)

mc = lambda m: cc[m].loc[common].mean() if (m in cc and len(common)) else np.nan
be_rows = []
for ml, offs in [("metalearn", {"low": p_wall + x_base_s, "high": p_wall * args.p_cores + x_base_s}),
                 ("ws_t30", {"low": p_wall + x_base_s + hpo_s, "high": p_wall * args.p_cores + x_base_s + hpo_s})]:
    if ml not in cc:
        continue
    for g in AG_NAMES:
        diff = mc(g) - mc(ml)
        be_rows.append(dict(ml=ml, ag=g, custo_ml_core_s=mc(ml), custo_ag_core_s=mc(g), offline_baixo_s=offs["low"],
                            offline_alto_s=offs["high"],
                            N_breakeven_baixo=offs["low"] / diff if diff > 0 else np.nan,
                            N_breakeven_alto=offs["high"] / diff if diff > 0 else np.nan))
be = pd.DataFrame(be_rows)
be.to_csv(OUT / "breakeven.csv", index=False)

# orcamento onde AG supera ML
cross_lines = []
for ml in [m for m in ["metalearn", "ws_t30"] if m in f1]:
    ml_f1 = fc[ml].mean() if len(common) else np.nan
    seq = [(mc(f"AG_main_{b}"), fc[f"AG_main_{b}"].mean(), b) for b in AG_MAIN_BUDGETS if f"AG_main_{b}" in f1]
    seq = [s for s in seq if not np.isnan(s[1])]
    msg = f"{label(ml)} (F1 medio {ml_f1:.4f}): "
    if not seq:
        msg += "sem AG main disponivel"
    elif seq[0][1] >= ml_f1:
        msg += f"AG ja supera no menor orcamento ({seq[0][2]}s, F1 {seq[0][1]:.4f})"
    else:
        found = False
        for (c0, f0, b0), (c1, f1_, b1) in zip(seq, seq[1:]):
            if f1_ >= ml_f1:
                fr = (ml_f1 - f0) / (f1_ - f0)
                cx = np.exp(np.log(c0) + fr * (np.log(c1) - np.log(c0)))
                msg += f"cruzamento entre {b0}s e {b1}s; custo interpolado (log) ~{cx:.0f} nucleo-s (~{cx / 4:.0f}s de relogio com 4 CPUs)"
                found = True
                break
        if not found:
            msg += f"AG nao supera ate o maior orcamento testado ({seq[-1][2]}s, F1 {seq[-1][1]:.4f})"
    cross_lines.append(msg)

# ------------------------------------------------------------------ 3. escolhas do AG
agdf = pd.DataFrame(ag_rows)
choice_md = []
if len(agdf):
    def top_fam(ens):
        if not ens:
            return None
        s = {}
        for f, w in ens:
            s[f] = s.get(f, 0) + w
        return max(s, key=s.get)
    agdf["top_family"] = agdf["_ens"].map(top_fam)
    agdf["analog_our_algo"] = agdf["top_family"].map(FAM_ANALOG)
    agdf["in_our_space"] = agdf["analog_our_algo"].map(lambda a: f"analogo aprox.: {a}" if isinstance(a, str) else "sem analogo")
    agdf["ml_algo"] = agdf["dataset_id"].map(rec["ml_algo"])
    agdf["ml_group"] = agdf["ml_algo"].map(OUR_GROUP)
    agdf["ag_group"] = agdf["top_family"].map(FAM_GROUP)
    agdf["agrees_with_ml"] = agdf["ag_group"] == agdf["ml_group"]
    out_ag = agdf.drop(columns="_ens")
    out_ag.to_csv(OUT / "ag_choices.csv", index=False)
    choice_md.append("Mapeamento AG -> nossos algoritmos (aproximado): RF/XT->svd_rf; GBM/XGB/CAT->svd_histgb; KNN->svd_knn; "
                     "LR->tfidf_word_lr; FASTAI/NN_TORCH->minilm_mlp. O AG so-texto usa n-gramas + arvores/NNs, "
                     "fora do espaco exato dos nossos pipelines; o mapa e apenas documentacao.\n")
    for rs, g in agdf.groupby("runset"):
        n = len(g)
        unw, wgt = {}, {}
        for ens in g["_ens"]:
            tot = sum(w for _, w in ens) or 1.0
            for f, w in ens:
                unw[f] = unw.get(f, 0) + 1 / max(len(ens), 1)
                wgt[f] = wgt.get(f, 0) + w / tot
        fam = pd.DataFrame({"freq_nao_ponderada": pd.Series(unw) / n, "freq_ponderada": pd.Series(wgt) / n}).fillna(0)
        fam = fam.sort_values("freq_ponderada", ascending=False).reset_index().rename(columns={"index": "familia"})
        top1 = g["top_family"].value_counts(normalize=True)
        feat_cols = [c for c in g.columns if c.startswith("feat_")]
        st = dict(N=n, frac_ensemble_gt1=g["is_ens"].mean(), niveis_stack_medio=pd.to_numeric(g["n_stack"], errors="coerce").mean(),
                  n_modelos_medio=pd.to_numeric(g["n_models"], errors="coerce").mean(),
                  concordancia_com_ML=g["agrees_with_ml"].mean())
        for c in feat_cols:
            st[c] = pd.to_numeric(g[c], errors="coerce").mean()
        choice_md.append(f"### {rs}\n\n" + md_table(pd.DataFrame([st])) + "\n\nFamilia (por dataset, media):\n\n" + md_table(fam) +
                         "\n\nFamilia top-1 (fracao de datasets): " + ", ".join(f"{k}={v:.2f}" for k, v in top1.items()) + "\n")
else:
    pd.DataFrame().to_csv(OUT / "ag_choices.csv", index=False)
    choice_md.append("Sem resultados AutoGluon ainda.\n")

# ganhos e correlatos
gain_rows, decomp_lines = [], []
big = next((f"AG_main_{b}" for b in sorted(AG_MAIN_BUDGETS, reverse=True) if f"AG_main_{b}" in f1), None)
gains = {}
if big and "ws_t30" in f1:
    gains[f"{big}-ws_t30"] = f1[big] - f1["ws_t30"]
if big and "metalearn" in f1:
    gains[f"{big}-metalearn"] = f1[big] - f1["metalearn"]
if "AG_emb_240" in f1 and "ws_t30" in f1:
    gains["AG_emb_240-ws_t30"] = f1["AG_emb_240"] - f1["ws_t30"]
feats = {c: mf_base[c].reindex(ids) for c in mf_base.columns}
feats_other = {"nn1_dist": ws_nn1,
               "meta_model_loss_cv": pd.Series({d: P.loc[d].max() - P.at[d, rec.at[d, "ml_algo"]] for d in ids if d in rec.index}),
               "n_instances_manifest": manifest["n_instances"].reindex(ids),
               "is_english": (manifest["language"].reindex(ids) == "en").astype(float)}
for gname, gs in gains.items():
    res = []
    for fn, fv in {**feats, **feats_other}.items():
        d = pd.concat([gs, fv], axis=1, keys=["g", "f"]).dropna()
        if len(d) >= 5 and d.f.nunique() > 1 and d.g.nunique() > 1:
            rho, p = spearmanr(d.g, d.f)
            res.append(dict(gain=gname, feature=fn, kind="meta" if fn in feats else "other", rho=rho, p=p, n=len(d)))
    res = pd.DataFrame(res)
    if len(res):
        meta = res[res.kind == "meta"].reindex(res[res.kind == "meta"].rho.abs().sort_values(ascending=False).index).head(8)
        gain_rows.append(pd.concat([meta, res[res.kind == "other"]]))
gc = pd.concat(gain_rows) if gain_rows else pd.DataFrame(columns=["gain", "feature", "kind", "rho", "p", "n"])
gc.to_csv(OUT / "gain_correlates.csv", index=False)

# decomposicao
for agm in [x for x in [big, "AG_emb_240"] if x and x in f1]:
    for mlm in [m for m in ["ws_t30", "metalearn"] if m in f1]:
        d = f1[[agm, mlm, "oracle"]].dropna()
        if len(d):
            beats = int((d[agm] > d.oracle + TOL).sum())
            between = int(((d[agm] > d[mlm] + TOL) & (d[agm] <= d.oracle + TOL)).sum())
            below = int((d[agm] <= d[mlm] + TOL).sum())
            decomp_lines.append(f"{label(agm)} vs {label(mlm)} (N={len(d)}): supera oracle={beats}; entre ML e oracle={between}; abaixo/igual a ML={below}")

# tabela AG vs ML/oracle por run set
agml_rows = []
for g in AG_NAMES:
    cols_ = [c for c in ["ws_t30", "metalearn"] if c in f1]
    d = f1[[g, "oracle"] + cols_].dropna()
    r = dict(run_set=g, N=len(d), f1_AG=d[g].mean() if len(d) else np.nan)
    for c in ["ws_t30", "metalearn"]:
        if c in d:
            pr = paired(d[g], d[c])
            r[f"f1_{c}"] = d[c].mean() if len(d) else np.nan
            r[f"W_vs_{c}"], r[f"T_vs_{c}"], r[f"L_vs_{c}"] = pr["wins"], pr["ties"], pr["losses"]
    r["f1_oracle"] = d["oracle"].mean() if len(d) else np.nan
    r["AG_gt_oracle"] = int((d[g] > d.oracle + TOL).sum())
    agml_rows.append(r)
agml = pd.DataFrame(agml_rows)
agml.to_csv(OUT / "ag_vs_ml.csv", index=False)

# escala de orcamento + gap val-teste
mains = [f"AG_main_{b}" for b in AG_MAIN_BUDGETS if f"AG_main_{b}" in f1]
bs_rows = []
if mains:
    cm = f1[mains].dropna().index
    for g in mains:
        v = pd.Series(AG_VAL.get(g, {})).reindex(cm)
        t_ = f1.loc[cm, g]
        bs_rows.append(dict(run_set=g, N_comum=len(cm), f1_teste=t_.mean(), score_val=v.mean(), gap_val_menos_teste=(v - t_).mean(),
                            gap_mediano=(v - t_).median(), frac_val_gt_teste=float((v > t_).mean())))
bscal = pd.DataFrame(bs_rows)
bscal.to_csv(OUT / "ag_budget_scaling.csv", index=False)
gap_rows = []
for g in AG_NAMES:
    v = pd.Series(AG_VAL.get(g, {}))
    t_ = f1[g].dropna()
    ix = v.index.intersection(t_.index)
    gap_rows.append(dict(run_set=g, N=len(ix), f1_teste=t_.loc[ix].mean(), score_val=v.loc[ix].mean(), gap_val_menos_teste=(v.loc[ix] - t_.loc[ix]).mean()))
gapdf = pd.DataFrame(gap_rows)
gapdf.to_csv(OUT / "ag_val_gap.csv", index=False)

# resumo termico
th_lines = []
if len(thdf):
    for rs, g in thdf.groupby("run_set"):
        pz = g[g.paused_s > 0]
        th_lines.append(dict(run_set=rs, n_runs=len(g), n_pausadas=len(pz), paused_frac_medio=g.paused_frac.mean(),
                             paused_frac_max=g.paused_frac.max(), paused_s_total=g.paused_s.sum(), n_parallel_3=int((g.n_parallel == 3).sum()),
                             n_parallel_2=int((g.n_parallel == 2).sum()), n_parallel_1=int((g.n_parallel == 1).sum())))
thsum = pd.DataFrame(th_lines)
n_pause_log = len(PAUSES)

# ablacoes
abl_lines = []
for _, r in wil[wil.group.isin(["ablation_meta", "ablation_repr"])].iterrows() if len(wil) else []:
    abl_lines.append(f"{label(r.a)} - {label(r.b)}: n={r.n}, dif media F1={r.mean_diff:+.4f}, vit/emp/der={r.wins}/{r.ties}/{r.losses}, p={r.p:.3g}, p_holm={r.p_holm:.3g}")

# ------------------------------------------------------------------ tex
if len(summ):
    t1 = summ[["metodo", "f1_medio", "perda_media", "perda_mediana", "vitorias", "rank_medio"]].copy()
    t1["metodo"] = t1.metodo.map(lambda m: "Oracle" if m == "oracle" else label(m))
    t1.columns = ["Método", "F1 médio", "Perda média", "Perda mediana", "Vitórias", "Rank médio"]
    (OUT / "tex" / "tab_desempenho.tex").write_text(tex_table(t1, f"Desempenho no conjunto de teste (N={len(common)} datasets).", "tab:automl-desempenho", "{:.3f}"))
    t2 = summ[["metodo", "custo_med_core_s", "custo_med_wall_s", "perda_media"]].copy()
    t2["metodo"] = t2.metodo.map(lambda m: "Oracle" if m == "oracle" else label(m))
    t2.columns = ["Método", "Custo mediano (núcleo-s)", "Custo mediano (s relógio)", "Perda média"]
    (OUT / "tex" / "tab_custo.tex").write_text(tex_table(t2, "Custo online por dataset e perda média.", "tab:automl-custo", "{:.2f}"))
if len(be):
    t3 = be[["ml", "ag", "custo_ml_core_s", "custo_ag_core_s", "N_breakeven_baixo", "N_breakeven_alto"]].copy()
    t3["ml"] = t3.ml.map(label)
    t3["ag"] = t3.ag.map(label)
    t3.columns = ["Meta-aprendizado", "AutoGluon", "Custo ML (núcleo-s)", "Custo AG (núcleo-s)", "N break-even (baixo)", "N break-even (alto)"]
    (OUT / "tex" / "tab_breakeven.tex").write_text(tex_table(t3, "Número de novos datasets para amortizar o custo offline.", "tab:automl-breakeven", "{:.1f}"))

# ------------------------------------------------------------------ summary.md
def sh(cmd_path, pat):
    try:
        for line in open(cmd_path):
            if line.startswith(pat):
                return line.split(":", 1)[1].strip()
    except Exception:
        return "?"


import os
ram = sh("/proc/meminfo", "MemTotal")
env = [f"- CPU: {sh('/proc/cpuinfo', 'model name')}; nproc={os.cpu_count()}; RAM total {ram}",
       "- AutoGluon: num_cpus=4 por execucao; 3 execucoes concorrentes ate 2026-11-16 08:30, 2 ate 12:10, 1 depois (ver tabela de pausas termicas); guarda termica: SIGSTOP a >=88C (>=86C de 12:10 a 13:05, >=89C depois), retoma <80C (<78C de 12:10 a 13:05); baselines ML: 1 thread, 8 workers concorrentes; encoding: n_threads registrado em encode_time.csv (core-s = encode_s x n_threads)"]
S = ["# AutoML vs meta-aprendizado: resumo\n", "## Ambiente\n", "\n".join(env), "\n## Cobertura\n",
     md_table(cov_df.reset_index().rename(columns={"index": "metodo"}), "{:.0f}"),
     f"\nIntersecao de datasets (todos os metodos): N={len(common)}\n"]
if WARN:
    S += ["\nAvisos:\n"] + [f"- {w}" for w in WARN] + [""]
S += ["\n## 1. Desempenho (intersecao)\n", md_table(summ) if len(summ) else "sem dados"]
S += ["\n### Friedman / Nemenyi\n", "```\n" + "\n".join(txt) + "\n```"]
S += ["\n### Wilcoxon (F1 teste; wins/ties/losses do primeiro)\n", md_table(wil) if len(wil) else "sem dados"]
S += ["\n## 2. Desempenho vs custo\n", "Figura: perf_cost.png\n", "### Custo offline (s)\n", md_table(off, "{:.1f}"),
      "\nBreak-even N = offline / (custo_AG - custo_ML), custos medios online em nucleo-s. Baixo: P=6697 s como CPU-s; alto: P x " + str(args.p_cores) + ".\n",
      md_table(be, "{:.1f}") if len(be) else "sem dados", "\n### Orcamento onde AG supera ML (interpolacao log-custo, F1 medio)\n",
      "\n".join("- " + l for l in cross_lines) or "sem dados"]
S += ["\n## 3. O que o AutoGluon escolheu\n", "\n".join(choice_md), "\n### Decomposicao do ganho do AG\n",
      "\n".join("- " + l for l in decomp_lines) or "sem dados",
      "\n### AG vs ML e oracle (datasets com AG, ws_t30, metalearn; W/T/L = AG vs o metodo)\n", md_table(agml) if len(agml) else "sem dados",
      "\n### Escala de orcamento AG main (datasets comuns aos 3 orcamentos) e gap validacao-teste\n",
      md_table(bscal) if len(bscal) else "sem dados", "\nGap val-teste por run set (datasets proprios):\n", md_table(gapdf) if len(gapdf) else "sem dados",
      f"\n### Pausas termicas (thermal.log: {n_pause_log} intervalos PAUSE->RESUME)\n",
      "Tempo de relogio dos runs AG sobrepostos a pausas inclui tempo parado (SIGSTOP); custo AG usa wall efetivo (total_wall_s - paused_s); custo bruto salvo em ag_cost_raw_core_s.csv. n_parallel inferido do mtime: 3 se < 2026-11-16 08:30; 2 se < 2026-11-16 12:10; 1 apos. Colunas n_parallel_3/n_parallel_2/n_parallel_1 = contagem de runs por valor. Detalhe: ag_thermal.csv\n",
      md_table(thsum) if len(thsum) else "sem dados",
      "\n### Correlatos do ganho (Spearman)\n", md_table(gc, "{:.3f}") if len(gc) else "sem dados",
      "\n### Ablacoes\n", "\n".join("- " + l for l in abl_lines) or "sem dados"]
(OUT / "summary.md").write_text("\n".join(S) + "\n")
(OUT / "ag_choices.md").write_text("\n".join(choice_md) + "\n")
print(f"[done] escrito em {OUT}")
