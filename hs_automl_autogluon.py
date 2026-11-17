#!/usr/bin/env python
"""AutoGluon-Tabular baseline on the 50 hate-speech datasets (text -> AG text features).

Variants: main (best_quality, zeroshot portfolio), nometa (best_quality, hyperparameters='default'),
          emb (main + 384 MiniLM float columns).
Usage:
  $AG_PYTHON hs_automl_autogluon.py --budget 600 --variant main \
      --num-cpus 4 --n-parallel 3 [--datasets id1 id2] [--out results_automl/raw/autogluon]
Output: {out}/{variant}_{budget}/{id}.json (resumable; error JSON has key "error").
"""
import argparse, json, os, shutil, sys, time, traceback, random
import concurrent.futures as cf
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data_hs")
SEED = 42


def get_split(did, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    tr, te = next(iter(skf.split(np.zeros(len(y)), y)))
    tr, te = [int(i) for i in tr], [int(i) for i in te]
    p = os.path.join(ROOT, "results_automl", "splits", f"{did}.json")
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        assert [int(i) for i in s["train_idx"]] == tr and [int(i) for i in s["test_idx"]] == te, \
            f"{did}: split file differs from local computation"
    return np.array(tr), np.array(te)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def feature_summary(predictor):
    """Group generated (post-preprocessing) features by prefix / type."""
    fg = predictor._learner.feature_generator
    names = list(fg.features_out)
    groups = Counter()
    for n in names:
        if n.startswith("text.__nlp__.") or "__nlp__" in n:
            groups["text_ngram"] += 1  # char/word ngram (CountVectorizer/TF-IDF) features
        elif "text" in n and n.endswith(("char_count", "word_count", "capital_ratio", "lower_ratio", "digit_ratio",
                                         "special_ratio", "symbol_count")) or "text." in n:
            groups["text_special"] += 1
        elif n.startswith("e") and n[1:].isdigit():
            groups["embedding_numeric"] += 1
        else:
            groups["other"] += 1
    ftypes, special, inmeta = {}, {}, {}
    try:
        fm = fg.feature_metadata
        ftypes = {str(k): v for k, v in Counter(fm.type_map_raw.values()).items()}
        for st in ("text_ngram", "text_special", "text_as_category", "binned", "datetime_as_int"):
            n = len(fm.get_features(required_special_types=[st]))
            if n:
                special[st] = n
    except Exception as e:
        special = {"error": repr(e)}
    try:
        fmi = predictor.feature_metadata_in
        inmeta = {"raw_types": {str(k): v for k, v in Counter(fmi.type_map_raw.values()).items()},
                  "text_features": fmi.get_features(required_special_types=["text"])}
    except Exception as e:
        inmeta = {"error": repr(e)}
    return {"n_features_out": len(names), "by_group": dict(groups), "out_raw_types": ftypes,
            "out_special_types": special, "feature_metadata_in": inmeta}


def ensemble_info(predictor, best):
    out = {"best_is_weighted_ensemble": False}
    try:
        trainer = predictor._trainer
        info = trainer.get_model_attribute_full if False else None
        mi = predictor.info()["model_info"].get(best, {})
        out["best_model_type"] = mi.get("model_type")
        out["best_stack_level"] = mi.get("stack_level")
        if mi.get("model_type") == "WeightedEnsembleModel" or "WeightedEnsemble" in best:
            out["best_is_weighted_ensemble"] = True
            w = None
            try:
                m = trainer.load_model(best)
                bm = getattr(m, "model_base", None)
                w = getattr(bm, "weights_", None)
                wd = m._get_model_weights()
                out["weights"] = {k: float(v) for k, v in wd.items() if v > 0}
                out["base_models"] = list(getattr(m, "base_model_names", None) or wd.keys())
                out["base_model_types"] = {k: predictor.info()["model_info"].get(k, {}).get("model_type") for k in out["weights"]}
            except Exception as e:
                out["weights_error"] = repr(e)
            if "weights" not in out:
                for k in ("children_info",):
                    ci = mi.get(k, {})
                    for cn, cinfo in ci.items():
                        out["weights"] = cinfo.get("model_weights")
                        out["base_models"] = cinfo.get("base_model_names")
        out["bagged_info"] = {k: mi.get(k) for k in ("num_models", "bagged_mode", "stack_level") if k in mi}
    except Exception as e:
        out["error"] = repr(e)
    return out


def run_one(did, variant, budget, num_cpus, out_dir):
    from sklearn.metrics import f1_score
    from autogluon.tabular import TabularPredictor
    from autogluon.common.features.feature_metadata import FeatureMetadata
    random.seed(SEED); np.random.seed(SEED)
    res_dir = os.path.join(out_dir, f"{variant}_{budget}")
    os.makedirs(res_dir, exist_ok=True)
    jpath = os.path.join(res_dir, f"{did}.json")
    tmp = os.path.join(ROOT, "results_automl", "ag_tmp", f"{variant}_{budget}", did)
    base = {"dataset_id": did, "variant": variant, "budget": budget, "num_cpus": num_cpus}
    t_start = time.perf_counter()
    try:
        df = pd.read_parquet(os.path.join(DATA, "datasets", f"{did}.parquet"))
        texts = df["text"].astype(str).to_numpy()
        y = df["label"].to_numpy()
        assert y.dtype.kind in "iu"
        tr, te = get_split(did, y)
        X = pd.DataFrame({"text": texts})
        if variant == "emb":
            E = np.load(os.path.join(DATA, "embeddings", f"{did}__minilm.npy")).astype(np.float32)
            assert E.shape[0] == len(df)
            X = pd.concat([X, pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])], axis=1)
        X["label"] = y
        train_df, test_df = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ncls = len(np.unique(y[tr]))
        ptype = "binary" if ncls == 2 else "multiclass"
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        predictor = TabularPredictor(label="label", eval_metric="f1_macro", problem_type=ptype, path=tmp,
                                     verbosity=1)
        kw = dict(presets="best_quality", time_limit=budget, num_cpus=num_cpus, num_gpus=0)
        if variant == "nometa":
            kw["hyperparameters"] = "default"
        fm = FeatureMetadata.from_df(train_df.drop(columns=["label"]))
        if fm.type_map_raw.get("text") != "object" or "text" not in fm.get_features(required_special_types=["text"]):
            fm = fm.add_special_types({"text": ["text"]})
        kw["feature_metadata"] = fm
        t0 = time.perf_counter()
        predictor.fit(train_df, **kw)
        fit_wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = predictor.predict(test_df.drop(columns=["label"]))
        pred_wall = time.perf_counter() - t0
        f1 = float(f1_score(test_df["label"].to_numpy(), pred.to_numpy(), average="macro"))
        best = predictor.model_best
        lb = predictor.leaderboard(silent=True)
        keep = ["model", "score_val", "fit_time", "pred_time_val", "stack_level", "can_infer"]
        lb_list = lb[[c for c in keep if c in lb.columns]].to_dict("records")
        info = predictor.info()
        model_types = {m: v.get("model_type") for m, v in info["model_info"].items()}
        fi = predictor.fit_summary if False else None
        dyn = None
        try:
            so = getattr(predictor, "_stacked_overfitting_occurred", None)
            dyn = {"dystack_ran": so is not None, "stacked_overfitting_occurred": so,
                   "n_dystack_subfits": len(getattr(predictor, "_sub_fits", []) or [])}
            dyn["num_stack_levels_run"] = int(max(lb["stack_level"]))
            dyn["any_stack_level_ge2"] = bool((lb["stack_level"] >= 2).any())
        except Exception as e:
            dyn = {"error": repr(e)}
        n_levels = int(lb["stack_level"].max())
        res = dict(base, n_train=int(len(tr)), n_test=int(len(te)), f1_test=f1,
                   fit_wall_s=fit_wall, predict_wall_s=pred_wall, total_wall_s=time.perf_counter() - t_start,
                   best_model=best, leaderboard=lb_list, model_types=model_types,
                   n_models=len(predictor.model_names()), n_stack_levels=n_levels,
                   ensemble=ensemble_info(predictor, best), features=feature_summary(predictor),
                   dynamic_stacking=dyn, ag_version=__import__("autogluon.tabular").tabular.__version__ if False else None)
    except Exception:
        res = dict(base, error=traceback.format_exc(), total_wall_s=time.perf_counter() - t_start)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(jpath, "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    return did, res.get("f1_test"), res.get("total_wall_s"), res.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--variant", choices=["main", "nometa", "emb"], default="main")
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--n-parallel", type=int, default=3)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join("results_automl", "raw", "autogluon"))
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    ids = pd.read_csv(os.path.join(DATA, "datasets_manifest.csv"))["dataset_id"].tolist()
    if a.datasets:
        ids = [d for d in ids if d in set(a.datasets)]
    todo = [d for d in ids if not os.path.exists(os.path.join(out, f"{a.variant}_{a.budget}", f"{d}.json"))]
    print(f"[autogluon] {a.variant} budget={a.budget}: {len(todo)}/{len(ids)} to run", flush=True)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=a.n_parallel, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = [ex.submit(run_one, d, a.variant, a.budget, a.num_cpus, out) for d in todo]
        for fu in cf.as_completed(futs):
            try:
                d, f1, w, err = fu.result()
                print(f"[autogluon] {d} variant={a.variant} f1={f1} wall={w:.1f}s" + (" ERROR" if err else ""), flush=True)
            except Exception as e:
                print(f"[autogluon] worker crash: {e!r}", flush=True)


if __name__ == "__main__":
    main()
