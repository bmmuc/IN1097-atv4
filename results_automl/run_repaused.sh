#!/bin/bash
# Refaz as execucoes do AutoGluon que ficaram >5% do tempo pausadas pela guarda termica
# (o time_limit do AutoGluon e de parede, entao a pausa consumiu orcamento).
# Os JSONs originais vao para raw/autogluon_paused/ (mantidos para registro).
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_fill.sh" >/dev/null; do sleep 30; done
python hs_automl_analysis.py > results_automl/logs/analysis.log 2>&1
python - <<'PY'
import pandas as pd, shutil, os
t = pd.read_csv('results_automl/analysis/ag_thermal.csv')
t = t[t.paused_frac > 0.05]
for _, r in t.iterrows():
    rs = r.run_set.replace('AG_', '')
    src = f'results_automl/raw/autogluon/{rs}/{r.dataset_id}.json'
    dst = f'results_automl/raw/autogluon_paused/{rs}/{r.dataset_id}.json'
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(src, dst); print('moved', src)
PY
AG=${AG_PYTHON:-python}
for spec in "nometa 240" "main 900"; do
  set -- $spec
  for try in 1 2 3; do
    n=$(ls results_automl/raw/autogluon/$1_$2 | wc -l); [ "$n" -ge 50 ] && break
    echo "[$(date +%T)] rerun $1 $2 (tem $n) tentativa $try"
    $AG hs_automl_autogluon.py --budget $2 --variant $1 --num-cpus 4 --n-parallel 2 >> results_automl/logs/ag_$1_$2.log 2>&1
  done
  echo "[$(date +%T)] $1 $2: $(ls results_automl/raw/autogluon/$1_$2 | wc -l) arquivos"
done
python hs_automl_analysis.py > results_automl/logs/analysis.log 2>&1
echo "[$(date +%T)] analysis exit $?"
echo REPAUSED_DONE
