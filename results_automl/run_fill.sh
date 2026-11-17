#!/bin/bash
# Completa execucoes do AutoGluon interrompidas (pool quebrado) e refaz a analise.
cd "$(dirname "$0")/.."
AG=${AG_PYTHON:-python}
for spec in "main 60" "main 240" "nometa 240" "emb 240" "main 900"; do
  set -- $spec
  for try in 1 2 3; do
    n=$(ls results_automl/raw/autogluon/$1_$2 2>/dev/null | wc -l)
    [ "$n" -ge 50 ] && break
    echo "[$(date +%T)] refill $1 $2 (tem $n) tentativa $try"
    $AG hs_automl_autogluon.py --budget $2 --variant $1 --num-cpus 4 --n-parallel 2 >> results_automl/logs/ag_$1_$2.log 2>&1
  done
  echo "[$(date +%T)] $1 $2: $(ls results_automl/raw/autogluon/$1_$2 | wc -l) arquivos"
done
python hs_automl_analysis.py > results_automl/logs/analysis.log 2>&1
echo "[$(date +%T)] analysis exit $?"
echo FILL_DONE
