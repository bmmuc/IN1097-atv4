#!/bin/bash
# Completa main 900 s com 1 execucao por vez (menos calor, menos pausas), depois analise.
cd "$(dirname "$0")/.."
AG=${AG_PYTHON:-python}
for try in 1 2 3; do
  n=$(ls results_automl/raw/autogluon/main_900 | wc -l); [ "$n" -ge 50 ] && break
  echo "[$(date +%T)] final main 900 (tem $n) tentativa $try"
  $AG hs_automl_autogluon.py --budget 900 --variant main --num-cpus 4 --n-parallel 1 >> results_automl/logs/ag_main_900.log 2>&1
done
echo "[$(date +%T)] main 900: $(ls results_automl/raw/autogluon/main_900 | wc -l) arquivos"
python hs_automl_analysis.py > results_automl/logs/analysis.log 2>&1
echo "[$(date +%T)] analysis exit $?"
echo FINAL_DONE
