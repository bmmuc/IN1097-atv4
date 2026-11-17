#!/bin/bash
# Encadeia: espera warm-start, encoding (12 threads), depois AutoGluon.
cd "$(dirname "$0")/.."
while pgrep -f hs_automl_warmstart_real.py >/dev/null; do sleep 20; done
python hs_automl_ml_baselines.py --n-jobs 1 --encoders minilm > results_automl/logs/encode.log 2>&1; echo "encode exit $?"
AG=${AG_PYTHON:-python}
for spec in "main 60" "main 240" "nometa 240" "emb 240" "main 900"; do
  set -- $spec
  $AG hs_automl_autogluon.py --budget $2 --variant $1 --num-cpus 4 --n-parallel 3 > results_automl/logs/ag_$1_$2.log 2>&1
  echo "ag $1 $2 exit $? $(ls results_automl/raw/autogluon/$1_$2 | wc -l) files"
done
echo CHAIN_DONE
