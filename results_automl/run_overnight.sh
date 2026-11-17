#!/bin/bash
# Rodada noturna da Atividade 4: encoding MiniLM (custo online) + AutoGluon + analise.
# Uso: nohup results_automl/run_overnight.sh >> results_automl/logs/overnight.log 2>&1 &
# Retomavel: cada etapa pula datasets ja concluidos.
cd "$(dirname "$0")/.."
while pgrep -f hs_automl_warmstart_real.py >/dev/null; do sleep 30; done
echo "[$(date +%T)] inicio"
python hs_automl_ml_baselines.py --n-jobs 1 --encoders minilm > results_automl/logs/encode.log 2>&1
echo "[$(date +%T)] encode exit $?"
AG=${AG_PYTHON:-python}
for spec in "main 60" "main 240" "nometa 240" "emb 240" "main 900"; do
  set -- $spec
  $AG hs_automl_autogluon.py --budget $2 --variant $1 --num-cpus 4 --n-parallel 3 > results_automl/logs/ag_$1_$2.log 2>&1
  echo "[$(date +%T)] ag $1 $2 exit $? $(ls results_automl/raw/autogluon/$1_$2 2>/dev/null | wc -l) arquivos"
done
python hs_automl_analysis.py > results_automl/logs/analysis.log 2>&1
echo "[$(date +%T)] analysis exit $?"
echo OVERNIGHT_DONE
