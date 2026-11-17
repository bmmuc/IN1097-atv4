# AutoML versus meta-aprendizagem na detecção de discurso de ódio

Código e resultados brutos da Atividade Prática 4 de IN1097 (Tópicos Avançados
em Agentes Inteligentes 2).

Compara, nos 50 datasets de discurso de ódio das atividades anteriores e sob o
mesmo protocolo, a solução de meta-aprendizagem (Ativ. 1 a 3) com o
**AutoGluon-Tabular 1.6.3** (`presets='best_quality'`).

## Protocolo

- Teste = fold 0 de `StratifiedKFold(5, shuffle=True, random_state=42)` por
  dataset (80/20). Índices em `results_automl/splits/<dataset>.json`.
- Todos os métodos retreinados só no treino; meta-características `X_base`
  do dataset alvo recalculadas no treino.
- Métrica: macro-F1 no teste. Custo em núcleo-segundos.

| método | script | saída bruta |
|---|---|---|
| Melhor médio (AverageRank em P, LODO) | `hs_automl_ml_baselines.py` | `results_automl/raw/recommendations.csv`, `algos_test.csv` |
| Meta-aprendizagem (RegressorOnP + X_base, LODO) | `hs_automl_ml_baselines.py` | idem |
| Meta-aprendizagem + warm-start (GP+EI, T=5/15/30) | `hs_automl_warmstart_real.py` | `results_automl/raw/ws_evals.csv`, `ws_test.csv` |
| AutoGluon 60/240/900 s, só texto | `hs_automl_autogluon.py --variant main` | `results_automl/raw/autogluon/main_<B>/<dataset>.json` |
| AG-nometa 240 s (sem portfólio zeroshot) | `--variant nometa` | `.../nometa_240/` |
| AG+MiniLM 240 s (texto + 384 embeddings) | `--variant emb` | `.../emb_240/` |
| Oráculo (melhor dos 10 algoritmos no teste) | `hs_automl_ml_baselines.py` | `algos_test.csv` |

Custos online: `results_automl/raw/encode_time.csv`, `mf_time.csv`, campos de
tempo nos JSON do AutoGluon. Pausas térmicas por execução:
`results_automl/analysis/ag_thermal.csv` e `results_automl/logs/thermal.log`.

## Resultados por dataset e análises

`results_automl/analysis/`:

- `per_dataset.csv`: F1 e perda de todos os métodos em cada um dos 50 datasets.
- `summary.md`, `summary_perf.csv`: tabela de desempenho e custo.
- `friedman.txt`, `cd_*.png`: Friedman + Nemenyi por orçamento.
- `wilcoxon.csv`: Wilcoxon pareado com Holm.
- `perf_cost.png`, `breakeven.csv`, `offline_cost.csv`: desempenho x custo e amortização.
- `ag_choices.csv/.md`: modelos escolhidos pelo AutoGluon; `gain_correlates.csv`: onde ele ganhou.
- `ag_val_gap.csv`, `ag_budget_scaling.csv`, `ag_vs_ml.csv`.

`results_automl/overlap_check.md`: verificação de sobreposição entre o
meta-conhecimento do AutoGluon (portfólio do TabRepo, datasets tabulares do
OpenML) e os datasets alvo.

## Reprodução

```bash
pip install -r requirements.txt                 # ambiente principal
conda create -n automl python=3.11 && conda activate automl
pip install -r requirements-autogluon.txt       # ambiente do AutoGluon
```

1. Datasets e embeddings (`data_hs/datasets/`, `data_hs/embeddings/`, não
   versionados): `python build_datasets.py` e `python hs_embed_worker.py`, como
   nas atividades anteriores. `data_hs/` já traz `P.csv`, `X.csv`, o manifest e
   a matriz de perdas de HPO (`data_hs/hpo/`).
2. Baselines e meta-aprendizagem:
   `python hs_automl_ml_baselines.py --n-jobs 1 --encoders minilm`
3. Warm-start real: `python hs_automl_warmstart_real.py`
4. AutoGluon (no env `automl`), por orçamento e variante:
   `python hs_automl_autogluon.py --budget 240 --variant main --num-cpus 4 --n-parallel 1`
   (`results_automl/run_overnight.sh` encadeia tudo; use `AG_PYTHON` para
   apontar o Python do env `automl`; `results_automl/thermal_guard.sh` pausa os
   processos acima de 89 °C).
5. Análise: `python hs_automl_analysis.py`

Todos os scripts são retomáveis: pulam datasets já concluídos.

Ambiente usado: AMD Ryzen 5 5500 (6 núcleos/12 threads), 46 GB RAM, só CPU.
