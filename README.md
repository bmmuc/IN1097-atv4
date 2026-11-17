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

## Dados

Só são versionados dados derivados e agregados por dataset (`P.csv`, `X.csv`,
manifest, matriz de perdas de HPO) e os resultados. Os corpora amostrados
(`data_hs/datasets/*.parquet`) e os embeddings (`data_hs/embeddings/*.npy`)
ficam fora do repositório: as fontes têm termos de uso próprios, que não
permitem redistribuir os textos.

32 dos 50 datasets vêm do Hugging Face e são baixados por `build_datasets.py`
(exige rede e `huggingface-cli login`):

| fonte | identificador no Hugging Face |
|---|---|
| Davidson et al. (2017) | `tdavidson/hate_speech_offensive` |
| ETHOS | `iamollas/ethos` |
| HateCheck | `Paul/hatecheck` |
| HateXplain | `Hate-speech-CNERG/hatexplain` |
| HateBR | `ruanchaves/hatebr` |
| MLMA | `nedjmaou/MLMA_hate_speech` |
| Measuring Hate Speech | `ucberkeley-dlab/measuring-hate-speech` |
| TweetEval | `cardiffnlp/tweet_eval` |
| toxic_conversations | `SetFit/toxic_conversations` |

Os outros 18 (IHC, SBIC, HateWiC, HatEval) vêm de cópias das fontes obtidas
junto aos autores originais, mediante aceite dos termos de cada uma. Elas são
lidas de `$TCC_ROOT` (padrão `./fontes_locais/`):

```
$TCC_ROOT/implicit-hate-corpus/   implicit_hate_v1_stg{1,2,3}_posts.tsv   (IHC)
$TCC_ROOT/sbic/                   sbic_{dev,test}_filled_v*.csv            (SBIC)
$TCC_ROOT/hatewic/                HateWiC_IndividualAnnos_with_def.csv     (HateWiC)
$TCC_ROOT/hateval/                arquivos do SemEval-2019 Task 5          (HatEval)
```

Sem elas, `build_datasets.py` constrói só os 32 datasets do Hugging Face e
registra os descartados; os scripts desta atividade rodam sobre o subconjunto
disponível. A coluna `provenance` de `data_hs/datasets_manifest.csv` indica a
origem de cada dataset. Todas as análises de `results_automl/analysis/` podem
ser refeitas sem os corpora, só com `results_automl/raw/`
(`python hs_automl_analysis.py`).

## Reprodução

```bash
pip install -r requirements.txt                 # ambiente principal
conda create -n automl python=3.11 && conda activate automl
pip install -r requirements-autogluon.txt       # ambiente do AutoGluon
```

Ordem dos passos (cada script é retomável e pula datasets já concluídos):

| passo | comando | saída |
|---|---|---|
| 1 | `python build_datasets.py` | `data_hs/datasets/*.parquet` |
| 2 | `python hs_build_metadataset.py` | `data_hs/embeddings/*.npy` (MiniLM, hateBERT, mBERT) |
| 3 | `python hs_automl_ml_baselines.py --n-jobs 1 --encoders minilm` | partições, `algos_test.csv`, `encode_time.csv`, `metafeatures_train.csv`, `mf_time.csv`, `recommendations.csv` |
| 4 | `python hs_automl_warmstart_real.py` | `ws_evals.csv`, `ws_test.csv` (precisa do passo 3) |
| 5 | `$AG_PYTHON hs_automl_autogluon.py --budget B --variant V --num-cpus 4 --n-parallel 1` para `main` 60/240/900, `nometa` 240 e `emb` 240 | `results_automl/raw/autogluon/V_B/*.json` |
| 6 | `python hs_automl_analysis.py` | `results_automl/analysis/` |

`AG_PYTHON` é o Python do env `automl`. O passo 2 também refaz a matriz `P` das
atividades anteriores (demorado); para esta atividade só os embeddings são
necessários, e `P.csv` já está versionado.

`results_automl/run_overnight.sh` é o script que usei para os passos 5 e 6 depois
do passo 4: ele espera o warm-start terminar, roda de novo o passo 3 só para
medir o custo de encoding, executa as cinco configurações do AutoGluon e a
análise. Os demais `run_*.sh` retomaram execuções interrompidas.
`results_automl/thermal_guard.sh` pausa os processos (SIGSTOP) quando a CPU passa
de 89 °C e os retoma abaixo de 80 °C; as pausas são descontadas do custo.

**Custo.** Os núcleo-segundos são estimados como tempo de relógio efetivo
(sem pausas) vezes núcleos alocados: 4 por execução do AutoGluon, 12 no
encoding, 1 nos demais passos. Não é uma medida direta do uso de CPU.

Ambiente usado: AMD Ryzen 5 5500 (6 núcleos/12 threads), 46 GB RAM, só CPU.
