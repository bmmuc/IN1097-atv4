# AutoML vs meta-aprendizado: resumo

## Ambiente

- CPU: AMD Ryzen 5 5500; nproc=12; RAM total 49195004 kB
- AutoGluon: num_cpus=4 por execucao; 3 execucoes concorrentes ate 2026-11-16 08:30, 2 ate 12:10, 1 depois (ver tabela de pausas termicas); guarda termica: SIGSTOP a >=88C (>=86C de 12:10 a 13:05, >=89C depois), retoma <80C (<78C de 12:10 a 13:05); baselines ML: 1 thread, 8 workers concorrentes; encoding: n_threads registrado em encode_time.csv (core-s = encode_s x n_threads)

## Cobertura

| metodo | n_datasets | n_error |
|---|---|---|
| best_avg | 50 | nan |
| metalearn | 50 | nan |
| ws_t5 | 50 | nan |
| ws_t15 | 50 | nan |
| ws_t30 | 50 | nan |
| AG_main_60 | 50 | 0 |
| AG_main_240 | 50 | 0 |
| AG_main_900 | 50 | 0 |
| AG_nometa_240 | 50 | 0 |
| AG_emb_240 | 50 | 0 |
| oracle | 50 | nan |

Intersecao de datasets (todos os metodos): N=50


## 1. Desempenho (intersecao)

| metodo | N | f1_medio | perda_media | perda_mediana | vitorias | rank_medio | n_proprio | f1_medio_proprio | custo_med_core_s | custo_med_wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| best_avg | 50 | 0.7366 | 0.0421 | 0.0380 | 4.0000 | 5.4600 | 50 | 0.7366 | 103.1774 | 10.7892 |
| metalearn | 50 | 0.7471 | 0.0315 | 0.0308 | 9.0000 | 4.5000 | 50 | 0.7471 | 112.7036 | 20.8478 |
| ws_t5 | 50 | 0.7452 | 0.0334 | 0.0283 | 7.0000 | 4.1600 | 50 | 0.7452 | 219.3505 | 119.9872 |
| ws_t15 | 50 | 0.7434 | 0.0352 | 0.0319 | 4.0000 | 4.5900 | 50 | 0.7434 | 377.1873 | 253.0447 |
| ws_t30 | 50 | 0.7414 | 0.0372 | 0.0335 | 4.0000 | 4.6200 | 50 | 0.7414 | 594.0775 | 476.6061 |
| AG_main_60 | 50 | 0.6730 | 0.1056 | 0.0691 | 2.0000 | 7.2600 | 50 | 0.6730 | 296.5422 | 74.1356 |
| AG_main_240 | 50 | 0.6691 | 0.1095 | 0.0745 | 1.0000 | 7.3700 | 50 | 0.6691 | 1117.4930 | 279.3733 |
| AG_main_900 | 50 | 0.6686 | 0.1100 | 0.0717 | 1.0000 | 7.0400 | 50 | 0.6686 | 3401.3907 | 850.3477 |
| AG_nometa_240 | 50 | 0.6654 | 0.1132 | 0.0759 | 3.0000 | 7.3400 | 50 | 0.6654 | 872.0739 | 218.0185 |
| AG_emb_240 | 50 | 0.7626 | 0.0160 | 0.0058 | 24.0000 | 2.6600 | 50 | 0.7626 | 1209.6717 | 281.7165 |
| oracle | 50 | 0.7610 | 0.0176 | 0.0131 | nan | nan | 50 | 0.7610 | 0.0000 | 0.0000 |

### Friedman / Nemenyi

```
== AG main 60s (N=50, k=4) ==
Friedman chi2=33.415 p=2.633e-07
CD(Nemenyi, 0.05)=0.663
ranks medios (1=melhor):
ws_t30        2.00
metalearn     2.17
best_avg      2.51
AG_main_60    3.32
pares significativos: [('Melhor media', 'AG main 60s', 0.81), ('Meta-learning', 'AG main 60s', 1.15), ('ML+WS (T=30)', 'AG main 60s', 1.32)]

== AG main 240s (N=50, k=4) ==
Friedman chi2=34.296 p=1.716e-07
CD(Nemenyi, 0.05)=0.663
ranks medios (1=melhor):
ws_t30         2.02
metalearn      2.15
best_avg       2.49
AG_main_240    3.34
pares significativos: [('Melhor media', 'AG main 240s', 0.85), ('Meta-learning', 'AG main 240s', 1.19), ('ML+WS (T=30)', 'AG main 240s', 1.32)]

== AG main 900s (N=50, k=4) ==
Friedman chi2=31.730 p=5.966e-07
CD(Nemenyi, 0.05)=0.663
ranks medios (1=melhor):
ws_t30         2.02
metalearn      2.17
best_avg       2.51
AG_main_900    3.30
pares significativos: [('Melhor media', 'AG main 900s', 0.79), ('Meta-learning', 'AG main 900s', 1.13), ('ML+WS (T=30)', 'AG main 900s', 1.28)]

== AG+MiniLM 240s (N=50, k=4) ==
Friedman chi2=29.922 p=1.433e-06
CD(Nemenyi, 0.05)=0.663
ranks medios (1=melhor):
AG_emb_240    1.72
ws_t30        2.58
metalearn     2.68
best_avg      3.02
pares significativos: [('Melhor media', 'AG+MiniLM 240s', 1.3), ('Meta-learning', 'AG+MiniLM 240s', 0.96), ('ML+WS (T=30)', 'AG+MiniLM 240s', 0.86)]

== todos os metodos (sem oracle) (N=50, k=10) ==
Friedman chi2=138.350 p=2.259e-25
CD(Nemenyi, 0.05)=1.916
ranks medios (1=melhor):
AG_emb_240       2.66
ws_t5            4.16
metalearn        4.50
ws_t15           4.59
ws_t30           4.62
best_avg         5.46
AG_main_900      7.04
AG_main_60       7.26
AG_nometa_240    7.34
AG_main_240      7.37
pares significativos: [('Melhor media', 'AG+MiniLM 240s', 2.8), ('Meta-learning', 'AG main 60s', 2.76), ('Meta-learning', 'AG main 240s', 2.87), ('Meta-learning', 'AG main 900s', 2.54), ('Meta-learning', 'AG nometa 240s', 2.84), ('ML+WS (T=5)', 'AG main 60s', 3.1), ('ML+WS (T=5)', 'AG main 240s', 3.21), ('ML+WS (T=5)', 'AG main 900s', 2.88), ('ML+WS (T=5)', 'AG nometa 240s', 3.18), ('ML+WS (T=15)', 'AG main 60s', 2.67), ('ML+WS (T=15)', 'AG main 240s', 2.78), ('ML+WS (T=15)', 'AG main 900s', 2.45), ('ML+WS (T=15)', 'AG nometa 240s', 2.75), ('ML+WS (T=15)', 'AG+MiniLM 240s', 1.93), ('ML+WS (T=30)', 'AG main 60s', 2.64), ('ML+WS (T=30)', 'AG main 240s', 2.75), ('ML+WS (T=30)', 'AG main 900s', 2.42), ('ML+WS (T=30)', 'AG nometa 240s', 2.72), ('ML+WS (T=30)', 'AG+MiniLM 240s', 1.96), ('AG main 60s', 'AG+MiniLM 240s', 4.6), ('AG main 240s', 'AG+MiniLM 240s', 4.71), ('AG main 900s', 'AG+MiniLM 240s', 4.38), ('AG nometa 240s', 'AG+MiniLM 240s', 4.68)]

cobertura (datasets por metodo):
               n_datasets  n_error
best_avg               50      NaN
metalearn              50      NaN
ws_t5                  50      NaN
ws_t15                 50      NaN
ws_t30                 50      NaN
AG_main_60             50      0.0
AG_main_240            50      0.0
AG_main_900            50      0.0
AG_nometa_240          50      0.0
AG_emb_240             50      0.0
oracle                 50      NaN

```

### Wilcoxon (F1 teste; wins/ties/losses do primeiro)

| group | a | b | n | mean_diff | wins | ties | losses | p | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| vs_AG | ws_t30 | AG_main_60 | 50 | 0.0684 | 39 | 0 | 11 | 0.0000 | 0.0000 |
| vs_AG | ws_t30 | AG_main_240 | 50 | 0.0724 | 38 | 0 | 12 | 0.0000 | 0.0000 |
| vs_AG | ws_t30 | AG_main_900 | 50 | 0.0728 | 38 | 0 | 12 | 0.0000 | 0.0000 |
| vs_AG | ws_t30 | AG_nometa_240 | 50 | 0.0761 | 38 | 0 | 12 | 0.0000 | 0.0000 |
| vs_AG | ws_t30 | AG_emb_240 | 50 | -0.0212 | 10 | 0 | 40 | 0.0000 | 0.0000 |
| vs_AG | metalearn | AG_main_60 | 50 | 0.0741 | 41 | 0 | 9 | 0.0000 | 0.0000 |
| vs_AG | metalearn | AG_main_240 | 50 | 0.0780 | 42 | 0 | 8 | 0.0000 | 0.0000 |
| vs_AG | metalearn | AG_main_900 | 50 | 0.0785 | 41 | 0 | 9 | 0.0000 | 0.0000 |
| vs_AG | metalearn | AG_nometa_240 | 50 | 0.0817 | 42 | 0 | 8 | 0.0000 | 0.0000 |
| vs_AG | metalearn | AG_emb_240 | 50 | -0.0156 | 14 | 2 | 34 | 0.0013 | 0.0025 |
| ablation_meta | AG_main_240 | AG_nometa_240 | 50 | 0.0037 | 20 | 10 | 20 | 0.6707 | 0.6707 |
| ablation_repr | AG_emb_240 | AG_main_240 | 50 | 0.0936 | 43 | 0 | 7 | 0.0000 | 0.0000 |

## 2. Desempenho vs custo

Figura: perf_cost.png

### Custo offline (s)

| item | segundos | nota |
|---|---|---|
| P (CV macro-F1 50x10), wall | 6697.2 | log nao informa n de cores |
| P, limite superior CPU-equiv (x12) | 80366.4 | assume build com n_cores workers |
| X base (grupos base_* de extraction_cost.csv) | 361.0 | soma 50 datasets |
| X (todos os grupos, extendido) | 2489.8 | nao usado no break-even |
| matriz de perdas HPO (CPU-s) | 227662.0 | soma build_cost.csv |

Break-even N = offline / (custo_AG - custo_ML), custos medios online em nucleo-s. Baixo: P=6697 s como CPU-s; alto: P x 12.

| ml | ag | custo_ml_core_s | custo_ag_core_s | offline_baixo_s | offline_alto_s | N_breakeven_baixo | N_breakeven_alto |
|---|---|---|---|---|---|---|---|
| metalearn | AG_main_60 | 116.8 | 296.7 | 7058.2 | 80727.4 | 39.2 | 448.7 |
| metalearn | AG_main_240 | 116.8 | 1117.0 | 7058.2 | 80727.4 | 7.1 | 80.7 |
| metalearn | AG_main_900 | 116.8 | 3438.0 | 7058.2 | 80727.4 | 2.1 | 24.3 |
| metalearn | AG_nometa_240 | 116.8 | 866.8 | 7058.2 | 80727.4 | 9.4 | 107.6 |
| metalearn | AG_emb_240 | 116.8 | 1198.8 | 7058.2 | 80727.4 | 6.5 | 74.6 |
| ws_t30 | AG_main_60 | 597.0 | 296.7 | 234720.1 | 308389.3 | nan | nan |
| ws_t30 | AG_main_240 | 597.0 | 1117.0 | 234720.1 | 308389.3 | 451.4 | 593.0 |
| ws_t30 | AG_main_900 | 597.0 | 3438.0 | 234720.1 | 308389.3 | 82.6 | 108.6 |
| ws_t30 | AG_nometa_240 | 597.0 | 866.8 | 234720.1 | 308389.3 | 869.9 | 1143.0 |
| ws_t30 | AG_emb_240 | 597.0 | 1198.8 | 234720.1 | 308389.3 | 390.0 | 512.4 |

### Orcamento onde AG supera ML (interpolacao log-custo, F1 medio)

- Meta-learning (F1 medio 0.7471): AG nao supera ate o maior orcamento testado (900s, F1 0.6686)
- ML+WS (T=30) (F1 medio 0.7414): AG nao supera ate o maior orcamento testado (900s, F1 0.6686)

## 3. O que o AutoGluon escolheu

Mapeamento AG -> nossos algoritmos (aproximado): RF/XT->svd_rf; GBM/XGB/CAT->svd_histgb; KNN->svd_knn; LR->tfidf_word_lr; FASTAI/NN_TORCH->minilm_mlp. O AG so-texto usa n-gramas + arvores/NNs, fora do espaco exato dos nossos pipelines; o mapa e apenas documentacao.

### AG_emb_240

| N | frac_ensemble_gt1 | niveis_stack_medio | n_modelos_medio | concordancia_com_ML | feat_text_special | feat_text_ngram | feat_embedding_numeric |
|---|---|---|---|---|---|---|---|
| 50.0000 | 0.3600 | 2.7600 | 13.8600 | 0.3200 | 23.5400 | 185.1200 | 381.9200 |

Familia (por dataset, media):

| familia | freq_nao_ponderada | freq_ponderada |
|---|---|---|
| GBM | 0.6361 | 0.6410 |
| FASTAI | 0.1542 | 0.2123 |
| NN_TORCH | 0.0600 | 0.0600 |
| CAT | 0.0607 | 0.0380 |
| XT | 0.0566 | 0.0277 |
| RF | 0.0324 | 0.0210 |

Familia top-1 (fracao de datasets): GBM=0.60, FASTAI=0.32, NN_TORCH=0.06, XT=0.02

### AG_main_240

| N | frac_ensemble_gt1 | niveis_stack_medio | n_modelos_medio | concordancia_com_ML | feat_text_special | feat_text_ngram | feat_embedding_numeric |
|---|---|---|---|---|---|---|---|
| 50.0000 | 0.2600 | 2.6800 | 21.8200 | 0.2800 | 23.5400 | 185.1200 | nan |

Familia (por dataset, media):

| familia | freq_nao_ponderada | freq_ponderada |
|---|---|---|
| GBM | 0.5151 | 0.5334 |
| FASTAI | 0.2479 | 0.3043 |
| XGB | 0.1417 | 0.1286 |
| XT | 0.0540 | 0.0184 |
| RF | 0.0151 | 0.0075 |
| CAT | 0.0140 | 0.0051 |
| NN_TORCH | 0.0122 | 0.0027 |

Familia top-1 (fracao de datasets): GBM=0.52, FASTAI=0.34, XGB=0.12, XT=0.02

### AG_main_60

| N | frac_ensemble_gt1 | niveis_stack_medio | n_modelos_medio | concordancia_com_ML | feat_text_special | feat_text_ngram | feat_embedding_numeric |
|---|---|---|---|---|---|---|---|
| 50.0000 | 0.0800 | 2.9800 | 8.4600 | 0.0800 | 23.5400 | 185.1200 | nan |

Familia (por dataset, media):

| familia | freq_nao_ponderada | freq_ponderada |
|---|---|---|
| GBM | 0.8800 | 0.8970 |
| FASTAI | 0.1000 | 0.0997 |
| CAT | 0.0100 | 0.0018 |
| RF | 0.0100 | 0.0015 |

Familia top-1 (fracao de datasets): GBM=0.90, FASTAI=0.10

### AG_main_900

| N | frac_ensemble_gt1 | niveis_stack_medio | n_modelos_medio | concordancia_com_ML | feat_text_special | feat_text_ngram | feat_embedding_numeric |
|---|---|---|---|---|---|---|---|
| 50.0000 | 0.4000 | 2.5800 | 54.2600 | 0.3600 | 23.5400 | 185.1200 | nan |

Familia (por dataset, media):

| familia | freq_nao_ponderada | freq_ponderada |
|---|---|---|
| GBM | 0.4030 | 0.3882 |
| FASTAI | 0.2102 | 0.2514 |
| XGB | 0.1717 | 0.1606 |
| NN_TORCH | 0.1128 | 0.1447 |
| XT | 0.0390 | 0.0286 |
| RF | 0.0375 | 0.0156 |
| CAT | 0.0258 | 0.0108 |

Familia top-1 (fracao de datasets): GBM=0.38, FASTAI=0.30, XGB=0.16, NN_TORCH=0.14, XT=0.02

### AG_nometa_240

| N | frac_ensemble_gt1 | niveis_stack_medio | n_modelos_medio | concordancia_com_ML | feat_text_special | feat_text_ngram | feat_embedding_numeric |
|---|---|---|---|---|---|---|---|
| 50.0000 | 0.3000 | 2.7200 | 19.2800 | 0.3400 | 23.5400 | 185.1200 | nan |

Familia (por dataset, media):

| familia | freq_nao_ponderada | freq_ponderada |
|---|---|---|
| GBM | 0.5442 | 0.5318 |
| FASTAI | 0.2608 | 0.3162 |
| NN_TORCH | 0.0742 | 0.0607 |
| XGB | 0.0375 | 0.0574 |
| XT | 0.0267 | 0.0157 |
| RF | 0.0475 | 0.0136 |
| CAT | 0.0092 | 0.0046 |

Familia top-1 (fracao de datasets): GBM=0.52, FASTAI=0.38, XGB=0.06, NN_TORCH=0.04


### Decomposicao do ganho do AG

- AG main 900s vs ML+WS (T=30) (N=50): supera oracle=3; entre ML e oracle=9; abaixo/igual a ML=38
- AG main 900s vs Meta-learning (N=50): supera oracle=3; entre ML e oracle=6; abaixo/igual a ML=41
- AG+MiniLM 240s vs ML+WS (T=30) (N=50): supera oracle=24; entre ML e oracle=16; abaixo/igual a ML=10
- AG+MiniLM 240s vs Meta-learning (N=50): supera oracle=24; entre ML e oracle=10; abaixo/igual a ML=16

### AG vs ML e oracle (datasets com AG, ws_t30, metalearn; W/T/L = AG vs o metodo)

| run_set | N | f1_AG | f1_ws_t30 | W_vs_ws_t30 | T_vs_ws_t30 | L_vs_ws_t30 | f1_metalearn | W_vs_metalearn | T_vs_metalearn | L_vs_metalearn | f1_oracle | AG_gt_oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AG_main_60 | 50 | 0.6730 | 0.7414 | 11 | 0 | 39 | 0.7471 | 9 | 0 | 41 | 0.7610 | 4 |
| AG_main_240 | 50 | 0.6691 | 0.7414 | 12 | 0 | 38 | 0.7471 | 8 | 0 | 42 | 0.7610 | 2 |
| AG_main_900 | 50 | 0.6686 | 0.7414 | 12 | 0 | 38 | 0.7471 | 9 | 0 | 41 | 0.7610 | 3 |
| AG_nometa_240 | 50 | 0.6654 | 0.7414 | 12 | 0 | 38 | 0.7471 | 8 | 0 | 42 | 0.7610 | 3 |
| AG_emb_240 | 50 | 0.7626 | 0.7414 | 40 | 0 | 10 | 0.7471 | 34 | 2 | 14 | 0.7610 | 24 |

### Escala de orcamento AG main (datasets comuns aos 3 orcamentos) e gap validacao-teste

| run_set | N_comum | f1_teste | score_val | gap_val_menos_teste | gap_mediano | frac_val_gt_teste |
|---|---|---|---|---|---|---|
| AG_main_60 | 50 | 0.6730 | 0.6858 | 0.0127 | 0.0097 | 0.6800 |
| AG_main_240 | 50 | 0.6691 | 0.7015 | 0.0324 | 0.0239 | 0.7600 |
| AG_main_900 | 50 | 0.6686 | 0.7107 | 0.0421 | 0.0302 | 0.7800 |

Gap val-teste por run set (datasets proprios):

| run_set | N | f1_teste | score_val | gap_val_menos_teste |
|---|---|---|---|---|
| AG_main_60 | 50 | 0.6730 | 0.6858 | 0.0127 |
| AG_main_240 | 50 | 0.6691 | 0.7015 | 0.0324 |
| AG_main_900 | 50 | 0.6686 | 0.7107 | 0.0421 |
| AG_nometa_240 | 50 | 0.6654 | 0.7038 | 0.0384 |
| AG_emb_240 | 50 | 0.7626 | 0.7680 | 0.0053 |

### Pausas termicas (thermal.log: 381 intervalos PAUSE->RESUME)

Tempo de relogio dos runs AG sobrepostos a pausas inclui tempo parado (SIGSTOP); custo AG usa wall efetivo (total_wall_s - paused_s); custo bruto salvo em ag_cost_raw_core_s.csv. n_parallel inferido do mtime: 3 se < 2026-11-16 08:30; 2 se < 2026-11-16 12:10; 1 apos. Colunas n_parallel_3/n_parallel_2/n_parallel_1 = contagem de runs por valor. Detalhe: ag_thermal.csv

| run_set | n_runs | n_pausadas | paused_frac_medio | paused_frac_max | paused_s_total | n_parallel_3 | n_parallel_2 | n_parallel_1 |
|---|---|---|---|---|---|---|---|---|
| AG_emb_240 | 50 | 0 | 0.0000 | 0.0000 | 0.0000 | 50 | 0 | 0 |
| AG_main_240 | 50 | 0 | 0.0000 | 0.0000 | 0.0000 | 50 | 0 | 0 |
| AG_main_60 | 50 | 0 | 0.0000 | 0.0000 | 0.0000 | 50 | 0 | 0 |
| AG_main_900 | 50 | 44 | 0.1245 | 0.2925 | 6080.8244 | 23 | 12 | 15 |
| AG_nometa_240 | 50 | 35 | 0.1070 | 0.2533 | 1307.9180 | 26 | 24 | 0 |

### Correlatos do ganho (Spearman)

| gain | feature | kind | rho | p | n |
|---|---|---|---|---|---|
| AG_main_900-ws_t30 | is_english | meta | -0.449 | 0.001 | 50 |
| AG_main_900-ws_t30 | is_multilingual_row | meta | 0.449 | 0.001 | 50 |
| AG_main_900-ws_t30 | vocab_size | meta | 0.365 | 0.009 | 50 |
| AG_main_900-ws_t30 | log_vocab_size | meta | 0.365 | 0.009 | 50 |
| AG_main_900-ws_t30 | instances_per_class | meta | 0.364 | 0.009 | 50 |
| AG_main_900-ws_t30 | mean_mentions_per_doc | meta | 0.358 | 0.011 | 50 |
| AG_main_900-ws_t30 | pct_docs_with_mention | meta | 0.357 | 0.011 | 50 |
| AG_main_900-ws_t30 | mean_mutual_info | meta | -0.348 | 0.013 | 50 |
| AG_main_900-ws_t30 | nn1_dist | other | -0.074 | 0.608 | 50 |
| AG_main_900-ws_t30 | meta_model_loss_cv | other | 0.119 | 0.411 | 50 |
| AG_main_900-ws_t30 | n_instances_manifest | other | 0.302 | 0.033 | 50 |
| AG_main_900-metalearn | std_doc_len_words | meta | -0.338 | 0.016 | 50 |
| AG_main_900-metalearn | is_twitter | meta | 0.323 | 0.022 | 50 |
| AG_main_900-metalearn | instances_per_class | meta | 0.316 | 0.025 | 50 |
| AG_main_900-metalearn | n_instances | meta | 0.304 | 0.032 | 50 |
| AG_main_900-metalearn | log_instances | meta | 0.304 | 0.032 | 50 |
| AG_main_900-metalearn | knn1_disagreement | meta | 0.300 | 0.034 | 50 |
| AG_main_900-metalearn | uppercase_char_ratio | meta | -0.288 | 0.043 | 50 |
| AG_main_900-metalearn | pct_docs_with_url | meta | 0.284 | 0.046 | 50 |
| AG_main_900-metalearn | nn1_dist | other | -0.143 | 0.323 | 50 |
| AG_main_900-metalearn | meta_model_loss_cv | other | 0.258 | 0.070 | 50 |
| AG_main_900-metalearn | n_instances_manifest | other | 0.304 | 0.032 | 50 |
| AG_emb_240-ws_t30 | vocab_size | meta | 0.389 | 0.005 | 50 |
| AG_emb_240-ws_t30 | log_vocab_size | meta | 0.389 | 0.005 | 50 |
| AG_emb_240-ws_t30 | n_instances | meta | 0.309 | 0.029 | 50 |
| AG_emb_240-ws_t30 | log_instances | meta | 0.309 | 0.029 | 50 |
| AG_emb_240-ws_t30 | pct_docs_with_url | meta | 0.291 | 0.040 | 50 |
| AG_emb_240-ws_t30 | is_twitter | meta | 0.274 | 0.054 | 50 |
| AG_emb_240-ws_t30 | instances_per_class | meta | 0.257 | 0.072 | 50 |
| AG_emb_240-ws_t30 | mean_hashtags_per_doc | meta | 0.253 | 0.077 | 50 |
| AG_emb_240-ws_t30 | nn1_dist | other | -0.083 | 0.567 | 50 |
| AG_emb_240-ws_t30 | meta_model_loss_cv | other | 0.057 | 0.695 | 50 |
| AG_emb_240-ws_t30 | n_instances_manifest | other | 0.309 | 0.029 | 50 |

### Ablacoes

- AG main 240s - AG nometa 240s: n=50, dif media F1=+0.0037, vit/emp/der=20/10/20, p=0.671, p_holm=0.671
- AG+MiniLM 240s - AG main 240s: n=50, dif media F1=+0.0936, vit/emp/der=43/0/7, p=5.61e-11, p_holm=6.73e-10
