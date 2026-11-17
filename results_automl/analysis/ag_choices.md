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

