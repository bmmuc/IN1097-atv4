# Sobreposição com o meta-conhecimento do AutoGluon

- Versão: AutoGluon 1.6.3 (`autogluon.tabular`), env conda `automl`.
- `presets='best_quality'` usa `hyperparameters='zeroshot'`, que em
  `autogluon/tabular/configs/hyperparameter_configs.py:151` é
  `hyperparameter_portfolio_zeroshot_2023`. O arquivo
  `configs/zeroshot/zeroshot_portfolio_2023.py` declara na linha 1:
  "Portfolio learned from zeroshot-HPO using TabRepo". O portfólio tem 110
  configurações: NN_TORCH 21, GBM 16, CAT 20, XGB 10, FASTAI 23, RF 10, XT 10.
- O TabRepo (Salinas & Erickson, 2023) foi construído sobre datasets
  **tabulares** do OpenML (suítes do AutoML Benchmark / OpenML-CC18). Nenhum
  deles é um corpus de texto, nem de discurso de ódio.
- Os 50 datasets daqui vêm do HuggingFace (32) e de cópias locais fornecidas
  pelos autores (18: IHC, SBIC, HateWiC, HatEval). Ver
  `data_hs/datasets_manifest.csv`, coluna `provenance`. Nenhum vem do OpenML.
- Conclusão: o AutoGluon nunca "viu" um dataset-alvo. O meta-conhecimento dele
  é um portfólio de hiperparâmetros independente do domínio, sem informação
  específica de texto. O leave-one-dataset-out fica preservado.
- Ablação (`variant=nometa`): `hyperparameters='default'`, as 13 configurações
  feitas à mão pela equipe do AutoGluon, sem o portfólio do TabRepo. Mesmo
  orçamento de 240 s.
  - Resultado: AG main 240 − AG nometa 240 = +0,0037 de F1 médio,
    20/10/20 vitórias/empates/derrotas, Wilcoxon p = 0,67 (n.s.).
  - Ressalva: 35/50 execuções `nometa` ficaram, em média, 10,7% do tempo
    pausadas pela guarda térmica, o que reduziu o orçamento efetivo
    (`results_automl/analysis/ag_thermal.csv`).
