# Calibracao de Rosenzweig-MacArthur

O trabalho usa tres etapas separadas: gerar dados, calibrar parametros e analisar os resultados.

## 1. Gerar um dataset

Execute a partir deste diretorio:

```bash
python3 codigo/gerar_dados.py
```

O comando cria, por padrao:

```text
dados/datasets/rosenzweig_piloto_trajectories.npz
dados/datasets/rosenzweig_piloto_metadata.csv
```

Use `--name`, `--output-dir`, `--candidates`, `--observations`, `--noise-fractions`,
`--noise-replicates` e `--seed` para definir um novo dataset. O arquivo NPZ contem
as trajetorias e observacoes; o CSV contem os metadados por cenario.

## 2. Calibrar um dataset

Cada metodo recebe um NPZ previamente gerado e grava os resultados no mesmo formato:

```bash
python3 codigo/executar_calibracao.py \
  --method genetic_algorithm \
  --dataset dados/datasets/rosenzweig_piloto_trajectories.npz \
  --case-start 0 \
  --case-stop 10
```

O algoritmo genetico aceita ainda `--workers`, `--population-size`, `--generations`,
`--noise-fractions`, `--noise-replicates` e `--optimizer-seeds`. Execucoes ja
registradas com a mesma configuracao sao ignoradas.

As saidas padrao sao:

```text
dados/resultados/metricas.csv
dados/resultados/lotes/
dados/resultados/historicos/
```

Todo novo metodo deve usar `experiment_metrics.py` para montar e salvar suas
metricas. Dessa forma, os resultados de AG, redes neurais e PINNs podem ser
comparados no mesmo arquivo.

## 3. Inspecionar e analisar

- `codigo/data_rosenzweig.ipynb` carrega e visualiza um dataset ja criado.
- `codigo/visualizar_resultados.ipynb` le somente os resultados consolidados.
- `codigo/otimizacao_genetica.ipynb` e uma alternativa interativa ao comando de
  calibracao para o algoritmo genetico.
