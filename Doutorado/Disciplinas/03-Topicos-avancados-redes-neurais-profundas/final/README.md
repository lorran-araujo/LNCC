# Treinamento de Redes Neurais para Classificacao de Imagens

Projeto base em PyTorch para classificacao de imagens com suporte a CIFAR-10 e datasets locais no formato `ImageFolder`.

## Uso

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute o treinamento:

```bash
python prepare_features.py --config config.yaml
python main.py --config config.yaml
```

## Dataset

Para usar CIFAR-10, mantenha no `config.yaml`:

```yaml
dataset:
  name: cifar10
  data_dir: ./data
  download: true
```

Para usar seu proprio dataset local, use `ImageFolder`:

```yaml
dataset:
  name: imagefolder
  data_dir: ./meu_dataset
```

O formato esperado e:

```text
meu_dataset/
├── classe_1/
│   ├── imagem_001.jpg
│   └── imagem_002.jpg
└── classe_2/
    ├── imagem_003.jpg
    └── imagem_004.jpg
```

Esse formato e adequado quando cada imagem possui exatamente uma classe e as classes podem ser representadas por diretorios.

## Validacao Cruzada

No `config.yaml`, ajuste:

```yaml
dataset:
  k_folds: 5
```

Com `k_folds > 1`, o codigo separa primeiro um conjunto de teste e faz validacao cruzada apenas sobre treino+validacao.

## Modelo

O pipeline usa um backbone convolucional pre-treinado, como `resnet18`, como extrator de caracteristicas congelado. Antes do treinamento do classificador, gere o cache de features:

```bash
python prepare_features.py --config config.yaml
```

Depois, escolha no YAML se o classificador sera uma MLP ou um circuito quantico variacional.

Para executar todos os experimentos configurados em `features.qubits` e `features.encodings`, alem do baseline MLP, use:

```bash
python run_experiments.py --config config.yaml
```

Os resultados serao salvos em subpastas de `outputs/experiments`, com um resumo consolidado em `experiment_summary.json` e `experiment_summary.csv`.

O script executa os classificadores quanticos para cada combinacao de `features.encodings` e `features.qubits`. As MLPs classicas sao executadas uma vez para cada dimensao PCA unica exigida por essas combinacoes, evitando duplicar experimentos classicos quando duas codificacoes usam a mesma dimensao, como `angle` e `reupload_angle`.

Para cada dimensao PCA unica, sao treinadas uma MLP forte e uma MLP pequena com cerca de `model.mlp_tiny.target_params` parametros treinaveis.

MLP sobre features congeladas:

```yaml
model:
  backbone: resnet18
  pretrained: true
  freeze_backbone: true
  classifier: mlp
  num_classes: 10

  mlp:
    feature_source: raw
    hidden_layers: [256]
    dropout: 0.3

  mlp_tiny:
    target_params: 150
    dropout: 0.0
```

Classificador quantico variacional:

```yaml
model:
  classifier: quantum

  quantum:
    encoding: angle
    num_qubits: 4
    num_layers: 2
    entanglement: linear
```

Para classificadores quanticos, as features da ResNet sao reduzidas por PCA. O PCA e ajustado apenas no conjunto de treino e aplicado em treino, validacao e teste. As codificacoes disponiveis sao:

| Codificacao | Dimensao PCA | Descricao |
|---|---:|---|
| `angle` | `num_qubits` | Uma feature por qubit, aplicada como rotacao `RY` antes das camadas variacionais. |
| `amplitude` | `2 ** num_qubits` | Vetor normalizado como amplitudes do estado quantico. |
| `dense_angle` | `3 * num_qubits` | Tres features por qubit, aplicadas como rotacoes `RX`, `RY` e `RZ` antes das camadas variacionais. |
| `reupload_angle` | `num_qubits` | Mesmo vetor de `angle`, reinserido antes de cada camada variacional. |
