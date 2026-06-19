# Vision Transformer Experiments

Projeto base em PyTorch para testar Vision Transformers em tarefas de classificacao.

## Estrutura

```text
configs/        Configuracoes dos experimentos
src/datasets/   Criacao de datasets e dataloaders
src/models/     Modelos ViT simples ou via timm
src/trainers/   Estrategias de treinamento
src/utils/      Seed, metricas, checkpoints e config
scripts/        Scripts auxiliares
outputs/        Resultados dos experimentos
data/           Datasets baixados pelo torchvision
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Treinar

```bash
python3 -m src.main --config configs/default.yaml
```

ou:

```bash
bash scripts/train.sh
```

## Escolher CPU/GPU

Configure o campo `training.device` em `configs/default.yaml`:

```yaml
training:
  device: auto    # usa cuda:0 se houver CUDA, senao cpu
```

Exemplos validos:

```yaml
training:
  device: cpu
```

```yaml
training:
  device: cuda:0
```

```yaml
training:
  device: cuda:1
```

Se uma GPU inexistente for selecionada, o programa interrompe com uma mensagem indicando quantas GPUs CUDA estao disponiveis.

## Usar ViT do timm

Altere `configs/default.yaml`:

```yaml
model:
  backend: timm
  name: vit_tiny_patch16_224
  pretrained: true
```

Os resultados sao salvos em `outputs/<experiment_name>/`.

Arquivos principais gerados:

```text
config.yaml
metrics.csv
test_metrics.csv
test_predictions.csv
confusion_matrix.csv
checkpoints/best.pt
checkpoints/last.pt
plots/loss.png
plots/accuracy.png
plots/confusion_matrix.png
plots/prediction_examples.png
```

O teste final usa automaticamente o checkpoint `checkpoints/best.pt`, escolhido pela melhor acuracia de validacao.

`test_predictions.csv` contem uma linha por imagem do conjunto de teste, com classe real, classe predita e se a predicao foi correta.
