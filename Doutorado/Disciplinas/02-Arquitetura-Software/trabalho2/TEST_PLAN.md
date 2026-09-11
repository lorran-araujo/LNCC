# Plano de Testes — IAProcessing

## Objetivo

O plano de testes verifica se as implementações dummy dos módulos do
`IAProcessing` respeitam as responsabilidades definidas no projeto detalhado.

Os testes não procuram validar algoritmos reais de inteligência artificial ou
geoprocessamento. O objetivo é validar as interfaces, responsabilidades e
interações entre os módulos, de forma que a implementação completa possa
substituir posteriormente as operações dummy sem alterar o fluxo arquitetural.

## Testes unitários

### UT01 — DataPreProcessing

**Objetivo:** verificar se o módulo obtém os novos dados, executa o
pré-processamento e armazena o resultado no repositório de dados
pré-processados.

**Resultado esperado:** `execute()` retorna sucesso e o
`DataPreprocessedRepository` passa a conter dados marcados como processados.

### UT02 — Training

**Objetivo:** verificar se o módulo utiliza os dados pré-processados e atualiza
o modelo armazenado.

**Resultado esperado:** após `execute()`, a versão do modelo armazenado é
incrementada.

### UT03 — Generate

**Objetivo:** verificar se o módulo gera um novo mapa a partir do modelo e dos
dados pré-processados e o disponibiliza por meio de
`LandUseCoverManagement`.

**Resultado esperado:** existe um novo mapa e sua versão de modelo corresponde
à versão atual do modelo armazenado.

### UT04 — Monitoring com irregularidade

**Objetivo:** verificar se uma irregularidade identificada durante a comparação
é registrada no `ControlledAccess`.

**Resultado esperado:** um alerta é registrado.

### UT05 — Monitoring sem irregularidade

**Objetivo:** verificar o comportamento do módulo quando nenhuma irregularidade
é identificada.

**Resultado esperado:** nenhum alerta é registrado.

### UT06 — PipelineManager

**Objetivo:** verificar se o coordenador executa as etapas na ordem definida
pela arquitetura.

**Resultado esperado:** pré-processamento, treinamento, geração e monitoramento
são executados nessa ordem.

### UT07 — IngestionScheduler com novos dados

**Objetivo:** verificar se a identificação de novos dados dispara o pipeline.

**Resultado esperado:** `PipelineManager.execute_pipeline()` é chamado.

### UT08 — IngestionScheduler sem novos dados

**Objetivo:** verificar se o pipeline não é executado quando o repositório
externo não apresenta novos dados.

**Resultado esperado:** o método retorna `False` e o `PipelineManager` não é
acionado.

## Testes de integração

### IT01 — Fluxo completo com irregularidade

**Módulos envolvidos:** `IngestionScheduler`, `PipelineManager`,
`DataPreProcessing`, `Training`, `Generate` e `Monitoring`.

**Objetivo:** verificar a interação ponta a ponta quando existem novos dados e
o monitoramento encontra uma irregularidade.

**Resultado esperado:**
- dados pré-processados são armazenados;
- o modelo é atualizado;
- um novo mapa é disponibilizado;
- o mapa referencia a versão atualizada do modelo;
- um alerta é registrado.

### IT02 — Fluxo completo sem irregularidade

**Objetivo:** verificar o fluxo de atualização quando o monitoramento não
identifica irregularidades.

**Resultado esperado:** o processamento e a geração do mapa são concluídos,
mas nenhum alerta é registrado.

### IT03 — Ausência de novos dados

**Objetivo:** verificar a integração entre `IngestionScheduler` e os demais
componentes quando não existem novos dados.

**Resultado esperado:** o pipeline não é iniciado, o modelo não é atualizado,
nenhum mapa é gerado e nenhum alerta é registrado.

## Execução

A partir da raiz do projeto:

```bash
python3 -m unittest discover -s tests -v
```
