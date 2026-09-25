# Plano de Testes - IAProcessing

## Objetivo

Verificar se as implementações dummy dos módulos respeitam as responsabilidades do projeto
detalhado. Os testes não validam algoritmos reais de IA ou geoprocessamento: validam
interfaces, responsabilidades e interações, de modo que a implementação completa possa
substituir as operações dummy sem alterar o fluxo arquitetural.

Execução (a partir da raiz): `python3 -m unittest discover -s tests -v`

## Testes unitários (`tests/test_unit.py`)

| ID | Módulo | Objetivo | Resultado esperado |
|----|--------|----------|--------------------|
| UT01 | DataPreProcessing | Obter imagens, pré-processar e armazenar | `execute()` retorna `True`; o repositório contém dados marcados como processados |
| UT02 | Training | Usar dados pré-processados e atualizar o modelo | Versão do modelo armazenado incrementada |
| UT03 | Generate | Gerar mapa a partir do modelo e publicá-lo | Mapa disponível em `LandUseCoverManagement` com `model_version` igual à versão do modelo |
| UT04 | Monitoring | Irregularidade identificada | Um alerta registrado em `AlertManagement` |
| UT05 | Monitoring | Nenhuma irregularidade | Nenhum alerta registrado |
| UT06 | Monitoring | Não há mapa disponível | `execute()` retorna `False`; nenhum alerta |
| UT07 | PipelineManager | Ordem de execução | Etapas executadas na ordem em que foram injetadas |
| UT08 | PipelineManager | Etapa falha | Execução interrompida; retorna `False`; `failed_stage` preenchido |
| UT09 | PipelineManager | Etapa lança exceção | Exceção tratada como falha (com log de erro) |
| UT10 | IngestionScheduler | Há novos dados | `PipelineManager.execute_pipeline()` chamado; retorna `True` |
| UT11 | IngestionScheduler | Não há novos dados | Retorna `False`; pipeline não acionado |
| UT12 | IngestionScheduler | Pipeline falha | Retorna `False` |
| UT13 | ControlledAccess | Consulta de alertas | `view_alerts()` devolve os alertas registrados em `AlertManagement` |

## Testes de integração (`tests/test_integration.py`)

Módulos envolvidos: `IngestionScheduler`, `PipelineManager`, `DataPreProcessing`, `Training`,
`Generate`, `Monitoring`, repositórios, `LandUseCoverManagement`, `ZoningManagement`,
`AlertManagement` e `ControlledAccess`.

| ID | Cenário | Resultado esperado |
|----|---------|--------------------|
| IT01 | Fluxo completo com novos dados e irregularidade | Dados pré-processados armazenados; modelo atualizado; mapa disponibilizado referenciando o novo modelo; alerta visível via `ControlledAccess` |
| IT02 | Fluxo completo sem irregularidade | Mapa gerado; nenhum alerta |
| IT03 | Ausência de novos dados | Pipeline não iniciado; modelo, mapa e alertas inalterados |
| IT04 | Falha na coleta durante uma segunda atualização (RNF3) | Retorna `False`; `failed_stage == "DataPreProcessing"`; mapa, modelo e alertas anteriores preservados |
