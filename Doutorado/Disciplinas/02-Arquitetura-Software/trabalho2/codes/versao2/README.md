# PIMUCS - módulos do IAProcessing (implementação dummy)

Implementação simplificada dos módulos do `IAProcessing` da PIMUCS, com os módulos
de `DataManagement`, `WebApplication` e o sistema externo necessários à integração.

## Estrutura

```
main.py                         ponto de montagem (injeção de dependências)
pimucs/
  models/                       entidades de domínio (dataclasses)
  external/                     ExternalDataRepository (sistema externo)
  ia_processing/                PipelineStage, IngestionScheduler, PipelineManager,
                                DataPreProcessing, Training, Generate, Monitoring,
                                DataPreprocessedRepository, ModelRepository
  data_management/              LandUseCoverManagement, ZoningManagement, AlertManagement
  web_application/              ControlledAccess
MUDANCAS.md                   inadequações da arquitetura e refatorações aplicadas
tests/                          testes unitários e de integração (unittest)
TEST_PLAN.md                    plano de testes
```

Dependências entre pacotes (verificadas a partir dos `import`s):
`ia_processing → data_management, external, models`;
`web_application → data_management, models`; `data_management → models`; `external → models`.

## Execução

Requer Python 3.10+ (sem dependências externas).

```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

Fluxo simulado: verificação de novos dados → pré-processamento → atualização do modelo →
geração do mapa → monitoramento → registro de alertas (consultados via `ControlledAccess`).
