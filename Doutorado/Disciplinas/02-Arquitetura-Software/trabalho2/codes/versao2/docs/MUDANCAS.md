# Mudanças em relação ao 1º trabalho e ao dummy anterior

## A. Inadequações da especificação arquitetural e como foram resolvidas

| # | Inadequação | Resolução |
|---|-------------|-----------|
| 1 | **Dependência de camada invertida**: `Monitoring` (IAProcessing) usava `ControlledAccess` (WebApplication); no código, `ControlledAccess` era apenas uma lista de alertas. | Criada `AlertManagement` no DataManagement. `Monitoring` grava nela; `ControlledAccess` apenas lê. Todas as dependências passam a apontar "para baixo" (verificado pelos imports). |
| 2 | **Acoplamento do `IngestionScheduler`**: recebia as quatro etapas só para repassá-las ao `PipelineManager`, que era *stateless*. | Extraída a classe abstrata `PipelineStage`; o `PipelineManager` recebe a lista de etapas no construtor. O scheduler depende apenas do `PipelineManager` (RNF2). |
| 3 | **Falhas ignoradas**: `execute_pipeline` descartava o retorno das etapas e sempre devolvia `True` (contraria RNF3 e o "retorna sucesso" da Figura 4). | O pipeline é interrompido na primeira falha (retorno `False` ou exceção) e registra `failed_stage`. `Monitoring` retorna `False` se não há mapa. Testes UT06, UT08, UT09, UT12, IT04. |
| 4 | `ExternalDataRepository` era uma classe interna do DataManagement no código, mas é um sistema externo na arquitetura. | Movida para o pacote `external`. |
| 5 | `DataPreprocessedRepository` e `ModelRepository` ficavam em módulo de DataManagement no código; a visão de implantação os aloca no IA Processing Server. | Movidos para `ia_processing/repositories.py`. |

Correção de leitura: o acionamento do `Monitoring` pelo `PipelineManager` (Figura 5) **não** é
inadequação; o código o implementa como quarta etapa do pipeline.

## B. Refatorações aplicadas

- **Move Class / Move Module**: arquivos organizados em pacotes `models`, `external`, `ia_processing`, `data_management` e `web_application`.
- **Extract Superclass (Extract Interface)**: `PipelineStage`.
- **Extract Class**: `AlertManagement` (responsabilidade retirada de `ControlledAccess`).
- **Split Module**: `repositories.py` dividido em módulos por pacote/responsabilidade.
- **Introduce Constructor Injection**: etapas passam de parâmetros de método para o construtor do `PipelineManager`.
- **Encapsulate Field**: atributos de colaboradores tornados privados (`_x`).
- **Replace Nested Conditional with Guard Clause**: verificações de retorno antecipado em `IngestionScheduler` e `Monitoring`.
- **Rename**: `DataPreprocessedRepository` padronizado; `simulate_irregularity` mantido como gancho de teste do dummy.
- Adicionadas *type hints* (rastreabilidade código ↔ diagrama).

## C. Padrões de projeto efetivamente usados

- **Injeção de dependências**: `main.py` é a raiz de composição; todos os colaboradores chegam pelos construtores.
- **Repository**: `DataPreprocessedRepository` e `ModelRepository`.
- **Fachada de acesso a dados**: `LandUseCoverManagement`, `ZoningManagement` e `AlertManagement` escondem o armazenamento (em memória no dummy).
- **Interface comum com polimorfismo (variação de Strategy)**: `PipelineStage` permite substituir/reordenar etapas sem alterar o `PipelineManager`.

Não foram aplicados *Template Method* nem *Observer*.

## D. Limitações conscientes do dummy

- RF3.1: não há estado de "data da última atualização"; `has_new_data` é um booleano simulado.
- RF1.1: `LandUseCoverManagement` guarda apenas o mapa vigente, sem histórico de versões.
- `simulate_irregularity` é um gancho de teste dentro de classe de produção; o ideal é injetar uma estratégia de detecção.
- Autenticação/autorização (RNF4) e administração (RF5.1, RF5.2) não implementadas.
- No diagrama de classes, os usos de tipos de `Models` em assinaturas não são desenhados (aparecem no diagrama de pacotes).
