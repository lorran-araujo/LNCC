# IAProcessing Dummy

Implementação simplificada dos módulos do IAProcessing da PIMUCS.

## Execução

```bash
python3 main.py
```

Fluxo simulado:
1. verificação de novos dados;
2. pré-processamento;
3. atualização do modelo;
4. geração de novo mapa;
5. monitoramento;
6. registro de alerta.


## Testes

Para executar os testes unitários e de integração:

```bash
python3 -m unittest discover -s tests -v
```

A descrição do plano de testes está em `TEST_PLAN.md`.
