# Síntese da calibração de Rosenzweig–MacArthur

Foram analisadas 210 calibrações em 7 realizações dos dados.

## Recuperação e ajuste

- **Ruído 0%**. Menor erro relativo absoluto médio para `r` (6,0%) e maior para `h` (57,8%). Erro médio dos seis parâmetros de 23,9% e NRMSE médio contra a trajetória limpa de 1,91%. 1 de 30 execuções cumpriram o critério de recuperação. A realização 1 é a representativa.
- **Ruído 2%**. Menor erro relativo absoluto médio para `r` (5,3%) e maior para `h` (56,1%). Erro médio dos seis parâmetros de 23,3% e NRMSE médio contra a trajetória limpa de 1,93%. 2 de 90 execuções cumpriram o critério de recuperação. A realização 2 é a representativa.
- **Ruído 5%**. Menor erro relativo absoluto médio para `r` (7,3%) e maior para `h` (59,7%). Erro médio dos seis parâmetros de 25,3% e NRMSE médio contra a trajetória limpa de 2,42%. 0 de 90 execuções cumpriram o critério de recuperação. A realização 1 é a representativa.
- 53 das 56 execuções no quartil de menor custo de suas respectivas realizações ficaram fora da tolerância em pelo menos um parâmetro.
- Em dados ruidosos, 17 de 180 estimativas obtiveram custo menor que o vetor verdadeiro. Destas, 16 ficaram fora da tolerância de recuperação em pelo menos um parâmetro.
- O custo final mede o ajuste às observações de cada realização. A comparação com a trajetória limpa e a recuperação dos parâmetros complementam esse indicador.

## Convergência e custo computacional

- **Ruído 0%**. Entre a geração inicial e a geração 50, a queda mediana do melhor custo foi 99,4% nas 30 execuções que chegaram a esse ponto. O custo médio mediano da população passou de 5e+10 para 0.023. Houve 8 paradas antecipadas. O tempo médio foi 159,0 s, com 14042 avaliações e 143,7 gerações em média.
- **Ruído 2%**. Entre a geração inicial e a geração 50, a queda mediana do melhor custo foi 98,8% nas 90 execuções que chegaram a esse ponto. O custo médio mediano da população passou de 5e+10 para 0.023. Houve 26 paradas antecipadas. O tempo médio foi 157,7 s, com 13921 avaliações e 142,5 gerações em média.
- **Ruído 5%**. Entre a geração inicial e a geração 50, a queda mediana do melhor custo foi 96,1% nas 90 execuções que chegaram a esse ponto. O custo médio mediano da população passou de 5e+10 para 0.025. Houve 15 paradas antecipadas. O tempo médio foi 158,9 s, com 14242 avaliações e 145,8 gerações em média.
- O máximo teórico foi de 14650 avaliações. A fórmula por geração reproduziu todos os totais salvos. 210 execuções tiveram pelo menos uma geração cujo custo médio sinaliza penalidade. 0 trajetórias vencedoras foram inválidas na reconstrução e 0 custos recalculados divergiram dos salvos.

## Compensações e limitações

- **Ruído 0%**. A maior correlação de Pearson em módulo foi entre `e` e `m` (r = 0,98, Spearman = 0,96). As matrizes por realização permitem consultar as associações com os dados fixos.
- **Ruído 2%**. A maior correlação de Pearson em módulo foi entre `K` e `h` (r = -0,98, Spearman = -0,99). As matrizes por realização permitem consultar as associações com os dados fixos.
- **Ruído 5%**. A maior correlação de Pearson em módulo foi entre `K` e `h` (r = -0,97, Spearman = -0,99). As matrizes por realização permitem consultar as associações com os dados fixos.
- No ruído de 5%, realização 3, as sementes 3118 e 3125 produziram custos com diferença relativa de 5,00% e vetores separados em média por 30,2% dos valores verdadeiros. O par é um indício descritivo de compensação entre parâmetros.
- Entre os vencedores, houve 0 coordenadas em limites de busca. As populações completas e seus custos não foram salvos. A diversidade residual e a concentração nos limites entre todos os indivíduos ficam indeterminadas.
- As comparações de ruído e realização são descritivas. Há três realizações para cada nível ruidoso, com sementes compartilhadas entre os dois níveis.
- Os valores SHAP individuais não foram salvos. A importância quantitativa fica pendente até a exportação de `tabela_shap`.

## Figuras e tabelas para os slides

- [Radar de erro relativo](figuras/parametros/radar_erro_relativo.pdf) e [tabela de recuperação](tabelas/latex/tabela_recuperacao_parametros.tex).
- [Presas](figuras/trajetorias/ajuste_representativo_presas.pdf) e [predadores](figuras/trajetorias/ajuste_representativo_predadores.pdf) nos três níveis de ruído.
- Ajuste das trajetórias nas realizações representativas:
  - [Ruído 0%, realização 1](figuras/trajetorias/ajuste_trajetorias_ruido_00_realizacao_01.pdf).
  - [Ruído 2%, realização 2](figuras/trajetorias/ajuste_trajetorias_ruido_02_realizacao_02.pdf).
  - [Ruído 5%, realização 1](figuras/trajetorias/ajuste_trajetorias_ruido_05_realizacao_01.pdf).
- [Convergência do melhor custo](figuras/convergencia/convergencia_melhor_custo.pdf) e [do custo médio](figuras/convergencia/convergencia_custo_medio.pdf), com contagem de execuções.
- [Custo final versus erro dos parâmetros](figuras/correlacao/custo_final_vs_erro_parametros.pdf).
- [Tabela de custo computacional](tabelas/latex/tabela_custo_computacional.tex).
- [Recorte do custo médio a partir da geração 5](figuras/convergencia/convergencia_custo_medio_apos_penalidades.pdf).
