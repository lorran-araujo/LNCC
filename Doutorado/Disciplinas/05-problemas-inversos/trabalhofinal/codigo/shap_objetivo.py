"""Valores SHAP exatos para uma referência empírica de parâmetros."""

from math import comb

import numpy as np


def calcular_shap(objetivo, estimativas, referencia):
    """Decompõe objetivo(estimativa) - média do objetivo na referência."""
    estimativas = np.asarray(estimativas, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    n_parametros = estimativas.shape[1]
    bits = 1 << np.arange(n_parametros)
    mascaras = np.arange(1 << n_parametros)[:, None] & bits != 0
    tamanhos = mascaras.sum(axis=1)
    pesos = np.array([1 / (n_parametros * comb(n_parametros - 1, k))
                      for k in range(n_parametros)])
    base = np.mean([objetivo(vetor) for vetor in referencia])
    valores = np.empty_like(estimativas)
    erros = np.empty(len(estimativas))

    for i, estimativa in enumerate(estimativas):
        medias = np.empty(len(mascaras))
        medias[0] = base
        medias[-1] = objetivo(estimativa)
        for j, mascara in enumerate(mascaras[1:-1], start=1):
            hibridos = np.where(mascara, estimativa, referencia)
            medias[j] = np.mean([objetivo(vetor) for vetor in hibridos])

        for k, bit in enumerate(bits):
            ausentes = np.flatnonzero(~mascaras[:, k])
            diferencas = medias[ausentes | bit] - medias[ausentes]
            valores[i, k] = pesos[tamanhos[ausentes]] @ diferencas
        erros[i] = medias[-1]

    np.testing.assert_allclose(base + valores.sum(axis=1), erros,
                               rtol=1e-9, atol=1e-12)
    return base, valores, erros
