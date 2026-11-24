
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def gerar_dados_sinteticos(
    modelo,
    parametros,
    y0,
    t_span=(0, 30),
    n_obs=100,
    ruido=0.05,
    seed=42,
    nomes_variaveis=None,
    metodo="RK45",
    rtol=1e-8,
    atol=1e-10,
):
    """
    Gera observações sintéticas de um sistema de EDOs.

    Parâmetros
    ----------
    modelo : callable
        Função da EDO no formato modelo(t, y, *parametros).

    parametros : array-like
        Parâmetros verdadeiros do modelo.

    y0 : array-like
        Condições iniciais.

    t_span : tuple
        Intervalo temporal (t0, tf).

    n_obs : int
        Número de instantes de observação, incluindo extremos.

    ruido : float
        Desvio-padrão relativo do ruído gaussiano.
        Exemplo: 0.05 representa 5% da escala de cada variável.

    seed : int
        Semente para reprodutibilidade.

    nomes_variaveis : list, optional
        Nomes das variáveis do sistema.

    Retorna
    -------
    pd.DataFrame
        Dataset com tempos, observações e valores verdadeiros.
    """

    if n_obs < 2:
        raise ValueError("n_obs deve ser pelo menos 2.")

    if ruido < 0:
        raise ValueError("ruido deve ser não negativo.")

    y0 = np.asarray(y0, dtype=float)
    parametros = np.asarray(parametros, dtype=float)

    if nomes_variaveis is None:
        nomes_variaveis = [
            f"y{i+1}" for i in range(len(y0))
        ]

    if len(nomes_variaveis) != len(y0):
        raise ValueError(
            "O número de nomes deve coincidir com o número de variáveis."
        )

    # Gerador aleatório local
    rng = np.random.default_rng(seed)

    # Instantes de observação
    t_obs = np.linspace(t_span[0], t_span[1], n_obs)

    # Solução numérica da EDO
    sol = solve_ivp(
        modelo,
        t_span,
        y0,
        t_eval=t_obs,
        args=tuple(parametros),
        method=metodo,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # Formato: (n_obs, n_variaveis)
    y_true = sol.y.T

    # Escala do ruído para cada variável
    escala = np.ptp(y_true, axis=0)

    # Evita escala nula em trajetórias constantes
    escala = np.where(
        escala > 0,
        escala,
        np.maximum(np.abs(y_true).max(axis=0), 1.0)
    )

    sigma = ruido * escala

    # Ruído gaussiano independente
    epsilon = rng.normal(
        loc=0.0,
        scale=sigma,
        size=y_true.shape
    )

    y_obs = y_true + epsilon

    # Construção do dataset
    dados = {"t": t_obs}

    for i, nome in enumerate(nomes_variaveis):
        dados[f"{nome}_obs"] = y_obs[:, i]
        dados[f"{nome}_true"] = y_true[:, i]

    return pd.DataFrame(dados)


def rosenzweig_macarthur(t, y, r, K, a, h, e, m):
    N, P = y

    predacao = (a * N * P) / (1 + a * h * N)

    dNdt = r * N * (1 - N / K) - predacao
    dPdt = e * predacao - m * P

    return [dNdt, dPdt]