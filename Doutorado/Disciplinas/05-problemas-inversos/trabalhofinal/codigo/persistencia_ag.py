"""Persistencia JSON de experimentos de calibracao, sem pickle."""
import json
from pathlib import Path

import numpy as np


def _serializar(valor):
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"Tipo nao serializavel: {type(valor).__name__}")


def salvar_experimento(pasta, metadados, resultados, completo=False):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "experimento.json"
    temporario = pasta / "experimento.json.tmp"
    conteudo = {"versao": 1, "completo": completo,
                "metadados": metadados, "resultados": resultados}
    temporario.write_text(json.dumps(conteudo, default=_serializar, allow_nan=False,
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporario.replace(destino)


def carregar_experimento(pasta):
    caminho = Path(pasta) / "experimento.json"
    if not caminho.is_file():
        raise FileNotFoundError(f"Experimento nao encontrado: {caminho}. Use optimize ou escolha outra pasta.")
    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    if conteudo.get("versao") != 1:
        raise ValueError("Versao de experimento nao suportada.")
    if not conteudo["resultados"]:
        raise ValueError("O experimento ainda nao possui calibracoes concluidas.")
    for resultado in conteudo["resultados"]:
        for chave in ("parametros", "historico_melhor", "historico_medio", "observacoes", "referencia"):
            resultado[chave] = np.asarray(resultado[chave], dtype=float)
    return conteudo
