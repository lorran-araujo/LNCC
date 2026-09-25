from __future__ import annotations

from pimucs.models import IrregularityAlert


class AlertManagement:
    """Armazena os alertas de irregularidade.

    Escrito pelo Monitoring (IAProcessing) e lido pelo ControlledAccess
    (WebApplication). Existe para manter as dependências apontando sempre
    "para baixo" na arquitetura em camadas.
    """

    def __init__(self) -> None:
        self._alerts: list[IrregularityAlert] = []

    def register_alert(self, alert: IrregularityAlert) -> None:
        self._alerts.append(alert)

    def get_alerts(self) -> list[IrregularityAlert]:
        return list(self._alerts)
