from __future__ import annotations

from pimucs.data_management import AlertManagement
from pimucs.models import IrregularityAlert


class ControlledAccess:
    """Funcionalidades restritas (dummy): consulta de alertas pelo fiscalizador.

    A autenticação/autorização (RNF4) não é implementada nesta versão.
    """

    def __init__(self, alert_management: AlertManagement) -> None:
        self._alert_management = alert_management

    def view_alerts(self) -> list[IrregularityAlert]:
        return self._alert_management.get_alerts()
