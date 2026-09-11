from models import IrregularityAlert

class Monitoring:
    def __init__(
        self,
        land_use_management,
        zoning_management,
        controlled_access,
        simulate_irregularity: bool = True
    ):
        self.land_use_management = land_use_management
        self.zoning_management = zoning_management
        self.controlled_access = controlled_access
        self.simulate_irregularity = simulate_irregularity

    def execute(self):
        land_use_map = self.land_use_management.get_current_map()
        zoning_map = self.zoning_management.get_zoning_map()
        irregularities = self.compare_maps(land_use_map, zoning_map)

        for irregularity in irregularities:
            self.controlled_access.register_alert(irregularity)

        return True

    def compare_maps(self, land_use_map, zoning_map):
        if not self.simulate_irregularity:
            return []

        return [
            IrregularityAlert(
                description="Possível uso incompatível detectado."
            )
        ]
