"""
Module for power system network modeling in OpenGrid.

This module provides classes and functions to define, import, and manipulate
power grid network models using libraries like pandapower or PyPSA.
"""
import structlog
import pandapower as pp
# import pypsa
from typing import Dict, Any

logger = structlog.get_logger(__name__)

class PowerNetwork:
    """Represents a power distribution or transmission network.

    Attributes:
        name (str): Name of the network.
        pandapower_net: A pandapower network object.
        # pypsa_net: A PyPSA network object (optional, if supporting both)
        is_empty (bool): True if the network has no components.
    """
    def __init__(self, name: str):
        self.name = name
        self.pandapower_net = pp.create_empty_network(name=name)
        # self.pypsa_net = None 
        self.is_empty = True
        logger.info("PowerNetwork created", name=self.name)

    def add_bus(self, vn_kv: float, name: str = None, **kwargs) -> int:
        """Adds a bus to the pandapower network."""
        bus_id = pp.create_bus(self.pandapower_net, vn_kv=vn_kv, name=name, **kwargs)
        self.is_empty = False
        logger.debug("Bus added", network=self.name, bus_id=bus_id, vn_kv=vn_kv, bus_name=name)
        return bus_id

    def add_line(self, from_bus: int, to_bus: int, length_km: float, std_type: str, name: str = None, **kwargs) -> int:
        """Adds a line to the pandapower network."""
        line_id = pp.create_line(self.pandapower_net, from_bus=from_bus, to_bus=to_bus, 
                                 length_km=length_km, std_type=std_type, name=name, **kwargs)
        self.is_empty = False
        logger.debug("Line added", network=self.name, line_id=line_id, from_bus=from_bus, to_bus=to_bus)
        return line_id

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the network components."""
        summary = {
            "name": self.name,
            "buses": len(self.pandapower_net.bus),
            "lines": len(self.pandapower_net.line),
            "transformers": len(self.pandapower_net.trafo),
            "loads": len(self.pandapower_net.load),
            "generators": len(self.pandapower_net.gen) + len(self.pandapower_net.sgen),
        }
        logger.info("Network summary requested", network_summary=summary)
        return summary

    # Add methods for other components (transformers, loads, generators, etc.)
    # Add methods for running simulations (load flow, short circuit) via pandapower/pypsa

logger.info("OpenGrid network_model module loaded.") 