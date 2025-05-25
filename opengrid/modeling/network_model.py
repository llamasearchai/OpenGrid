"""
Module for power system network modeling in OpenGrid.

This module provides classes and functions to define, import, and manipulate
power grid network models using libraries like pandapower or PyPSA.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import structlog
import pandapower as pp
import pypsa
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

logger = structlog.get_logger(__name__)


class NetworkComponentManager:
    """Manages network components and their relationships."""
    
    def __init__(self):
        self.components = {}
        self.relationships = {}
    
    def add_component(self, component_id: str, component_type: str, properties: Dict[str, Any]) -> None:
        """Add a component to the manager."""
        self.components[component_id] = {
            "type": component_type,
            "properties": properties,
            "created_at": datetime.now().isoformat()
        }
        logger.debug("Component added", component_id=component_id, component_type=component_type)
    
    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a component by ID."""
        return self.components.get(component_id)
    
    def list_components_by_type(self, component_type: str) -> List[Dict[str, Any]]:
        """List all components of a specific type."""
        return [
            {"id": cid, **comp} 
            for cid, comp in self.components.items() 
            if comp["type"] == component_type
        ]


class PowerNetwork:
    """Represents a power distribution or transmission network.

    Attributes:
        name (str): Name of the network.
        pandapower_net: A pandapower network object.
        pypsa_net: A PyPSA network object.
        is_empty (bool): True if the network has no components.
        component_manager: Manages network components.
    """
    
    def __init__(self, name: str, use_pypsa: bool = False):
        self.name = name
        self.pandapower_net = pp.create_empty_network(name=name)
        self.pypsa_net = pypsa.Network(name=name) if use_pypsa else None
        self.use_pypsa = use_pypsa
        self.is_empty = True
        self.component_manager = NetworkComponentManager()
        self.created_at = datetime.now()
        logger.info("PowerNetwork created", name=self.name, use_pypsa=use_pypsa)

    def add_bus(self, vn_kv: float, name: str = None, zone: str = None, **kwargs) -> int:
        """Adds a bus to the network."""
        if self.use_pypsa and self.pypsa_net:
            bus_id = len(self.pypsa_net.buses)
            self.pypsa_net.add("Bus", name or f"Bus_{bus_id}", v_nom=vn_kv, **kwargs)
        else:
            bus_id = pp.create_bus(
                self.pandapower_net, 
                vn_kv=vn_kv, 
                name=name,
                zone=zone,
                **kwargs
            )
        
        self.component_manager.add_component(
            f"bus_{bus_id}",
            "bus",
            {"vn_kv": vn_kv, "name": name, "zone": zone, **kwargs}
        )
        
        self.is_empty = False
        logger.debug(
            "Bus added", 
            network=self.name, 
            bus_id=bus_id, 
            vn_kv=vn_kv, 
            bus_name=name,
            zone=zone
        )
        return bus_id

    def add_line(
        self, 
        from_bus: int, 
        to_bus: int, 
        length_km: float, 
        std_type: str, 
        name: str = None,
        max_i_ka: float = None,
        **kwargs
    ) -> int:
        """Adds a line to the network."""
        if self.use_pypsa and self.pypsa_net:
            line_id = len(self.pypsa_net.lines)
            self.pypsa_net.add(
                "Line",
                name or f"Line_{line_id}",
                bus0=f"Bus_{from_bus}",
                bus1=f"Bus_{to_bus}",
                length=length_km,
                **kwargs
            )
        else:
            line_id = pp.create_line(
                self.pandapower_net, 
                from_bus=from_bus, 
                to_bus=to_bus,
                length_km=length_km, 
                std_type=std_type, 
                name=name,
                max_i_ka=max_i_ka,
                **kwargs
            )
        
        self.component_manager.add_component(
            f"line_{line_id}",
            "line",
            {
                "from_bus": from_bus,
                "to_bus": to_bus,
                "length_km": length_km,
                "std_type": std_type,
                "name": name,
                "max_i_ka": max_i_ka,
                **kwargs
            }
        )
        
        self.is_empty = False
        logger.debug(
            "Line added", 
            network=self.name, 
            line_id=line_id, 
            from_bus=from_bus, 
            to_bus=to_bus,
            length_km=length_km
        )
        return line_id

    def add_transformer(
        self,
        hv_bus: int,
        lv_bus: int,
        std_type: str,
        name: str = None,
        **kwargs
    ) -> int:
        """Adds a transformer to the network."""
        trafo_id = pp.create_transformer(
            self.pandapower_net,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            std_type=std_type,
            name=name,
            **kwargs
        )
        
        self.component_manager.add_component(
            f"transformer_{trafo_id}",
            "transformer",
            {
                "hv_bus": hv_bus,
                "lv_bus": lv_bus,
                "std_type": std_type,
                "name": name,
                **kwargs
            }
        )
        
        self.is_empty = False
        logger.debug(
            "Transformer added",
            network=self.name,
            trafo_id=trafo_id,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            std_type=std_type
        )
        return trafo_id

    def add_load(
        self,
        bus: int,
        p_mw: float,
        q_mvar: float = 0,
        name: str = None,
        **kwargs
    ) -> int:
        """Adds a load to the network."""
        load_id = pp.create_load(
            self.pandapower_net,
            bus=bus,
            p_mw=p_mw,
            q_mvar=q_mvar,
            name=name,
            **kwargs
        )
        
        self.component_manager.add_component(
            f"load_{load_id}",
            "load",
            {
                "bus": bus,
                "p_mw": p_mw,
                "q_mvar": q_mvar,
                "name": name,
                **kwargs
            }
        )
        
        self.is_empty = False
        logger.debug(
            "Load added",
            network=self.name,
            load_id=load_id,
            bus=bus,
            p_mw=p_mw,
            q_mvar=q_mvar
        )
        return load_id

    def add_generator(
        self,
        bus: int,
        p_mw: float,
        vm_pu: float = 1.0,
        name: str = None,
        **kwargs
    ) -> int:
        """Adds a generator to the network."""
        gen_id = pp.create_gen(
            self.pandapower_net,
            bus=bus,
            p_mw=p_mw,
            vm_pu=vm_pu,
            name=name,
            **kwargs
        )
        
        self.component_manager.add_component(
            f"generator_{gen_id}",
            "generator",
            {
                "bus": bus,
                "p_mw": p_mw,
                "vm_pu": vm_pu,
                "name": name,
                **kwargs
            }
        )
        
        self.is_empty = False
        logger.debug(
            "Generator added",
            network=self.name,
            gen_id=gen_id,
            bus=bus,
            p_mw=p_mw,
            vm_pu=vm_pu
        )
        return gen_id

    def run_powerflow(self, **kwargs) -> Dict[str, Any]:
        """Runs power flow analysis."""
        try:
            if self.use_pypsa and self.pypsa_net:
                self.pypsa_net.pf(**kwargs)
                results = {
                    "converged": True,
                    "bus_voltages": self.pypsa_net.buses_t.v_mag_pu.iloc[-1].to_dict(),
                    "line_loading": self.pypsa_net.lines_t.p0.iloc[-1].to_dict(),
                }
            else:
                pp.runpp(self.pandapower_net, **kwargs)
                results = {
                    "converged": self.pandapower_net.converged,
                    "bus_voltages": self.pandapower_net.res_bus.vm_pu.to_dict(),
                    "line_loading": self.pandapower_net.res_line.loading_percent.to_dict(),
                    "bus_angles": self.pandapower_net.res_bus.va_degree.to_dict(),
                    "line_losses": self.pandapower_net.res_line.pl_mw.to_dict(),
                }
            
            logger.info("Power flow analysis completed", network=self.name, converged=results["converged"])
            return results
            
        except Exception as e:
            logger.error("Power flow analysis failed", network=self.name, error=str(e))
            return {"converged": False, "error": str(e)}

    def get_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive summary of the network."""
        summary = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "is_empty": self.is_empty,
            "use_pypsa": self.use_pypsa,
            "components": {
                "buses": len(self.pandapower_net.bus),
                "lines": len(self.pandapower_net.line),
                "transformers": len(self.pandapower_net.trafo),
                "loads": len(self.pandapower_net.load),
                "generators": len(self.pandapower_net.gen) + len(self.pandapower_net.sgen),
            },
            "total_load_mw": float(self.pandapower_net.load.p_mw.sum()) if not self.pandapower_net.load.empty else 0,
            "total_generation_mw": float(self.pandapower_net.gen.p_mw.sum()) if not self.pandapower_net.gen.empty else 0,
        }
        
        # Add voltage levels
        if not self.pandapower_net.bus.empty:
            summary["voltage_levels"] = sorted(self.pandapower_net.bus.vn_kv.unique().tolist())
        
        logger.info("Network summary requested", network_summary=summary)
        return summary

    def export_to_json(self) -> str:
        """Export network to JSON format."""
        export_data = {
            "metadata": {
                "name": self.name,
                "created_at": self.created_at.isoformat(),
                "version": "0.2.0"
            },
            "components": self.component_manager.components,
            "summary": self.get_summary()
        }
        return json.dumps(export_data, indent=2)

    def validate_network(self) -> Dict[str, Any]:
        """Validate network connectivity and parameters."""
        issues = []
        warnings = []
        
        # Check for isolated buses
        if not self.pandapower_net.bus.empty:
            connected_buses = set()
            for _, line in self.pandapower_net.line.iterrows():
                connected_buses.add(line.from_bus)
                connected_buses.add(line.to_bus)
            
            all_buses = set(self.pandapower_net.bus.index)
            isolated_buses = all_buses - connected_buses
            
            if isolated_buses:
                issues.append(f"Isolated buses found: {list(isolated_buses)}")
        
        # Check for buses without loads or generators
        buses_with_loads = set(self.pandapower_net.load.bus) if not self.pandapower_net.load.empty else set()
        buses_with_gens = set(self.pandapower_net.gen.bus) if not self.pandapower_net.gen.empty else set()
        
        all_buses = set(self.pandapower_net.bus.index) if not self.pandapower_net.bus.empty else set()
        buses_without_injection = all_buses - buses_with_loads - buses_with_gens
        
        if len(buses_without_injection) > len(all_buses) * 0.1:  # More than 10% of buses
            warnings.append(f"Many buses without loads or generators: {len(buses_without_injection)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


logger.info("OpenGrid enhanced network_model module loaded.") 