"""
Mock Power System Networks
Provides sample networks for testing, demonstration, and development.

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandapower as pp
import pypsa


@dataclass
class NetworkMetadata:
    """Metadata for sample networks."""
    name: str
    description: str
    voltage_levels: List[float]  # kV
    num_buses: int
    num_lines: int
    num_generators: int
    num_loads: int
    total_load_mw: float
    total_generation_mw: float
    network_type: str  # "transmission", "distribution", "microgrid", "industrial"
    complexity: str  # "simple", "medium", "complex"


class MockNetworkGenerator:
    """Generates mock power system networks for testing and demonstration."""
    
    def __init__(self):
        self.networks = {}
        self._initialize_sample_networks()
    
    def _initialize_sample_networks(self):
        """Initialize all sample networks."""
        self.networks = {
            "ieee_9_bus": self._create_ieee_9_bus(),
            "ieee_14_bus": self._create_ieee_14_bus(),
            "ieee_30_bus": self._create_ieee_30_bus(),
            "simple_microgrid": self._create_simple_microgrid(),
            "industrial_plant": self._create_industrial_plant(),
            "distribution_feeder": self._create_distribution_feeder(),
            "renewable_grid": self._create_renewable_grid(),
            "dc_microgrid": self._create_dc_microgrid(),
        }
    
    def get_network(self, name: str) -> Dict[str, Any]:
        """Get a sample network by name."""
        if name not in self.networks:
            raise ValueError(f"Network '{name}' not found. Available: {list(self.networks.keys())}")
        return self.networks[name]
    
    def list_networks(self) -> List[NetworkMetadata]:
        """List all available sample networks."""
        return [network["metadata"] for network in self.networks.values()]
    
    def _create_ieee_9_bus(self) -> Dict[str, Any]:
        """Create IEEE 9-bus test system."""
        net = pp.create_empty_network()
        
        # Buses
        buses = [
            {"name": "Bus 1", "vn_kv": 16.5, "type": "b"},
            {"name": "Bus 2", "vn_kv": 18.0, "type": "b"},
            {"name": "Bus 3", "vn_kv": 13.8, "type": "b"},
            {"name": "Bus 4", "vn_kv": 230.0, "type": "b"},
            {"name": "Bus 5", "vn_kv": 230.0, "type": "b"},
            {"name": "Bus 6", "vn_kv": 230.0, "type": "b"},
            {"name": "Bus 7", "vn_kv": 230.0, "type": "b"},
            {"name": "Bus 8", "vn_kv": 230.0, "type": "b"},
            {"name": "Bus 9", "vn_kv": 230.0, "type": "b"},
        ]
        
        bus_indices = []
        for bus_data in buses:
            bus_idx = pp.create_bus(net, **bus_data)
            bus_indices.append(bus_idx)
        
        # Generators
        pp.create_gen(net, bus_indices[0], p_mw=71.6, vm_pu=1.04, name="Gen 1")
        pp.create_gen(net, bus_indices[1], p_mw=163.0, vm_pu=1.025, name="Gen 2")
        pp.create_gen(net, bus_indices[2], p_mw=85.0, vm_pu=1.025, name="Gen 3")
        
        # Transformers
        pp.create_transformer(net, bus_indices[0], bus_indices[3], std_type="25 MVA 16.5/230 kV", name="T1")
        pp.create_transformer(net, bus_indices[1], bus_indices[6], std_type="25 MVA 18/230 kV", name="T2")
        pp.create_transformer(net, bus_indices[2], bus_indices[8], std_type="25 MVA 13.8/230 kV", name="T3")
        
        # Lines
        line_data = [
            (3, 4, 146.0, "Line 4-5"),
            (3, 5, 48.0, "Line 4-6"),
            (4, 5, 73.0, "Line 5-6"),
            (5, 6, 120.0, "Line 6-7"),
            (5, 7, 161.0, "Line 6-8"),
            (6, 7, 32.0, "Line 7-8"),
            (6, 8, 85.0, "Line 7-9"),
            (7, 8, 119.0, "Line 8-9"),
        ]
        
        for from_bus, to_bus, length_km, name in line_data:
            pp.create_line_from_parameters(
                net, bus_indices[from_bus], bus_indices[to_bus],
                length_km=length_km, r_ohm_per_km=0.01938, x_ohm_per_km=0.05917,
                c_nf_per_km=264.0, max_i_ka=0.4, name=name
            )
        
        # Loads
        pp.create_load(net, bus_indices[4], p_mw=125.0, q_mvar=50.0, name="Load 5")
        pp.create_load(net, bus_indices[5], p_mw=90.0, q_mvar=30.0, name="Load 6")
        pp.create_load(net, bus_indices[7], p_mw=100.0, q_mvar=35.0, name="Load 8")
        
        metadata = NetworkMetadata(
            name="IEEE 9-Bus Test System",
            description="Standard IEEE 9-bus transmission test system",
            voltage_levels=[13.8, 16.5, 18.0, 230.0],
            num_buses=9,
            num_lines=8,
            num_generators=3,
            num_loads=3,
            total_load_mw=315.0,
            total_generation_mw=319.6,
            network_type="transmission",
            complexity="simple"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("ieee_9_bus")
        }
    
    def _create_ieee_14_bus(self) -> Dict[str, Any]:
        """Create IEEE 14-bus test system."""
        net = pp.networks.case14()
        
        metadata = NetworkMetadata(
            name="IEEE 14-Bus Test System",
            description="Standard IEEE 14-bus transmission test system",
            voltage_levels=[69.0, 138.0],
            num_buses=14,
            num_lines=20,
            num_generators=5,
            num_loads=11,
            total_load_mw=259.0,
            total_generation_mw=272.4,
            network_type="transmission",
            complexity="medium"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("ieee_14_bus")
        }
    
    def _create_ieee_30_bus(self) -> Dict[str, Any]:
        """Create IEEE 30-bus test system."""
        net = pp.networks.case30()
        
        metadata = NetworkMetadata(
            name="IEEE 30-Bus Test System",
            description="Standard IEEE 30-bus transmission test system",
            voltage_levels=[132.0],
            num_buses=30,
            num_lines=41,
            num_generators=6,
            num_loads=21,
            total_load_mw=283.4,
            total_generation_mw=300.0,
            network_type="transmission",
            complexity="complex"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("ieee_30_bus")
        }
    
    def _create_simple_microgrid(self) -> Dict[str, Any]:
        """Create a simple microgrid with renewable sources."""
        net = pp.create_empty_network()
        
        # Main grid connection
        slack_bus = pp.create_bus(net, vn_kv=11.0, name="Grid Connection")
        pp.create_ext_grid(net, slack_bus, vm_pu=1.0, name="Main Grid")
        
        # Microgrid buses
        pcc_bus = pp.create_bus(net, vn_kv=0.4, name="PCC Bus")
        solar_bus = pp.create_bus(net, vn_kv=0.4, name="Solar Bus")
        wind_bus = pp.create_bus(net, vn_kv=0.4, name="Wind Bus")
        load_bus1 = pp.create_bus(net, vn_kv=0.4, name="Load Bus 1")
        load_bus2 = pp.create_bus(net, vn_kv=0.4, name="Load Bus 2")
        battery_bus = pp.create_bus(net, vn_kv=0.4, name="Battery Bus")
        
        # Transformer
        pp.create_transformer(net, slack_bus, pcc_bus, std_type="0.25 MVA 10/0.4 kV", name="Main Transformer")
        
        # Distribution lines
        pp.create_line_from_parameters(net, pcc_bus, solar_bus, length_km=0.2, 
                                     r_ohm_per_km=0.164, x_ohm_per_km=0.174, c_nf_per_km=315.0, max_i_ka=0.3)
        pp.create_line_from_parameters(net, pcc_bus, wind_bus, length_km=0.5,
                                     r_ohm_per_km=0.164, x_ohm_per_km=0.174, c_nf_per_km=315.0, max_i_ka=0.3)
        pp.create_line_from_parameters(net, pcc_bus, load_bus1, length_km=0.1,
                                     r_ohm_per_km=0.164, x_ohm_per_km=0.174, c_nf_per_km=315.0, max_i_ka=0.3)
        pp.create_line_from_parameters(net, pcc_bus, load_bus2, length_km=0.3,
                                     r_ohm_per_km=0.164, x_ohm_per_km=0.174, c_nf_per_km=315.0, max_i_ka=0.3)
        pp.create_line_from_parameters(net, pcc_bus, battery_bus, length_km=0.05,
                                     r_ohm_per_km=0.164, x_ohm_per_km=0.174, c_nf_per_km=315.0, max_i_ka=0.3)
        
        # Renewable generators
        pp.create_sgen(net, solar_bus, p_mw=0.1, q_mvar=0.02, name="Solar PV", type="PV")
        pp.create_sgen(net, wind_bus, p_mw=0.05, q_mvar=0.01, name="Wind Turbine", type="WP")
        
        # Loads
        pp.create_load(net, load_bus1, p_mw=0.08, q_mvar=0.02, name="Residential Load")
        pp.create_load(net, load_bus2, p_mw=0.05, q_mvar=0.015, name="Commercial Load")
        
        # Battery storage
        pp.create_storage(net, battery_bus, p_mw=0.05, max_e_mwh=0.2, name="Battery Storage")
        
        metadata = NetworkMetadata(
            name="Simple Microgrid",
            description="Small microgrid with renewable generation and storage",
            voltage_levels=[0.4, 11.0],
            num_buses=7,
            num_lines=5,
            num_generators=2,
            num_loads=2,
            total_load_mw=0.13,
            total_generation_mw=0.15,
            network_type="microgrid",
            complexity="simple"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("microgrid")
        }
    
    def _create_industrial_plant(self) -> Dict[str, Any]:
        """Create an industrial plant power system."""
        net = pp.create_empty_network()
        
        # Utility connection
        utility_bus = pp.create_bus(net, vn_kv=138.0, name="Utility Bus")
        pp.create_ext_grid(net, utility_bus, vm_pu=1.0, name="Utility Grid")
        
        # Plant buses
        main_bus = pp.create_bus(net, vn_kv=13.8, name="Main Bus")
        motor_bus1 = pp.create_bus(net, vn_kv=4.16, name="Motor Bus 1")
        motor_bus2 = pp.create_bus(net, vn_kv=4.16, name="Motor Bus 2")
        aux_bus = pp.create_bus(net, vn_kv=0.48, name="Auxiliary Bus")
        gen_bus = pp.create_bus(net, vn_kv=13.8, name="Generator Bus")
        
        # Transformers
        pp.create_transformer(net, utility_bus, main_bus, std_type="25 MVA 138/13.8 kV", name="Main Transformer")
        pp.create_transformer(net, main_bus, motor_bus1, std_type="5 MVA 13.8/4.16 kV", name="Motor Transformer 1")
        pp.create_transformer(net, main_bus, motor_bus2, std_type="5 MVA 13.8/4.16 kV", name="Motor Transformer 2")
        pp.create_transformer(net, main_bus, aux_bus, std_type="1 MVA 13.8/0.48 kV", name="Auxiliary Transformer")
        
        # Emergency generator
        pp.create_gen(net, gen_bus, p_mw=2.0, vm_pu=1.0, name="Emergency Generator")
        pp.create_switch(net, main_bus, gen_bus, et="b", name="Generator Breaker", closed=False)
        
        # Industrial loads (motors)
        pp.create_load(net, motor_bus1, p_mw=3.5, q_mvar=1.2, name="Large Motor 1")
        pp.create_load(net, motor_bus2, p_mw=3.0, q_mvar=1.0, name="Large Motor 2")
        pp.create_load(net, aux_bus, p_mw=0.5, q_mvar=0.15, name="Auxiliary Loads")
        
        # Arc furnace (variable load)
        pp.create_load(net, main_bus, p_mw=8.0, q_mvar=2.5, name="Arc Furnace")
        
        metadata = NetworkMetadata(
            name="Industrial Plant",
            description="Typical industrial plant with large motors and variable loads",
            voltage_levels=[0.48, 4.16, 13.8, 138.0],
            num_buses=6,
            num_lines=0,
            num_generators=1,
            num_loads=4,
            total_load_mw=15.0,
            total_generation_mw=2.0,
            network_type="industrial",
            complexity="medium"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("industrial")
        }
    
    def _create_distribution_feeder(self) -> Dict[str, Any]:
        """Create a distribution feeder network."""
        net = pp.create_empty_network()
        
        # Substation
        substation_hv = pp.create_bus(net, vn_kv=69.0, name="Substation HV")
        substation_lv = pp.create_bus(net, vn_kv=12.47, name="Substation LV")
        pp.create_ext_grid(net, substation_hv, vm_pu=1.0, name="Transmission System")
        pp.create_transformer(net, substation_hv, substation_lv, std_type="25 MVA 69/12.47 kV", name="Substation Transformer")
        
        # Main feeder
        buses = [substation_lv]
        for i in range(1, 11):
            bus = pp.create_bus(net, vn_kv=12.47, name=f"Feeder Bus {i}")
            buses.append(bus)
            
            # Create line to previous bus
            pp.create_line_from_parameters(
                net, buses[i-1], bus, length_km=1.0,
                r_ohm_per_km=0.4013, x_ohm_per_km=0.4013, c_nf_per_km=8.5, max_i_ka=0.4,
                name=f"Feeder Line {i-1}-{i}"
            )
            
            # Add loads
            if i % 2 == 0:  # Every other bus
                load_power = np.random.uniform(0.5, 1.5)
                pp.create_load(net, bus, p_mw=load_power, q_mvar=load_power*0.3, name=f"Load {i}")
        
        # Lateral branches
        for i in [3, 6, 9]:
            lateral_bus = pp.create_bus(net, vn_kv=12.47, name=f"Lateral Bus {i}")
            pp.create_line_from_parameters(
                net, buses[i], lateral_bus, length_km=0.5,
                r_ohm_per_km=0.4013, x_ohm_per_km=0.4013, c_nf_per_km=8.5, max_i_ka=0.4,
                name=f"Lateral Line {i}"
            )
            pp.create_load(net, lateral_bus, p_mw=0.3, q_mvar=0.1, name=f"Lateral Load {i}")
        
        # Distributed generation
        pp.create_sgen(net, buses[5], p_mw=0.5, q_mvar=0.0, name="Rooftop Solar", type="PV")
        pp.create_sgen(net, buses[8], p_mw=0.2, q_mvar=0.0, name="Small Wind", type="WP")
        
        metadata = NetworkMetadata(
            name="Distribution Feeder",
            description="Typical radial distribution feeder with DG",
            voltage_levels=[12.47, 69.0],
            num_buses=14,
            num_lines=13,
            num_generators=2,
            num_loads=8,
            total_load_mw=6.4,
            total_generation_mw=0.7,
            network_type="distribution",
            complexity="medium"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("distribution")
        }
    
    def _create_renewable_grid(self) -> Dict[str, Any]:
        """Create a grid with high renewable penetration."""
        net = pp.create_empty_network()
        
        # Main grid
        grid_bus = pp.create_bus(net, vn_kv=230.0, name="Grid Bus")
        pp.create_ext_grid(net, grid_bus, vm_pu=1.0, name="Main Grid")
        
        # Renewable farm buses
        wind_farm_bus = pp.create_bus(net, vn_kv=34.5, name="Wind Farm Bus")
        solar_farm_bus = pp.create_bus(net, vn_kv=34.5, name="Solar Farm Bus")
        load_center_bus = pp.create_bus(net, vn_kv=138.0, name="Load Center Bus")
        
        # Step-up transformers
        pp.create_transformer(net, grid_bus, load_center_bus, std_type="100 MVA 230/138 kV", name="Grid Transformer")
        wind_tx_bus = pp.create_bus(net, vn_kv=230.0, name="Wind TX Bus")
        solar_tx_bus = pp.create_bus(net, vn_kv=230.0, name="Solar TX Bus")
        pp.create_transformer(net, wind_tx_bus, wind_farm_bus, std_type="50 MVA 230/34.5 kV", name="Wind Transformer")
        pp.create_transformer(net, solar_tx_bus, solar_farm_bus, std_type="50 MVA 230/34.5 kV", name="Solar Transformer")
        
        # Transmission lines
        pp.create_line_from_parameters(net, grid_bus, wind_tx_bus, length_km=50.0,
                                     r_ohm_per_km=0.01938, x_ohm_per_km=0.05917, c_nf_per_km=264.0, max_i_ka=0.4)
        pp.create_line_from_parameters(net, grid_bus, solar_tx_bus, length_km=30.0,
                                     r_ohm_per_km=0.01938, x_ohm_per_km=0.05917, c_nf_per_km=264.0, max_i_ka=0.4)
        
        # Wind farm (multiple turbines)
        for i in range(5):
            pp.create_sgen(net, wind_farm_bus, p_mw=8.0, q_mvar=0.0, name=f"Wind Turbine {i+1}", type="WP")
        
        # Solar farm (multiple inverters)
        for i in range(10):
            pp.create_sgen(net, solar_farm_bus, p_mw=5.0, q_mvar=0.0, name=f"Solar Inverter {i+1}", type="PV")
        
        # Load centers
        pp.create_load(net, load_center_bus, p_mw=60.0, q_mvar=20.0, name="City Load")
        
        # Energy storage
        storage_bus = pp.create_bus(net, vn_kv=34.5, name="Storage Bus")
        pp.create_transformer(net, load_center_bus, storage_bus, std_type="25 MVA 138/34.5 kV", name="Storage Transformer")
        pp.create_storage(net, storage_bus, p_mw=20.0, max_e_mwh=80.0, name="Grid Storage")
        
        metadata = NetworkMetadata(
            name="Renewable Grid",
            description="High renewable penetration grid with storage",
            voltage_levels=[34.5, 138.0, 230.0],
            num_buses=8,
            num_lines=2,
            num_generators=15,
            num_loads=1,
            total_load_mw=60.0,
            total_generation_mw=90.0,
            network_type="transmission",
            complexity="complex"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": self._convert_to_pypsa(net),
            "analysis_cases": self._generate_analysis_cases("renewable")
        }
    
    def _create_dc_microgrid(self) -> Dict[str, Any]:
        """Create a DC microgrid network."""
        # Note: pandapower has limited DC support, so this is a simplified representation
        net = pp.create_empty_network()
        
        # DC buses (represented as AC buses with special naming)
        dc_main = pp.create_bus(net, vn_kv=0.8, name="DC Main Bus (+800V)")
        dc_solar = pp.create_bus(net, vn_kv=0.8, name="DC Solar Bus (+800V)")
        dc_battery = pp.create_bus(net, vn_kv=0.8, name="DC Battery Bus (+800V)")
        dc_load1 = pp.create_bus(net, vn_kv=0.4, name="DC Load Bus 1 (+400V)")
        dc_load2 = pp.create_bus(net, vn_kv=0.4, name="DC Load Bus 2 (+400V)")
        ac_grid = pp.create_bus(net, vn_kv=0.4, name="AC Grid Interface")
        
        # AC grid connection
        pp.create_ext_grid(net, ac_grid, vm_pu=1.0, name="AC Grid")
        
        # DC "lines" (represented as low impedance AC lines)
        pp.create_line_from_parameters(net, dc_main, dc_solar, length_km=0.1,
                                     r_ohm_per_km=0.01, x_ohm_per_km=0.001, c_nf_per_km=0.0, max_i_ka=1.0)
        pp.create_line_from_parameters(net, dc_main, dc_battery, length_km=0.05,
                                     r_ohm_per_km=0.01, x_ohm_per_km=0.001, c_nf_per_km=0.0, max_i_ka=1.0)
        pp.create_line_from_parameters(net, dc_main, dc_load1, length_km=0.2,
                                     r_ohm_per_km=0.02, x_ohm_per_km=0.001, c_nf_per_km=0.0, max_i_ka=0.5)
        pp.create_line_from_parameters(net, dc_main, dc_load2, length_km=0.15,
                                     r_ohm_per_km=0.02, x_ohm_per_km=0.001, c_nf_per_km=0.0, max_i_ka=0.5)
        
        # Converters (represented as generators/loads)
        pp.create_sgen(net, dc_solar, p_mw=0.05, q_mvar=0.0, name="Solar DC Source", type="PV")
        pp.create_storage(net, dc_battery, p_mw=0.03, max_e_mwh=0.1, name="DC Battery")
        pp.create_sgen(net, dc_main, p_mw=-0.02, q_mvar=0.0, name="AC-DC Converter", type="conv")
        
        # DC loads
        pp.create_load(net, dc_load1, p_mw=0.02, q_mvar=0.0, name="LED Lighting")
        pp.create_load(net, dc_load2, p_mw=0.015, q_mvar=0.0, name="DC Motor Drive")
        
        metadata = NetworkMetadata(
            name="DC Microgrid",
            description="DC microgrid with solar, battery, and DC loads",
            voltage_levels=[0.4, 0.8],
            num_buses=6,
            num_lines=4,
            num_generators=2,
            num_loads=2,
            total_load_mw=0.035,
            total_generation_mw=0.05,
            network_type="microgrid",
            complexity="simple"
        )
        
        return {
            "pandapower": net,
            "metadata": metadata,
            "pypsa": None,  # PyPSA handles DC better, but omitted for simplicity
            "analysis_cases": self._generate_analysis_cases("dc_microgrid")
        }
    
    def _convert_to_pypsa(self, pp_net) -> Optional[Any]:
        """Convert pandapower network to PyPSA (simplified)."""
        try:
            # This is a placeholder for actual conversion logic
            # In practice, you'd need proper conversion between the two formats
            pypsa_net = pypsa.Network()
            pypsa_net.name = f"Converted from pandapower network"
            return pypsa_net
        except Exception:
            return None
    
    def _generate_analysis_cases(self, network_type: str) -> List[Dict[str, Any]]:
        """Generate typical analysis cases for each network type."""
        base_cases = [
            {
                "name": "Normal Operation",
                "description": "System under normal operating conditions",
                "type": "load_flow",
                "parameters": {"tolerance": 1e-6, "max_iterations": 50}
            },
            {
                "name": "N-1 Contingency",
                "description": "Single line outage analysis",
                "type": "contingency",
                "parameters": {"contingency_type": "line", "severity": "single"}
            },
            {
                "name": "Short Circuit Analysis",
                "description": "Three-phase fault analysis",
                "type": "short_circuit",
                "parameters": {"fault_type": "3ph", "fault_impedance": 0.0}
            }
        ]
        
        # Add network-specific cases
        if network_type in ["microgrid", "renewable", "dc_microgrid"]:
            base_cases.extend([
                {
                    "name": "Renewable Variability",
                    "description": "Analysis with variable renewable generation",
                    "type": "time_series",
                    "parameters": {"duration_hours": 24, "resolution_minutes": 15}
                },
                {
                    "name": "Storage Operation",
                    "description": "Energy storage charge/discharge cycles",
                    "type": "storage_analysis",
                    "parameters": {"soc_initial": 0.5, "efficiency": 0.95}
                }
            ])
        
        if network_type == "industrial":
            base_cases.extend([
                {
                    "name": "Motor Starting",
                    "description": "Large motor starting transient",
                    "type": "transient",
                    "parameters": {"motor_size_mw": 3.5, "starting_method": "DOL"}
                },
                {
                    "name": "Harmonic Analysis",
                    "description": "Power quality assessment with nonlinear loads",
                    "type": "harmonic",
                    "parameters": {"max_harmonic": 50, "thd_limit": 0.05}
                }
            ])
        
        return base_cases


# Singleton instance for easy access
sample_networks = MockNetworkGenerator() 