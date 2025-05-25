"""
Sample Analysis Cases
Provides pre-configured analysis scenarios for comprehensive power system studies.

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta
import numpy as np

from .mock_networks import sample_networks, NetworkMetadata
from .mock_data import MockDataGenerator


@dataclass
class AnalysisCase:
    """Analysis case definition."""
    case_id: str
    name: str
    description: str
    network_name: str
    analysis_type: str
    parameters: Dict[str, Any]
    expected_results: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    difficulty: str = "medium"  # "easy", "medium", "hard"
    estimated_runtime_seconds: float = 60.0


@dataclass
class StudyPlan:
    """Collection of related analysis cases."""
    plan_id: str
    name: str
    description: str
    cases: List[AnalysisCase]
    objective: str
    deliverables: List[str]
    estimated_duration_hours: float


class SampleCases:
    """Provides comprehensive analysis cases for testing and education."""
    
    def __init__(self):
        self.mock_data = MockDataGenerator()
        self.cases = {}
        self.study_plans = {}
        self._initialize_cases()
        self._initialize_study_plans()
    
    def _initialize_cases(self):
        """Initialize all sample analysis cases."""
        self.cases = {
            # Basic Load Flow Studies
            "lf_ieee9_basic": self._create_basic_load_flow("ieee_9_bus"),
            "lf_ieee14_basic": self._create_basic_load_flow("ieee_14_bus"),
            "lf_ieee30_basic": self._create_basic_load_flow("ieee_30_bus"),
            
            # Contingency Analysis
            "cont_n1_lines": self._create_n1_contingency("ieee_14_bus", "lines"),
            "cont_n1_generators": self._create_n1_contingency("ieee_14_bus", "generators"),
            "cont_n2_critical": self._create_n2_contingency("ieee_30_bus"),
            
            # Short Circuit Analysis
            "sc_3phase_fault": self._create_short_circuit_case("ieee_14_bus", "3ph"),
            "sc_line_ground": self._create_short_circuit_case("ieee_14_bus", "lg"),
            "sc_line_line": self._create_short_circuit_case("ieee_14_bus", "ll"),
            
            # Stability Studies
            "stab_small_signal": self._create_stability_case("ieee_9_bus", "small_signal"),
            "stab_transient": self._create_stability_case("ieee_9_bus", "transient"),
            "stab_voltage": self._create_voltage_stability_case("ieee_14_bus"),
            
            # Harmonic Analysis
            "harm_industrial": self._create_harmonic_case("industrial_plant"),
            "harm_renewable": self._create_harmonic_case("renewable_grid"),
            
            # Microgrid Studies
            "mg_islanding": self._create_microgrid_islanding("simple_microgrid"),
            "mg_resync": self._create_microgrid_resync("simple_microgrid"),
            "mg_energy_management": self._create_energy_management("simple_microgrid"),
            
            # Renewable Integration
            "ren_variability": self._create_renewable_variability("renewable_grid"),
            "ren_hosting_capacity": self._create_hosting_capacity("distribution_feeder"),
            "ren_storage_sizing": self._create_storage_sizing("renewable_grid"),
            
            # Distribution Studies
            "dist_voltage_regulation": self._create_voltage_regulation("distribution_feeder"),
            "dist_protection_coordination": self._create_protection_coordination("distribution_feeder"),
            "dist_dg_interconnection": self._create_dg_interconnection("distribution_feeder"),
            
            # Industrial Applications
            "ind_motor_starting": self._create_motor_starting("industrial_plant"),
            "ind_power_quality": self._create_power_quality("industrial_plant"),
            "ind_emergency_backup": self._create_emergency_backup("industrial_plant"),
            
            # Optimization Studies
            "opt_economic_dispatch": self._create_economic_dispatch("ieee_30_bus"),
            "opt_opf_basic": self._create_optimal_power_flow("ieee_14_bus"),
            "opt_unit_commitment": self._create_unit_commitment("ieee_30_bus"),
            
            # Advanced Studies
            "adv_pmu_placement": self._create_pmu_placement("ieee_30_bus"),
            "adv_state_estimation": self._create_state_estimation("ieee_14_bus"),
            "adv_market_clearing": self._create_market_clearing("ieee_30_bus"),
        }
    
    def _initialize_study_plans(self):
        """Initialize comprehensive study plans."""
        self.study_plans = {
            "transmission_planning": self._create_transmission_planning_study(),
            "distribution_modernization": self._create_distribution_modernization_study(),
            "microgrid_design": self._create_microgrid_design_study(),
            "renewable_integration": self._create_renewable_integration_study(),
            "industrial_power_study": self._create_industrial_power_study(),
            "system_reliability": self._create_system_reliability_study(),
            "power_quality_assessment": self._create_power_quality_study(),
            "grid_modernization": self._create_grid_modernization_study(),
        }
    
    def get_case(self, case_id: str) -> AnalysisCase:
        """Get a specific analysis case."""
        if case_id not in self.cases:
            raise ValueError(f"Case '{case_id}' not found. Available: {list(self.cases.keys())}")
        return self.cases[case_id]
    
    def get_study_plan(self, plan_id: str) -> StudyPlan:
        """Get a specific study plan."""
        if plan_id not in self.study_plans:
            raise ValueError(f"Study plan '{plan_id}' not found. Available: {list(self.study_plans.keys())}")
        return self.study_plans[plan_id]
    
    def list_cases(self, analysis_type: Optional[str] = None, difficulty: Optional[str] = None) -> List[str]:
        """List available cases with optional filtering."""
        cases = []
        for case_id, case in self.cases.items():
            if analysis_type and case.analysis_type != analysis_type:
                continue
            if difficulty and case.difficulty != difficulty:
                continue
            cases.append(case_id)
        return cases
    
    def list_study_plans(self) -> List[str]:
        """List available study plans."""
        return list(self.study_plans.keys())
    
    def export_case(self, case_id: str, filepath: str):
        """Export a case to JSON file."""
        case = self.get_case(case_id)
        with open(filepath, 'w') as f:
            json.dump(asdict(case), f, indent=2, default=str)
    
    def export_study_plan(self, plan_id: str, filepath: str):
        """Export a study plan to JSON file."""
        plan = self.get_study_plan(plan_id)
        plan_dict = asdict(plan)
        with open(filepath, 'w') as f:
            json.dump(plan_dict, f, indent=2, default=str)
    
    # Case Creation Methods
    def _create_basic_load_flow(self, network_name: str) -> AnalysisCase:
        """Create basic load flow case."""
        return AnalysisCase(
            case_id=f"lf_{network_name}_basic",
            name=f"Basic Load Flow - {network_name}",
            description=f"Steady-state power flow analysis for {network_name} network",
            network_name=network_name,
            analysis_type="load_flow",
            parameters={
                "algorithm": "newton_raphson",
                "tolerance": 1e-6,
                "max_iterations": 50,
                "flat_start": True,
                "distributed_slack": False,
                "check_connectivity": True
            },
            expected_results={
                "convergence": True,
                "max_voltage_deviation": 0.05,
                "total_losses_percent": 5.0,
                "min_voltage_pu": 0.95,
                "max_voltage_pu": 1.05
            },
            tags=["load_flow", "basic", "steady_state"],
            difficulty="easy",
            estimated_runtime_seconds=5.0
        )
    
    def _create_n1_contingency(self, network_name: str, element_type: str) -> AnalysisCase:
        """Create N-1 contingency case."""
        return AnalysisCase(
            case_id=f"cont_n1_{element_type}_{network_name}",
            name=f"N-1 {element_type.title()} Contingency - {network_name}",
            description=f"Single {element_type[:-1]} outage contingency analysis",
            network_name=network_name,
            analysis_type="contingency",
            parameters={
                "contingency_type": "n_minus_1",
                "element_types": [element_type[:-1]],  # Remove 's' from plural
                "voltage_limits": {"min": 0.9, "max": 1.1},
                "thermal_limits": {"max_loading": 100},
                "include_cascading": False,
                "post_contingency_actions": True
            },
            expected_results={
                "total_contingencies": 10,
                "violations": 2,
                "critical_contingencies": 1,
                "system_secure": False
            },
            tags=["contingency", "n-1", "reliability"],
            difficulty="medium",
            estimated_runtime_seconds=30.0
        )
    
    def _create_n2_contingency(self, network_name: str) -> AnalysisCase:
        """Create N-2 contingency case."""
        return AnalysisCase(
            case_id=f"cont_n2_{network_name}",
            name=f"N-2 Critical Contingency - {network_name}",
            description="Double contingency analysis for critical system elements",
            network_name=network_name,
            analysis_type="contingency",
            parameters={
                "contingency_type": "n_minus_k",
                "k_value": 2,
                "element_types": ["line", "transformer"],
                "voltage_limits": {"min": 0.85, "max": 1.15},
                "thermal_limits": {"max_loading": 120},
                "include_cascading": True,
                "critical_pairs_only": True
            },
            expected_results={
                "total_contingencies": 45,
                "violations": 8,
                "critical_contingencies": 3,
                "cascading_events": 1
            },
            tags=["contingency", "n-2", "critical", "cascading"],
            difficulty="hard",
            estimated_runtime_seconds=120.0
        )
    
    def _create_short_circuit_case(self, network_name: str, fault_type: str) -> AnalysisCase:
        """Create short circuit analysis case."""
        fault_names = {
            "3ph": "Three-Phase",
            "lg": "Line-to-Ground",
            "ll": "Line-to-Line"
        }
        
        return AnalysisCase(
            case_id=f"sc_{fault_type}_{network_name}",
            name=f"{fault_names[fault_type]} Short Circuit - {network_name}",
            description=f"{fault_names[fault_type]} fault current analysis",
            network_name=network_name,
            analysis_type="short_circuit",
            parameters={
                "fault_type": fault_type,
                "case": "max",
                "fault_impedance": 0.0,
                "include_motors": True,
                "motor_contribution": True,
                "method": "iec",
                "calculate_voltages": True
            },
            expected_results={
                "max_fault_current_ka": 25.0,
                "min_fault_current_ka": 8.0,
                "critical_buses": [1, 5, 8],
                "equipment_adequacy": True
            },
            tags=["short_circuit", fault_type, "protection"],
            difficulty="medium",
            estimated_runtime_seconds=15.0
        )
    
    def _create_stability_case(self, network_name: str, stability_type: str) -> AnalysisCase:
        """Create stability analysis case."""
        if stability_type == "small_signal":
            return AnalysisCase(
                case_id=f"stab_ss_{network_name}",
                name=f"Small Signal Stability - {network_name}",
                description="Small signal stability and oscillatory mode analysis",
                network_name=network_name,
                analysis_type="stability",
                parameters={
                    "stability_type": "small_signal",
                    "operating_point": "nominal",
                    "eigenvalue_analysis": True,
                    "participation_factors": True,
                    "frequency_range": [0.1, 2.0],
                    "damping_threshold": 0.05
                },
                expected_results={
                    "stable": True,
                    "critical_modes": 2,
                    "min_damping": 0.08,
                    "oscillatory_modes": 5
                },
                tags=["stability", "small_signal", "eigenvalue"],
                difficulty="hard",
                estimated_runtime_seconds=60.0
            )
        else:  # transient
            return AnalysisCase(
                case_id=f"stab_trans_{network_name}",
                name=f"Transient Stability - {network_name}",
                description="Transient stability analysis with fault clearing",
                network_name=network_name,
                analysis_type="stability",
                parameters={
                    "stability_type": "transient",
                    "simulation_time": 10.0,
                    "time_step": 0.01,
                    "fault_location": "bus_5",
                    "fault_clearing_time": 0.15,
                    "post_fault_actions": []
                },
                expected_results={
                    "stable": True,
                    "critical_clearing_time": 0.18,
                    "max_rotor_angle": 45.0,
                    "settling_time": 3.5
                },
                tags=["stability", "transient", "fault_clearing"],
                difficulty="hard",
                estimated_runtime_seconds=90.0
            )
    
    def _create_voltage_stability_case(self, network_name: str) -> AnalysisCase:
        """Create voltage stability case."""
        return AnalysisCase(
            case_id=f"vstab_{network_name}",
            name=f"Voltage Stability - {network_name}",
            description="Voltage stability and collapse point analysis",
            network_name=network_name,
            analysis_type="stability",
            parameters={
                "stability_type": "voltage",
                "load_increase_direction": "uniform",
                "pv_curve_analysis": True,
                "critical_bus_analysis": True,
                "contingency_screening": True,
                "reactive_limits": True
            },
            expected_results={
                "voltage_margin": 25.0,
                "critical_buses": [7, 12],
                "collapse_point_mw": 450.0,
                "weak_areas": ["load_center_1"]
            },
            tags=["stability", "voltage", "pv_curve"],
            difficulty="hard",
            estimated_runtime_seconds=45.0
        )
    
    def _create_harmonic_case(self, network_name: str) -> AnalysisCase:
        """Create harmonic analysis case."""
        return AnalysisCase(
            case_id=f"harm_{network_name}",
            name=f"Harmonic Analysis - {network_name}",
            description="Power quality harmonic distortion analysis",
            network_name=network_name,
            analysis_type="harmonic",
            parameters={
                "harmonic_orders": [3, 5, 7, 9, 11, 13, 15, 17, 19],
                "thd_limits": {"voltage": 5.0, "current": 8.0},
                "individual_limits": {"voltage": 3.0, "current": 4.0},
                "resonance_analysis": True,
                "filter_design": True,
                "standards": "ieee_519"
            },
            expected_results={
                "max_voltage_thd": 4.2,
                "max_current_thd": 7.8,
                "compliance": True,
                "resonance_frequencies": [350, 750],
                "filter_required": False
            },
            tags=["harmonic", "power_quality", "thd"],
            difficulty="medium",
            estimated_runtime_seconds=40.0
        )
    
    def _create_microgrid_islanding(self, network_name: str) -> AnalysisCase:
        """Create microgrid islanding case."""
        return AnalysisCase(
            case_id=f"mg_island_{network_name}",
            name=f"Microgrid Islanding - {network_name}",
            description="Planned islanding operation and control",
            network_name=network_name,
            analysis_type="microgrid",
            parameters={
                "islanding_type": "planned",
                "trigger_event": "utility_outage",
                "load_shedding": True,
                "generation_control": "droop",
                "frequency_limits": [59.3, 60.7],
                "voltage_limits": [0.95, 1.05],
                "transition_time": 2.0
            },
            expected_results={
                "successful_islanding": True,
                "load_shed_mw": 0.02,
                "frequency_nadir": 59.4,
                "voltage_dip": 0.05,
                "stability_maintained": True
            },
            tags=["microgrid", "islanding", "control"],
            difficulty="hard",
            estimated_runtime_seconds=75.0
        )
    
    def _create_microgrid_resync(self, network_name: str) -> AnalysisCase:
        """Create microgrid resynchronization case."""
        return AnalysisCase(
            case_id=f"mg_resync_{network_name}",
            name=f"Microgrid Resynchronization - {network_name}",
            description="Grid reconnection and synchronization procedure",
            network_name=network_name,
            analysis_type="microgrid",
            parameters={
                "sync_method": "automatic",
                "sync_check_parameters": {
                    "voltage_diff": 0.1,
                    "frequency_diff": 0.1,
                    "phase_angle_diff": 10.0
                },
                "ramp_rate": 0.5,
                "load_transfer": "gradual",
                "timeout": 300.0
            },
            expected_results={
                "sync_successful": True,
                "sync_time": 45.0,
                "transient_overvoltage": 1.02,
                "phase_angle_error": 5.0,
                "power_transfer_smooth": True
            },
            tags=["microgrid", "resync", "synchronization"],
            difficulty="hard",
            estimated_runtime_seconds=60.0
        )
    
    def _create_energy_management(self, network_name: str) -> AnalysisCase:
        """Create energy management case."""
        return AnalysisCase(
            case_id=f"mg_ems_{network_name}",
            name=f"Energy Management System - {network_name}",
            description="Optimal energy management and storage control",
            network_name=network_name,
            analysis_type="optimization",
            parameters={
                "optimization_horizon": 24,
                "time_resolution": 0.25,
                "objective": "cost_minimization",
                "storage_control": True,
                "demand_response": True,
                "renewable_forecasting": True,
                "price_signals": True
            },
            expected_results={
                "cost_savings": 15.2,
                "storage_cycles": 1.2,
                "renewable_utilization": 0.85,
                "peak_reduction": 0.03,
                "objective_value": 450.0
            },
            tags=["microgrid", "ems", "optimization"],
            difficulty="medium",
            estimated_runtime_seconds=50.0
        )
    
    def _create_renewable_variability(self, network_name: str) -> AnalysisCase:
        """Create renewable variability case."""
        return AnalysisCase(
            case_id=f"ren_var_{network_name}",
            name=f"Renewable Variability Impact - {network_name}",
            description="Impact of renewable generation variability on grid operations",
            network_name=network_name,
            analysis_type="time_series",
            parameters={
                "duration_hours": 168,
                "resolution_minutes": 15,
                "renewable_scenarios": ["high_wind", "low_solar", "mixed"],
                "load_following": True,
                "ramping_analysis": True,
                "reserve_requirements": True
            },
            expected_results={
                "max_ramp_rate": 15.0,
                "reserve_shortage_hours": 8,
                "voltage_violations": 12,
                "curtailment_mwh": 45.0,
                "flexibility_required": 25.0
            },
            tags=["renewable", "variability", "time_series"],
            difficulty="medium",
            estimated_runtime_seconds=90.0
        )
    
    def _create_hosting_capacity(self, network_name: str) -> AnalysisCase:
        """Create hosting capacity case."""
        return AnalysisCase(
            case_id=f"host_cap_{network_name}",
            name=f"DG Hosting Capacity - {network_name}",
            description="Distributed generation hosting capacity analysis",
            network_name=network_name,
            analysis_type="hosting_capacity",
            parameters={
                "dg_type": "solar_pv",
                "penetration_step": 1.0,
                "max_penetration": 100.0,
                "voltage_limits": [0.95, 1.05],
                "thermal_limits": 100.0,
                "protection_coordination": True,
                "power_quality_limits": True
            },
            expected_results={
                "hosting_capacity_mw": 12.5,
                "limiting_factor": "voltage_rise",
                "critical_buses": [8, 12],
                "mitigation_required": True,
                "upgrade_cost": 45000.0
            },
            tags=["hosting_capacity", "dg", "penetration"],
            difficulty="medium",
            estimated_runtime_seconds=120.0
        )
    
    def _create_storage_sizing(self, network_name: str) -> AnalysisCase:
        """Create storage sizing case."""
        return AnalysisCase(
            case_id=f"storage_{network_name}",
            name=f"Energy Storage Sizing - {network_name}",
            description="Optimal energy storage system sizing and placement",
            network_name=network_name,
            analysis_type="optimization",
            parameters={
                "storage_technologies": ["lithium_ion", "flow_battery"],
                "power_range": [1.0, 50.0],
                "energy_range": [2.0, 200.0],
                "efficiency": 0.90,
                "cycle_life": 6000,
                "applications": ["peak_shaving", "renewable_smoothing", "backup"]
            },
            expected_results={
                "optimal_power_mw": 15.0,
                "optimal_energy_mwh": 60.0,
                "optimal_location": "bus_7",
                "payback_years": 8.5,
                "npv": 125000.0
            },
            tags=["storage", "sizing", "optimization"],
            difficulty="hard",
            estimated_runtime_seconds=180.0
        )
    
    def _create_voltage_regulation(self, network_name: str) -> AnalysisCase:
        """Create voltage regulation case."""
        return AnalysisCase(
            case_id=f"vreg_{network_name}",
            name=f"Voltage Regulation - {network_name}",
            description="Voltage regulation and reactive power control",
            network_name=network_name,
            analysis_type="voltage_control",
            parameters={
                "control_devices": ["tap_changer", "capacitor", "reactor"],
                "voltage_targets": [1.0, 1.0, 1.0],
                "load_scenarios": ["light", "nominal", "heavy"],
                "reactive_limits": True,
                "coordination": True,
                "automatic_control": True
            },
            expected_results={
                "voltage_deviation": 0.02,
                "reactive_support": 25.0,
                "tap_operations": 15,
                "capacitor_switching": 8,
                "compliance_achieved": True
            },
            tags=["voltage_regulation", "reactive_power", "control"],
            difficulty="medium",
            estimated_runtime_seconds=35.0
        )
    
    def _create_protection_coordination(self, network_name: str) -> AnalysisCase:
        """Create protection coordination case."""
        return AnalysisCase(
            case_id=f"prot_{network_name}",
            name=f"Protection Coordination - {network_name}",
            description="Protective relay coordination and arc flash analysis",
            network_name=network_name,
            analysis_type="protection",
            parameters={
                "device_types": ["overcurrent", "differential", "distance"],
                "coordination_curves": True,
                "arc_flash_analysis": True,
                "working_distance": 0.61,
                "incident_energy_limit": 40.0,
                "ppe_categories": True
            },
            expected_results={
                "coordination_achieved": True,
                "miscoordinated_pairs": 2,
                "max_incident_energy": 35.0,
                "ppe_category": 4,
                "settings_updated": 8
            },
            tags=["protection", "coordination", "arc_flash"],
            difficulty="hard",
            estimated_runtime_seconds=150.0
        )
    
    def _create_dg_interconnection(self, network_name: str) -> AnalysisCase:
        """Create DG interconnection case."""
        return AnalysisCase(
            case_id=f"dg_intercon_{network_name}",
            name=f"DG Interconnection Study - {network_name}",
            description="Distributed generation interconnection impact study",
            network_name=network_name,
            analysis_type="interconnection",
            parameters={
                "dg_capacity": 5.0,
                "dg_location": "bus_8",
                "dg_technology": "solar_pv",
                "study_type": "impact_study",
                "ieee1547_compliance": True,
                "grid_code_requirements": True,
                "flicker_analysis": True
            },
            expected_results={
                "interconnection_approved": True,
                "voltage_impact": 0.03,
                "flicker_severity": 0.8,
                "protection_changes": True,
                "upgrade_required": False
            },
            tags=["dg", "interconnection", "ieee1547"],
            difficulty="medium",
            estimated_runtime_seconds=70.0
        )
    
    def _create_motor_starting(self, network_name: str) -> AnalysisCase:
        """Create motor starting case."""
        return AnalysisCase(
            case_id=f"motor_{network_name}",
            name=f"Motor Starting Study - {network_name}",
            description="Large motor starting transient analysis",
            network_name=network_name,
            analysis_type="transient",
            parameters={
                "motor_power": 3.5,
                "motor_location": "bus_motor_1",
                "starting_method": "DOL",
                "motor_model": "detailed",
                "load_flow_integration": True,
                "voltage_drop_limit": 0.15,
                "flicker_analysis": True
            },
            expected_results={
                "voltage_drop": 0.12,
                "starting_time": 4.5,
                "inrush_current": 6.8,
                "flicker_level": 0.6,
                "acceptable_starting": True
            },
            tags=["motor_starting", "transient", "industrial"],
            difficulty="medium",
            estimated_runtime_seconds=25.0
        )
    
    def _create_power_quality(self, network_name: str) -> AnalysisCase:
        """Create power quality case."""
        return AnalysisCase(
            case_id=f"pq_{network_name}",
            name=f"Power Quality Assessment - {network_name}",
            description="Comprehensive power quality analysis",
            network_name=network_name,
            analysis_type="power_quality",
            parameters={
                "phenomena": ["harmonics", "flicker", "unbalance", "sag_swell"],
                "measurement_period": 168,
                "standards": ["ieee_519", "iec_61000"],
                "load_characterization": True,
                "mitigation_analysis": True,
                "cost_assessment": True
            },
            expected_results={
                "thd_compliance": True,
                "flicker_violations": 2,
                "unbalance_level": 1.8,
                "sag_frequency": 0.5,
                "mitigation_cost": 75000.0
            },
            tags=["power_quality", "harmonics", "flicker"],
            difficulty="hard",
            estimated_runtime_seconds=120.0
        )
    
    def _create_emergency_backup(self, network_name: str) -> AnalysisCase:
        """Create emergency backup case."""
        return AnalysisCase(
            case_id=f"backup_{network_name}",
            name=f"Emergency Backup System - {network_name}",
            description="Emergency generator backup system analysis",
            network_name=network_name,
            analysis_type="emergency_backup",
            parameters={
                "backup_capacity": 2.0,
                "load_priority": ["critical", "essential", "non_essential"],
                "transfer_time": 10.0,
                "fuel_duration": 8.0,
                "automatic_transfer": True,
                "load_shedding": True
            },
            expected_results={
                "backup_adequacy": True,
                "critical_load_supported": 1.8,
                "fuel_consumption": 150.0,
                "transfer_successful": True,
                "runtime_hours": 6.5
            },
            tags=["backup", "emergency", "generator"],
            difficulty="medium",
            estimated_runtime_seconds=30.0
        )
    
    def _create_economic_dispatch(self, network_name: str) -> AnalysisCase:
        """Create economic dispatch case."""
        return AnalysisCase(
            case_id=f"econ_disp_{network_name}",
            name=f"Economic Dispatch - {network_name}",
            description="Economic dispatch optimization",
            network_name=network_name,
            analysis_type="optimization",
            parameters={
                "time_horizon": 24,
                "generation_costs": {"gen_1": 45.0, "gen_2": 52.0, "gen_3": 38.0},
                "emission_factors": {"gen_1": 0.8, "gen_2": 0.6, "gen_3": 1.2},
                "transmission_losses": True,
                "reserve_requirements": 0.15,
                "ramp_limits": True
            },
            expected_results={
                "total_cost": 25400.0,
                "emission_tons": 145.0,
                "generator_dispatch": {"gen_1": 85.0, "gen_2": 120.0, "gen_3": 60.0},
                "losses_mw": 12.5,
                "reserve_provided": 40.0
            },
            tags=["economic_dispatch", "optimization", "cost"],
            difficulty="medium",
            estimated_runtime_seconds=20.0
        )
    
    def _create_optimal_power_flow(self, network_name: str) -> AnalysisCase:
        """Create optimal power flow case."""
        return AnalysisCase(
            case_id=f"opf_{network_name}",
            name=f"Optimal Power Flow - {network_name}",
            description="AC optimal power flow with security constraints",
            network_name=network_name,
            analysis_type="optimization",
            parameters={
                "objective": "minimize_cost",
                "ac_formulation": True,
                "security_constraints": True,
                "voltage_constraints": [0.95, 1.05],
                "thermal_constraints": True,
                "reactive_limits": True,
                "contingency_constraints": ["n_minus_1"]
            },
            expected_results={
                "optimal_cost": 28500.0,
                "convergence": True,
                "voltage_margins": 0.03,
                "loading_margins": 15.0,
                "reactive_reserves": 25.0
            },
            tags=["opf", "optimization", "security"],
            difficulty="hard",
            estimated_runtime_seconds=45.0
        )
    
    def _create_unit_commitment(self, network_name: str) -> AnalysisCase:
        """Create unit commitment case."""
        return AnalysisCase(
            case_id=f"uc_{network_name}",
            name=f"Unit Commitment - {network_name}",
            description="Day-ahead unit commitment optimization",
            network_name=network_name,
            analysis_type="optimization",
            parameters={
                "time_horizon": 24,
                "load_forecast": "day_ahead",
                "startup_costs": {"gen_1": 5000, "gen_2": 8000, "gen_3": 3000},
                "min_up_time": {"gen_1": 4, "gen_2": 6, "gen_3": 2},
                "min_down_time": {"gen_1": 4, "gen_2": 6, "gen_3": 2},
                "reserve_requirements": 0.15
            },
            expected_results={
                "total_cost": 125000.0,
                "units_committed": 18,
                "startup_cost": 15000.0,
                "reserve_shortfall": 0.0,
                "schedule_feasible": True
            },
            tags=["unit_commitment", "optimization", "scheduling"],
            difficulty="hard",
            estimated_runtime_seconds=180.0
        )
    
    def _create_pmu_placement(self, network_name: str) -> AnalysisCase:
        """Create PMU placement case."""
        return AnalysisCase(
            case_id=f"pmu_{network_name}",
            name=f"PMU Placement Optimization - {network_name}",
            description="Optimal phasor measurement unit placement",
            network_name=network_name,
            analysis_type="optimization",
            parameters={
                "observability_requirement": "full",
                "redundancy_level": 1,
                "communication_constraints": True,
                "budget_limit": 500000.0,
                "pmu_cost": 25000.0,
                "reliability_weighting": True
            },
            expected_results={
                "optimal_locations": [1, 5, 9, 12],
                "total_cost": 100000.0,
                "observability_index": 1.0,
                "redundancy_achieved": 0.8,
                "communication_feasible": True
            },
            tags=["pmu", "placement", "observability"],
            difficulty="hard",
            estimated_runtime_seconds=60.0
        )
    
    def _create_state_estimation(self, network_name: str) -> AnalysisCase:
        """Create state estimation case."""
        return AnalysisCase(
            case_id=f"se_{network_name}",
            name=f"State Estimation - {network_name}",
            description="Weighted least squares state estimation",
            network_name=network_name,
            analysis_type="state_estimation",
            parameters={
                "measurement_types": ["power_flow", "power_injection", "voltage_magnitude"],
                "measurement_redundancy": 2.5,
                "bad_data_detection": True,
                "topology_processor": True,
                "pmu_measurements": False,
                "estimation_method": "wls"
            },
            expected_results={
                "convergence": True,
                "chi_square_test": True,
                "bad_data_detected": 2,
                "estimation_accuracy": 0.98,
                "processing_time": 1.5
            },
            tags=["state_estimation", "wls", "scada"],
            difficulty="hard",
            estimated_runtime_seconds=40.0
        )
    
    def _create_market_clearing(self, network_name: str) -> AnalysisCase:
        """Create market clearing case."""
        return AnalysisCase(
            case_id=f"market_{network_name}",
            name=f"Market Clearing - {network_name}",
            description="Day-ahead market clearing and LMP calculation",
            network_name=network_name,
            analysis_type="market",
            parameters={
                "market_type": "day_ahead",
                "bid_format": "linear",
                "demand_elasticity": 0.1,
                "transmission_constraints": True,
                "market_power_mitigation": True,
                "price_cap": 1000.0
            },
            expected_results={
                "market_clearing_price": 65.0,
                "total_surplus": 85000.0,
                "congestion_cost": 2500.0,
                "loss_cost": 1200.0,
                "market_efficiency": 0.95
            },
            tags=["market", "lmp", "clearing"],
            difficulty="hard",
            estimated_runtime_seconds=90.0
        )
    
    # Study Plan Creation Methods
    def _create_transmission_planning_study(self) -> StudyPlan:
        """Create transmission planning study."""
        cases = [
            self.cases["lf_ieee30_basic"],
            self.cases["cont_n1_lines"],
            self.cases["cont_n2_critical"],
            self.cases["stab_voltage"],
            self.cases["opt_opf_basic"]
        ]
        
        return StudyPlan(
            plan_id="transmission_planning",
            name="Transmission System Planning Study",
            description="Comprehensive transmission planning analysis including reliability and economics",
            cases=cases,
            objective="Assess transmission system adequacy and identify reinforcement needs",
            deliverables=[
                "Load flow analysis report",
                "Contingency analysis results",
                "Voltage stability assessment",
                "Economic optimization study",
                "Transmission expansion plan"
            ],
            estimated_duration_hours=8.0
        )
    
    def _create_distribution_modernization_study(self) -> StudyPlan:
        """Create distribution modernization study."""
        cases = [
            self.cases["lf_ieee14_basic"],
            self.cases["ren_hosting_capacity"],
            self.cases["dist_voltage_regulation"],
            self.cases["dist_protection_coordination"],
            self.cases["harm_renewable"]
        ]
        
        return StudyPlan(
            plan_id="distribution_modernization",
            name="Distribution System Modernization Study",
            description="Analysis for smart grid and renewable integration in distribution systems",
            cases=cases,
            objective="Evaluate distribution system modernization requirements and benefits",
            deliverables=[
                "Hosting capacity analysis",
                "Voltage regulation study",
                "Protection coordination review",
                "Power quality assessment",
                "Smart grid integration plan"
            ],
            estimated_duration_hours=12.0
        )
    
    def _create_microgrid_design_study(self) -> StudyPlan:
        """Create microgrid design study."""
        cases = [
            self.cases["mg_islanding"],
            self.cases["mg_resync"],
            self.cases["mg_energy_management"],
            self.cases["ren_storage_sizing"],
            self.cases["dist_protection_coordination"]
        ]
        
        return StudyPlan(
            plan_id="microgrid_design",
            name="Microgrid Design and Analysis Study",
            description="Complete microgrid design including controls, protection, and economics",
            cases=cases,
            objective="Design optimal microgrid configuration with reliable islanding capability",
            deliverables=[
                "Islanding analysis report",
                "Resynchronization procedures",
                "Energy management strategy",
                "Storage system design",
                "Protection system design"
            ],
            estimated_duration_hours=16.0
        )
    
    def _create_renewable_integration_study(self) -> StudyPlan:
        """Create renewable integration study."""
        cases = [
            self.cases["ren_variability"],
            self.cases["ren_hosting_capacity"],
            self.cases["ren_storage_sizing"],
            self.cases["stab_voltage"],
            self.cases["harm_renewable"]
        ]
        
        return StudyPlan(
            plan_id="renewable_integration",
            name="Renewable Energy Integration Study",
            description="Comprehensive analysis of renewable energy integration impacts",
            cases=cases,
            objective="Assess grid impacts and mitigation measures for renewable integration",
            deliverables=[
                "Variability impact assessment",
                "Hosting capacity determination",
                "Storage requirements analysis",
                "Stability impact study",
                "Power quality analysis"
            ],
            estimated_duration_hours=14.0
        )
    
    def _create_industrial_power_study(self) -> StudyPlan:
        """Create industrial power study."""
        cases = [
            self.cases["ind_motor_starting"],
            self.cases["ind_power_quality"],
            self.cases["ind_emergency_backup"],
            self.cases["sc_3phase_fault"],
            self.cases["dist_protection_coordination"]
        ]
        
        return StudyPlan(
            plan_id="industrial_power_study",
            name="Industrial Power System Study",
            description="Complete electrical analysis for industrial facilities",
            cases=cases,
            objective="Ensure safe and reliable industrial power system operation",
            deliverables=[
                "Motor starting analysis",
                "Power quality assessment",
                "Emergency backup evaluation",
                "Short circuit study",
                "Protection coordination study"
            ],
            estimated_duration_hours=10.0
        )
    
    def _create_system_reliability_study(self) -> StudyPlan:
        """Create system reliability study."""
        cases = [
            self.cases["cont_n1_lines"],
            self.cases["cont_n1_generators"],
            self.cases["cont_n2_critical"],
            self.cases["stab_transient"],
            self.cases["stab_voltage"]
        ]
        
        return StudyPlan(
            plan_id="system_reliability",
            name="Power System Reliability Assessment",
            description="Comprehensive reliability analysis including contingency and stability",
            cases=cases,
            objective="Evaluate system reliability and identify critical vulnerabilities",
            deliverables=[
                "N-1 contingency analysis",
                "N-2 contingency screening",
                "Transient stability assessment",
                "Voltage stability analysis",
                "Reliability improvement recommendations"
            ],
            estimated_duration_hours=12.0
        )
    
    def _create_power_quality_study(self) -> StudyPlan:
        """Create power quality study."""
        cases = [
            self.cases["harm_industrial"],
            self.cases["harm_renewable"],
            self.cases["ind_power_quality"],
            self.cases["ind_motor_starting"],
            self.cases["dist_voltage_regulation"]
        ]
        
        return StudyPlan(
            plan_id="power_quality_assessment",
            name="Power Quality Assessment Study",
            description="Comprehensive power quality analysis and mitigation strategies",
            cases=cases,
            objective="Assess power quality compliance and develop mitigation measures",
            deliverables=[
                "Harmonic analysis report",
                "Power quality compliance assessment",
                "Mitigation recommendations",
                "Filter design requirements",
                "Standards compliance verification"
            ],
            estimated_duration_hours=8.0
        )
    
    def _create_grid_modernization_study(self) -> StudyPlan:
        """Create grid modernization study."""
        cases = [
            self.cases["adv_pmu_placement"],
            self.cases["adv_state_estimation"],
            self.cases["adv_market_clearing"],
            self.cases["opt_opf_basic"],
            self.cases["ren_hosting_capacity"]
        ]
        
        return StudyPlan(
            plan_id="grid_modernization",
            name="Smart Grid Modernization Study",
            description="Advanced grid modernization with smart technologies and market integration",
            cases=cases,
            objective="Design smart grid infrastructure with advanced monitoring and control",
            deliverables=[
                "PMU placement optimization",
                "State estimation implementation",
                "Market integration analysis",
                "Advanced control strategies",
                "Technology roadmap"
            ],
            estimated_duration_hours=20.0
        )


# Singleton instance for easy access
sample_cases = SampleCases() 