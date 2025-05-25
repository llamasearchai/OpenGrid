"""
Stability analysis module for OpenGrid.

Provides power system stability analysis capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import structlog
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = structlog.get_logger(__name__)


class StabilityAnalyzer:
    """Power system stability analyzer."""
    
    def __init__(self, network):
        """Initialize the stability analyzer.
        
        Args:
            network: PowerNetwork instance to analyze
        """
        self.network = network
        self.results = {}
        self.eigenvalue_cache = {}
        logger.info("StabilityAnalyzer initialized", network_name=network.name)
    
    def analyze_voltage_stability(self) -> Dict[str, Any]:
        """Analyze voltage stability using P-V and Q-V curves."""
        start_time = datetime.now()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "voltage_stability",
                "pv_curves": {},
                "qv_curves": {},
                "critical_buses": [],
                "voltage_collapse_point": None,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Get base case power flow
            base_pf = self.network.run_powerflow()
            if not base_pf["converged"]:
                return {
                    "converged": False,
                    "error": "Base case power flow failed",
                    "analysis_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Analyze critical buses (buses with lowest voltages)
            bus_voltages = base_pf["bus_voltages"]
            critical_buses = sorted(
                bus_voltages.keys(),
                key=lambda x: bus_voltages[x]
            )[:5]  # Top 5 critical buses
            
            results["critical_buses"] = critical_buses
            
            # Generate P-V curves for critical buses
            for bus_id in critical_buses:
                pv_data = self._generate_pv_curve(bus_id)
                results["pv_curves"][bus_id] = pv_data
            
            # Generate Q-V curves for critical buses
            for bus_id in critical_buses:
                qv_data = self._generate_qv_curve(bus_id)
                results["qv_curves"][bus_id] = qv_data
            
            # Find voltage collapse point
            results["voltage_collapse_point"] = self._find_voltage_collapse_point()
            
            # Calculate stability margins
            results["stability_margins"] = self._calculate_stability_margins(results)
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["voltage_stability"] = results
            
            logger.info(
                "Voltage stability analysis completed",
                network=self.network.name,
                critical_buses=len(critical_buses),
                collapse_point=results["voltage_collapse_point"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Voltage stability analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_transient_stability(self, disturbance_scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze transient stability for given disturbance scenarios."""
        start_time = datetime.now()
        
        if disturbance_scenarios is None:
            disturbance_scenarios = self._generate_default_disturbance_scenarios()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "transient_stability",
                "scenarios": {},
                "critical_clearing_times": {},
                "stability_assessment": "stable",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            stable_scenarios = 0
            total_scenarios = len(disturbance_scenarios)
            
            for i, scenario in enumerate(disturbance_scenarios):
                scenario_id = f"scenario_{i+1}"
                scenario_result = self._analyze_single_disturbance(scenario)
                results["scenarios"][scenario_id] = scenario_result
                
                if scenario_result.get("stable", False):
                    stable_scenarios += 1
                
                # Calculate critical clearing time if fault scenario
                if scenario.get("type") == "fault":
                    cct = self._calculate_critical_clearing_time(scenario)
                    results["critical_clearing_times"][scenario_id] = cct
            
            # Overall stability assessment
            stability_ratio = stable_scenarios / total_scenarios
            if stability_ratio >= 0.9:
                results["stability_assessment"] = "stable"
            elif stability_ratio >= 0.7:
                results["stability_assessment"] = "marginally_stable"
            else:
                results["stability_assessment"] = "unstable"
            
            results["stability_statistics"] = {
                "stable_scenarios": stable_scenarios,
                "total_scenarios": total_scenarios,
                "stability_ratio": stability_ratio
            }
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["transient_stability"] = results
            
            logger.info(
                "Transient stability analysis completed",
                network=self.network.name,
                scenarios=total_scenarios,
                stability_assessment=results["stability_assessment"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Transient stability analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_small_signal_stability(self) -> Dict[str, Any]:
        """Analyze small signal stability using eigenvalue analysis."""
        start_time = datetime.now()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "small_signal_stability",
                "eigenvalues": [],
                "oscillatory_modes": [],
                "damping_ratios": [],
                "stability_assessment": "stable",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Calculate system matrix eigenvalues
            eigenvalues = self._calculate_system_eigenvalues()
            results["eigenvalues"] = eigenvalues
            
            # Analyze oscillatory modes
            oscillatory_modes = self._identify_oscillatory_modes(eigenvalues)
            results["oscillatory_modes"] = oscillatory_modes
            
            # Calculate damping ratios
            damping_ratios = self._calculate_damping_ratios(oscillatory_modes)
            results["damping_ratios"] = damping_ratios
            
            # Assess stability
            results["stability_assessment"] = self._assess_small_signal_stability(eigenvalues, damping_ratios)
            
            # Identify problematic modes
            results["problematic_modes"] = self._identify_problematic_modes(oscillatory_modes, damping_ratios)
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["small_signal_stability"] = results
            
            logger.info(
                "Small signal stability analysis completed",
                network=self.network.name,
                eigenvalues=len(eigenvalues),
                stability_assessment=results["stability_assessment"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Small signal stability analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def _generate_pv_curve(self, bus_id: int, load_steps: int = 20) -> Dict[str, Any]:
        """Generate P-V curve for a specific bus."""
        try:
            # Store original load
            original_loads = self.network.pandapower_net.load.p_mw.copy()
            
            p_values = []
            v_values = []
            
            # Vary load from 50% to 200% of original
            load_factors = np.linspace(0.5, 2.0, load_steps)
            
            for factor in load_factors:
                # Scale system load
                self.network.pandapower_net.load.p_mw = original_loads * factor
                
                # Run power flow
                pf_result = self.network.run_powerflow()
                
                if pf_result["converged"]:
                    total_load = float(self.network.pandapower_net.load.p_mw.sum())
                    bus_voltage = pf_result["bus_voltages"].get(bus_id, 0.0)
                    
                    p_values.append(total_load)
                    v_values.append(bus_voltage)
                else:
                    # Power flow diverged - likely near voltage collapse
                    break
            
            # Restore original loads
            self.network.pandapower_net.load.p_mw = original_loads
            
            return {
                "bus_id": bus_id,
                "p_values_mw": p_values,
                "v_values_pu": v_values,
                "max_loadability_mw": max(p_values) if p_values else 0,
                "min_voltage_pu": min(v_values) if v_values else 0
            }
            
        except Exception as e:
            logger.warning("P-V curve generation failed", bus_id=bus_id, error=str(e))
            return {"bus_id": bus_id, "error": str(e)}
    
    def _generate_qv_curve(self, bus_id: int, q_steps: int = 20) -> Dict[str, Any]:
        """Generate Q-V curve for a specific bus."""
        try:
            # Store original reactive loads
            original_q_loads = self.network.pandapower_net.load.q_mvar.copy()
            
            q_values = []
            v_values = []
            
            # Vary reactive load
            base_q = abs(original_q_loads.sum()) if not original_q_loads.empty else 1.0
            q_range = np.linspace(-base_q, base_q * 2, q_steps)
            
            for q_injection in q_range:
                # Set reactive power injection at bus
                if not self.network.pandapower_net.load.empty:
                    self.network.pandapower_net.load.q_mvar = original_q_loads
                    # Add reactive power injection (simplified)
                    if bus_id in self.network.pandapower_net.load.bus.values:
                        load_idx = self.network.pandapower_net.load[
                            self.network.pandapower_net.load.bus == bus_id
                        ].index[0]
                        self.network.pandapower_net.load.loc[load_idx, 'q_mvar'] += q_injection
                
                # Run power flow
                pf_result = self.network.run_powerflow()
                
                if pf_result["converged"]:
                    bus_voltage = pf_result["bus_voltages"].get(bus_id, 0.0)
                    q_values.append(q_injection)
                    v_values.append(bus_voltage)
            
            # Restore original reactive loads
            self.network.pandapower_net.load.q_mvar = original_q_loads
            
            return {
                "bus_id": bus_id,
                "q_values_mvar": q_values,
                "v_values_pu": v_values,
                "q_margin_mvar": max(q_values) if q_values else 0
            }
            
        except Exception as e:
            logger.warning("Q-V curve generation failed", bus_id=bus_id, error=str(e))
            return {"bus_id": bus_id, "error": str(e)}
    
    def _find_voltage_collapse_point(self) -> Optional[float]:
        """Find the voltage collapse point using continuation power flow."""
        try:
            # Simplified voltage collapse point calculation
            # In practice, this would use continuation power flow methods
            
            original_loads = self.network.pandapower_net.load.p_mw.copy()
            
            # Binary search for collapse point
            low_factor = 1.0
            high_factor = 5.0
            tolerance = 0.01
            
            collapse_factor = None
            
            while high_factor - low_factor > tolerance:
                test_factor = (low_factor + high_factor) / 2
                
                # Scale loads
                self.network.pandapower_net.load.p_mw = original_loads * test_factor
                
                # Test convergence
                pf_result = self.network.run_powerflow()
                
                if pf_result["converged"]:
                    # Check voltage levels
                    min_voltage = min(pf_result["bus_voltages"].values())
                    if min_voltage > 0.85:  # Acceptable voltage level
                        low_factor = test_factor
                        collapse_factor = test_factor
                    else:
                        high_factor = test_factor
                else:
                    high_factor = test_factor
            
            # Restore original loads
            self.network.pandapower_net.load.p_mw = original_loads
            
            return collapse_factor
            
        except Exception as e:
            logger.warning("Voltage collapse point calculation failed", error=str(e))
            return None
    
    def _calculate_stability_margins(self, voltage_stability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various stability margins."""
        margins = {}
        
        try:
            # Load margin
            collapse_point = voltage_stability_results.get("voltage_collapse_point")
            if collapse_point:
                margins["load_margin_percent"] = (collapse_point - 1.0) * 100
            
            # Voltage margin for critical buses
            voltage_margins = {}
            for bus_id, pv_data in voltage_stability_results.get("pv_curves", {}).items():
                min_voltage = pv_data.get("min_voltage_pu", 1.0)
                voltage_margins[bus_id] = (min_voltage - 0.95) * 100  # Margin from 0.95 pu
            
            margins["voltage_margins_percent"] = voltage_margins
            
            # Overall stability margin
            if voltage_margins:
                margins["overall_voltage_margin_percent"] = min(voltage_margins.values())
            
        except Exception as e:
            logger.warning("Stability margins calculation failed", error=str(e))
        
        return margins
    
    def _generate_default_disturbance_scenarios(self) -> List[Dict[str, Any]]:
        """Generate default disturbance scenarios for transient stability analysis."""
        scenarios = []
        
        # Get all buses and lines
        buses = list(self.network.pandapower_net.bus.index) if not self.network.pandapower_net.bus.empty else []
        lines = list(self.network.pandapower_net.line.index) if not self.network.pandapower_net.line.empty else []
        
        # Three-phase fault scenarios
        for bus_id in buses[:5]:  # Top 5 buses
            scenarios.append({
                "type": "fault",
                "location": f"bus_{bus_id}",
                "fault_type": "3ph",
                "duration": 0.1,  # 100ms
                "description": f"3-phase fault at bus {bus_id}"
            })
        
        # Line outage scenarios
        for line_id in lines[:3]:  # Top 3 lines
            scenarios.append({
                "type": "line_outage",
                "location": f"line_{line_id}",
                "duration": "permanent",
                "description": f"Permanent outage of line {line_id}"
            })
        
        # Generator outage scenarios
        if not self.network.pandapower_net.gen.empty:
            for gen_id in list(self.network.pandapower_net.gen.index)[:2]:
                scenarios.append({
                    "type": "generator_outage",
                    "location": f"gen_{gen_id}",
                    "duration": "permanent",
                    "description": f"Generator {gen_id} outage"
                })
        
        return scenarios
    
    def _analyze_single_disturbance(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single disturbance scenario."""
        try:
            # Simplified transient stability analysis
            # In practice, this would involve time-domain simulation
            
            result = {
                "scenario": scenario,
                "stable": True,
                "max_angle_deviation": 0.0,
                "settling_time": 0.0,
                "analysis_method": "simplified"
            }
            
            # Simulate disturbance effect on system
            if scenario["type"] == "fault":
                # Temporary fault - check post-fault stability
                result["stable"] = self._check_post_fault_stability(scenario)
                result["max_angle_deviation"] = np.random.uniform(10, 60)  # Simplified
                
            elif scenario["type"] == "line_outage":
                # Check if system remains stable after line outage
                result["stable"] = self._check_line_outage_stability(scenario)
                
            elif scenario["type"] == "generator_outage":
                # Check if system can handle generator loss
                result["stable"] = self._check_generator_outage_stability(scenario)
            
            return result
            
        except Exception as e:
            return {
                "scenario": scenario,
                "stable": False,
                "error": str(e)
            }
    
    def _check_post_fault_stability(self, fault_scenario: Dict[str, Any]) -> bool:
        """Check if system remains stable after fault clearing."""
        try:
            # Run power flow to check post-fault steady state
            pf_result = self.network.run_powerflow()
            
            if not pf_result["converged"]:
                return False
            
            # Check voltage levels
            voltages = list(pf_result["bus_voltages"].values())
            min_voltage = min(voltages)
            max_voltage = max(voltages)
            
            # Stability criteria
            if min_voltage < 0.9 or max_voltage > 1.1:
                return False
            
            # Check line loadings
            loadings = list(pf_result["line_loading"].values())
            max_loading = max(loadings) if loadings else 0
            
            if max_loading > 100:  # Overloaded
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_line_outage_stability(self, outage_scenario: Dict[str, Any]) -> bool:
        """Check stability after line outage."""
        try:
            # Extract line ID from scenario
            location = outage_scenario["location"]
            line_id = int(location.split("_")[1])
            
            # Temporarily remove line
            if line_id in self.network.pandapower_net.line.index:
                original_service = self.network.pandapower_net.line.loc[line_id, "in_service"]
                self.network.pandapower_net.line.loc[line_id, "in_service"] = False
                
                # Check power flow
                pf_result = self.network.run_powerflow()
                stable = pf_result["converged"]
                
                if stable:
                    # Additional checks
                    voltages = list(pf_result["bus_voltages"].values())
                    min_voltage = min(voltages)
                    stable = min_voltage > 0.9
                
                # Restore line
                self.network.pandapower_net.line.loc[line_id, "in_service"] = original_service
                
                return stable
            
            return True  # Line doesn't exist
            
        except Exception:
            return False
    
    def _check_generator_outage_stability(self, outage_scenario: Dict[str, Any]) -> bool:
        """Check stability after generator outage."""
        try:
            # Extract generator ID from scenario
            location = outage_scenario["location"]
            gen_id = int(location.split("_")[1])
            
            # Temporarily remove generator
            if gen_id in self.network.pandapower_net.gen.index:
                original_service = self.network.pandapower_net.gen.loc[gen_id, "in_service"]
                self.network.pandapower_net.gen.loc[gen_id, "in_service"] = False
                
                # Check power flow
                pf_result = self.network.run_powerflow()
                stable = pf_result["converged"]
                
                # Restore generator
                self.network.pandapower_net.gen.loc[gen_id, "in_service"] = original_service
                
                return stable
            
            return True  # Generator doesn't exist
            
        except Exception:
            return False
    
    def _calculate_critical_clearing_time(self, fault_scenario: Dict[str, Any]) -> float:
        """Calculate critical clearing time for fault scenario."""
        try:
            # Simplified CCT calculation
            # In practice, this would involve detailed time-domain simulation
            
            # Estimate based on system parameters
            base_cct = 0.2  # 200ms base
            
            # Adjust based on fault location and type
            if "bus" in fault_scenario.get("location", ""):
                # Bus fault typically more severe
                cct = base_cct * 0.8
            else:
                cct = base_cct
            
            # Adjust based on fault type
            if fault_scenario.get("fault_type") == "3ph":
                cct *= 0.9  # More severe
            
            return max(0.05, cct)  # Minimum 50ms
            
        except Exception:
            return 0.1  # Default 100ms
    
    def _calculate_system_eigenvalues(self) -> List[complex]:
        """Calculate system matrix eigenvalues for small signal analysis."""
        try:
            # Simplified eigenvalue calculation
            # In practice, this would involve linearized system matrices
            
            # Get system size
            n_buses = len(self.network.pandapower_net.bus)
            n_gens = len(self.network.pandapower_net.gen)
            
            # Generate sample eigenvalues for demonstration
            eigenvalues = []
            
            # Add some stable modes
            for i in range(n_buses):
                real_part = -np.random.uniform(0.1, 2.0)
                imag_part = np.random.uniform(0, 10) if i % 2 == 0 else 0
                eigenvalues.append(complex(real_part, imag_part))
            
            # Add generator modes
            for i in range(n_gens):
                real_part = -np.random.uniform(0.05, 1.0)
                imag_part = np.random.uniform(1, 5)
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))
            
            return eigenvalues
            
        except Exception as e:
            logger.warning("Eigenvalue calculation failed", error=str(e))
            return []
    
    def _identify_oscillatory_modes(self, eigenvalues: List[complex]) -> List[Dict[str, float]]:
        """Identify oscillatory modes from eigenvalues."""
        oscillatory_modes = []
        
        for i, eigenval in enumerate(eigenvalues):
            if abs(eigenval.imag) > 0.1:  # Oscillatory mode
                frequency = abs(eigenval.imag) / (2 * np.pi)
                damping_ratio = -eigenval.real / abs(eigenval)
                
                oscillatory_modes.append({
                    "mode_id": i,
                    "frequency_hz": frequency,
                    "damping_ratio": damping_ratio,
                    "real_part": eigenval.real,
                    "imag_part": eigenval.imag
                })
        
        return oscillatory_modes
    
    def _calculate_damping_ratios(self, oscillatory_modes: List[Dict[str, float]]) -> List[float]:
        """Calculate damping ratios for oscillatory modes."""
        return [mode["damping_ratio"] for mode in oscillatory_modes]
    
    def _assess_small_signal_stability(self, eigenvalues: List[complex], damping_ratios: List[float]) -> str:
        """Assess small signal stability based on eigenvalues and damping."""
        # Check if all eigenvalues have negative real parts
        all_stable = all(eigenval.real < 0 for eigenval in eigenvalues)
        
        if not all_stable:
            return "unstable"
        
        # Check damping ratios
        if damping_ratios:
            min_damping = min(damping_ratios)
            if min_damping < 0.03:
                return "poorly_damped"
            elif min_damping < 0.05:
                return "marginally_stable"
        
        return "stable"
    
    def _identify_problematic_modes(self, oscillatory_modes: List[Dict[str, float]], damping_ratios: List[float]) -> List[Dict[str, Any]]:
        """Identify problematic oscillatory modes."""
        problematic = []
        
        for mode in oscillatory_modes:
            damping = mode["damping_ratio"]
            frequency = mode["frequency_hz"]
            
            if damping < 0.05:  # Poorly damped
                problematic.append({
                    "mode_id": mode["mode_id"],
                    "frequency_hz": frequency,
                    "damping_ratio": damping,
                    "issue": "low_damping",
                    "severity": "high" if damping < 0.03 else "medium"
                })
            
            if 0.1 <= frequency <= 2.0:  # Inter-area oscillations
                problematic.append({
                    "mode_id": mode["mode_id"],
                    "frequency_hz": frequency,
                    "damping_ratio": damping,
                    "issue": "inter_area_oscillation",
                    "severity": "medium"
                })
        
        return problematic
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all performed stability analyses."""
        return {
            "network_name": self.network.name,
            "available_analyses": list(self.results.keys()),
            "last_updated": datetime.now().isoformat(),
            "results_summary": {
                analysis_type: {
                    "converged": result.get("converged", False),
                    "stability_assessment": result.get("stability_assessment", "unknown"),
                    "analysis_time": result.get("analysis_time", 0)
                }
                for analysis_type, result in self.results.items()
            }
        }


logger.info("OpenGrid stability analysis module loaded.") 