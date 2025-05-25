"""
Harmonic analysis module for OpenGrid.

Provides harmonic distortion analysis for power systems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import structlog
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = structlog.get_logger(__name__)


class HarmonicAnalyzer:
    """Harmonic distortion analyzer for power systems."""
    
    def __init__(self, network):
        """Initialize the harmonic analyzer.
        
        Args:
            network: PowerNetwork instance to analyze
        """
        self.network = network
        self.results = {}
        self.harmonic_sources = []
        logger.info("HarmonicAnalyzer initialized", network_name=network.name)
    
    def analyze_thd(self, harmonic_orders: List[int] = None) -> Dict[str, Any]:
        """Analyze Total Harmonic Distortion (THD) at all buses."""
        start_time = datetime.now()
        
        if harmonic_orders is None:
            harmonic_orders = list(range(2, 51))  # 2nd to 50th harmonics
        
        try:
            results = {
                "converged": True,
                "analysis_type": "thd_analysis",
                "harmonic_orders": harmonic_orders,
                "bus_thd_voltage": {},
                "bus_thd_current": {},
                "thd_summary": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Get bus information
            buses = self.network.pandapower_net.bus.index.tolist() if not self.network.pandapower_net.bus.empty else []
            
            for bus_id in buses:
                # Calculate voltage THD
                voltage_thd = self._calculate_voltage_thd(bus_id, harmonic_orders)
                results["bus_thd_voltage"][bus_id] = voltage_thd
                
                # Calculate current THD
                current_thd = self._calculate_current_thd(bus_id, harmonic_orders)
                results["bus_thd_current"][bus_id] = current_thd
            
            # Calculate summary statistics
            voltage_thds = list(results["bus_thd_voltage"].values())
            current_thds = list(results["bus_thd_current"].values())
            
            results["thd_summary"] = {
                "max_voltage_thd_percent": max(voltage_thds) if voltage_thds else 0,
                "min_voltage_thd_percent": min(voltage_thds) if voltage_thds else 0,
                "avg_voltage_thd_percent": np.mean(voltage_thds) if voltage_thds else 0,
                "max_current_thd_percent": max(current_thds) if current_thds else 0,
                "min_current_thd_percent": min(current_thds) if current_thds else 0,
                "avg_current_thd_percent": np.mean(current_thds) if current_thds else 0,
                "buses_exceeding_voltage_limit": [
                    bus_id for bus_id, thd in results["bus_thd_voltage"].items()
                    if thd > 5.0  # IEEE 519 limit for voltage THD
                ],
                "buses_exceeding_current_limit": [
                    bus_id for bus_id, thd in results["bus_thd_current"].items()
                    if thd > 20.0  # Typical current THD limit
                ]
            }
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["thd_analysis"] = results
            
            logger.info(
                "THD analysis completed",
                network=self.network.name,
                buses_analyzed=len(buses),
                max_voltage_thd=results["thd_summary"]["max_voltage_thd_percent"],
                max_current_thd=results["thd_summary"]["max_current_thd_percent"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("THD analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_individual_harmonics(self, harmonic_orders: List[int] = None) -> Dict[str, Any]:
        """Analyze individual harmonic components."""
        start_time = datetime.now()
        
        if harmonic_orders is None:
            harmonic_orders = [3, 5, 7, 9, 11, 13]  # Common harmonics
        
        try:
            results = {
                "converged": True,
                "analysis_type": "individual_harmonics",
                "harmonic_orders": harmonic_orders,
                "bus_harmonics": {},
                "dominant_harmonics": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            buses = self.network.pandapower_net.bus.index.tolist() if not self.network.pandapower_net.bus.empty else []
            
            for bus_id in buses:
                bus_harmonics = {}
                
                for order in harmonic_orders:
                    harmonic_data = self._calculate_individual_harmonic(bus_id, order)
                    bus_harmonics[f"h{order}"] = harmonic_data
                
                results["bus_harmonics"][bus_id] = bus_harmonics
                
                # Find dominant harmonics at this bus
                dominant = self._find_dominant_harmonics(bus_harmonics)
                results["dominant_harmonics"][bus_id] = dominant
            
            # Overall system analysis
            results["system_harmonics"] = self._analyze_system_harmonics(results["bus_harmonics"])
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["individual_harmonics"] = results
            
            logger.info(
                "Individual harmonics analysis completed",
                network=self.network.name,
                harmonic_orders=len(harmonic_orders),
                buses_analyzed=len(buses)
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Individual harmonics analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_harmonic_resonance(self) -> Dict[str, Any]:
        """Analyze harmonic resonance conditions."""
        start_time = datetime.now()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "harmonic_resonance",
                "resonance_frequencies": [],
                "resonance_points": {},
                "critical_frequencies": [],
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Calculate frequency response
            frequency_range = np.linspace(50, 2500, 100)  # 50 Hz to 2.5 kHz
            
            for freq in frequency_range:
                impedance_data = self._calculate_system_impedance(freq)
                
                # Check for resonance (high impedance or phase changes)
                if self._is_resonance_frequency(freq, impedance_data):
                    results["resonance_frequencies"].append(freq)
                    results["resonance_points"][freq] = impedance_data
            
            # Identify critical frequencies (near common harmonic frequencies)
            critical_freqs = [50 * h for h in [3, 5, 7, 9, 11, 13]]  # 150, 250, 350, etc.
            
            for freq in critical_freqs:
                impedance_data = self._calculate_system_impedance(freq)
                if impedance_data["max_impedance"] > 1000:  # High impedance threshold
                    results["critical_frequencies"].append({
                        "frequency_hz": freq,
                        "harmonic_order": freq / 50,
                        "max_impedance_ohm": impedance_data["max_impedance"],
                        "resonance_risk": "high" if impedance_data["max_impedance"] > 5000 else "medium"
                    })
            
            # Resonance analysis summary
            results["resonance_summary"] = {
                "total_resonance_frequencies": len(results["resonance_frequencies"]),
                "critical_frequencies_count": len(results["critical_frequencies"]),
                "resonance_risk_assessment": self._assess_resonance_risk(results)
            }
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["harmonic_resonance"] = results
            
            logger.info(
                "Harmonic resonance analysis completed",
                network=self.network.name,
                resonance_frequencies=len(results["resonance_frequencies"]),
                critical_frequencies=len(results["critical_frequencies"])
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Harmonic resonance analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_filter_requirements(self, target_thd_limit: float = 5.0) -> Dict[str, Any]:
        """Analyze harmonic filter requirements."""
        start_time = datetime.now()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "filter_requirements",
                "target_thd_limit_percent": target_thd_limit,
                "filter_recommendations": {},
                "cost_analysis": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Get current THD analysis
            thd_results = self.analyze_thd()
            
            if not thd_results["converged"]:
                return {
                    "converged": False,
                    "error": "Could not analyze current THD levels",
                    "analysis_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Identify buses requiring filtering
            problem_buses = thd_results["thd_summary"]["buses_exceeding_voltage_limit"]
            
            for bus_id in problem_buses:
                current_thd = thd_results["bus_thd_voltage"][bus_id]
                filter_requirement = self._calculate_filter_requirement(bus_id, current_thd, target_thd_limit)
                results["filter_recommendations"][bus_id] = filter_requirement
            
            # Cost analysis
            results["cost_analysis"] = self._estimate_filter_costs(results["filter_recommendations"])
            
            # System-wide recommendations
            results["system_recommendations"] = self._generate_system_filter_recommendations(results)
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["filter_requirements"] = results
            
            logger.info(
                "Filter requirements analysis completed",
                network=self.network.name,
                buses_requiring_filters=len(problem_buses),
                estimated_cost=results["cost_analysis"].get("total_cost_usd", 0)
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Filter requirements analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def _calculate_voltage_thd(self, bus_id: int, harmonic_orders: List[int]) -> float:
        """Calculate voltage THD at a specific bus."""
        try:
            # Simplified THD calculation
            # In practice, this would require detailed harmonic analysis
            
            # Base voltage magnitude
            base_voltage = self._get_bus_base_voltage(bus_id)
            
            # Calculate harmonic voltages (simplified model)
            harmonic_voltages_squared = 0.0
            
            for order in harmonic_orders:
                # Simplified harmonic voltage calculation
                harmonic_voltage = base_voltage * self._get_harmonic_factor(order) * 0.01
                harmonic_voltages_squared += harmonic_voltage ** 2
            
            # THD calculation
            thd_percent = (np.sqrt(harmonic_voltages_squared) / base_voltage) * 100
            
            return min(thd_percent, 50.0)  # Cap at 50% for realistic values
            
        except Exception as e:
            logger.warning("Voltage THD calculation failed", bus_id=bus_id, error=str(e))
            return 0.0
    
    def _calculate_current_thd(self, bus_id: int, harmonic_orders: List[int]) -> float:
        """Calculate current THD at a specific bus."""
        try:
            # Get fundamental current
            fundamental_current = self._get_bus_fundamental_current(bus_id)
            
            # Calculate harmonic currents
            harmonic_currents_squared = 0.0
            
            for order in harmonic_orders:
                # Simplified harmonic current calculation
                harmonic_current = fundamental_current * self._get_harmonic_factor(order) * 0.02
                harmonic_currents_squared += harmonic_current ** 2
            
            # THD calculation
            if fundamental_current > 0:
                thd_percent = (np.sqrt(harmonic_currents_squared) / fundamental_current) * 100
            else:
                thd_percent = 0.0
            
            return min(thd_percent, 100.0)  # Cap at 100%
            
        except Exception as e:
            logger.warning("Current THD calculation failed", bus_id=bus_id, error=str(e))
            return 0.0
    
    def _calculate_individual_harmonic(self, bus_id: int, harmonic_order: int) -> Dict[str, float]:
        """Calculate individual harmonic component."""
        try:
            base_voltage = self._get_bus_base_voltage(bus_id)
            fundamental_current = self._get_bus_fundamental_current(bus_id)
            
            # Simplified harmonic calculation
            harmonic_factor = self._get_harmonic_factor(harmonic_order)
            
            harmonic_voltage = base_voltage * harmonic_factor * 0.01
            harmonic_current = fundamental_current * harmonic_factor * 0.02
            
            return {
                "voltage_magnitude_v": harmonic_voltage,
                "current_magnitude_a": harmonic_current,
                "voltage_percent": (harmonic_voltage / base_voltage) * 100,
                "current_percent": (harmonic_current / fundamental_current) * 100 if fundamental_current > 0 else 0,
                "phase_angle_deg": np.random.uniform(-180, 180)  # Simplified
            }
            
        except Exception as e:
            logger.warning("Individual harmonic calculation failed", bus_id=bus_id, harmonic_order=harmonic_order, error=str(e))
            return {
                "voltage_magnitude_v": 0.0,
                "current_magnitude_a": 0.0,
                "voltage_percent": 0.0,
                "current_percent": 0.0,
                "phase_angle_deg": 0.0
            }
    
    def _get_harmonic_factor(self, harmonic_order: int) -> float:
        """Get harmonic amplitude factor based on order."""
        # Typical harmonic spectrum for power systems
        if harmonic_order == 3:
            return 2.0
        elif harmonic_order == 5:
            return 1.5
        elif harmonic_order == 7:
            return 1.0
        elif harmonic_order == 9:
            return 0.8
        elif harmonic_order == 11:
            return 0.6
        elif harmonic_order == 13:
            return 0.5
        else:
            # General decay for higher order harmonics
            return max(0.1, 1.0 / harmonic_order)
    
    def _get_bus_base_voltage(self, bus_id: int) -> float:
        """Get base voltage for a bus."""
        try:
            if bus_id in self.network.pandapower_net.bus.index:
                return float(self.network.pandapower_net.bus.loc[bus_id, "vn_kv"]) * 1000  # Convert to V
            else:
                return 400.0  # Default 400V
        except:
            return 400.0
    
    def _get_bus_fundamental_current(self, bus_id: int) -> float:
        """Get fundamental frequency current at bus."""
        try:
            # Simplified current calculation based on connected loads
            total_current = 0.0
            
            # Get loads connected to this bus
            bus_loads = self.network.pandapower_net.load[
                self.network.pandapower_net.load.bus == bus_id
            ] if not self.network.pandapower_net.load.empty else pd.DataFrame()
            
            if not bus_loads.empty:
                total_power = bus_loads.p_mw.sum() * 1000  # Convert to kW
                bus_voltage = self._get_bus_base_voltage(bus_id) / 1000  # Convert to kV
                
                if bus_voltage > 0:
                    total_current = total_power / (bus_voltage * np.sqrt(3))  # 3-phase current
            
            return max(total_current, 1.0)  # Minimum 1A
            
        except Exception:
            return 10.0  # Default 10A
    
    def _find_dominant_harmonics(self, bus_harmonics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Find dominant harmonic components."""
        dominant = []
        
        for harmonic_name, harmonic_data in bus_harmonics.items():
            voltage_percent = harmonic_data.get("voltage_percent", 0)
            current_percent = harmonic_data.get("current_percent", 0)
            
            # Consider harmonic dominant if voltage > 1% or current > 5%
            if voltage_percent > 1.0 or current_percent > 5.0:
                order = int(harmonic_name[1:])  # Remove 'h' prefix
                dominant.append({
                    "harmonic_order": order,
                    "voltage_percent": voltage_percent,
                    "current_percent": current_percent,
                    "dominance_score": voltage_percent + current_percent
                })
        
        # Sort by dominance score
        dominant.sort(key=lambda x: x["dominance_score"], reverse=True)
        
        return dominant[:5]  # Return top 5 dominant harmonics
    
    def _analyze_system_harmonics(self, bus_harmonics: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Analyze system-wide harmonic characteristics."""
        system_analysis = {}
        
        try:
            # Aggregate harmonic data across all buses
            harmonic_orders = set()
            for bus_data in bus_harmonics.values():
                harmonic_orders.update([int(h[1:]) for h in bus_data.keys()])
            
            for order in sorted(harmonic_orders):
                voltage_values = []
                current_values = []
                
                for bus_data in bus_harmonics.values():
                    harmonic_key = f"h{order}"
                    if harmonic_key in bus_data:
                        voltage_values.append(bus_data[harmonic_key]["voltage_percent"])
                        current_values.append(bus_data[harmonic_key]["current_percent"])
                
                if voltage_values and current_values:
                    system_analysis[f"h{order}"] = {
                        "max_voltage_percent": max(voltage_values),
                        "avg_voltage_percent": np.mean(voltage_values),
                        "max_current_percent": max(current_values),
                        "avg_current_percent": np.mean(current_values),
                        "buses_affected": len(voltage_values)
                    }
        
        except Exception as e:
            logger.warning("System harmonics analysis failed", error=str(e))
        
        return system_analysis
    
    def _calculate_system_impedance(self, frequency: float) -> Dict[str, float]:
        """Calculate system impedance at given frequency."""
        try:
            # Simplified impedance calculation
            # In practice, this would involve detailed frequency domain analysis
            
            fundamental_freq = 50.0  # Hz
            freq_ratio = frequency / fundamental_freq
            
            # Simplified impedance model
            # Resistance increases slightly with frequency
            base_resistance = 1.0  # Ohms
            resistance = base_resistance * (1 + 0.1 * np.sqrt(freq_ratio))
            
            # Reactance increases with frequency
            base_reactance = 0.5  # Ohms at fundamental
            reactance = base_reactance * freq_ratio
            
            # Total impedance
            impedance_magnitude = np.sqrt(resistance**2 + reactance**2)
            
            # Check for resonance conditions (simplified)
            # Resonance occurs when inductive and capacitive reactances are equal
            if abs(reactance - 2.0) < 0.1:  # Simplified resonance condition
                impedance_magnitude *= 10  # Amplification at resonance
            
            return {
                "frequency_hz": frequency,
                "resistance_ohm": resistance,
                "reactance_ohm": reactance,
                "impedance_magnitude_ohm": impedance_magnitude,
                "max_impedance": impedance_magnitude
            }
            
        except Exception as e:
            logger.warning("System impedance calculation failed", frequency=frequency, error=str(e))
            return {
                "frequency_hz": frequency,
                "resistance_ohm": 1.0,
                "reactance_ohm": 1.0,
                "impedance_magnitude_ohm": 1.4,
                "max_impedance": 1.4
            }
    
    def _is_resonance_frequency(self, frequency: float, impedance_data: Dict[str, float]) -> bool:
        """Check if frequency is a resonance frequency."""
        # High impedance indicates resonance
        if impedance_data["max_impedance"] > 10.0:
            return True
        
        # Check for phase changes (simplified)
        # In practice, this would check for 90-degree phase shifts
        return False
    
    def _assess_resonance_risk(self, resonance_results: Dict[str, Any]) -> str:
        """Assess overall resonance risk."""
        critical_count = len(resonance_results["critical_frequencies"])
        resonance_count = len(resonance_results["resonance_frequencies"])
        
        if critical_count >= 3 or resonance_count >= 10:
            return "high"
        elif critical_count >= 1 or resonance_count >= 5:
            return "medium"
        else:
            return "low"
    
    def _calculate_filter_requirement(self, bus_id: int, current_thd: float, target_thd: float) -> Dict[str, Any]:
        """Calculate filter requirements for a specific bus."""
        try:
            thd_reduction_needed = current_thd - target_thd
            
            if thd_reduction_needed <= 0:
                return {
                    "filter_required": False,
                    "current_thd_percent": current_thd,
                    "target_thd_percent": target_thd
                }
            
            # Determine filter type based on THD level
            if thd_reduction_needed < 5:
                filter_type = "passive_lc"
                cost_factor = 1.0
            elif thd_reduction_needed < 15:
                filter_type = "active_filter"
                cost_factor = 3.0
            else:
                filter_type = "hybrid_filter"
                cost_factor = 5.0
            
            # Estimate filter rating
            bus_power = self._get_bus_total_power(bus_id)
            filter_rating = bus_power * 0.5  # 50% of bus power
            
            return {
                "filter_required": True,
                "current_thd_percent": current_thd,
                "target_thd_percent": target_thd,
                "thd_reduction_needed_percent": thd_reduction_needed,
                "filter_type": filter_type,
                "filter_rating_kvar": filter_rating,
                "estimated_cost_usd": filter_rating * cost_factor * 1000,  # $1000/kVAR base
                "installation_complexity": "medium" if filter_type == "passive_lc" else "high"
            }
            
        except Exception as e:
            logger.warning("Filter requirement calculation failed", bus_id=bus_id, error=str(e))
            return {"filter_required": False, "error": str(e)}
    
    def _get_bus_total_power(self, bus_id: int) -> float:
        """Get total power at a bus."""
        try:
            bus_loads = self.network.pandapower_net.load[
                self.network.pandapower_net.load.bus == bus_id
            ] if not self.network.pandapower_net.load.empty else pd.DataFrame()
            
            if not bus_loads.empty:
                return float(bus_loads.p_mw.sum() * 1000)  # Convert to kW
            else:
                return 100.0  # Default 100 kW
                
        except Exception:
            return 100.0
    
    def _estimate_filter_costs(self, filter_recommendations: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate total costs for harmonic filtering."""
        total_cost = 0.0
        filter_count = 0
        cost_breakdown = {}
        
        for bus_id, recommendation in filter_recommendations.items():
            if recommendation.get("filter_required", False):
                cost = recommendation.get("estimated_cost_usd", 0)
                total_cost += cost
                filter_count += 1
                
                filter_type = recommendation.get("filter_type", "unknown")
                if filter_type not in cost_breakdown:
                    cost_breakdown[filter_type] = {"count": 0, "cost": 0}
                
                cost_breakdown[filter_type]["count"] += 1
                cost_breakdown[filter_type]["cost"] += cost
        
        return {
            "total_cost_usd": total_cost,
            "total_filters": filter_count,
            "cost_breakdown": cost_breakdown,
            "average_cost_per_filter_usd": total_cost / filter_count if filter_count > 0 else 0
        }
    
    def _generate_system_filter_recommendations(self, filter_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system-wide filter recommendations."""
        recommendations = {
            "priority_actions": [],
            "cost_optimization": [],
            "implementation_phases": []
        }
        
        # Prioritize high THD buses
        high_priority_buses = [
            bus_id for bus_id, rec in filter_results["filter_recommendations"].items()
            if rec.get("thd_reduction_needed_percent", 0) > 10
        ]
        
        if high_priority_buses:
            recommendations["priority_actions"].append({
                "action": "immediate_filtering",
                "buses": high_priority_buses,
                "reason": "THD exceeds 10% above target"
            })
        
        # Cost optimization suggestions
        total_cost = filter_results["cost_analysis"]["total_cost_usd"]
        if total_cost > 50000:  # High cost threshold
            recommendations["cost_optimization"].append({
                "suggestion": "Consider centralized filtering solution",
                "potential_savings": total_cost * 0.3,
                "description": "Install filters at distribution level instead of individual buses"
            })
        
        # Implementation phases
        if filter_results["filter_recommendations"]:
            filter_count = len(filter_results["filter_recommendations"])
            if filter_count > 5:
                recommendations["implementation_phases"] = [
                    {
                        "phase": 1,
                        "description": "Critical buses with THD > 10%",
                        "bus_count": len(high_priority_buses),
                        "timeline": "0-3 months"
                    },
                    {
                        "phase": 2,
                        "description": "Remaining buses with THD violations",
                        "bus_count": filter_count - len(high_priority_buses),
                        "timeline": "3-12 months"
                    }
                ]
        
        return recommendations
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all performed harmonic analyses."""
        return {
            "network_name": self.network.name,
            "available_analyses": list(self.results.keys()),
            "last_updated": datetime.now().isoformat(),
            "results_summary": {
                analysis_type: {
                    "converged": result.get("converged", False),
                    "analysis_time": result.get("analysis_time", 0)
                }
                for analysis_type, result in self.results.items()
            }
        }


logger.info("OpenGrid harmonic analysis module loaded.") 