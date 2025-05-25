"""
Short circuit analysis module for OpenGrid.

Provides comprehensive fault analysis capabilities for power systems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import structlog
import pandapower as pp
import pandapower.shortcircuit as sc
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = structlog.get_logger(__name__)


class ShortCircuitAnalyzer:
    """Comprehensive short circuit analyzer for power systems."""
    
    def __init__(self, network):
        """Initialize the short circuit analyzer.
        
        Args:
            network: PowerNetwork instance to analyze
        """
        self.network = network
        self.results = {}
        self.fault_scenarios = []
        logger.info("ShortCircuitAnalyzer initialized", network_name=network.name)
    
    def run_max_fault_current(
        self,
        bus: Optional[int] = None,
        fault_type: str = "3ph",
        case: str = "max"
    ) -> Dict[str, Any]:
        """Calculate maximum fault currents.
        
        Args:
            bus: Specific bus to analyze (None for all buses)
            fault_type: Type of fault ('3ph', 'line_to_ground', 'line_to_line')
            case: Calculation case ('max' or 'min')
            
        Returns:
            Dictionary containing fault current results
        """
        start_time = datetime.now()
        
        try:
            # Run short circuit calculation
            sc.calc_sc(
                self.network.pandapower_net,
                fault=fault_type,
                case=case,
                bus=bus
            )
            
            # Extract results
            results = {
                "converged": True,
                "fault_type": fault_type,
                "case": case,
                "bus_fault_currents": {},
                "line_fault_currents": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Bus fault currents
            if hasattr(self.network.pandapower_net, 'res_bus_sc'):
                results["bus_fault_currents"] = {
                    "ikss_ka": self.network.pandapower_net.res_bus_sc.ikss_ka.to_dict(),
                    "ip_ka": self.network.pandapower_net.res_bus_sc.ip_ka.to_dict() if 'ip_ka' in self.network.pandapower_net.res_bus_sc.columns else {},
                    "rk_ohm": self.network.pandapower_net.res_bus_sc.rk_ohm.to_dict() if 'rk_ohm' in self.network.pandapower_net.res_bus_sc.columns else {},
                    "xk_ohm": self.network.pandapower_net.res_bus_sc.xk_ohm.to_dict() if 'xk_ohm' in self.network.pandapower_net.res_bus_sc.columns else {}
                }
            
            # Line fault currents
            if hasattr(self.network.pandapower_net, 'res_line_sc'):
                results["line_fault_currents"] = {
                    "ikss_ka_from": self.network.pandapower_net.res_line_sc.ikss_ka_from.to_dict() if 'ikss_ka_from' in self.network.pandapower_net.res_line_sc.columns else {},
                    "ikss_ka_to": self.network.pandapower_net.res_line_sc.ikss_ka_to.to_dict() if 'ikss_ka_to' in self.network.pandapower_net.res_line_sc.columns else {}
                }
            
            # Calculate fault analysis metrics
            if results["bus_fault_currents"].get("ikss_ka"):
                fault_currents = list(results["bus_fault_currents"]["ikss_ka"].values())
                results["fault_analysis"] = {
                    "max_fault_current_ka": float(max(fault_currents)),
                    "min_fault_current_ka": float(min(fault_currents)),
                    "avg_fault_current_ka": float(np.mean(fault_currents)),
                    "critical_buses": self._identify_critical_fault_buses(results["bus_fault_currents"]["ikss_ka"])
                }
            
            self.results[f"{fault_type}_{case}"] = results
            
            logger.info(
                "Short circuit analysis completed",
                network=self.network.name,
                fault_type=fault_type,
                case=case,
                max_fault_current=results.get("fault_analysis", {}).get("max_fault_current_ka", 0)
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "fault_type": fault_type,
                "case": case,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Short circuit analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def run_min_fault_current(
        self,
        bus: Optional[int] = None,
        fault_type: str = "3ph"
    ) -> Dict[str, Any]:
        """Calculate minimum fault currents.
        
        Args:
            bus: Specific bus to analyze (None for all buses)
            fault_type: Type of fault ('3ph', 'line_to_ground', 'line_to_line')
            
        Returns:
            Dictionary containing minimum fault current results
        """
        return self.run_max_fault_current(bus=bus, fault_type=fault_type, case="min")
    
    def analyze_fault_at_bus(self, bus_id: int, fault_types: List[str] = None) -> Dict[str, Any]:
        """Analyze various fault types at a specific bus.
        
        Args:
            bus_id: Bus ID to analyze
            fault_types: List of fault types to analyze
            
        Returns:
            Dictionary containing fault analysis results for all fault types
        """
        if fault_types is None:
            fault_types = ["3ph", "1ph", "2ph"]
        
        start_time = datetime.now()
        
        try:
            results = {
                "bus_id": bus_id,
                "fault_analyses": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            for fault_type in fault_types:
                try:
                    # Run fault analysis for each type
                    fault_result = self.run_max_fault_current(bus=bus_id, fault_type=fault_type)
                    results["fault_analyses"][fault_type] = fault_result
                    
                except Exception as e:
                    results["fault_analyses"][fault_type] = {
                        "converged": False,
                        "error": str(e)
                    }
                    logger.warning(
                        "Fault analysis failed for specific type",
                        bus_id=bus_id,
                        fault_type=fault_type,
                        error=str(e)
                    )
            
            # Calculate comparative analysis
            results["comparison"] = self._compare_fault_types(results["fault_analyses"])
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                "Bus fault analysis completed",
                network=self.network.name,
                bus_id=bus_id,
                fault_types=fault_types
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "bus_id": bus_id,
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Bus fault analysis failed", network=self.network.name, bus_id=bus_id, error=str(e))
            return error_result
    
    def analyze_protection_coordination(self) -> Dict[str, Any]:
        """Analyze protection device coordination based on fault currents.
        
        Returns:
            Dictionary containing protection coordination analysis
        """
        start_time = datetime.now()
        
        try:
            # Get maximum fault currents for all buses
            max_fault_results = self.run_max_fault_current(fault_type="3ph", case="max")
            
            if not max_fault_results["converged"]:
                return {
                    "converged": False,
                    "error": "Could not calculate fault currents for protection coordination",
                    "analysis_time": (datetime.now() - start_time).total_seconds()
                }
            
            fault_currents = max_fault_results["bus_fault_currents"]["ikss_ka"]
            
            # Protection device recommendations
            protection_recommendations = {}
            
            for bus_id, fault_current in fault_currents.items():
                # Determine protection device ratings based on fault current
                protection_recommendations[bus_id] = self._recommend_protection_device(
                    fault_current, 
                    bus_id
                )
            
            # Check for coordination issues
            coordination_issues = self._check_coordination_issues(protection_recommendations)
            
            results = {
                "converged": True,
                "protection_recommendations": protection_recommendations,
                "coordination_issues": coordination_issues,
                "summary": {
                    "total_buses": len(protection_recommendations),
                    "coordination_issues_count": len(coordination_issues),
                    "max_fault_current_ka": max(fault_currents.values()),
                    "min_fault_current_ka": min(fault_currents.values())
                },
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.results["protection_coordination"] = results
            
            logger.info(
                "Protection coordination analysis completed",
                network=self.network.name,
                coordination_issues=len(coordination_issues)
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Protection coordination analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def calculate_arc_flash_hazard(self, working_distance: float = 0.61) -> Dict[str, Any]:
        """Calculate arc flash hazard levels for buses.
        
        Args:
            working_distance: Working distance in meters (default: 610mm)
            
        Returns:
            Dictionary containing arc flash analysis results
        """
        start_time = datetime.now()
        
        try:
            # Get fault current data
            fault_results = self.run_max_fault_current(fault_type="3ph", case="max")
            
            if not fault_results["converged"]:
                return {
                    "converged": False,
                    "error": "Could not calculate fault currents for arc flash analysis",
                    "analysis_time": (datetime.now() - start_time).total_seconds()
                }
            
            fault_currents = fault_results["bus_fault_currents"]["ikss_ka"]
            
            # Calculate arc flash energy for each bus
            arc_flash_results = {}
            
            for bus_id, fault_current in fault_currents.items():
                # Get bus voltage
                bus_voltage = self._get_bus_voltage(bus_id)
                
                # Calculate arc flash energy using IEEE 1584 methodology
                arc_energy = self._calculate_arc_flash_energy(
                    fault_current * 1000,  # Convert to A
                    bus_voltage,
                    working_distance
                )
                
                # Determine hazard category
                hazard_category = self._determine_hazard_category(arc_energy)
                
                arc_flash_results[bus_id] = {
                    "fault_current_ka": fault_current,
                    "bus_voltage_kv": bus_voltage,
                    "arc_energy_cal_cm2": arc_energy,
                    "hazard_category": hazard_category,
                    "ppe_required": self._get_ppe_requirements(hazard_category),
                    "working_distance_m": working_distance
                }
            
            # Summary statistics
            arc_energies = [result["arc_energy_cal_cm2"] for result in arc_flash_results.values()]
            
            results = {
                "converged": True,
                "arc_flash_analysis": arc_flash_results,
                "summary": {
                    "max_arc_energy_cal_cm2": max(arc_energies),
                    "min_arc_energy_cal_cm2": min(arc_energies),
                    "avg_arc_energy_cal_cm2": np.mean(arc_energies),
                    "high_hazard_buses": [
                        bus_id for bus_id, result in arc_flash_results.items()
                        if result["hazard_category"] >= 3
                    ]
                },
                "working_distance_m": working_distance,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.results["arc_flash"] = results
            
            logger.info(
                "Arc flash analysis completed",
                network=self.network.name,
                high_hazard_buses=len(results["summary"]["high_hazard_buses"]),
                max_arc_energy=results["summary"]["max_arc_energy_cal_cm2"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Arc flash analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def _identify_critical_fault_buses(self, fault_currents: Dict[int, float], threshold: float = 50.0) -> List[int]:
        """Identify buses with high fault currents."""
        critical_buses = []
        for bus_id, current in fault_currents.items():
            if current > threshold:
                critical_buses.append(bus_id)
        return critical_buses
    
    def _compare_fault_types(self, fault_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare different fault types at a bus."""
        comparison = {}
        
        for fault_type, analysis in fault_analyses.items():
            if analysis.get("converged") and "fault_analysis" in analysis:
                comparison[fault_type] = {
                    "max_current_ka": analysis["fault_analysis"]["max_fault_current_ka"],
                    "min_current_ka": analysis["fault_analysis"]["min_fault_current_ka"]
                }
        
        if comparison:
            all_currents = [data["max_current_ka"] for data in comparison.values()]
            comparison["overall"] = {
                "highest_fault_current_ka": max(all_currents),
                "lowest_fault_current_ka": min(all_currents),
                "most_severe_fault": max(comparison.keys(), key=lambda x: comparison[x]["max_current_ka"])
            }
        
        return comparison
    
    def _recommend_protection_device(self, fault_current: float, bus_id: int) -> Dict[str, Any]:
        """Recommend protection device based on fault current."""
        bus_voltage = self._get_bus_voltage(bus_id)
        
        # Simple protection device selection logic
        if fault_current < 1.0:  # < 1 kA
            device_type = "Miniature Circuit Breaker (MCB)"
            rating = "32A"
        elif fault_current < 10.0:  # < 10 kA
            device_type = "Molded Case Circuit Breaker (MCCB)"
            rating = f"{min(int(fault_current * 2000), 6300)}A"
        elif fault_current < 50.0:  # < 50 kA
            device_type = "Air Circuit Breaker (ACB)"
            rating = f"{min(int(fault_current * 2000), 50000)}A"
        else:  # >= 50 kA
            device_type = "SF6 Circuit Breaker"
            rating = f"{min(int(fault_current * 2000), 100000)}A"
        
        return {
            "device_type": device_type,
            "current_rating": rating,
            "fault_current_ka": fault_current,
            "interrupting_capacity_ka": fault_current * 1.5,  # Safety margin
            "voltage_rating_kv": bus_voltage
        }
    
    def _check_coordination_issues(self, protection_recommendations: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for protection coordination issues."""
        issues = []
        
        # Simple coordination check - more sophisticated analysis would consider time-current curves
        for bus_id, recommendation in protection_recommendations.items():
            fault_current = recommendation["fault_current_ka"]
            
            # Check if downstream devices can handle upstream faults
            # This is a simplified check
            if fault_current > 25.0:  # High fault current
                issues.append({
                    "bus_id": bus_id,
                    "issue_type": "high_fault_current",
                    "description": f"Bus {bus_id} has very high fault current ({fault_current:.1f} kA) requiring careful coordination",
                    "severity": "high" if fault_current > 50.0 else "medium"
                })
        
        return issues
    
    def _get_bus_voltage(self, bus_id: int) -> float:
        """Get bus voltage level."""
        try:
            return float(self.network.pandapower_net.bus.loc[bus_id, "vn_kv"])
        except:
            return 0.4  # Default to LV
    
    def _calculate_arc_flash_energy(self, fault_current: float, voltage: float, distance: float) -> float:
        """Calculate arc flash energy using simplified IEEE 1584 methodology."""
        # Simplified calculation - actual IEEE 1584 is more complex
        try:
            # Log of normalized current
            lg_i = np.log10(fault_current)
            
            # Arc current (simplified)
            if voltage < 1.0:  # LV system
                lg_ia = lg_i + 0.00402 + 0.983 * lg_i
                ia = 10 ** lg_ia
            else:  # MV system
                lg_ia = 0.00402 + 0.983 * lg_i
                ia = 10 ** lg_ia
            
            # Arc energy (simplified calculation)
            # Assuming 0.2 second arcing time for circuit breaker operation
            t = 0.2  # seconds
            
            if voltage < 1.0:  # LV
                energy = 4.184 * ia * t * (0.0016 * (distance ** -1.4738))
            else:  # MV
                energy = 4.184 * ia * t * (0.0016 * (distance ** -1.4738))
            
            return max(0.0, energy)
            
        except Exception as e:
            logger.warning("Arc flash energy calculation failed", error=str(e))
            return 0.0
    
    def _determine_hazard_category(self, arc_energy: float) -> int:
        """Determine hazard/risk category based on arc energy."""
        if arc_energy < 1.2:
            return 0  # No hazard
        elif arc_energy < 4.0:
            return 1  # Low hazard
        elif arc_energy < 8.0:
            return 2  # Medium hazard
        elif arc_energy < 25.0:
            return 3  # High hazard
        elif arc_energy < 40.0:
            return 4  # Very high hazard
        else:
            return 5  # Extreme hazard
    
    def _get_ppe_requirements(self, hazard_category: int) -> Dict[str, str]:
        """Get PPE requirements based on hazard category."""
        ppe_requirements = {
            0: {"description": "No special PPE required", "arc_rating": "N/A"},
            1: {"description": "Arc-rated long sleeve shirt and pants", "arc_rating": "4 cal/cm²"},
            2: {"description": "Arc-rated shirt and pants + hard hat + safety glasses", "arc_rating": "8 cal/cm²"},
            3: {"description": "Arc flash suit + hard hat + face shield", "arc_rating": "25 cal/cm²"},
            4: {"description": "Arc flash suit + hard hat + arc-rated face shield", "arc_rating": "40 cal/cm²"},
            5: {"description": "Special arc flash suit + remote operation recommended", "arc_rating": "40+ cal/cm²"}
        }
        return ppe_requirements.get(hazard_category, ppe_requirements[5])
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all performed short circuit analyses."""
        return {
            "network_name": self.network.name,
            "available_analyses": list(self.results.keys()),
            "fault_scenarios": len(self.fault_scenarios),
            "last_updated": datetime.now().isoformat(),
            "results_summary": {
                analysis_type: {
                    "converged": result.get("converged", False),
                    "analysis_time": result.get("analysis_time", 0)
                }
                for analysis_type, result in self.results.items()
            }
        }


logger.info("OpenGrid short circuit analysis module loaded.") 