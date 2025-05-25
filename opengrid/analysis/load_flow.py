"""
Load flow analysis module for OpenGrid.

Provides comprehensive power flow analysis capabilities using pandapower and PyPSA.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import structlog
import pandapower as pp
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import warnings

logger = structlog.get_logger(__name__)


class LoadFlowAnalyzer:
    """Comprehensive load flow analyzer for power systems."""
    
    def __init__(self, network):
        """Initialize the load flow analyzer.
        
        Args:
            network: PowerNetwork instance to analyze
        """
        self.network = network
        self.results = {}
        self.analysis_history = []
        logger.info("LoadFlowAnalyzer initialized", network_name=network.name)
    
    def run_newton_raphson(
        self,
        tolerance_mva: float = 1e-6,
        max_iteration: int = 10,
        calculate_voltage_angles: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Run Newton-Raphson power flow analysis.
        
        Args:
            tolerance_mva: Convergence tolerance in MVA
            max_iteration: Maximum number of iterations
            calculate_voltage_angles: Whether to calculate voltage angles
            **kwargs: Additional pandapower arguments
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = datetime.now()
        
        try:
            # Run power flow
            pp.runpp(
                self.network.pandapower_net,
                algorithm="nr",
                tolerance_mva=tolerance_mva,
                max_iteration=max_iteration,
                calculate_voltage_angles=calculate_voltage_angles,
                **kwargs
            )
            
            # Extract results
            results = self._extract_results("newton_raphson")
            results.update({
                "algorithm": "newton_raphson",
                "tolerance_mva": tolerance_mva,
                "max_iteration": max_iteration,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            })
            
            self.results["newton_raphson"] = results
            self._add_to_history("newton_raphson", results)
            
            logger.info(
                "Newton-Raphson analysis completed",
                network=self.network.name,
                converged=results["converged"],
                iterations=results.get("iterations", "unknown")
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "algorithm": "newton_raphson",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Newton-Raphson analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def run_fast_decoupled(
        self,
        tolerance_mva: float = 1e-6,
        max_iteration: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """Run Fast Decoupled power flow analysis.
        
        Args:
            tolerance_mva: Convergence tolerance in MVA
            max_iteration: Maximum number of iterations
            **kwargs: Additional pandapower arguments
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = datetime.now()
        
        try:
            # Run power flow
            pp.runpp(
                self.network.pandapower_net,
                algorithm="fdpf",
                tolerance_mva=tolerance_mva,
                max_iteration=max_iteration,
                **kwargs
            )
            
            # Extract results
            results = self._extract_results("fast_decoupled")
            results.update({
                "algorithm": "fast_decoupled",
                "tolerance_mva": tolerance_mva,
                "max_iteration": max_iteration,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            })
            
            self.results["fast_decoupled"] = results
            self._add_to_history("fast_decoupled", results)
            
            logger.info(
                "Fast Decoupled analysis completed",
                network=self.network.name,
                converged=results["converged"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "algorithm": "fast_decoupled",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Fast Decoupled analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def run_dc_power_flow(self) -> Dict[str, Any]:
        """Run DC power flow analysis.
        
        Returns:
            Dictionary containing DC power flow results
        """
        start_time = datetime.now()
        
        try:
            # Run DC power flow
            pp.rundcpp(self.network.pandapower_net)
            
            # Extract DC results
            results = {
                "converged": True,
                "algorithm": "dc_power_flow",
                "bus_angles": self.network.pandapower_net.res_bus.va_degree.to_dict(),
                "line_flows": self.network.pandapower_net.res_line.p_from_mw.to_dict(),
                "generator_dispatch": self.network.pandapower_net.res_gen.p_mw.to_dict(),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.results["dc_power_flow"] = results
            self._add_to_history("dc_power_flow", results)
            
            logger.info("DC power flow analysis completed", network=self.network.name)
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "algorithm": "dc_power_flow",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("DC power flow analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def run_optimal_power_flow(
        self,
        objective: str = "cost",
        **kwargs
    ) -> Dict[str, Any]:
        """Run Optimal Power Flow analysis.
        
        Args:
            objective: Optimization objective ('cost', 'losses', 'emissions')
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary containing OPF results
        """
        start_time = datetime.now()
        
        try:
            # Run OPF
            pp.runopp(self.network.pandapower_net, **kwargs)
            
            # Extract OPF results
            results = self._extract_results("optimal_power_flow")
            results.update({
                "algorithm": "optimal_power_flow",
                "objective": objective,
                "total_cost": float(self.network.pandapower_net.res_cost) if hasattr(self.network.pandapower_net, 'res_cost') else None,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            })
            
            self.results["optimal_power_flow"] = results
            self._add_to_history("optimal_power_flow", results)
            
            logger.info(
                "Optimal Power Flow analysis completed",
                network=self.network.name,
                converged=results["converged"],
                objective=objective
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "algorithm": "optimal_power_flow",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Optimal Power Flow analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_voltage_stability(self) -> Dict[str, Any]:
        """Analyze voltage stability margins.
        
        Returns:
            Dictionary containing voltage stability analysis results
        """
        start_time = datetime.now()
        
        try:
            # Get base case results
            base_results = self.run_newton_raphson()
            if not base_results["converged"]:
                return {
                    "converged": False,
                    "error": "Base case did not converge",
                    "analysis_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Calculate voltage stability metrics
            bus_voltages = np.array(list(base_results["bus_voltages"].values()))
            
            stability_metrics = {
                "min_voltage_pu": float(np.min(bus_voltages)),
                "max_voltage_pu": float(np.max(bus_voltages)),
                "voltage_deviation": float(np.std(bus_voltages)),
                "voltage_unbalance": float((np.max(bus_voltages) - np.min(bus_voltages)) / np.mean(bus_voltages)),
                "critical_buses": self._identify_critical_buses(base_results["bus_voltages"])
            }
            
            # Load margin analysis
            load_margin = self._calculate_load_margin()
            
            results = {
                "converged": True,
                "algorithm": "voltage_stability",
                "stability_metrics": stability_metrics,
                "load_margin_percent": load_margin,
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.results["voltage_stability"] = results
            self._add_to_history("voltage_stability", results)
            
            logger.info(
                "Voltage stability analysis completed",
                network=self.network.name,
                load_margin=load_margin,
                min_voltage=stability_metrics["min_voltage_pu"]
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "algorithm": "voltage_stability",
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Voltage stability analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def _extract_results(self, analysis_type: str) -> Dict[str, Any]:
        """Extract results from pandapower network."""
        try:
            results = {
                "converged": bool(self.network.pandapower_net.converged),
                "bus_voltages": self.network.pandapower_net.res_bus.vm_pu.to_dict(),
                "bus_angles": self.network.pandapower_net.res_bus.va_degree.to_dict(),
                "line_loading": self.network.pandapower_net.res_line.loading_percent.to_dict(),
                "line_losses": self.network.pandapower_net.res_line.pl_mw.to_dict(),
                "total_losses_mw": float(self.network.pandapower_net.res_line.pl_mw.sum()),
                "generator_dispatch": self.network.pandapower_net.res_gen.p_mw.to_dict() if not self.network.pandapower_net.res_gen.empty else {},
            }
            
            # Add transformer results if available
            if not self.network.pandapower_net.res_trafo.empty:
                results["transformer_loading"] = self.network.pandapower_net.res_trafo.loading_percent.to_dict()
                results["transformer_losses"] = self.network.pandapower_net.res_trafo.pl_mw.to_dict()
            
            return results
            
        except Exception as e:
            logger.error("Failed to extract results", analysis_type=analysis_type, error=str(e))
            return {"converged": False, "error": f"Result extraction failed: {str(e)}"}
    
    def _identify_critical_buses(self, bus_voltages: Dict[int, float], threshold: float = 0.95) -> List[int]:
        """Identify buses with voltages below threshold."""
        critical_buses = []
        for bus_id, voltage in bus_voltages.items():
            if voltage < threshold:
                critical_buses.append(bus_id)
        return critical_buses
    
    def _calculate_load_margin(self) -> float:
        """Calculate load margin using continuation power flow approximation."""
        try:
            # Simple load margin calculation based on voltage collapse point
            # This is a simplified version - a full implementation would use continuation power flow
            
            # Get original loads
            original_loads = self.network.pandapower_net.load.p_mw.copy()
            
            # Binary search for maximum loadability
            low_factor, high_factor = 1.0, 3.0
            tolerance = 0.01
            
            while high_factor - low_factor > tolerance:
                test_factor = (low_factor + high_factor) / 2
                
                # Scale loads
                self.network.pandapower_net.load.p_mw = original_loads * test_factor
                
                try:
                    pp.runpp(self.network.pandapower_net, algorithm="nr", max_iteration=10)
                    if self.network.pandapower_net.converged:
                        low_factor = test_factor
                    else:
                        high_factor = test_factor
                except:
                    high_factor = test_factor
            
            # Restore original loads
            self.network.pandapower_net.load.p_mw = original_loads
            
            load_margin = (low_factor - 1.0) * 100  # Convert to percentage
            return float(max(0, load_margin))
            
        except Exception as e:
            logger.warning("Load margin calculation failed", error=str(e))
            return 0.0
    
    def _add_to_history(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """Add analysis to history."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": analysis_type,
            "converged": results.get("converged", False),
            "analysis_time": results.get("analysis_time", 0),
            "summary": {
                "total_losses_mw": results.get("total_losses_mw", 0),
                "min_voltage": min(results.get("bus_voltages", {1: 1.0}).values()),
                "max_voltage": max(results.get("bus_voltages", {1: 1.0}).values()),
                "max_loading": max(results.get("line_loading", {1: 0.0}).values()) if results.get("line_loading") else 0
            }
        }
        self.analysis_history.append(history_entry)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all performed analyses."""
        return {
            "network_name": self.network.name,
            "total_analyses": len(self.analysis_history),
            "available_results": list(self.results.keys()),
            "analysis_history": self.analysis_history[-10:],  # Last 10 analyses
            "last_updated": datetime.now().isoformat()
        }
    
    def export_results(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export analysis results in specified format."""
        export_data = {
            "network_summary": self.network.get_summary(),
            "analysis_results": self.results,
            "analysis_history": self.analysis_history,
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format_type.lower() == "json":
            import json
            return json.dumps(export_data, indent=2)
        else:
            return export_data


logger.info("OpenGrid load flow analysis module loaded.") 