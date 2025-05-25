"""
Contingency analysis module for OpenGrid.

Provides N-1 and N-k contingency analysis for power systems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import structlog
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from itertools import combinations

logger = structlog.get_logger(__name__)


class ContingencyAnalyzer:
    """Contingency analyzer for power system reliability studies."""
    
    def __init__(self, network):
        """Initialize the contingency analyzer.
        
        Args:
            network: PowerNetwork instance to analyze
        """
        self.network = network
        self.results = {}
        self.contingency_list = []
        logger.info("ContingencyAnalyzer initialized", network_name=network.name)
    
    def analyze_n_minus_1(self, component_types: List[str] = None) -> Dict[str, Any]:
        """Perform N-1 contingency analysis."""
        start_time = datetime.now()
        
        if component_types is None:
            component_types = ["line", "transformer", "generator"]
        
        try:
            results = {
                "converged": True,
                "analysis_type": "n_minus_1",
                "component_types": component_types,
                "contingencies": {},
                "violations": [],
                "critical_contingencies": [],
                "system_reliability": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Generate contingency list
            contingency_list = self._generate_n_minus_1_contingencies(component_types)
            results["total_contingencies"] = len(contingency_list)
            
            # Analyze each contingency
            violation_count = 0
            critical_count = 0
            
            for i, contingency in enumerate(contingency_list):
                contingency_id = f"N1_{i+1}"
                contingency_result = self._analyze_single_contingency(contingency)
                results["contingencies"][contingency_id] = contingency_result
                
                # Check for violations
                if contingency_result.get("has_violations", False):
                    violation_count += 1
                    results["violations"].append({
                        "contingency_id": contingency_id,
                        "contingency": contingency,
                        "violations": contingency_result["violations"]
                    })
                
                # Check for critical contingencies
                if contingency_result.get("severity", "low") in ["high", "critical"]:
                    critical_count += 1
                    results["critical_contingencies"].append({
                        "contingency_id": contingency_id,
                        "contingency": contingency,
                        "severity": contingency_result["severity"],
                        "description": contingency_result.get("description", "")
                    })
            
            # Calculate system reliability metrics
            results["system_reliability"] = self._calculate_reliability_metrics(
                len(contingency_list), violation_count, critical_count
            )
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["n_minus_1"] = results
            
            logger.info(
                "N-1 contingency analysis completed",
                network=self.network.name,
                total_contingencies=len(contingency_list),
                violations=violation_count,
                critical_contingencies=critical_count
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("N-1 contingency analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_n_minus_k(self, k: int = 2, component_types: List[str] = None) -> Dict[str, Any]:
        """Perform N-k contingency analysis."""
        start_time = datetime.now()
        
        if component_types is None:
            component_types = ["line", "generator"]
        
        try:
            results = {
                "converged": True,
                "analysis_type": f"n_minus_{k}",
                "k_value": k,
                "component_types": component_types,
                "contingencies": {},
                "violations": [],
                "critical_contingencies": [],
                "system_reliability": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Generate N-k contingency list
            contingency_list = self._generate_n_minus_k_contingencies(k, component_types)
            results["total_contingencies"] = len(contingency_list)
            
            # Limit analysis for computational efficiency
            max_contingencies = 100  # Limit for N-k analysis
            if len(contingency_list) > max_contingencies:
                contingency_list = self._prioritize_contingencies(contingency_list, max_contingencies)
                results["note"] = f"Analysis limited to top {max_contingencies} contingencies"
            
            # Analyze each contingency
            violation_count = 0
            critical_count = 0
            
            for i, contingency in enumerate(contingency_list):
                contingency_id = f"N{k}_{i+1}"
                contingency_result = self._analyze_multiple_outage_contingency(contingency)
                results["contingencies"][contingency_id] = contingency_result
                
                # Check for violations
                if contingency_result.get("has_violations", False):
                    violation_count += 1
                    results["violations"].append({
                        "contingency_id": contingency_id,
                        "contingency": contingency,
                        "violations": contingency_result["violations"]
                    })
                
                # Check for critical contingencies
                if contingency_result.get("severity", "low") in ["high", "critical"]:
                    critical_count += 1
                    results["critical_contingencies"].append({
                        "contingency_id": contingency_id,
                        "contingency": contingency,
                        "severity": contingency_result["severity"]
                    })
            
            # Calculate system reliability metrics
            results["system_reliability"] = self._calculate_reliability_metrics(
                len(contingency_list), violation_count, critical_count
            )
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results[f"n_minus_{k}"] = results
            
            logger.info(
                f"N-{k} contingency analysis completed",
                network=self.network.name,
                total_contingencies=len(contingency_list),
                violations=violation_count,
                critical_contingencies=critical_count
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error(f"N-{k} contingency analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_cascading_failures(self, initial_contingency: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze potential cascading failure scenarios."""
        start_time = datetime.now()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "cascading_failures",
                "initial_contingency": initial_contingency,
                "cascade_scenarios": [],
                "system_collapse_risk": "low",
                "mitigation_strategies": [],
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            if initial_contingency is None:
                # Use most critical contingency from N-1 analysis
                initial_contingency = self._get_most_critical_contingency()
            
            # Simulate cascading failure
            cascade_scenario = self._simulate_cascading_failure(initial_contingency)
            results["cascade_scenarios"].append(cascade_scenario)
            
            # Assess system collapse risk
            results["system_collapse_risk"] = self._assess_collapse_risk(cascade_scenario)
            
            # Generate mitigation strategies
            results["mitigation_strategies"] = self._generate_mitigation_strategies(cascade_scenario)
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["cascading_failures"] = results
            
            logger.info(
                "Cascading failure analysis completed",
                network=self.network.name,
                collapse_risk=results["system_collapse_risk"],
                mitigation_strategies=len(results["mitigation_strategies"])
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Cascading failure analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def analyze_transfer_limits(self, interface_definitions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze transfer limits between areas/interfaces."""
        start_time = datetime.now()
        
        try:
            results = {
                "converged": True,
                "analysis_type": "transfer_limits",
                "interfaces": {},
                "transfer_capabilities": {},
                "limiting_contingencies": {},
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            
            if interface_definitions is None:
                interface_definitions = self._auto_define_interfaces()
            
            for interface in interface_definitions:
                interface_id = interface["name"]
                
                # Calculate transfer limits for this interface
                transfer_limits = self._calculate_interface_transfer_limits(interface)
                results["interfaces"][interface_id] = interface
                results["transfer_capabilities"][interface_id] = transfer_limits
                
                # Find limiting contingencies
                limiting_contingencies = self._find_limiting_contingencies(interface, transfer_limits)
                results["limiting_contingencies"][interface_id] = limiting_contingencies
            
            results["analysis_time"] = (datetime.now() - start_time).total_seconds()
            self.results["transfer_limits"] = results
            
            logger.info(
                "Transfer limits analysis completed",
                network=self.network.name,
                interfaces=len(interface_definitions)
            )
            
            return results
            
        except Exception as e:
            error_result = {
                "converged": False,
                "error": str(e),
                "analysis_time": (datetime.now() - start_time).total_seconds()
            }
            logger.error("Transfer limits analysis failed", network=self.network.name, error=str(e))
            return error_result
    
    def _generate_n_minus_1_contingencies(self, component_types: List[str]) -> List[Dict[str, Any]]:
        """Generate N-1 contingency list."""
        contingencies = []
        
        for component_type in component_types:
            if component_type == "line" and not self.network.pandapower_net.line.empty:
                for line_id in self.network.pandapower_net.line.index:
                    contingencies.append({
                        "type": "line_outage",
                        "component_type": "line",
                        "component_id": line_id,
                        "description": f"Line {line_id} outage"
                    })
            
            elif component_type == "transformer" and not self.network.pandapower_net.trafo.empty:
                for trafo_id in self.network.pandapower_net.trafo.index:
                    contingencies.append({
                        "type": "transformer_outage",
                        "component_type": "transformer",
                        "component_id": trafo_id,
                        "description": f"Transformer {trafo_id} outage"
                    })
            
            elif component_type == "generator" and not self.network.pandapower_net.gen.empty:
                for gen_id in self.network.pandapower_net.gen.index:
                    contingencies.append({
                        "type": "generator_outage",
                        "component_type": "generator",
                        "component_id": gen_id,
                        "description": f"Generator {gen_id} outage"
                    })
        
        return contingencies
    
    def _generate_n_minus_k_contingencies(self, k: int, component_types: List[str]) -> List[Dict[str, Any]]:
        """Generate N-k contingency list."""
        # Get all single components
        single_contingencies = self._generate_n_minus_1_contingencies(component_types)
        
        # Generate combinations of k components
        contingencies = []
        
        if len(single_contingencies) >= k:
            for combo in combinations(single_contingencies, k):
                contingencies.append({
                    "type": f"multiple_outage_{k}",
                    "components": list(combo),
                    "description": f"{k} component outage: " + ", ".join([c["description"] for c in combo])
                })
        
        return contingencies
    
    def _prioritize_contingencies(self, contingency_list: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
        """Prioritize contingencies based on criticality."""
        # Simple prioritization - in practice would use network topology analysis
        scored_contingencies = []
        
        for contingency in contingency_list:
            score = self._calculate_contingency_priority_score(contingency)
            scored_contingencies.append((score, contingency))
        
        # Sort by score (descending) and take top max_count
        scored_contingencies.sort(key=lambda x: x[0], reverse=True)
        return [cont for _, cont in scored_contingencies[:max_count]]
    
    def _calculate_contingency_priority_score(self, contingency: Dict[str, Any]) -> float:
        """Calculate priority score for contingency."""
        score = 0.0
        
        if contingency["type"] == "line_outage":
            # Higher score for transmission lines
            line_id = contingency["component_id"]
            if line_id in self.network.pandapower_net.line.index:
                # Consider line loading and voltage level
                score += 1.0
        
        elif contingency["type"] == "generator_outage":
            # Higher score for larger generators
            gen_id = contingency["component_id"]
            if gen_id in self.network.pandapower_net.gen.index:
                gen_power = self.network.pandapower_net.gen.loc[gen_id, "p_mw"]
                score += gen_power / 100  # Scale by 100 MW
        
        elif contingency["type"].startswith("multiple_outage"):
            # Multiple outages get higher base score
            score += len(contingency.get("components", []))
        
        return score
    
    def _analyze_single_contingency(self, contingency: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single contingency."""
        try:
            # Store original state
            original_state = self._save_network_state()
            
            # Apply contingency
            self._apply_contingency(contingency)
            
            # Run power flow
            pf_result = self.network.run_powerflow()
            
            # Analyze results
            result = {
                "contingency": contingency,
                "converged": pf_result["converged"],
                "has_violations": False,
                "violations": [],
                "severity": "low",
                "load_shed_mw": 0.0,
                "voltage_violations": [],
                "thermal_violations": []
            }
            
            if pf_result["converged"]:
                # Check for violations
                violations = self._check_violations(pf_result)
                result.update(violations)
                
                # Assess severity
                result["severity"] = self._assess_contingency_severity(violations)
            else:
                result["has_violations"] = True
                result["severity"] = "critical"
                result["violations"].append("Power flow did not converge")
            
            # Restore original state
            self._restore_network_state(original_state)
            
            return result
            
        except Exception as e:
            # Restore original state in case of error
            try:
                self._restore_network_state(original_state)
            except:
                pass
            
            return {
                "contingency": contingency,
                "converged": False,
                "has_violations": True,
                "severity": "critical",
                "error": str(e)
            }
    
    def _analyze_multiple_outage_contingency(self, contingency: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a multiple component outage contingency."""
        try:
            # Store original state
            original_state = self._save_network_state()
            
            # Apply multiple contingencies
            for component_contingency in contingency.get("components", []):
                self._apply_contingency(component_contingency)
            
            # Run power flow
            pf_result = self.network.run_powerflow()
            
            # Analyze results
            result = {
                "contingency": contingency,
                "converged": pf_result["converged"],
                "has_violations": False,
                "violations": [],
                "severity": "low",
                "components_out": len(contingency.get("components", [])),
                "load_shed_mw": 0.0
            }
            
            if pf_result["converged"]:
                violations = self._check_violations(pf_result)
                result.update(violations)
                result["severity"] = self._assess_contingency_severity(violations)
            else:
                result["has_violations"] = True
                result["severity"] = "critical"
                result["violations"].append("Power flow did not converge - possible system separation")
            
            # Restore original state
            self._restore_network_state(original_state)
            
            return result
            
        except Exception as e:
            try:
                self._restore_network_state(original_state)
            except:
                pass
            
            return {
                "contingency": contingency,
                "converged": False,
                "has_violations": True,
                "severity": "critical",
                "error": str(e)
            }
    
    def _apply_contingency(self, contingency: Dict[str, Any]) -> None:
        """Apply a contingency to the network."""
        contingency_type = contingency["type"]
        component_id = contingency["component_id"]
        
        if contingency_type == "line_outage":
            if component_id in self.network.pandapower_net.line.index:
                self.network.pandapower_net.line.loc[component_id, "in_service"] = False
        
        elif contingency_type == "transformer_outage":
            if component_id in self.network.pandapower_net.trafo.index:
                self.network.pandapower_net.trafo.loc[component_id, "in_service"] = False
        
        elif contingency_type == "generator_outage":
            if component_id in self.network.pandapower_net.gen.index:
                self.network.pandapower_net.gen.loc[component_id, "in_service"] = False
    
    def _save_network_state(self) -> Dict[str, Any]:
        """Save current network state."""
        state = {}
        
        if not self.network.pandapower_net.line.empty:
            state["line_service"] = self.network.pandapower_net.line["in_service"].copy()
        
        if not self.network.pandapower_net.trafo.empty:
            state["trafo_service"] = self.network.pandapower_net.trafo["in_service"].copy()
        
        if not self.network.pandapower_net.gen.empty:
            state["gen_service"] = self.network.pandapower_net.gen["in_service"].copy()
        
        return state
    
    def _restore_network_state(self, state: Dict[str, Any]) -> None:
        """Restore network state."""
        if "line_service" in state and not self.network.pandapower_net.line.empty:
            self.network.pandapower_net.line["in_service"] = state["line_service"]
        
        if "trafo_service" in state and not self.network.pandapower_net.trafo.empty:
            self.network.pandapower_net.trafo["in_service"] = state["trafo_service"]
        
        if "gen_service" in state and not self.network.pandapower_net.gen.empty:
            self.network.pandapower_net.gen["in_service"] = state["gen_service"]
    
    def _check_violations(self, pf_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check for violations in power flow results."""
        violations = {
            "has_violations": False,
            "violations": [],
            "voltage_violations": [],
            "thermal_violations": [],
            "load_shed_mw": 0.0
        }
        
        # Check voltage violations
        bus_voltages = pf_result.get("bus_voltages", {})
        for bus_id, voltage in bus_voltages.items():
            if voltage < 0.95 or voltage > 1.05:  # ±5% voltage limits
                violations["voltage_violations"].append({
                    "bus_id": bus_id,
                    "voltage_pu": voltage,
                    "violation_type": "low" if voltage < 0.95 else "high"
                })
                violations["has_violations"] = True
        
        # Check thermal violations
        line_loadings = pf_result.get("line_loading", {})
        for line_id, loading in line_loadings.items():
            if loading > 100:  # Overload
                violations["thermal_violations"].append({
                    "line_id": line_id,
                    "loading_percent": loading,
                    "violation_magnitude": loading - 100
                })
                violations["has_violations"] = True
        
        # Compile violation list
        if violations["voltage_violations"]:
            violations["violations"].append(f"{len(violations['voltage_violations'])} voltage violations")
        
        if violations["thermal_violations"]:
            violations["violations"].append(f"{len(violations['thermal_violations'])} thermal violations")
        
        return violations
    
    def _assess_contingency_severity(self, violations: Dict[str, Any]) -> str:
        """Assess the severity of a contingency."""
        if not violations["has_violations"]:
            return "low"
        
        voltage_violation_count = len(violations["voltage_violations"])
        thermal_violation_count = len(violations["thermal_violations"])
        
        # Critical violations
        severe_voltage_violations = sum(
            1 for v in violations["voltage_violations"]
            if v["voltage_pu"] < 0.9 or v["voltage_pu"] > 1.1
        )
        
        severe_thermal_violations = sum(
            1 for v in violations["thermal_violations"]
            if v["loading_percent"] > 150
        )
        
        if severe_voltage_violations > 0 or severe_thermal_violations > 0:
            return "critical"
        elif voltage_violation_count > 5 or thermal_violation_count > 3:
            return "high"
        elif voltage_violation_count > 2 or thermal_violation_count > 1:
            return "medium"
        else:
            return "low"
    
    def _calculate_reliability_metrics(self, total_contingencies: int, violations: int, critical: int) -> Dict[str, float]:
        """Calculate system reliability metrics."""
        if total_contingencies == 0:
            return {
                "contingency_pass_rate_percent": 100.0,
                "critical_contingency_rate_percent": 0.0,
                "system_reliability_index": 1.0
            }
        
        pass_rate = ((total_contingencies - violations) / total_contingencies) * 100
        critical_rate = (critical / total_contingencies) * 100
        reliability_index = max(0.0, 1.0 - (violations / total_contingencies) - (critical / total_contingencies) * 0.5)
        
        return {
            "contingency_pass_rate_percent": pass_rate,
            "critical_contingency_rate_percent": critical_rate,
            "system_reliability_index": reliability_index,
            "total_contingencies": total_contingencies,
            "violations": violations,
            "critical_contingencies": critical
        }
    
    def _simulate_cascading_failure(self, initial_contingency: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate cascading failure scenario."""
        try:
            scenario = {
                "initial_contingency": initial_contingency,
                "cascade_steps": [],
                "total_load_lost_mw": 0.0,
                "total_components_lost": 1,
                "cascade_stopped": False,
                "system_collapse": False
            }
            
            # Start with initial contingency
            original_state = self._save_network_state()
            self._apply_contingency(initial_contingency)
            
            step = 1
            max_steps = 10  # Limit cascade steps
            
            while step <= max_steps:
                # Run power flow
                pf_result = self.network.run_powerflow()
                
                if not pf_result["converged"]:
                    scenario["system_collapse"] = True
                    scenario["cascade_steps"].append({
                        "step": step,
                        "event": "System collapse - power flow diverged",
                        "components_lost": [],
                        "load_lost_mw": 0.0
                    })
                    break
                
                # Check for overloads that could trigger more outages
                overloaded_components = self._find_overloaded_components(pf_result)
                
                if not overloaded_components:
                    scenario["cascade_stopped"] = True
                    scenario["cascade_steps"].append({
                        "step": step,
                        "event": "Cascade stopped - no more overloads",
                        "components_lost": [],
                        "load_lost_mw": 0.0
                    })
                    break
                
                # Trip overloaded components
                components_tripped = []
                load_lost = 0.0
                
                for component in overloaded_components[:3]:  # Limit to 3 per step
                    self._trip_component(component)
                    components_tripped.append(component)
                    load_lost += self._estimate_load_loss(component)
                
                scenario["cascade_steps"].append({
                    "step": step,
                    "event": f"Overload protection operated",
                    "components_lost": components_tripped,
                    "load_lost_mw": load_lost
                })
                
                scenario["total_components_lost"] += len(components_tripped)
                scenario["total_load_lost_mw"] += load_lost
                
                step += 1
            
            # Restore original state
            self._restore_network_state(original_state)
            
            return scenario
            
        except Exception as e:
            return {
                "initial_contingency": initial_contingency,
                "error": str(e),
                "cascade_steps": [],
                "total_load_lost_mw": 0.0,
                "system_collapse": False
            }
    
    def _find_overloaded_components(self, pf_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find components that are overloaded and could trip."""
        overloaded = []
        
        # Check line loadings
        line_loadings = pf_result.get("line_loading", {})
        for line_id, loading in line_loadings.items():
            if loading > 120:  # 120% threshold for protection operation
                overloaded.append({
                    "type": "line",
                    "id": line_id,
                    "loading_percent": loading
                })
        
        # Sort by loading level (most overloaded first)
        overloaded.sort(key=lambda x: x["loading_percent"], reverse=True)
        
        return overloaded
    
    def _trip_component(self, component: Dict[str, Any]) -> None:
        """Trip an overloaded component."""
        if component["type"] == "line":
            line_id = component["id"]
            if line_id in self.network.pandapower_net.line.index:
                self.network.pandapower_net.line.loc[line_id, "in_service"] = False
    
    def _estimate_load_loss(self, component: Dict[str, Any]) -> float:
        """Estimate load loss from component outage."""
        # Simplified load loss estimation
        if component["type"] == "line":
            # Estimate based on line loading
            loading = component.get("loading_percent", 0)
            return max(0, loading * 0.1)  # Simplified
        return 0.0
    
    def _get_most_critical_contingency(self) -> Dict[str, Any]:
        """Get the most critical contingency from previous analysis."""
        # Default critical contingency if no previous analysis
        if not self.network.pandapower_net.line.empty:
            line_id = self.network.pandapower_net.line.index[0]
            return {
                "type": "line_outage",
                "component_type": "line",
                "component_id": line_id,
                "description": f"Line {line_id} outage"
            }
        else:
            return {
                "type": "generator_outage",
                "component_type": "generator",
                "component_id": 0,
                "description": "Generator 0 outage"
            }
    
    def _assess_collapse_risk(self, cascade_scenario: Dict[str, Any]) -> str:
        """Assess system collapse risk from cascade scenario."""
        if cascade_scenario.get("system_collapse", False):
            return "critical"
        
        load_lost_percent = cascade_scenario.get("total_load_lost_mw", 0) / 100  # Simplified
        components_lost = cascade_scenario.get("total_components_lost", 0)
        
        if load_lost_percent > 50 or components_lost > 10:
            return "high"
        elif load_lost_percent > 20 or components_lost > 5:
            return "medium"
        else:
            return "low"
    
    def _generate_mitigation_strategies(self, cascade_scenario: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate mitigation strategies for cascading failures."""
        strategies = []
        
        # Based on cascade characteristics
        if cascade_scenario.get("system_collapse", False):
            strategies.append({
                "strategy": "Emergency load shedding",
                "description": "Implement automatic load shedding schemes to prevent voltage collapse",
                "priority": "high"
            })
        
        if cascade_scenario.get("total_components_lost", 0) > 3:
            strategies.append({
                "strategy": "Special protection schemes",
                "description": "Install remedial action schemes to isolate disturbances",
                "priority": "medium"
            })
        
        strategies.append({
            "strategy": "System reinforcement",
            "description": "Add transmission capacity to reduce loading levels",
            "priority": "low"
        })
        
        return strategies
    
    def _auto_define_interfaces(self) -> List[Dict[str, Any]]:
        """Automatically define transfer interfaces."""
        # Simplified interface definition
        interfaces = []
        
        if not self.network.pandapower_net.line.empty:
            # Create a simple interface between voltage levels
            interfaces.append({
                "name": "HV_LV_Interface",
                "description": "Interface between HV and LV systems",
                "monitored_lines": list(self.network.pandapower_net.line.index[:3]),  # First 3 lines
                "direction": "export"
            })
        
        return interfaces
    
    def _calculate_interface_transfer_limits(self, interface: Dict[str, Any]) -> Dict[str, float]:
        """Calculate transfer limits for an interface."""
        # Simplified transfer limit calculation
        monitored_lines = interface.get("monitored_lines", [])
        
        # Base transfer capability
        base_limit = len(monitored_lines) * 100  # 100 MW per line
        
        # Calculate limits under different conditions
        return {
            "base_transfer_limit_mw": base_limit,
            "emergency_limit_mw": base_limit * 1.2,
            "n_minus_1_limit_mw": base_limit * 0.8,
            "thermal_limit_mw": base_limit,
            "voltage_limit_mw": base_limit * 0.9,
            "stability_limit_mw": base_limit * 0.85
        }
    
    def _find_limiting_contingencies(self, interface: Dict[str, Any], transfer_limits: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find contingencies that limit interface transfer capability."""
        limiting_contingencies = []
        
        # Check each monitored line
        for line_id in interface.get("monitored_lines", []):
            limiting_contingencies.append({
                "contingency": {
                    "type": "line_outage",
                    "component_id": line_id,
                    "description": f"Line {line_id} outage"
                },
                "limit_type": "thermal",
                "limit_value_mw": transfer_limits.get("n_minus_1_limit_mw", 0),
                "severity": "medium"
            })
        
        return limiting_contingencies
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all performed contingency analyses."""
        return {
            "network_name": self.network.name,
            "available_analyses": list(self.results.keys()),
            "last_updated": datetime.now().isoformat(),
            "results_summary": {
                analysis_type: {
                    "converged": result.get("converged", False),
                    "total_contingencies": result.get("total_contingencies", 0),
                    "violations": len(result.get("violations", [])),
                    "critical_contingencies": len(result.get("critical_contingencies", [])),
                    "analysis_time": result.get("analysis_time", 0)
                }
                for analysis_type, result in self.results.items()
            }
        }


logger.info("OpenGrid contingency analysis module loaded.") 