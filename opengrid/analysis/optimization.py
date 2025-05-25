"""Optimization Engine for OpenGrid

Provides comprehensive optimization capabilities for power system planning and operation.
Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import structlog
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

logger = structlog.get_logger(__name__)

@dataclass
class OptimizationProblem:
    """Definition of an optimization problem"""
    name: str
    objective_type: str  # minimize, maximize
    variables: Dict[str, Dict[str, Any]]  # variable definitions
    constraints: List[Dict[str, Any]]  # constraint definitions
    parameters: Dict[str, Any]  # problem parameters
    metadata: Dict[str, Any] = None

@dataclass
class OptimizationResult:
    """Result of an optimization problem"""
    converged: bool
    objective_value: float
    variables: Dict[str, float]
    constraints_satisfied: bool
    solution_time: float
    iterations: int
    optimality_gap: float
    metadata: Dict[str, Any]

class OptimizerBase(ABC):
    """Base class for optimization algorithms"""
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve the optimization problem"""
        pass

class LinearProgrammingSolver(OptimizerBase):
    """Linear programming solver using scipy"""
    
    def __init__(self):
        self.method = "highs"
        
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve linear programming problem"""
        start_time = datetime.now()
        
        try:
            from scipy.optimize import linprog
            
            # Extract problem data
            variables = problem.variables
            constraints = problem.constraints
            
            # Build coefficient matrices
            var_names = list(variables.keys())
            n_vars = len(var_names)
            
            # Objective function coefficients
            c = np.zeros(n_vars)
            for i, var_name in enumerate(var_names):
                var_data = variables[var_name]
                if 'objective_coeff' in var_data:
                    c[i] = var_data['objective_coeff']
            
            if problem.objective_type == "maximize":
                c = -c  # Convert to minimization
            
            # Inequality constraints (A_ub @ x <= b_ub)
            A_ub = []
            b_ub = []
            
            # Equality constraints (A_eq @ x == b_eq)
            A_eq = []
            b_eq = []
            
            for constraint in constraints:
                coeffs = np.zeros(n_vars)
                for var_name, coeff in constraint.get('coefficients', {}).items():
                    if var_name in var_names:
                        coeffs[var_names.index(var_name)] = coeff
                
                if constraint['type'] == 'equality':
                    A_eq.append(coeffs)
                    b_eq.append(constraint['rhs'])
                elif constraint['type'] == 'inequality':
                    A_ub.append(coeffs)
                    b_ub.append(constraint['rhs'])
            
            # Variable bounds
            bounds = []
            for var_name in var_names:
                var_data = variables[var_name]
                lower = var_data.get('lower_bound', 0)
                upper = var_data.get('upper_bound', None)
                bounds.append((lower, upper))
            
            # Solve
            result = linprog(
                c=c,
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=bounds,
                method=self.method
            )
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            solution_vars = {}
            if result.success:
                for i, var_name in enumerate(var_names):
                    solution_vars[var_name] = float(result.x[i])
            
            objective_value = float(result.fun) if result.success else 0.0
            if problem.objective_type == "maximize":
                objective_value = -objective_value
            
            return OptimizationResult(
                converged=bool(result.success),
                objective_value=objective_value,
                variables=solution_vars,
                constraints_satisfied=bool(result.success),
                solution_time=solve_time,
                iterations=getattr(result, 'nit', 0),
                optimality_gap=0.0 if result.success else 1.0,
                metadata={
                    'solver': 'scipy_linprog',
                    'method': self.method,
                    'status': result.message,
                    'n_variables': n_vars,
                    'n_constraints': len(constraints)
                }
            )
            
        except Exception as e:
            solve_time = (datetime.now() - start_time).total_seconds()
            logger.error("Linear programming solver failed", error=str(e))
            
            return OptimizationResult(
                converged=False,
                objective_value=0.0,
                variables={},
                constraints_satisfied=False,
                solution_time=solve_time,
                iterations=0,
                optimality_gap=1.0,
                metadata={'error': str(e)}
            )

class OptimizationEngine:
    """Main optimization engine for power system problems"""
    
    def __init__(self, network):
        """Initialize optimization engine
        
        Args:
            network: PowerNetwork instance
        """
        self.network = network
        self.solvers = {
            'linear': LinearProgrammingSolver(),
        }
        self.results = {}
        self.problem_library = {}
        
        logger.info("OptimizationEngine initialized", network_name=network.name)
    
    def solve_economic_dispatch(
        self,
        generators: Dict[int, Dict[str, float]],
        load_demand: float,
        time_horizon: int = 1
    ) -> OptimizationResult:
        """Solve economic dispatch problem
        
        Args:
            generators: Generator data with cost coefficients
            load_demand: Total load demand in MW
            time_horizon: Time horizon (hours)
            
        Returns:
            Optimization result
        """
        try:
            # Create optimization problem
            variables = {}
            constraints = []
            
            # Generator variables
            for gen_id, gen_data in generators.items():
                var_name = f"P_gen_{gen_id}"
                variables[var_name] = {
                    'type': 'continuous',
                    'lower_bound': gen_data.get('min_power', 0),
                    'upper_bound': gen_data.get('max_power', 100),
                    'objective_coeff': gen_data.get('cost_per_mwh', 50),
                    'description': f"Power output of generator {gen_id}"
                }
            
            # Power balance constraint
            balance_constraint = {
                'name': 'power_balance',
                'type': 'equality',
                'coefficients': {f"P_gen_{gen_id}": 1.0 for gen_id in generators.keys()},
                'rhs': load_demand,
                'description': 'Power supply must equal demand'
            }
            constraints.append(balance_constraint)
            
            problem = OptimizationProblem(
                name="economic_dispatch",
                objective_type="minimize",
                variables=variables,
                constraints=constraints,
                parameters={
                    'load_demand': load_demand,
                    'time_horizon': time_horizon,
                    'generator_count': len(generators)
                }
            )
            
            # Solve problem
            result = self.solvers['linear'].solve(problem)
            
            # Store result
            self.results['economic_dispatch'] = result
            
            logger.info(
                "Economic dispatch completed",
                converged=result.converged,
                objective_value=result.objective_value,
                solution_time=result.solution_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Economic dispatch failed", error=str(e))
            return OptimizationResult(
                converged=False,
                objective_value=0.0,
                variables={},
                constraints_satisfied=False,
                solution_time=0.0,
                iterations=0,
                optimality_gap=1.0,
                metadata={'error': str(e)}
            )
    
    def solve_optimal_power_flow(
        self,
        include_voltage_constraints: bool = True,
        include_thermal_constraints: bool = True
    ) -> OptimizationResult:
        """Solve optimal power flow problem
        
        Args:
            include_voltage_constraints: Include bus voltage constraints
            include_thermal_constraints: Include line thermal constraints
            
        Returns:
            Optimization result
        """
        try:
            # Get network data
            buses = self.network.pandapower_net.bus.index.tolist() if not self.network.pandapower_net.bus.empty else []
            lines = self.network.pandapower_net.line.index.tolist() if not self.network.pandapower_net.line.empty else []
            generators = self.network.pandapower_net.gen.index.tolist() if not self.network.pandapower_net.gen.empty else []
            
            variables = {}
            constraints = []
            
            # Generator power variables
            for gen_id in generators:
                gen_data = self.network.pandapower_net.gen.loc[gen_id]
                
                variables[f"P_gen_{gen_id}"] = {
                    'type': 'continuous',
                    'lower_bound': gen_data.get('min_p_mw', 0),
                    'upper_bound': gen_data.get('max_p_mw', gen_data.get('p_mw', 100)),
                    'objective_coeff': 30.0,  # Default cost
                    'description': f"Active power generation at generator {gen_id}"
                }
                
                variables[f"Q_gen_{gen_id}"] = {
                    'type': 'continuous',
                    'lower_bound': gen_data.get('min_q_mvar', -50),
                    'upper_bound': gen_data.get('max_q_mvar', 50),
                    'objective_coeff': 0.0,  # No cost for reactive power
                    'description': f"Reactive power generation at generator {gen_id}"
                }
            
            # Bus voltage magnitude variables
            if include_voltage_constraints:
                for bus_id in buses:
                    variables[f"V_mag_{bus_id}"] = {
                        'type': 'continuous',
                        'lower_bound': 0.95,
                        'upper_bound': 1.05,
                        'objective_coeff': 0.0,
                        'description': f"Voltage magnitude at bus {bus_id}"
                    }
            
            # Power balance constraints
            loads = self.network.pandapower_net.load if not self.network.pandapower_net.load.empty else pd.DataFrame()
            
            for bus_id in buses:
                # Active power balance
                p_balance = {
                    'name': f'p_balance_bus_{bus_id}',
                    'type': 'equality',
                    'coefficients': {},
                    'rhs': 0.0,
                    'description': f'Active power balance at bus {bus_id}'
                }
                
                # Add generator contributions
                bus_generators = [gen_id for gen_id in generators 
                                if self.network.pandapower_net.gen.loc[gen_id, 'bus'] == bus_id]
                for gen_id in bus_generators:
                    p_balance['coefficients'][f"P_gen_{gen_id}"] = 1.0
                
                # Add load demand
                bus_loads = loads[loads.bus == bus_id] if not loads.empty else pd.DataFrame()
                if not bus_loads.empty:
                    total_load = bus_loads.p_mw.sum()
                    p_balance['rhs'] = float(total_load)
                
                if p_balance['coefficients']:  # Only add if there are variables
                    constraints.append(p_balance)
            
            # Thermal constraints
            if include_thermal_constraints:
                for line_id in lines:
                    line_data = self.network.pandapower_net.line.loc[line_id]
                    max_loading = line_data.get('max_i_ka', 1.0) * line_data.get('vn_kv', 0.4) * np.sqrt(3)
                    
                    thermal_constraint = {
                        'name': f'thermal_line_{line_id}',
                        'type': 'inequality',
                        'coefficients': {},  # Would need power flow equations
                        'rhs': max_loading,
                        'description': f'Thermal limit for line {line_id}'
                    }
                    # Simplified - would need full AC power flow modeling
                    # constraints.append(thermal_constraint)
            
            problem = OptimizationProblem(
                name="optimal_power_flow",
                objective_type="minimize",
                variables=variables,
                constraints=constraints,
                parameters={
                    'buses': len(buses),
                    'lines': len(lines),
                    'generators': len(generators),
                    'include_voltage_constraints': include_voltage_constraints,
                    'include_thermal_constraints': include_thermal_constraints
                }
            )
            
            # Solve problem (simplified version)
            result = self.solvers['linear'].solve(problem)
            
            self.results['optimal_power_flow'] = result
            
            logger.info(
                "Optimal power flow completed",
                converged=result.converged,
                objective_value=result.objective_value
            )
            
            return result
            
        except Exception as e:
            logger.error("Optimal power flow failed", error=str(e))
            return OptimizationResult(
                converged=False,
                objective_value=0.0,
                variables={},
                constraints_satisfied=False,
                solution_time=0.0,
                iterations=0,
                optimality_gap=1.0,
                metadata={'error': str(e)}
            )
    
    def solve_unit_commitment(
        self,
        time_horizon: int = 24,
        load_forecast: List[float] = None
    ) -> OptimizationResult:
        """Solve unit commitment problem
        
        Args:
            time_horizon: Planning horizon in hours
            load_forecast: Hourly load forecast
            
        Returns:
            Optimization result
        """
        try:
            if load_forecast is None:
                # Generate sample load profile
                base_load = 100.0
                load_forecast = [
                    base_load * (0.7 + 0.3 * np.sin(2 * np.pi * h / 24))
                    for h in range(time_horizon)
                ]
            
            generators = self.network.pandapower_net.gen.index.tolist() if not self.network.pandapower_net.gen.empty else [0]
            
            variables = {}
            constraints = []
            
            # Variables for each generator and time period
            for gen_id in generators:
                gen_data = self.network.pandapower_net.gen.loc[gen_id] if gen_id in self.network.pandapower_net.gen.index else {
                    'p_mw': 50, 'min_p_mw': 10, 'max_p_mw': 100
                }
                
                for t in range(time_horizon):
                    # Unit commitment variable (binary)
                    variables[f"u_{gen_id}_{t}"] = {
                        'type': 'binary',
                        'lower_bound': 0,
                        'upper_bound': 1,
                        'objective_coeff': 1000,  # Start-up cost
                        'description': f"Unit {gen_id} commitment at hour {t}"
                    }
                    
                    # Power output variable
                    variables[f"p_{gen_id}_{t}"] = {
                        'type': 'continuous',
                        'lower_bound': 0,
                        'upper_bound': gen_data.get('max_p_mw', 100),
                        'objective_coeff': 50,  # Variable cost
                        'description': f"Power output of unit {gen_id} at hour {t}"
                    }
            
            # Load balance constraints
            for t in range(time_horizon):
                load_constraint = {
                    'name': f'load_balance_{t}',
                    'type': 'equality',
                    'coefficients': {f"p_{gen_id}_{t}": 1.0 for gen_id in generators},
                    'rhs': load_forecast[t],
                    'description': f'Load balance at hour {t}'
                }
                constraints.append(load_constraint)
            
            # Generation limits
            for gen_id in generators:
                gen_data = self.network.pandapower_net.gen.loc[gen_id] if gen_id in self.network.pandapower_net.gen.index else {
                    'min_p_mw': 10, 'max_p_mw': 100
                }
                
                for t in range(time_horizon):
                    # Minimum generation when online
                    min_gen_constraint = {
                        'name': f'min_gen_{gen_id}_{t}',
                        'type': 'inequality',
                        'coefficients': {
                            f"p_{gen_id}_{t}": 1.0,
                            f"u_{gen_id}_{t}": -gen_data.get('min_p_mw', 10)
                        },
                        'rhs': 0,
                        'description': f'Minimum generation for unit {gen_id} at hour {t}'
                    }
                    constraints.append(min_gen_constraint)
                    
                    # Maximum generation when online
                    max_gen_constraint = {
                        'name': f'max_gen_{gen_id}_{t}',
                        'type': 'inequality',
                        'coefficients': {
                            f"p_{gen_id}_{t}": 1.0,
                            f"u_{gen_id}_{t}": -gen_data.get('max_p_mw', 100)
                        },
                        'rhs': 0,
                        'description': f'Maximum generation for unit {gen_id} at hour {t}'
                    }
                    constraints.append(max_gen_constraint)
            
            problem = OptimizationProblem(
                name="unit_commitment",
                objective_type="minimize",
                variables=variables,
                constraints=constraints,
                parameters={
                    'time_horizon': time_horizon,
                    'generators': len(generators),
                    'load_forecast': load_forecast
                }
            )
            
            # Note: This would require a MILP solver for binary variables
            # For demonstration, we'll solve a relaxed version
            result = self.solvers['linear'].solve(problem)
            
            self.results['unit_commitment'] = result
            
            logger.info(
                "Unit commitment completed",
                converged=result.converged,
                time_horizon=time_horizon
            )
            
            return result
            
        except Exception as e:
            logger.error("Unit commitment failed", error=str(e))
            return OptimizationResult(
                converged=False,
                objective_value=0.0,
                variables={},
                constraints_satisfied=False,
                solution_time=0.0,
                iterations=0,
                optimality_gap=1.0,
                metadata={'error': str(e)}
            )
    
    def solve_transmission_expansion(
        self,
        candidate_lines: List[Dict[str, Any]],
        planning_horizon: int = 10,
        demand_growth: float = 0.03
    ) -> OptimizationResult:
        """Solve transmission expansion planning problem
        
        Args:
            candidate_lines: List of candidate transmission lines
            planning_horizon: Planning horizon in years
            demand_growth: Annual demand growth rate
            
        Returns:
            Optimization result
        """
        try:
            variables = {}
            constraints = []
            
            # Decision variables for each candidate line
            for i, line in enumerate(candidate_lines):
                variables[f"x_line_{i}"] = {
                    'type': 'binary',
                    'lower_bound': 0,
                    'upper_bound': 1,
                    'objective_coeff': line.get('investment_cost', 1000000),
                    'description': f"Build candidate line {i}"
                }
            
            # Add operational variables for each year
            for year in range(planning_horizon):
                load_multiplier = (1 + demand_growth) ** year
                
                # Generator dispatch variables
                generators = self.network.pandapower_net.gen.index.tolist() if not self.network.pandapower_net.gen.empty else []
                
                for gen_id in generators:
                    variables[f"p_gen_{gen_id}_year_{year}"] = {
                        'type': 'continuous',
                        'lower_bound': 0,
                        'upper_bound': 100,
                        'objective_coeff': 50 * planning_horizon,  # Operational cost
                        'description': f"Generation from unit {gen_id} in year {year}"
                    }
            
            # Reliability constraints
            n_scenarios = 3  # N-1, N-2, base case
            
            for scenario in range(n_scenarios):
                reliability_constraint = {
                    'name': f'reliability_scenario_{scenario}',
                    'type': 'inequality',
                    'coefficients': {},
                    'rhs': 0,
                    'description': f'Reliability constraint for scenario {scenario}'
                }
                
                # Add transmission line capacity contributions
                for i, line in enumerate(candidate_lines):
                    capacity = line.get('capacity_mw', 100)
                    reliability_constraint['coefficients'][f"x_line_{i}"] = capacity
                
                constraints.append(reliability_constraint)
            
            problem = OptimizationProblem(
                name="transmission_expansion",
                objective_type="minimize",
                variables=variables,
                constraints=constraints,
                parameters={
                    'candidate_lines': len(candidate_lines),
                    'planning_horizon': planning_horizon,
                    'demand_growth': demand_growth
                }
            )
            
            result = self.solvers['linear'].solve(problem)
            
            self.results['transmission_expansion'] = result
            
            logger.info(
                "Transmission expansion planning completed",
                converged=result.converged,
                candidate_lines=len(candidate_lines)
            )
            
            return result
            
        except Exception as e:
            logger.error("Transmission expansion planning failed", error=str(e))
            return OptimizationResult(
                converged=False,
                objective_value=0.0,
                variables={},
                constraints_satisfied=False,
                solution_time=0.0,
                iterations=0,
                optimality_gap=1.0,
                metadata={'error': str(e)}
            )
    
    def solve_renewable_integration(
        self,
        renewable_sites: List[Dict[str, Any]],
        storage_options: List[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Solve renewable energy integration optimization
        
        Args:
            renewable_sites: Potential renewable energy sites
            storage_options: Energy storage options
            
        Returns:
            Optimization result
        """
        try:
            variables = {}
            constraints = []
            
            # Renewable capacity variables
            for i, site in enumerate(renewable_sites):
                variables[f"cap_renewable_{i}"] = {
                    'type': 'continuous',
                    'lower_bound': 0,
                    'upper_bound': site.get('max_capacity', 100),
                    'objective_coeff': site.get('capex_per_mw', 2000000),
                    'description': f"Renewable capacity at site {i}"
                }
            
            # Storage capacity variables
            if storage_options:
                for i, storage in enumerate(storage_options):
                    variables[f"cap_storage_{i}"] = {
                        'type': 'continuous',
                        'lower_bound': 0,
                        'upper_bound': storage.get('max_capacity', 50),
                        'objective_coeff': storage.get('capex_per_mwh', 500000),
                        'description': f"Storage capacity option {i}"
                    }
            
            # Grid integration constraints
            total_capacity_constraint = {
                'name': 'total_renewable_capacity',
                'type': 'inequality',
                'coefficients': {f"cap_renewable_{i}": 1.0 for i in range(len(renewable_sites))},
                'rhs': 500,  # Maximum total renewable capacity
                'description': 'Total renewable capacity limit'
            }
            constraints.append(total_capacity_constraint)
            
            # Grid stability constraint
            stability_constraint = {
                'name': 'grid_stability',
                'type': 'inequality',
                'coefficients': {f"cap_renewable_{i}": 1.0 for i in range(len(renewable_sites))},
                'rhs': 200,  # Stability limit
                'description': 'Grid stability limit for renewables'
            }
            constraints.append(stability_constraint)
            
            problem = OptimizationProblem(
                name="renewable_integration",
                objective_type="minimize",
                variables=variables,
                constraints=constraints,
                parameters={
                    'renewable_sites': len(renewable_sites),
                    'storage_options': len(storage_options) if storage_options else 0
                }
            )
            
            result = self.solvers['linear'].solve(problem)
            
            self.results['renewable_integration'] = result
            
            logger.info(
                "Renewable integration optimization completed",
                converged=result.converged,
                renewable_sites=len(renewable_sites)
            )
            
            return result
            
        except Exception as e:
            logger.error("Renewable integration optimization failed", error=str(e))
            return OptimizationResult(
                converged=False,
                objective_value=0.0,
                variables={},
                constraints_satisfied=False,
                solution_time=0.0,
                iterations=0,
                optimality_gap=1.0,
                metadata={'error': str(e)}
            )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        return {
            "network_name": self.network.name,
            "available_optimizations": list(self.results.keys()),
            "last_updated": datetime.now().isoformat(),
            "results_summary": {
                opt_type: {
                    "converged": result.converged,
                    "objective_value": result.objective_value,
                    "solution_time": result.solution_time,
                    "variables_count": len(result.variables)
                }
                for opt_type, result in self.results.items()
            }
        }
    
    def export_results(self, format_type: str = "json") -> str:
        """Export optimization results"""
        export_data = {
            "network_summary": self.network.get_summary(),
            "optimization_results": {
                name: {
                    "converged": result.converged,
                    "objective_value": result.objective_value,
                    "variables": result.variables,
                    "solution_time": result.solution_time,
                    "metadata": result.metadata
                }
                for name, result in self.results.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            return str(export_data)


logger.info("OpenGrid optimization engine module loaded.") 