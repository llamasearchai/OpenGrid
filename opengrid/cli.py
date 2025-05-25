#!/usr/bin/env python3
"""
OpenGrid Command Line Interface
Provides command-line access to OpenGrid power systems analysis capabilities.

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import structlog

from .modeling import PowerNetwork
from .analysis import (
    LoadFlowAnalyzer, ShortCircuitAnalyzer, StabilityAnalyzer,
    HarmonicAnalyzer, ContingencyAnalyzer, OptimizationEngine
)
from .ai import OpenAIAgent
from .data import sample_networks, sample_cases
from .api.app import start_server

logger = structlog.get_logger(__name__)


class OpenGridCLI:
    """Command Line Interface for OpenGrid"""
    
    def __init__(self):
        self.networks = {}
        self.results = {}
        self.ai_agent = None
        
    def setup_ai_agent(self, api_key: Optional[str] = None):
        """Setup AI agent if API key is available"""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            try:
                self.ai_agent = OpenAIAgent(api_key=api_key)
                print("[SUCCESS] AI agent initialized successfully")
            except Exception as e:
                print(f"[WARNING] Failed to initialize AI agent: {e}")
        else:
            print("[WARNING] No OpenAI API key provided - AI features disabled")
    
    def create_network(self, name: str, network_type: str = "custom") -> str:
        """Create a new power network"""
        try:
            if network_type == "custom":
                network = PowerNetwork(name=name)
                network_id = f"custom_{len(self.networks) + 1}"
            else:
                # Load from sample networks
                network_data = sample_networks.get_network(network_type)
                network = PowerNetwork(name=name)
                network.pandapower_net = network_data["pandapower"]
                network_id = f"sample_{network_type}"
            
            self.networks[network_id] = network
            print(f"[SUCCESS] Network '{name}' created with ID: {network_id}")
            return network_id
            
        except Exception as e:
            print(f"[ERROR] Error creating network: {e}")
            return ""
    
    def list_networks(self):
        """List all available networks"""
        if not self.networks:
            print("No networks loaded")
            return
        
        print("\nAvailable Networks:")
        print("-" * 50)
        for net_id, network in self.networks.items():
            summary = network.get_summary()
            print(f"ID: {net_id}")
            print(f"Name: {network.name}")
            print(f"Buses: {summary.get('total_buses', 0)}")
            print(f"Lines: {summary.get('total_lines', 0)}")
            print(f"Generators: {summary.get('total_generators', 0)}")
            print(f"Loads: {summary.get('total_loads', 0)}")
            print("-" * 30)
    
    def list_sample_networks(self):
        """List available sample networks"""
        networks = sample_networks.list_networks()
        
        print("\nAvailable Sample Networks:")
        print("-" * 60)
        for metadata in networks:
            print(f"Name: {metadata.name}")
            print(f"Type: {metadata.network_type}")
            print(f"Complexity: {metadata.complexity}")
            print(f"Buses: {metadata.num_buses}")
            print(f"Voltage Levels: {metadata.voltage_levels} kV")
            print(f"Description: {metadata.description}")
            print("-" * 40)
    
    def run_analysis(self, network_id: str, analysis_type: str, **kwargs):
        """Run power system analysis"""
        if network_id not in self.networks:
            print(f"[ERROR] Network '{network_id}' not found")
            return
        
        network = self.networks[network_id]
        print(f"[INFO] Running {analysis_type} analysis on {network.name}...")
        
        try:
            start_time = time.time()
            
            if analysis_type == "load_flow":
                analyzer = LoadFlowAnalyzer(network)
                algorithm = kwargs.get('algorithm', 'newton_raphson')
                
                if algorithm == "newton_raphson":
                    results = analyzer.run_newton_raphson(
                        tolerance_mva=kwargs.get('tolerance', 1e-6),
                        max_iteration=kwargs.get('max_iterations', 50)
                    )
                elif algorithm == "dc_power_flow":
                    results = analyzer.run_dc_power_flow()
                else:
                    results = analyzer.run_newton_raphson()
                    
            elif analysis_type == "short_circuit":
                analyzer = ShortCircuitAnalyzer(network)
                results = analyzer.run_max_fault_current(
                    bus=kwargs.get('bus'),
                    fault_type=kwargs.get('fault_type', '3ph'),
                    case=kwargs.get('case', 'max')
                )
                
            elif analysis_type == "contingency":
                analyzer = ContingencyAnalyzer(network)
                contingency_type = kwargs.get('contingency_type', 'n_minus_1')
                
                if contingency_type == "n_minus_1":
                    results = analyzer.analyze_n_minus_1(
                        kwargs.get('element_types', ['line'])
                    )
                else:
                    results = analyzer.analyze_n_minus_k(
                        k_value=kwargs.get('k_value', 2),
                        element_types=kwargs.get('element_types', ['line'])
                    )
                    
            elif analysis_type == "stability":
                analyzer = StabilityAnalyzer(network)
                stability_type = kwargs.get('stability_type', 'voltage')
                
                if stability_type == "voltage":
                    results = analyzer.analyze_voltage_stability()
                elif stability_type == "transient":
                    results = analyzer.analyze_transient_stability()
                else:
                    results = analyzer.analyze_small_signal_stability()
                    
            elif analysis_type == "harmonic":
                analyzer = HarmonicAnalyzer(network)
                results = analyzer.analyze_harmonic_distortion(
                    harmonic_orders=kwargs.get('harmonic_orders', [3, 5, 7, 9, 11])
                )
                
            elif analysis_type == "optimization":
                optimizer = OptimizationEngine(network)
                opt_type = kwargs.get('optimization_type', 'economic_dispatch')
                
                if opt_type == "economic_dispatch":
                    generators = kwargs.get('generators', {0: {'max_power': 100, 'cost_per_mwh': 50}})
                    load_demand = kwargs.get('load_demand', 80.0)
                    results = optimizer.solve_economic_dispatch(generators, load_demand)
                else:
                    results = optimizer.solve_optimal_power_flow()
                    
            else:
                print(f"[ERROR] Unknown analysis type: {analysis_type}")
                return
            
            elapsed_time = time.time() - start_time
            
            # Store results
            result_id = f"{network_id}_{analysis_type}_{int(time.time())}"
            self.results[result_id] = {
                'network_id': network_id,
                'analysis_type': analysis_type,
                'results': results,
                'elapsed_time': elapsed_time,
                'parameters': kwargs
            }
            
            print(f"[SUCCESS] Analysis completed in {elapsed_time:.2f} seconds")
            
            # Display results summary
            self._display_results(analysis_type, results)
            
            return result_id
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            logger.error("Analysis failed", analysis_type=analysis_type, error=str(e))
            return None

    def _display_results(self, analysis_type: str, results: Dict[str, Any]):
        """Display analysis results summary"""
        if not results:
            print("[WARNING] No results to display")
            return
        
        print("\nResults Summary:")
        print("-" * 40)
        
        if analysis_type == "load_flow":
            if results.get("converged", False):
                print(f"Status: Converged in {results.get('iteration_count', 0)} iterations")
                print(f"Total Losses: {results.get('total_losses_mw', 0):.2f} MW")
                voltages = results.get('bus_voltages', {})
                if voltages:
                    min_v = min(voltages.values())
                    max_v = max(voltages.values())
                    print(f"Voltage Range: {min_v:.3f} - {max_v:.3f} pu")
            else:
                print("Status: Failed to converge")
                
        elif analysis_type == "short_circuit":
            fault_currents = results.get('fault_currents', {})
            if fault_currents:
                max_current = max(fault_currents.values())
                print(f"Maximum Fault Current: {max_current:.2f} kA")
                print(f"Buses Analyzed: {len(fault_currents)}")
                
        elif analysis_type == "contingency":
            total = results.get('total_contingencies', 0)
            violations = results.get('violations', [])
            critical = results.get('critical_contingencies', [])
            print(f"Total Contingencies: {total}")
            print(f"Violations Found: {len(violations)}")
            print(f"Critical Contingencies: {len(critical)}")
            
            if critical:
                print("\nCritical Contingencies:")
                for i, cont in enumerate(critical[:5], 1):
                    print(f"  {i}. {cont}")
                    
        elif analysis_type == "stability":
            stability_margin = results.get('stability_margin', 0)
            critical_modes = results.get('critical_modes', [])
            print(f"Stability Margin: {stability_margin:.3f}")
            if critical_modes:
                print(f"Critical Modes: {len(critical_modes)}")
                
        elif analysis_type == "harmonic":
            thd_values = results.get('thd_values', {})
            if thd_values:
                max_thd = max(thd_values.values())
                print(f"Maximum THD: {max_thd:.2f}%")
                print(f"Buses Analyzed: {len(thd_values)}")
                
        elif analysis_type == "optimization":
            if results.get('optimal', False):
                total_cost = results.get('total_cost', 0)
                dispatch = results.get('dispatch', {})
                print(f"Status: Optimal solution found")
                print(f"Total Cost: ${total_cost:.2f}")
                if dispatch:
                    print("Dispatch Schedule:")
                    for gen, power in dispatch.items():
                        print(f"  Generator {gen}: {power:.2f} MW")
            else:
                print("Status: No optimal solution found")

    async def run_ai_analysis(self, result_id: str, analysis_type: str = None):
        """Run AI analysis on results"""
        if not self.ai_agent:
            print("[WARNING] AI agent not initialized. Set OPENAI_API_KEY environment variable.")
            return
        
        if result_id not in self.results:
            print(f"[ERROR] Result '{result_id}' not found")
            return
        
        result_data = self.results[result_id]
        if not analysis_type:
            analysis_type = result_data['analysis_type']
        
        network_id = result_data['network_id']
        network = self.networks[network_id]
        results = result_data['results']
        
        print(f"[INFO] Running AI analysis for {analysis_type}...")
        
        try:
            if analysis_type == "load_flow":
                network_info = network.get_summary()
                ai_response = await self.ai_agent.analyze_power_flow(results, network_info)
            elif analysis_type == "contingency":
                scenarios = []  # Could be enhanced with specific scenarios
                ai_response = await self.ai_agent.analyze_contingency(results, scenarios)
            else:
                print(f"[WARNING] AI analysis not implemented for {analysis_type}")
                return
            
            print(f"[SUCCESS] AI analysis completed (confidence: {ai_response.confidence_score:.2f})")
            
            print(f"\nKey Insights:")
            for i, insight in enumerate(ai_response.insights[:5], 1):
                print(f"  {i}. {insight}")
            
            if ai_response.recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(ai_response.recommendations[:5], 1):
                    print(f"  {i}. {rec}")
            
            if hasattr(ai_response, 'warnings') and ai_response.warnings:
                print(f"\nWarnings:")
                for i, warning in enumerate(ai_response.warnings[:3], 1):
                    print(f"  {i}. {warning}")
            
            # Store AI results
            ai_result_id = f"{result_id}_ai"
            self.results[ai_result_id] = {
                'base_result_id': result_id,
                'ai_response': ai_response,
                'timestamp': time.time()
            }
            
            return ai_result_id
            
        except Exception as e:
            print(f"[ERROR] AI analysis failed: {e}")
            logger.error("AI analysis failed", result_id=result_id, error=str(e))

    def run_case_study(self, case_id: str, network_id: Optional[str] = None):
        """Run a predefined case study"""
        try:
            case = sample_cases.get_case(case_id)
            print(f"[INFO] Running case study: {case.name}")
            print(f"Description: {case.description}")
            print(f"Network: {case.network_name}")
            print(f"Analysis: {case.analysis_type}")
            print(f"Difficulty: {case.difficulty}")
            print(f"Estimated Runtime: {case.estimated_runtime_seconds:.1f} seconds")
            
            # Load network if not provided
            if not network_id:
                network_id = self.create_network(
                    name=f"Case Study Network - {case.network_name}",
                    network_type=case.network_name
                )
                
                if not network_id:
                    print("[ERROR] Failed to create network for case study")
                    return
            
            # Run analysis with case parameters
            result_id = self.run_analysis(
                network_id=network_id,
                analysis_type=case.analysis_type,
                **case.parameters
            )
            
            if result_id:
                print(f"[SUCCESS] Case study completed with result ID: {result_id}")
                
                # Compare with expected results if available
                if case.expected_results:
                    print("\nExpected vs Actual Results:")
                    actual_results = self.results[result_id]['results']
                    for key, expected in case.expected_results.items():
                        actual = actual_results.get(key, "N/A")
                        print(f"  {key}: Expected={expected}, Actual={actual}")
                        
                return result_id
            else:
                print("[ERROR] Case study analysis failed")
                return None
                
        except Exception as e:
            print(f"[ERROR] Case study failed: {e}")
            logger.error("Case study failed", case_id=case_id, error=str(e))
            return None

    def list_case_studies(self, analysis_type: Optional[str] = None, difficulty: Optional[str] = None):
        """List available case studies"""
        case_ids = sample_cases.list_cases(analysis_type=analysis_type, difficulty=difficulty)
        
        print(f"\nAvailable Case Studies:")
        print("-" * 60)
        for case_id in case_ids:
            try:
                case = sample_cases.get_case(case_id)
                print(f"ID: {case.case_id}")
                print(f"Name: {case.name}")
                print(f"Type: {case.analysis_type}")
                print(f"Network: {case.network_name}")
                print(f"Difficulty: {case.difficulty}")
                print(f"Runtime: {case.estimated_runtime_seconds:.1f}s")
                print(f"Description: {case.description}")
                if hasattr(case, 'tags') and case.tags:
                    print(f"Tags: {', '.join(case.tags[:3])}")
                print("-" * 40)
            except Exception as e:
                print(f"[WARNING] Could not load case {case_id}: {e}")
                continue

    def export_results(self, result_id: str, filepath: str, format_type: str = "json"):
        """Export analysis results to file"""
        if result_id not in self.results:
            print(f"[ERROR] Result '{result_id}' not found")
            return
        
        try:
            result_data = self.results[result_id]
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
            else:
                print(f"[ERROR] Unsupported format: {format_type}")
                return
            
            print(f"[SUCCESS] Results exported to {output_path}")
            
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")

    def start_api_server(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
        """Start the OpenGrid API server"""
        print(f"[INFO] Starting API server at http://{host}:{port}")
        start_server(host=host, port=port, reload=reload)

def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="OpenGrid Power Systems Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  opengrid server

  # List available networks
  opengrid network list-samples

  # Create and run analysis
  opengrid network create --name "Test" --type ieee_14_bus
  opengrid analysis run --network custom_1 --type load_flow

  # Run case study
  opengrid case run --case lf_ieee14_basic

  # Export results
  opengrid export --result result_123 --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Network commands
    network_parser = subparsers.add_parser('network', help='Network operations')
    network_subparsers = network_parser.add_subparsers(dest='network_command')
    
    # Network create
    create_parser = network_subparsers.add_parser('create', help='Create network')
    create_parser.add_argument('--name', required=True, help='Network name')
    create_parser.add_argument('--type', default='custom', help='Network type')
    
    # Network list
    network_subparsers.add_parser('list', help='List loaded networks')
    network_subparsers.add_parser('list-samples', help='List sample networks')
    
    # Analysis commands
    analysis_parser = subparsers.add_parser('analysis', help='Analysis operations')
    analysis_subparsers = analysis_parser.add_subparsers(dest='analysis_command')
    
    # Analysis run
    run_parser = analysis_subparsers.add_parser('run', help='Run analysis')
    run_parser.add_argument('--network', required=True, help='Network ID')
    run_parser.add_argument('--type', required=True, 
                          choices=['load_flow', 'short_circuit', 'contingency', 
                                 'stability', 'harmonic', 'optimization'],
                          help='Analysis type')
    run_parser.add_argument('--algorithm', help='Algorithm for load flow')
    run_parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    run_parser.add_argument('--max-iterations', type=int, default=50, help='Maximum iterations')
    
    # AI analysis
    ai_parser = analysis_subparsers.add_parser('ai', help='Run AI analysis')
    ai_parser.add_argument('--result', required=True, help='Result ID to analyze')
    
    # Case study commands
    case_parser = subparsers.add_parser('case', help='Case study operations')
    case_subparsers = case_parser.add_subparsers(dest='case_command')
    
    # Case run
    case_run_parser = case_subparsers.add_parser('run', help='Run case study')
    case_run_parser.add_argument('--case', required=True, help='Case study ID')
    case_run_parser.add_argument('--network', help='Network ID (optional)')
    
    # Case list
    case_list_parser = case_subparsers.add_parser('list', help='List case studies')
    case_list_parser.add_argument('--type', help='Filter by analysis type')
    case_list_parser.add_argument('--difficulty', help='Filter by difficulty')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results')
    export_parser.add_argument('--result', required=True, help='Result ID to export')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', default='json', help='Export format')
    
    return parser

async def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = OpenGridCLI()
    cli.setup_ai_agent()
    
    try:
        if args.command == 'server':
            cli.start_api_server(host=args.host, port=args.port, reload=args.reload)
            
        elif args.command == 'network':
            if args.network_command == 'create':
                cli.create_network(name=args.name, network_type=args.type)
            elif args.network_command == 'list':
                cli.list_networks()
            elif args.network_command == 'list-samples':
                cli.list_sample_networks()
                
        elif args.command == 'analysis':
            if args.analysis_command == 'run':
                cli.run_analysis(
                    network_id=args.network,
                    analysis_type=args.type,
                    algorithm=args.algorithm,
                    tolerance=args.tolerance,
                    max_iterations=args.max_iterations
                )
            elif args.analysis_command == 'ai':
                await cli.run_ai_analysis(args.result)
                
        elif args.command == 'case':
            if args.case_command == 'run':
                cli.run_case_study(case_id=args.case, network_id=args.network)
            elif args.case_command == 'list':
                cli.list_case_studies(analysis_type=args.type, difficulty=args.difficulty)
                
        elif args.command == 'export':
            cli.export_results(
                result_id=args.result,
                filepath=args.output,
                format_type=args.format
            )
            
    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user")
    except Exception as e:
        print(f"[ERROR] Command failed: {e}")
        logger.error("CLI command failed", command=args.command, error=str(e))

def cli_entry_point():
    """Entry point for CLI from setuptools"""
    asyncio.run(main()) 