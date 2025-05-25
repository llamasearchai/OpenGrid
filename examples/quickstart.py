#!/usr/bin/env python3
"""
OpenGrid Quickstart Example
Demonstrates the main features of the OpenGrid power systems analysis platform.

This example shows how to:
1. Load a sample network
2. Run various types of analysis
3. Use AI-powered interpretation
4. Export results

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import asyncio
import json
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from opengrid.modeling import PowerNetwork
from opengrid.analysis import (
    LoadFlowAnalyzer, ShortCircuitAnalyzer, StabilityAnalyzer,
    HarmonicAnalyzer, ContingencyAnalyzer, OptimizationEngine
)
from opengrid.ai import OpenAIAgent
from opengrid.data import sample_networks, sample_cases


def print_banner():
    """Print welcome banner."""
    banner = """
    ========================================
    OpenGrid Quickstart Example
    ========================================
    
    This example demonstrates the main features of OpenGrid:
    • Loading sample networks
    • Running power system analysis  
    • AI-powered result interpretation
    • Exporting results
    ========================================
    """
    print(banner)


def load_sample_network(network_name: str = "ieee_14_bus") -> PowerNetwork:
    """Load a sample network for analysis."""
    print(f"\n[NETWORK] Loading sample network: {network_name}")
    
    # Get network data
    network_data = sample_networks.get_network(network_name)
    
    # Create PowerNetwork instance
    network = PowerNetwork(name=f"Sample {network_name}")
    network.pandapower_net = network_data["pandapower"]
    
    # Display network summary
    summary = network.get_summary()
    print(f"[SUCCESS] Network loaded successfully!")
    print(f"   Buses: {summary.get('total_buses', 0)}")
    print(f"   Lines: {summary.get('total_lines', 0)}")
    print(f"   Generators: {summary.get('total_generators', 0)}")
    print(f"   Loads: {summary.get('total_loads', 0)}")
    
    return network


def run_load_flow_analysis(network: PowerNetwork) -> dict:
    """Demonstrate load flow analysis."""
    print("\n[ANALYSIS] Running Load Flow Analysis")
    print("-" * 40)
    
    analyzer = LoadFlowAnalyzer(network)
    
    # Run Newton-Raphson load flow
    results = analyzer.run_newton_raphson(
        tolerance_mva=1e-6,
        max_iteration=50
    )
    
    # Display results
    if results.get("converged", False):
        print("[SUCCESS] Load flow converged successfully!")
        print(f"   Iterations: {results.get('iteration_count', 0)}")
        print(f"   Total losses: {results.get('total_losses_mw', 0):.2f} MW")
        
        # Show voltage profile
        voltages = results.get('bus_voltages', {})
        if voltages:
            min_v = min(voltages.values())
            max_v = max(voltages.values())
            print(f"   Voltage range: {min_v:.3f} - {max_v:.3f} pu")
    else:
        print("[ERROR] Load flow failed to converge")
    
    return results


def run_short_circuit_analysis(network: PowerNetwork) -> dict:
    """Demonstrate short circuit analysis."""
    print("\n[ANALYSIS] Running Short Circuit Analysis")
    print("-" * 40)
    
    analyzer = ShortCircuitAnalyzer(network)
    
    # Run three-phase fault analysis
    results = analyzer.run_max_fault_current(
        bus=None,  # All buses
        fault_type="3ph",
        case="max"
    )
    
    # Display results
    if results.get("converged", False):
        print("[SUCCESS] Short circuit analysis completed!")
        fault_currents = results.get('fault_currents', {})
        if fault_currents:
            max_current = max(fault_currents.values())
            print(f"   Maximum fault current: {max_current:.2f} kA")
            print(f"   Fault currents calculated for {len(fault_currents)} buses")
    else:
        print("[ERROR] Short circuit analysis failed")
    
    return results


def run_contingency_analysis(network: PowerNetwork) -> dict:
    """Demonstrate contingency analysis."""
    print("\n[ANALYSIS] Running Contingency Analysis")
    print("-" * 40)
    
    analyzer = ContingencyAnalyzer(network)
    
    # Run N-1 contingency analysis
    results = analyzer.analyze_n_minus_1(element_types=["line"])
    
    # Display results
    total_contingencies = results.get('total_contingencies', 0)
    violations = results.get('violations', [])
    critical_contingencies = results.get('critical_contingencies', [])
    
    print(f"[SUCCESS] Contingency analysis completed!")
    print(f"   Total contingencies analyzed: {total_contingencies}")
    print(f"   Violations found: {len(violations)}")
    print(f"   Critical contingencies: {len(critical_contingencies)}")
    
    if critical_contingencies:
        print("   [WARNING] Critical contingencies detected!")
        for contingency in critical_contingencies[:3]:  # Show first 3
            print(f"      - {contingency}")
    
    return results


async def run_ai_analysis(network: PowerNetwork, analysis_results: dict, 
                         analysis_type: str) -> dict:
    """Demonstrate AI-powered analysis interpretation."""
    print(f"\n[AI] Running AI Analysis for {analysis_type}")
    print("-" * 40)
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OpenAI API key not found. Skipping AI analysis.")
        print("   Set OPENAI_API_KEY environment variable to enable AI features.")
        return {}
    
    try:
        # Initialize AI agent
        ai_agent = OpenAIAgent(api_key=api_key)
        
        # Run AI analysis based on type
        if analysis_type == "load_flow":
            network_info = network.get_summary()
            ai_response = await ai_agent.analyze_power_flow(analysis_results, network_info)
        elif analysis_type == "contingency":
            scenarios = []  # Could include specific scenarios
            ai_response = await ai_agent.analyze_contingency(analysis_results, scenarios)
        else:
            print(f"   AI analysis not implemented for {analysis_type}")
            return {}
        
        # Display AI insights
        print("[SUCCESS] AI analysis completed!")
        print(f"   Confidence score: {ai_response.confidence_score:.2f}")
        print(f"\n[SUMMARY]")
        print(f"   {ai_response.summary}")
        
        if ai_response.insights:
            print(f"\n[INSIGHTS] Key Insights:")
            for i, insight in enumerate(ai_response.insights[:3], 1):
                print(f"   {i}. {insight}")
        
        if ai_response.recommendations:
            print(f"\n[RECOMMENDATIONS]")
            for i, rec in enumerate(ai_response.recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        return {
            "summary": ai_response.summary,
            "insights": ai_response.insights,
            "recommendations": ai_response.recommendations,
            "confidence_score": ai_response.confidence_score
        }
        
    except Exception as e:
        print(f"[ERROR] AI analysis failed: {e}")
        return {}


def run_case_study_example():
    """Demonstrate running a predefined case study."""
    print("\n[CASE STUDY] Running Case Study Example")
    print("-" * 40)
    
    # Get a sample case
    case = sample_cases.get_case("lf_ieee14_basic")
    
    print(f"Case: {case.name}")
    print(f"Description: {case.description}")
    print(f"Network: {case.network_name}")
    print(f"Analysis type: {case.analysis_type}")
    print(f"Difficulty: {case.difficulty}")
    print(f"Estimated runtime: {case.estimated_runtime_seconds:.1f} seconds")
    
    # Show parameters
    print("\nParameters:")
    for key, value in case.parameters.items():
        print(f"   {key}: {value}")
    
    # Show expected results if available
    if case.expected_results:
        print("\nExpected results:")
        for key, value in case.expected_results.items():
            print(f"   {key}: {value}")


def export_results(results: dict, filename: str):
    """Export analysis results to JSON file."""
    print(f"\n[EXPORT] Exporting results to {filename}")
    
    try:
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[SUCCESS] Results exported successfully to {filepath}")
        
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")


async def main():
    """Main demonstration function."""
    print_banner()
    
    # 1. Load sample network
    network = load_sample_network("ieee_14_bus")
    
    # 2. Run various analyses
    print("\n" + "="*60)
    print("POWER SYSTEM ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Load flow analysis
    lf_results = run_load_flow_analysis(network)
    
    # Short circuit analysis
    sc_results = run_short_circuit_analysis(network)
    
    # Contingency analysis
    cont_results = run_contingency_analysis(network)
    
    # 3. AI Analysis (if API key available)
    print("\n" + "="*60)
    print("AI-POWERED ANALYSIS DEMONSTRATION")
    print("="*60)
    
    ai_lf_results = await run_ai_analysis(network, lf_results, "load_flow")
    ai_cont_results = await run_ai_analysis(network, cont_results, "contingency")
    
    # 4. Case study example
    print("\n" + "="*60)
    print("CASE STUDY DEMONSTRATION")
    print("="*60)
    
    run_case_study_example()
    
    # 5. Export results
    print("\n" + "="*60)
    print("RESULTS EXPORT DEMONSTRATION")
    print("="*60)
    
    # Combine all results
    all_results = {
        "network_summary": network.get_summary(),
        "load_flow": lf_results,
        "short_circuit": sc_results,
        "contingency": cont_results,
        "ai_load_flow": ai_lf_results,
        "ai_contingency": ai_cont_results,
        "timestamp": "2024-12-01T00:00:00Z"
    }
    
    export_results(all_results, "quickstart_results.json")
    
    # Final summary
    print("\n" + "="*60)
    print("QUICKSTART COMPLETE")
    print("="*60)
    print("[SUCCESS] Successfully demonstrated:")
    print("   • Network loading and modeling")
    print("   • Load flow analysis")
    print("   • Short circuit analysis") 
    print("   • Contingency analysis")
    if ai_lf_results:
        print("   • AI-powered analysis interpretation")
    print("   • Case study framework")
    print("   • Results export")
    print("\n[NEXT STEPS]")
    print("   • Explore more analysis types")
    print("   • Try different sample networks")
    print("   • Run comprehensive case studies")
    print("   • Use the CLI interface: python main.py --help")
    print("   • Start the API server: python main.py")
    print("\n[DOCUMENTATION] Available at /docs when API server is running")
    print("[REPOSITORY] https://github.com/llamasearchai/OpenGrid")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 