"""Prompt Templates for OpenAI Power Systems Analysis

Contains specialized prompts for different types of power system analysis.
Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

from typing import Dict, Any, List
import json

class PromptTemplates:
    """Template manager for AI prompts in power systems analysis"""
    
    SYSTEM_PROMPT = """You are an expert power systems engineer with deep knowledge of electrical grid analysis, 
    power flow studies, protection systems, stability analysis, and power quality. You provide clear, actionable 
    insights and recommendations based on power system analysis results. Your responses should be technically 
    accurate, practical, and prioritize safety and reliability.

    Always structure your responses with:
    1. Executive Summary
    2. Key Insights
    3. Recommendations
    4. Warnings/Concerns
    5. Next Steps

    Use engineering judgment and industry standards (IEEE, IEC, NERC) in your analysis."""

    def get_power_flow_prompt(self, results: Dict[str, Any], network_info: Dict[str, Any]) -> str:
        """Generate prompt for power flow analysis"""
        return f"""
        Analyze the following power flow study results for a {network_info.get('name', 'power system')}:

        NETWORK INFORMATION:
        - Network Name: {network_info.get('name', 'N/A')}
        - Voltage Levels: {network_info.get('voltage_levels', 'N/A')}
        - Total Buses: {network_info.get('components', {}).get('buses', 0)}
        - Total Lines: {network_info.get('components', {}).get('lines', 0)}
        - Total Load: {network_info.get('total_load_mw', 0):.1f} MW
        - Total Generation: {network_info.get('total_generation_mw', 0):.1f} MW

        POWER FLOW RESULTS:
        - Converged: {results.get('converged', False)}
        - Algorithm: {results.get('algorithm', 'N/A')}
        - Total Losses: {results.get('total_losses_mw', 0):.2f} MW
        - Bus Voltages (pu): Min={min(results.get('bus_voltages', {1.0: 1.0}).values()):.3f}, Max={max(results.get('bus_voltages', {1.0: 1.0}).values()):.3f}
        - Line Loading (%): Max={max(results.get('line_loading', {1: 0}).values()):.1f}%

        DETAILED DATA:
        {json.dumps({
            'bus_voltages': dict(list(results.get('bus_voltages', {}).items())[:10]),
            'line_loading': dict(list(results.get('line_loading', {}).items())[:10]),
            'losses_by_line': dict(list(results.get('line_losses', {}).items())[:5])
        }, indent=2)}

        Provide a comprehensive analysis including:
        1. Overall system health assessment
        2. Voltage profile analysis and any violations
        3. Loading conditions and thermal concerns
        4. Power losses evaluation
        5. Operational recommendations
        6. Any reliability concerns
        """

    def get_contingency_prompt(self, results: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> str:
        """Generate prompt for contingency analysis"""
        return f"""
        Analyze the following contingency analysis results:

        ANALYSIS OVERVIEW:
        - Analysis Type: {results.get('analysis_type', 'N/A')}
        - Total Contingencies: {results.get('total_contingencies', 0)}
        - Violations Found: {len(results.get('violations', []))}
        - Critical Contingencies: {len(results.get('critical_contingencies', []))}
        - Pass Rate: {results.get('system_reliability', {}).get('contingency_pass_rate_percent', 0):.1f}%

        RELIABILITY METRICS:
        {json.dumps(results.get('system_reliability', {}), indent=2)}

        CRITICAL CONTINGENCIES:
        {json.dumps(results.get('critical_contingencies', [])[:5], indent=2)}

        VIOLATIONS SUMMARY:
        {json.dumps(results.get('violations', [])[:5], indent=2)}

        SCENARIOS ANALYZED:
        {json.dumps(scenarios[:10], indent=2)}

        Provide analysis covering:
        1. System reliability assessment
        2. Most critical contingencies and their impacts
        3. Voltage and thermal violations analysis
        4. Recommendations for system reinforcement
        5. Protection and operational strategies
        6. Risk prioritization matrix
        """

    def get_stability_prompt(self, results: Dict[str, Any], disturbances: List[Dict[str, Any]]) -> str:
        """Generate prompt for stability analysis"""
        return f"""
        Analyze the following power system stability study results:

        STABILITY ANALYSIS:
        - Analysis Type: {results.get('analysis_type', 'N/A')}
        - Overall Assessment: {results.get('stability_assessment', 'N/A')}
        - Converged: {results.get('converged', False)}

        VOLTAGE STABILITY:
        - Critical Buses: {results.get('critical_buses', [])}
        - Voltage Collapse Point: {results.get('voltage_collapse_point', 'N/A')}
        - Load Margin: {results.get('stability_margins', {}).get('load_margin_percent', 'N/A')}%

        TRANSIENT STABILITY:
        - Stable Scenarios: {results.get('stability_statistics', {}).get('stable_scenarios', 0)}
        - Total Scenarios: {results.get('stability_statistics', {}).get('total_scenarios', 0)}
        - Critical Clearing Times: {json.dumps(results.get('critical_clearing_times', {}), indent=2)}

        SMALL SIGNAL STABILITY:
        - Eigenvalues Count: {len(results.get('eigenvalues', []))}
        - Oscillatory Modes: {len(results.get('oscillatory_modes', []))}
        - Problematic Modes: {len(results.get('problematic_modes', []))}

        DISTURBANCE SCENARIOS:
        {json.dumps(disturbances[:5], indent=2)}

        Provide comprehensive stability analysis including:
        1. Overall system stability assessment
        2. Voltage stability margins and critical areas
        3. Transient stability performance
        4. Small signal stability and oscillatory behavior
        5. Recommended stability improvements
        6. Special protection scheme requirements
        """

    def get_harmonic_prompt(self, results: Dict[str, Any], standards: Dict[str, Any]) -> str:
        """Generate prompt for harmonic analysis"""
        return f"""
        Analyze the following harmonic distortion study results:

        HARMONIC ANALYSIS OVERVIEW:
        - Analysis Type: {results.get('analysis_type', 'N/A')}
        - Converged: {results.get('converged', False)}
        - Harmonic Orders Analyzed: {results.get('harmonic_orders', [])}

        THD ANALYSIS:
        - Max Voltage THD: {results.get('thd_summary', {}).get('max_voltage_thd_percent', 0):.2f}%
        - Max Current THD: {results.get('thd_summary', {}).get('max_current_thd_percent', 0):.2f}%
        - Buses Exceeding Voltage Limits: {len(results.get('thd_summary', {}).get('buses_exceeding_voltage_limit', []))}
        - Buses Exceeding Current Limits: {len(results.get('thd_summary', {}).get('buses_exceeding_current_limit', []))}

        RESONANCE ANALYSIS:
        - Resonance Frequencies: {len(results.get('resonance_frequencies', []))}
        - Critical Frequencies: {len(results.get('critical_frequencies', []))}
        - Resonance Risk: {results.get('resonance_summary', {}).get('resonance_risk_assessment', 'N/A')}

        FILTER REQUIREMENTS:
        - Buses Requiring Filters: {len(results.get('filter_recommendations', {}))}
        - Estimated Total Cost: ${results.get('cost_analysis', {}).get('total_cost_usd', 0):,.0f}

        STANDARDS COMPLIANCE:
        {json.dumps(standards, indent=2)}

        DETAILED HARMONIC DATA:
        {json.dumps({
            'system_harmonics': results.get('system_harmonics', {}),
            'filter_recommendations': dict(list(results.get('filter_recommendations', {}).items())[:3])
        }, indent=2)}

        Provide harmonic analysis including:
        1. Power quality assessment and IEEE 519 compliance
        2. Harmonic source identification and impact
        3. Resonance conditions and risks
        4. Filter requirements and recommendations
        5. Cost-benefit analysis for mitigation
        6. Operational guidelines for harmonic control
        """

    def get_protection_prompt(self, results: Dict[str, Any], settings: Dict[str, Any]) -> str:
        """Generate prompt for protection coordination analysis"""
        return f"""
        Analyze the following protection coordination study results:

        PROTECTION ANALYSIS:
        - Analysis Type: {results.get('analysis_type', 'N/A')}
        - Converged: {results.get('converged', False)}
        - Total Buses Analyzed: {len(results.get('protection_recommendations', {}))}
        - Coordination Issues: {len(results.get('coordination_issues', []))}

        FAULT CURRENT DATA:
        - Max Fault Current: {results.get('fault_analysis', {}).get('max_fault_current_ka', 0):.1f} kA
        - Min Fault Current: {results.get('fault_analysis', {}).get('min_fault_current_ka', 0):.1f} kA
        - Critical Buses: {results.get('fault_analysis', {}).get('critical_buses', [])}

        ARC FLASH ANALYSIS:
        - High Hazard Buses: {len(results.get('summary', {}).get('high_hazard_buses', []))}
        - Max Arc Energy: {results.get('summary', {}).get('max_arc_energy_cal_cm2', 0):.1f} cal/cm²
        - Working Distance: {results.get('working_distance_m', 0.61):.2f} m

        PROTECTION RECOMMENDATIONS:
        {json.dumps(dict(list(results.get('protection_recommendations', {}).items())[:5]), indent=2)}

        COORDINATION ISSUES:
        {json.dumps(results.get('coordination_issues', [])[:5], indent=2)}

        PROTECTION SETTINGS:
        {json.dumps(settings, indent=2)}

        Provide protection analysis including:
        1. Overall protection coordination assessment
        2. Fault current adequacy and interrupting capacity
        3. Arc flash hazard evaluation and PPE requirements
        4. Protection device recommendations and settings
        5. Coordination issues and resolution strategies
        6. Safety procedures and operational guidelines
        """

    def get_optimization_prompt(self, current_state: Dict[str, Any], objectives: List[str], constraints: Dict[str, Any]) -> str:
        """Generate prompt for optimization recommendations"""
        return f"""
        Provide optimization recommendations for the following power system:

        CURRENT SYSTEM STATE:
        {json.dumps(current_state, indent=2)}

        OPTIMIZATION OBJECTIVES:
        {json.dumps(objectives, indent=2)}

        SYSTEM CONSTRAINTS:
        {json.dumps(constraints, indent=2)}

        Provide optimization recommendations including:
        1. Priority ranking of optimization opportunities
        2. Economic analysis and cost-benefit evaluation
        3. Technical feasibility assessment
        4. Implementation timeline and phases
        5. Risk assessment and mitigation strategies
        6. Performance metrics and monitoring requirements
        7. Regulatory and compliance considerations
        8. Long-term strategic planning recommendations
        """

    def get_emergency_response_prompt(self, incident_data: Dict[str, Any], system_status: Dict[str, Any]) -> str:
        """Generate prompt for emergency response analysis"""
        return f"""
        Analyze the following power system emergency situation and provide immediate response recommendations:

        INCIDENT DETAILS:
        {json.dumps(incident_data, indent=2)}

        CURRENT SYSTEM STATUS:
        {json.dumps(system_status, indent=2)}

        Provide emergency response analysis including:
        1. Immediate actions required (next 5-15 minutes)
        2. System isolation and switching procedures
        3. Load shedding priorities and procedures
        4. Safety considerations and personnel protection
        5. Communication requirements and stakeholder notifications
        6. System restoration planning and sequencing
        7. Lessons learned and preventive measures
        """

    def get_maintenance_planning_prompt(self, equipment_data: Dict[str, Any], outage_constraints: Dict[str, Any]) -> str:
        """Generate prompt for maintenance planning optimization"""
        return f"""
        Optimize the maintenance planning for the following power system equipment:

        EQUIPMENT DATA:
        {json.dumps(equipment_data, indent=2)}

        OUTAGE CONSTRAINTS:
        {json.dumps(outage_constraints, indent=2)}

        Provide maintenance optimization recommendations including:
        1. Prioritized maintenance schedule
        2. Equipment condition assessment
        3. Risk-based maintenance strategies
        4. Outage coordination and system impact analysis
        5. Resource allocation and crew scheduling
        6. Spare parts inventory optimization
        7. Predictive maintenance opportunities
        8. Cost optimization strategies
        """

    def get_renewable_integration_prompt(self, renewable_data: Dict[str, Any], grid_impact: Dict[str, Any]) -> str:
        """Generate prompt for renewable energy integration analysis"""
        return f"""
        Analyze renewable energy integration impacts and provide recommendations:

        RENEWABLE ENERGY DATA:
        {json.dumps(renewable_data, indent=2)}

        GRID IMPACT ANALYSIS:
        {json.dumps(grid_impact, indent=2)}

        Provide renewable integration analysis including:
        1. Grid stability and power quality impacts
        2. Voltage regulation and reactive power requirements
        3. Protection system modifications needed
        4. Energy storage requirements and sizing
        5. Grid code compliance assessment
        6. Forecasting and dispatch optimization
        7. Transmission and distribution upgrades
        8. Economic and environmental benefits analysis
        """ 