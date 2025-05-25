"""Pydantic Models for OpenGrid API

Defines request and response models for all API endpoints.
Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

# Enums for better validation
class AnalysisType(str, Enum):
    LOAD_FLOW = "load_flow"
    SHORT_CIRCUIT = "short_circuit"
    STABILITY = "stability"
    HARMONIC = "harmonic"
    CONTINGENCY = "contingency"
    OPTIMIZATION = "optimization"

class AlgorithmType(str, Enum):
    NEWTON_RAPHSON = "newton_raphson"
    FAST_DECOUPLED = "fast_decoupled"
    DC_POWER_FLOW = "dc_power_flow"
    OPTIMAL_POWER_FLOW = "optimal_power_flow"

class FaultType(str, Enum):
    THREE_PHASE = "3ph"
    LINE_TO_GROUND = "1ph"
    LINE_TO_LINE = "2ph"

class CaseType(str, Enum):
    MAX = "max"
    MIN = "min"

class ContingencyType(str, Enum):
    N_MINUS_1 = "n_minus_1"
    N_MINUS_K = "n_minus_k"
    CASCADING = "cascading"

# Network Models
class NetworkCreateRequest(BaseModel):
    """Request to create a new network"""
    name: str = Field(..., description="Network name")
    use_pypsa: bool = Field(False, description="Use PyPSA instead of pandapower")
    description: Optional[str] = Field(None, description="Network description")

class NetworkResponse(BaseModel):
    """Network information response"""
    network_id: str = Field(..., description="Unique network identifier")
    name: str = Field(..., description="Network name")
    created_at: datetime = Field(..., description="Creation timestamp")
    summary: Dict[str, Any] = Field(..., description="Network summary")
    status: str = Field(..., description="Network status")

# Component Models
class BusCreateRequest(BaseModel):
    """Request to create a bus"""
    vn_kv: float = Field(..., description="Nominal voltage in kV", gt=0)
    name: Optional[str] = Field(None, description="Bus name")
    zone: Optional[str] = Field(None, description="Bus zone")
    
    @validator('vn_kv')
    def validate_voltage(cls, v):
        if v <= 0 or v > 1000:  # Reasonable voltage range
            raise ValueError('Voltage must be between 0 and 1000 kV')
        return v

class LineCreateRequest(BaseModel):
    """Request to create a line"""
    from_bus: int = Field(..., description="From bus ID", ge=0)
    to_bus: int = Field(..., description="To bus ID", ge=0)
    length_km: float = Field(..., description="Line length in km", gt=0)
    std_type: str = Field("NAYY 4x50 SE", description="Standard line type")
    name: Optional[str] = Field(None, description="Line name")
    
    @validator('to_bus')
    def validate_different_buses(cls, v, values):
        if 'from_bus' in values and v == values['from_bus']:
            raise ValueError('from_bus and to_bus must be different')
        return v

class LoadCreateRequest(BaseModel):
    """Request to create a load"""
    bus: int = Field(..., description="Bus ID", ge=0)
    p_mw: float = Field(..., description="Active power in MW", gt=0)
    q_mvar: float = Field(0.0, description="Reactive power in MVAR")
    name: Optional[str] = Field(None, description="Load name")

class GeneratorCreateRequest(BaseModel):
    """Request to create a generator"""
    bus: int = Field(..., description="Bus ID", ge=0)
    p_mw: float = Field(..., description="Active power in MW", gt=0)
    vm_pu: float = Field(1.0, description="Voltage magnitude in per unit", gt=0, le=1.5)
    name: Optional[str] = Field(None, description="Generator name")

class ComponentResponse(BaseModel):
    """Component creation response"""
    component_id: int = Field(..., description="Component ID")
    component_type: str = Field(..., description="Component type")
    properties: Dict[str, Any] = Field(..., description="Component properties")
    status: str = Field(..., description="Creation status")

# Analysis Models
class LoadFlowRequest(BaseModel):
    """Load flow analysis request"""
    algorithm: AlgorithmType = Field(AlgorithmType.NEWTON_RAPHSON, description="Solution algorithm")
    tolerance_mva: float = Field(1e-6, description="Convergence tolerance in MVA", gt=0)
    max_iteration: int = Field(10, description="Maximum iterations", gt=0, le=100)
    calculate_voltage_angles: bool = Field(True, description="Calculate voltage angles")

class ShortCircuitRequest(BaseModel):
    """Short circuit analysis request"""
    bus: Optional[int] = Field(None, description="Specific bus ID for analysis")
    fault_type: FaultType = Field(FaultType.THREE_PHASE, description="Fault type")
    case: CaseType = Field(CaseType.MAX, description="Calculation case")

class ContingencyRequest(BaseModel):
    """Contingency analysis request"""
    analysis_type: ContingencyType = Field(ContingencyType.N_MINUS_1, description="Contingency type")
    component_types: List[str] = Field(["line", "transformer"], description="Component types to analyze")
    k_value: Optional[int] = Field(2, description="K value for N-k analysis", ge=2, le=5)

class StabilityRequest(BaseModel):
    """Stability analysis request"""
    analysis_type: str = Field("voltage_stability", description="Stability analysis type")
    disturbance_scenarios: Optional[List[Dict[str, Any]]] = Field(None, description="Disturbance scenarios")

class HarmonicRequest(BaseModel):
    """Harmonic analysis request"""
    harmonic_orders: List[int] = Field([3, 5, 7, 9, 11, 13], description="Harmonic orders to analyze")
    target_thd_limit: float = Field(5.0, description="Target THD limit in %", gt=0, le=20)

class AnalysisResponse(BaseModel):
    """Generic analysis response"""
    analysis_type: str = Field(..., description="Type of analysis")
    network_id: str = Field(..., description="Network ID")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    status: str = Field(..., description="Analysis status")

# AI Analysis Models
class AIAnalysisRequest(BaseModel):
    """AI analysis request"""
    analysis_type: str = Field(..., description="Type of analysis for AI interpretation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for AI")
    priority: str = Field("normal", description="Priority level")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ["power_flow", "contingency", "stability", "harmonic", "protection"]
        if v not in valid_types:
            raise ValueError(f'analysis_type must be one of {valid_types}')
        return v

class AIAnalysisResponse(BaseModel):
    """AI analysis response"""
    analysis_type: str = Field(..., description="Type of analysis")
    network_id: str = Field(..., description="Network ID")
    summary: str = Field(..., description="AI analysis summary")
    insights: List[str] = Field(..., description="Key insights")
    recommendations: List[str] = Field(..., description="Recommendations")
    warnings: List[str] = Field(..., description="Warnings")
    confidence_score: float = Field(..., description="AI confidence score", ge=0, le=1)
    timestamp: datetime = Field(..., description="Analysis timestamp")
    status: str = Field(..., description="Analysis status")

# Optimization Models
class EconomicDispatchRequest(BaseModel):
    """Economic dispatch optimization request"""
    generators: Dict[int, Dict[str, float]] = Field(..., description="Generator data")
    load_demand: float = Field(..., description="Total load demand in MW", gt=0)
    time_horizon: int = Field(1, description="Time horizon in hours", gt=0, le=168)
    
    @validator('generators')
    def validate_generators(cls, v):
        if not v:
            raise ValueError('At least one generator must be specified')
        for gen_id, gen_data in v.items():
            if 'max_power' not in gen_data:
                raise ValueError(f'Generator {gen_id} missing max_power')
            if gen_data['max_power'] <= 0:
                raise ValueError(f'Generator {gen_id} max_power must be positive')
        return v

class OptimalPowerFlowRequest(BaseModel):
    """Optimal power flow request"""
    include_voltage_constraints: bool = Field(True, description="Include voltage constraints")
    include_thermal_constraints: bool = Field(True, description="Include thermal constraints")
    objective: str = Field("cost", description="Optimization objective")

class UnitCommitmentRequest(BaseModel):
    """Unit commitment optimization request"""
    time_horizon: int = Field(24, description="Time horizon in hours", gt=0, le=168)
    load_forecast: Optional[List[float]] = Field(None, description="Hourly load forecast")
    reserve_margin: float = Field(0.15, description="Reserve margin", ge=0, le=0.5)

class TransmissionExpansionRequest(BaseModel):
    """Transmission expansion planning request"""
    candidate_lines: List[Dict[str, Any]] = Field(..., description="Candidate transmission lines")
    planning_horizon: int = Field(10, description="Planning horizon in years", gt=0, le=50)
    demand_growth: float = Field(0.03, description="Annual demand growth rate", ge=0, le=0.2)

class RenewableIntegrationRequest(BaseModel):
    """Renewable integration optimization request"""
    renewable_sites: List[Dict[str, Any]] = Field(..., description="Renewable energy sites")
    storage_options: Optional[List[Dict[str, Any]]] = Field(None, description="Storage options")
    integration_target: float = Field(0.3, description="Renewable integration target", ge=0, le=1)

class OptimizationResponse(BaseModel):
    """Optimization result response"""
    optimization_type: str = Field(..., description="Type of optimization")
    network_id: str = Field(..., description="Network ID")
    converged: bool = Field(..., description="Convergence status")
    objective_value: float = Field(..., description="Objective function value")
    variables: Dict[str, float] = Field(..., description="Solution variables")
    solution_time: float = Field(..., description="Solution time in seconds")
    timestamp: datetime = Field(..., description="Optimization timestamp")
    status: str = Field(..., description="Optimization status")

# Data Models
class DataExportRequest(BaseModel):
    """Data export request"""
    format: str = Field("json", description="Export format")
    include_results: bool = Field(True, description="Include analysis results")
    include_metadata: bool = Field(True, description="Include metadata")

class DataImportRequest(BaseModel):
    """Data import request"""
    format: str = Field("json", description="Import format")
    data: Dict[str, Any] = Field(..., description="Data to import")
    overwrite: bool = Field(False, description="Overwrite existing data")

# Mock Data Models
class MockNetworkRequest(BaseModel):
    """Request to generate mock network data"""
    network_type: str = Field("distribution", description="Network type")
    size: str = Field("medium", description="Network size")
    complexity: str = Field("medium", description="Network complexity")
    include_renewables: bool = Field(True, description="Include renewable sources")
    include_storage: bool = Field(False, description="Include energy storage")
    
    @validator('network_type')
    def validate_network_type(cls, v):
        valid_types = ["transmission", "distribution", "microgrid", "industrial"]
        if v not in valid_types:
            raise ValueError(f'network_type must be one of {valid_types}')
        return v
    
    @validator('size')
    def validate_size(cls, v):
        valid_sizes = ["small", "medium", "large"]
        if v not in valid_sizes:
            raise ValueError(f'size must be one of {valid_sizes}')
        return v

class MockDataResponse(BaseModel):
    """Mock data generation response"""
    network_id: str = Field(..., description="Generated network ID")
    components_created: Dict[str, int] = Field(..., description="Count of created components")
    network_summary: Dict[str, Any] = Field(..., description="Network summary")
    generation_time: float = Field(..., description="Generation time in seconds")
    status: str = Field(..., description="Generation status")

# Batch Processing Models
class BatchAnalysisRequest(BaseModel):
    """Batch analysis request"""
    network_ids: List[str] = Field(..., description="List of network IDs")
    analysis_types: List[str] = Field(..., description="Types of analysis to run")
    parallel: bool = Field(True, description="Run analyses in parallel")
    priority: str = Field("normal", description="Batch priority")

class BatchAnalysisResponse(BaseModel):
    """Batch analysis response"""
    batch_id: str = Field(..., description="Batch job ID")
    network_count: int = Field(..., description="Number of networks")
    analysis_count: int = Field(..., description="Number of analyses")
    status: str = Field(..., description="Batch status")
    progress: float = Field(0.0, description="Progress percentage", ge=0, le=100)
    results: List[AnalysisResponse] = Field([], description="Completed analysis results")

# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Status Models
class SystemStatus(BaseModel):
    """System status response"""
    status: str = Field(..., description="System status")
    uptime_seconds: float = Field(..., description="System uptime")
    active_networks: int = Field(..., description="Number of active networks")
    active_analyses: int = Field(..., description="Number of running analyses")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    ai_status: str = Field(..., description="AI service status")

# WebSocket Models
class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

class ProgressUpdate(BaseModel):
    """Progress update message"""
    job_id: str = Field(..., description="Job ID")
    progress: float = Field(..., description="Progress percentage", ge=0, le=100)
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")
    eta_seconds: Optional[float] = Field(None, description="Estimated time to completion")

# Validation utilities
def validate_positive_float(v: float, field_name: str) -> float:
    """Validate positive float value"""
    if v <= 0:
        raise ValueError(f'{field_name} must be positive')
    return v

def validate_percentage(v: float, field_name: str) -> float:
    """Validate percentage value (0-100)"""
    if v < 0 or v > 100:
        raise ValueError(f'{field_name} must be between 0 and 100')
    return v

def validate_per_unit(v: float, field_name: str) -> float:
    """Validate per unit value (0-2)"""
    if v < 0 or v > 2:
        raise ValueError(f'{field_name} must be between 0 and 2 per unit')
    return v 