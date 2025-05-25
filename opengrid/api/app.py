"""Main FastAPI Application for OpenGrid

Provides REST API endpoints for power systems analysis and modeling.
Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import os
import structlog

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..modeling import PowerNetwork
from ..analysis import (
    LoadFlowAnalyzer, ShortCircuitAnalyzer, StabilityAnalyzer,
    HarmonicAnalyzer, ContingencyAnalyzer, OptimizationEngine
)
from ..ai import OpenAIAgent
from .models import *

logger = structlog.get_logger(__name__)

# Global state
app_state = {
    "networks": {},
    "analyzers": {},
    "ai_agent": None,
    "start_time": datetime.now()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting OpenGrid API server")
    
    # Initialize AI agent if API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            app_state["ai_agent"] = OpenAIAgent(api_key=openai_api_key)
            logger.info("AI agent initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize AI agent", error=str(e))
    else:
        logger.warning("No OpenAI API key provided - AI features disabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenGrid API server")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="OpenGrid API",
        description="AI-Powered Power Systems Analysis and Design Platform",
        version="0.2.0",
        contact={
            "name": "Nik Jois",
            "email": "nikjois@llamasearch.ai"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        },
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"}
        )
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """API root endpoint with system information"""
        uptime = (datetime.now() - app_state["start_time"]).total_seconds()
        
        return {
            "name": "OpenGrid API",
            "version": "0.2.0",
            "description": "AI-Powered Power Systems Analysis Platform",
            "author": "Nik Jois (nikjois@llamasearch.ai)",
            "uptime_seconds": uptime,
            "status": "healthy",
            "features": {
                "ai_enabled": app_state["ai_agent"] is not None,
                "active_networks": len(app_state["networks"]),
                "supported_analyses": [
                    "load_flow", "short_circuit", "stability", 
                    "harmonic", "contingency", "optimization"
                ]
            }
        }
    
    # Health check endpoint
    @app.get("/health", response_model=Dict[str, str])
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    # Network management endpoints
    @app.post("/networks", response_model=NetworkResponse)
    async def create_network(request: NetworkCreateRequest):
        """Create a new power network"""
        try:
            network_id = f"network_{len(app_state['networks']) + 1}"
            
            # Create network
            network = PowerNetwork(
                name=request.name,
                use_pypsa=request.use_pypsa
            )
            
            # Store network and initialize analyzers
            app_state["networks"][network_id] = network
            app_state["analyzers"][network_id] = {
                "load_flow": LoadFlowAnalyzer(network),
                "short_circuit": ShortCircuitAnalyzer(network),
                "stability": StabilityAnalyzer(network),
                "harmonic": HarmonicAnalyzer(network),
                "contingency": ContingencyAnalyzer(network),
                "optimization": OptimizationEngine(network)
            }
            
            logger.info("Network created", network_id=network_id, name=request.name)
            
            return NetworkResponse(
                network_id=network_id,
                name=request.name,
                created_at=datetime.now(),
                summary=network.get_summary(),
                status="created"
            )
            
        except Exception as e:
            logger.error("Network creation failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Network creation failed: {str(e)}")
    
    @app.get("/networks", response_model=List[NetworkResponse])
    async def list_networks():
        """List all available networks"""
        networks = []
        for network_id, network in app_state["networks"].items():
            networks.append(NetworkResponse(
                network_id=network_id,
                name=network.name,
                created_at=network.created_at,
                summary=network.get_summary(),
                status="active"
            ))
        return networks
    
    @app.get("/networks/{network_id}", response_model=NetworkResponse)
    async def get_network(network_id: str):
        """Get specific network details"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        network = app_state["networks"][network_id]
        return NetworkResponse(
            network_id=network_id,
            name=network.name,
            created_at=network.created_at,
            summary=network.get_summary(),
            status="active"
        )
    
    @app.delete("/networks/{network_id}")
    async def delete_network(network_id: str):
        """Delete a network"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        del app_state["networks"][network_id]
        del app_state["analyzers"][network_id]
        
        logger.info("Network deleted", network_id=network_id)
        return {"message": "Network deleted successfully"}
    
    # Component management endpoints
    @app.post("/networks/{network_id}/buses", response_model=ComponentResponse)
    async def add_bus(network_id: str, request: BusCreateRequest):
        """Add a bus to the network"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            network = app_state["networks"][network_id]
            bus_id = network.add_bus(
                vn_kv=request.vn_kv,
                name=request.name,
                zone=request.zone
            )
            
            return ComponentResponse(
                component_id=bus_id,
                component_type="bus",
                properties={"vn_kv": request.vn_kv, "name": request.name},
                status="created"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bus creation failed: {str(e)}")
    
    @app.post("/networks/{network_id}/lines", response_model=ComponentResponse)
    async def add_line(network_id: str, request: LineCreateRequest):
        """Add a line to the network"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            network = app_state["networks"][network_id]
            line_id = network.add_line(
                from_bus=request.from_bus,
                to_bus=request.to_bus,
                length_km=request.length_km,
                std_type=request.std_type,
                name=request.name
            )
            
            return ComponentResponse(
                component_id=line_id,
                component_type="line",
                properties={
                    "from_bus": request.from_bus,
                    "to_bus": request.to_bus,
                    "length_km": request.length_km
                },
                status="created"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Line creation failed: {str(e)}")
    
    @app.post("/networks/{network_id}/loads", response_model=ComponentResponse)
    async def add_load(network_id: str, request: LoadCreateRequest):
        """Add a load to the network"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            network = app_state["networks"][network_id]
            load_id = network.add_load(
                bus=request.bus,
                p_mw=request.p_mw,
                q_mvar=request.q_mvar,
                name=request.name
            )
            
            return ComponentResponse(
                component_id=load_id,
                component_type="load",
                properties={
                    "bus": request.bus,
                    "p_mw": request.p_mw,
                    "q_mvar": request.q_mvar
                },
                status="created"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Load creation failed: {str(e)}")
    
    @app.post("/networks/{network_id}/generators", response_model=ComponentResponse)
    async def add_generator(network_id: str, request: GeneratorCreateRequest):
        """Add a generator to the network"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            network = app_state["networks"][network_id]
            gen_id = network.add_generator(
                bus=request.bus,
                p_mw=request.p_mw,
                vm_pu=request.vm_pu,
                name=request.name
            )
            
            return ComponentResponse(
                component_id=gen_id,
                component_type="generator",
                properties={
                    "bus": request.bus,
                    "p_mw": request.p_mw,
                    "vm_pu": request.vm_pu
                },
                status="created"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generator creation failed: {str(e)}")
    
    # Analysis endpoints
    @app.post("/networks/{network_id}/analysis/load-flow", response_model=AnalysisResponse)
    async def run_load_flow(network_id: str, request: LoadFlowRequest):
        """Run load flow analysis"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            analyzer = app_state["analyzers"][network_id]["load_flow"]
            
            if request.algorithm == "newton_raphson":
                results = analyzer.run_newton_raphson(
                    tolerance_mva=request.tolerance_mva,
                    max_iteration=request.max_iteration
                )
            elif request.algorithm == "fast_decoupled":
                results = analyzer.run_fast_decoupled(
                    tolerance_mva=request.tolerance_mva,
                    max_iteration=request.max_iteration
                )
            elif request.algorithm == "dc_power_flow":
                results = analyzer.run_dc_power_flow()
            else:
                results = analyzer.run_newton_raphson()
            
            return AnalysisResponse(
                analysis_type="load_flow",
                network_id=network_id,
                results=results,
                timestamp=datetime.now(),
                status="completed" if results.get("converged", False) else "failed"
            )
            
        except Exception as e:
            logger.error("Load flow analysis failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Load flow analysis failed: {str(e)}")
    
    @app.post("/networks/{network_id}/analysis/short-circuit", response_model=AnalysisResponse)
    async def run_short_circuit(network_id: str, request: ShortCircuitRequest):
        """Run short circuit analysis"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            analyzer = app_state["analyzers"][network_id]["short_circuit"]
            
            results = analyzer.run_max_fault_current(
                bus=request.bus,
                fault_type=request.fault_type,
                case=request.case
            )
            
            return AnalysisResponse(
                analysis_type="short_circuit",
                network_id=network_id,
                results=results,
                timestamp=datetime.now(),
                status="completed" if results.get("converged", False) else "failed"
            )
            
        except Exception as e:
            logger.error("Short circuit analysis failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Short circuit analysis failed: {str(e)}")
    
    @app.post("/networks/{network_id}/analysis/contingency", response_model=AnalysisResponse)
    async def run_contingency(network_id: str, request: ContingencyRequest):
        """Run contingency analysis"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            analyzer = app_state["analyzers"][network_id]["contingency"]
            
            if request.analysis_type == "n_minus_1":
                results = analyzer.analyze_n_minus_1(request.component_types)
            elif request.analysis_type == "n_minus_k":
                results = analyzer.analyze_n_minus_k(request.k_value, request.component_types)
            elif request.analysis_type == "cascading":
                results = analyzer.analyze_cascading_failures()
            else:
                results = analyzer.analyze_n_minus_1()
            
            return AnalysisResponse(
                analysis_type="contingency",
                network_id=network_id,
                results=results,
                timestamp=datetime.now(),
                status="completed" if results.get("converged", False) else "failed"
            )
            
        except Exception as e:
            logger.error("Contingency analysis failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Contingency analysis failed: {str(e)}")
    
    # AI Analysis endpoints
    @app.post("/networks/{network_id}/ai-analysis", response_model=AIAnalysisResponse)
    async def run_ai_analysis(network_id: str, request: AIAnalysisRequest):
        """Run AI-powered analysis"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        if not app_state["ai_agent"]:
            raise HTTPException(status_code=503, detail="AI agent not available")
        
        try:
            ai_agent = app_state["ai_agent"]
            network = app_state["networks"][network_id]
            
            # Get latest analysis results
            analyzer = app_state["analyzers"][network_id][request.analysis_type]
            
            if request.analysis_type == "power_flow":
                # Run power flow if not available
                if "newton_raphson" not in analyzer.results:
                    analyzer.run_newton_raphson()
                
                results = analyzer.results["newton_raphson"]
                network_info = network.get_summary()
                
                ai_response = await ai_agent.analyze_power_flow(results, network_info)
                
            elif request.analysis_type == "contingency":
                # Run contingency if not available
                if "n_minus_1" not in analyzer.results:
                    analyzer.analyze_n_minus_1()
                
                results = analyzer.results["n_minus_1"]
                scenarios = []  # Would get from request or generate
                
                ai_response = await ai_agent.analyze_contingency(results, scenarios)
                
            else:
                raise HTTPException(status_code=400, detail=f"AI analysis not supported for {request.analysis_type}")
            
            return AIAnalysisResponse(
                analysis_type=request.analysis_type,
                network_id=network_id,
                summary=ai_response.summary,
                insights=ai_response.insights,
                recommendations=ai_response.recommendations,
                warnings=ai_response.warnings,
                confidence_score=ai_response.confidence_score,
                timestamp=datetime.now(),
                status="completed"
            )
            
        except Exception as e:
            logger.error("AI analysis failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
    
    # Optimization endpoints
    @app.post("/networks/{network_id}/optimization/economic-dispatch", response_model=OptimizationResponse)
    async def run_economic_dispatch(network_id: str, request: EconomicDispatchRequest):
        """Run economic dispatch optimization"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            optimizer = app_state["analyzers"][network_id]["optimization"]
            
            results = optimizer.solve_economic_dispatch(
                generators=request.generators,
                load_demand=request.load_demand,
                time_horizon=request.time_horizon
            )
            
            return OptimizationResponse(
                optimization_type="economic_dispatch",
                network_id=network_id,
                converged=results.converged,
                objective_value=results.objective_value,
                variables=results.variables,
                solution_time=results.solution_time,
                timestamp=datetime.now(),
                status="completed" if results.converged else "failed"
            )
            
        except Exception as e:
            logger.error("Economic dispatch failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Economic dispatch failed: {str(e)}")
    
    # Data export endpoints
    @app.get("/networks/{network_id}/export")
    async def export_network(network_id: str, format: str = "json"):
        """Export network data"""
        if network_id not in app_state["networks"]:
            raise HTTPException(status_code=404, detail="Network not found")
        
        try:
            network = app_state["networks"][network_id]
            exported_data = network.export_to_json()
            
            if format.lower() == "json":
                return JSONResponse(content=exported_data)
            else:
                raise HTTPException(status_code=400, detail="Unsupported format")
                
        except Exception as e:
            logger.error("Network export failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    
    return app

# Function to start the server (used by CLI)
def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    import uvicorn
    
    uvicorn.run(
        "opengrid.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload
    ) 