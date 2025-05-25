"""OpenAI Agent for Power Systems Analysis

Provides AI-powered insights, recommendations, and analysis interpretation.
Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from openai import AsyncOpenAI
import structlog
from .prompt_templates import PromptTemplates
from .analysis_interpreter import AnalysisInterpreter

logger = structlog.get_logger(__name__)

@dataclass
class AIAnalysisRequest:
    """Request for AI analysis of power system data"""
    analysis_type: str
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # low, normal, high, critical
    
@dataclass
class AIAnalysisResponse:
    """Response from AI analysis"""
    summary: str
    insights: List[str]
    recommendations: List[str]
    warnings: List[str]
    confidence_score: float
    metadata: Dict[str, Any]

class OpenAIAgent:
    """AI agent for power systems analysis using OpenAI"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ):
        """Initialize OpenAI agent
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            max_tokens: Maximum tokens per response
            temperature: Model temperature (0-1)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_templates = PromptTemplates()
        self.interpreter = AnalysisInterpreter()
        
        logger.info("OpenAI agent initialized", model=model)
        
    async def analyze_power_flow(
        self,
        results: Dict[str, Any],
        network_info: Dict[str, Any]
    ) -> AIAnalysisResponse:
        """Analyze power flow results using AI
        
        Args:
            results: Power flow analysis results
            network_info: Network configuration information
            
        Returns:
            AI analysis response
        """
        try:
            prompt = self.prompt_templates.get_power_flow_prompt(results, network_info)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            parsed_response = self._parse_ai_response(content)
            
            logger.info("Power flow analysis completed", 
                       confidence=parsed_response.confidence_score)
            
            return parsed_response
            
        except Exception as e:
            logger.error("Error in power flow analysis", error=str(e))
            return self._create_error_response(str(e))
    
    async def analyze_contingency(
        self,
        results: Dict[str, Any],
        scenarios: List[Dict[str, Any]]
    ) -> AIAnalysisResponse:
        """Analyze contingency analysis results
        
        Args:
            results: Contingency analysis results
            scenarios: Contingency scenarios analyzed
            
        Returns:
            AI analysis response
        """
        try:
            prompt = self.prompt_templates.get_contingency_prompt(results, scenarios)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            parsed_response = self._parse_ai_response(content)
            
            logger.info("Contingency analysis completed",
                       confidence=parsed_response.confidence_score)
            
            return parsed_response
            
        except Exception as e:
            logger.error("Error in contingency analysis", error=str(e))
            return self._create_error_response(str(e))
    
    async def analyze_stability(
        self,
        results: Dict[str, Any],
        disturbances: List[Dict[str, Any]]
    ) -> AIAnalysisResponse:
        """Analyze stability analysis results
        
        Args:
            results: Stability analysis results
            disturbances: Disturbance scenarios
            
        Returns:
            AI analysis response
        """
        try:
            prompt = self.prompt_templates.get_stability_prompt(results, disturbances)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            parsed_response = self._parse_ai_response(content)
            
            logger.info("Stability analysis completed",
                       confidence=parsed_response.confidence_score)
            
            return parsed_response
            
        except Exception as e:
            logger.error("Error in stability analysis", error=str(e))
            return self._create_error_response(str(e))
    
    async def analyze_harmonics(
        self,
        results: Dict[str, Any],
        standards: Dict[str, Any]
    ) -> AIAnalysisResponse:
        """Analyze harmonic analysis results
        
        Args:
            results: Harmonic analysis results
            standards: Applicable standards (IEEE 519, etc.)
            
        Returns:
            AI analysis response
        """
        try:
            prompt = self.prompt_templates.get_harmonic_prompt(results, standards)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            parsed_response = self._parse_ai_response(content)
            
            logger.info("Harmonic analysis completed",
                       confidence=parsed_response.confidence_score)
            
            return parsed_response
            
        except Exception as e:
            logger.error("Error in harmonic analysis", error=str(e))
            return self._create_error_response(str(e))
    
    async def analyze_protection(
        self,
        results: Dict[str, Any],
        settings: Dict[str, Any]
    ) -> AIAnalysisResponse:
        """Analyze protection coordination results
        
        Args:
            results: Protection analysis results
            settings: Protection device settings
            
        Returns:
            AI analysis response
        """
        try:
            prompt = self.prompt_templates.get_protection_prompt(results, settings)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            parsed_response = self._parse_ai_response(content)
            
            logger.info("Protection analysis completed",
                       confidence=parsed_response.confidence_score)
            
            return parsed_response
            
        except Exception as e:
            logger.error("Error in protection analysis", error=str(e))
            return self._create_error_response(str(e))
    
    async def get_optimization_recommendations(
        self,
        current_state: Dict[str, Any],
        objectives: List[str],
        constraints: Dict[str, Any]
    ) -> AIAnalysisResponse:
        """Get AI recommendations for system optimization
        
        Args:
            current_state: Current system state
            objectives: Optimization objectives
            constraints: System constraints
            
        Returns:
            AI optimization recommendations
        """
        try:
            prompt = self.prompt_templates.get_optimization_prompt(
                current_state, objectives, constraints
            )
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            parsed_response = self._parse_ai_response(content)
            
            logger.info("Optimization recommendations generated",
                       confidence=parsed_response.confidence_score)
            
            return parsed_response
            
        except Exception as e:
            logger.error("Error generating optimization recommendations", error=str(e))
            return self._create_error_response(str(e))
    
    async def batch_analyze(
        self,
        requests: List[AIAnalysisRequest]
    ) -> List[AIAnalysisResponse]:
        """Perform batch analysis of multiple requests
        
        Args:
            requests: List of analysis requests
            
        Returns:
            List of AI analysis responses
        """
        try:
            # Sort by priority
            priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
            sorted_requests = sorted(
                requests, 
                key=lambda x: priority_order.get(x.priority, 3)
            )
            
            # Process in parallel with rate limiting
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
            
            async def process_request(request: AIAnalysisRequest) -> AIAnalysisResponse:
                async with semaphore:
                    if request.analysis_type == "power_flow":
                        return await self.analyze_power_flow(
                            request.data, 
                            request.context or {}
                        )
                    elif request.analysis_type == "contingency":
                        return await self.analyze_contingency(
                            request.data,
                            request.context.get("scenarios", [])
                        )
                    elif request.analysis_type == "stability":
                        return await self.analyze_stability(
                            request.data,
                            request.context.get("disturbances", [])
                        )
                    elif request.analysis_type == "harmonics":
                        return await self.analyze_harmonics(
                            request.data,
                            request.context.get("standards", {})
                        )
                    elif request.analysis_type == "protection":
                        return await self.analyze_protection(
                            request.data,
                            request.context.get("settings", {})
                        )
                    else:
                        return self._create_error_response(
                            f"Unknown analysis type: {request.analysis_type}"
                        )
            
            responses = await asyncio.gather(
                *[process_request(req) for req in sorted_requests],
                return_exceptions=True
            )
            
            # Handle exceptions
            final_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    final_responses.append(
                        self._create_error_response(str(response))
                    )
                else:
                    final_responses.append(response)
            
            logger.info("Batch analysis completed", 
                       total_requests=len(requests),
                       successful=len([r for r in final_responses 
                                     if r.confidence_score > 0]))
            
            return final_responses
            
        except Exception as e:
            logger.error("Error in batch analysis", error=str(e))
            return [self._create_error_response(str(e)) for _ in requests]
    
    def _parse_ai_response(self, content: str) -> AIAnalysisResponse:
        """Parse AI response content into structured format
        
        Args:
            content: Raw AI response content
            
        Returns:
            Structured AI analysis response
        """
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                data = json.loads(content)
                return AIAnalysisResponse(
                    summary=data.get('summary', ''),
                    insights=data.get('insights', []),
                    recommendations=data.get('recommendations', []),
                    warnings=data.get('warnings', []),
                    confidence_score=data.get('confidence_score', 0.8),
                    metadata=data.get('metadata', {})
                )
            
            # Parse structured text format
            return self.interpreter.parse_response(content)
            
        except Exception as e:
            logger.warning("Error parsing AI response", error=str(e))
            return AIAnalysisResponse(
                summary=content[:500],
                insights=[],
                recommendations=[],
                warnings=["Could not parse AI response fully"],
                confidence_score=0.5,
                metadata={"parse_error": str(e)}
            )
    
    def _create_error_response(self, error_msg: str) -> AIAnalysisResponse:
        """Create error response
        
        Args:
            error_msg: Error message
            
        Returns:
            Error AI analysis response
        """
        return AIAnalysisResponse(
            summary=f"Analysis failed: {error_msg}",
            insights=[],
            recommendations=[],
            warnings=[f"Error occurred: {error_msg}"],
            confidence_score=0.0,
            metadata={"error": error_msg}
        )
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model
        
        Returns:
            Model information
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "status": "active"
        }
    
    async def validate_api_key(self) -> bool:
        """Validate OpenAI API key
        
        Returns:
            True if API key is valid
        """
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False 