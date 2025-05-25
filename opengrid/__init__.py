"""
OpenGrid: AI-Powered Power Systems Analysis and Design Platform

A comprehensive power systems analysis platform integrating AI capabilities
for advanced grid modeling, analysis, and optimization.

Author: Nik Jois <nikjois@llamasearch.ai>
License: MIT
"""

__version__ = "0.2.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

from .modeling import PowerNetwork
from .analysis import LoadFlowAnalyzer, ShortCircuitAnalyzer
from .ai import OpenAIAgent
from .api import create_app

__all__ = [
    "PowerNetwork",
    "LoadFlowAnalyzer", 
    "ShortCircuitAnalyzer",
    "OpenAIAgent",
    "create_app",
] 