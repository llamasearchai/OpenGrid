#!/usr/bin/env python3
"""
OpenGrid - AI-Powered Power Systems Analysis Platform

A comprehensive platform for power system modeling, analysis, and AI-powered insights.
Supports load flow, short circuit, stability, and contingency analysis with modern APIs.

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import argparse
import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print application banner."""
    banner = """
    ========================================
    OpenGrid - AI-Powered Power Systems Analysis Platform
    ========================================
    
    Advanced power system modeling and analysis
    AI-powered insights and recommendations
    Comprehensive analysis tools and case studies
    REST API and command-line interfaces
    
    Author: Nik Jois (nikjois@llamasearch.ai)
    License: MIT
    ========================================
    """
    print(banner)

def create_app():
    """Create and configure the FastAPI application."""
    from opengrid.api.app import create_app
    return create_app()

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="OpenGrid Power Systems Analysis Platform")
    parser.add_argument("command", nargs="?", help="Command to run (default: start API server)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args, unknown = parser.parse_known_args()
    
    print_banner()
    
    # If CLI command provided, delegate to CLI module
    if args.command and args.command not in ["serve", "server"]:
        print("Starting OpenGrid CLI...")
        from opengrid.cli import cli_entry_point
        return cli_entry_point()
    
    # Start API server
    print("Starting OpenGrid API server...")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("AI features will be disabled. Set this variable to enable AI analysis.")
    
    # Get configuration from environment
    host = os.getenv("HOST", args.host)
    port = int(os.getenv("PORT", args.port))
    reload = os.getenv("RELOAD", "false").lower() == "true" or args.reload
    workers = int(os.getenv("WORKERS", args.workers))
    
    print(f"Server starting at http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Interactive API: http://{host}:{port}/redoc")
    print(f"Tip: Use 'python main.py --help' for CLI options")
    
    # Create and run the application
    app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 