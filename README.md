# OpenGrid - AI-Powered Power Systems Analysis Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)

> **Advanced power systems analysis with AI-powered insights**

OpenGrid is a comprehensive, AI-enhanced power systems analysis platform that combines cutting-edge electrical engineering tools with artificial intelligence to provide intelligent insights and recommendations for power grid operations, planning, and optimization.

## Features

### **Comprehensive Analysis Capabilities**
- **Load Flow Analysis** - Newton-Raphson, Fast-Decoupled, DC Power Flow
- **Short Circuit Analysis** - 3-phase, Line-to-Ground, Line-to-Line faults
- **Contingency Analysis** - N-1, N-2, and cascading failure analysis
- **Stability Studies** - Small signal, transient, and voltage stability
- **Harmonic Analysis** - Power quality assessment and filter design
- **Protection Coordination** - Relay coordination and arc flash analysis
- **Optimization** - Economic dispatch, OPF, unit commitment

### **AI-Powered Intelligence**
- **Automated Analysis Interpretation** - AI explains complex results in plain language
- **Intelligent Recommendations** - Context-aware suggestions for system improvements
- **Risk Assessment** - AI-driven evaluation of system vulnerabilities
- **Predictive Insights** - Pattern recognition for proactive maintenance

### **Pre-Built Network Models**
- **IEEE Test Systems** - 9-bus, 14-bus, 30-bus standard networks
- **Microgrid Models** - Renewable energy and storage systems
- **Industrial Plants** - Motor starting and power quality scenarios
- **Distribution Feeders** - Hosting capacity and DG integration
- **Renewable Grids** - High penetration renewable scenarios

### **Extensive Case Study Library**
- **35+ Analysis Cases** - Pre-configured scenarios for learning and testing
- **8 Comprehensive Study Plans** - End-to-end analysis workflows
- **Educational Content** - Perfect for power engineering education
- **Industry Applications** - Real-world scenarios and best practices

### **Multiple Interfaces**
- **REST API** - Complete programmatic access with FastAPI
- **Command Line Interface** - Powerful CLI for automation and scripting
- **Interactive Documentation** - Auto-generated API docs with examples
- **Docker Support** - Containerized deployment for any environment

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenGrid.git
cd OpenGrid

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (optional, for AI features)
export OPENAI_API_KEY="your-api-key-here"

# Start the API server
python main.py

# Or use the CLI
python main.py network list-samples
```

### Docker Installation

```bash
# Build the image
docker build -t opengrid .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" opengrid
```

### Package Installation

```bash
# Install from source
pip install -e .

# Use the CLI directly
opengrid --help
```

## Command Line Usage

### Server Mode
```bash
# Start API server
python main.py
# or
opengrid server --port 8000
```

### Analysis Mode
```bash
# List available sample networks
opengrid network list-samples

# Create a network from IEEE 14-bus system
opengrid network create --name "Test Grid" --type ieee_14_bus

# Run load flow analysis
opengrid analysis run --network sample_ieee_14_bus --type load_flow --algorithm newton_raphson

# Run AI analysis on results
opengrid analysis ai --result network_1_load_flow_1234567890
```

### Case Studies
```bash
# List available case studies
opengrid case list --type load_flow --difficulty easy

# Run a specific case study
opengrid case run --case lf_ieee14_basic

# Export results
opengrid export --result case_result_123 --file results.json
```

## API Usage

### Start API Server
```bash
python main.py
# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Example API Calls

```python
import requests

# Create a network
response = requests.post("http://localhost:8000/networks", json={
    "name": "My Test Network",
    "use_pypsa": False
})
network_id = response.json()["network_id"]

# Run load flow analysis
response = requests.post(f"http://localhost:8000/networks/{network_id}/analysis/load-flow", json={
    "algorithm": "newton_raphson",
    "tolerance_mva": 1e-6,
    "max_iteration": 50
})

# Get AI insights
response = requests.post(f"http://localhost:8000/networks/{network_id}/ai-analysis", json={
    "analysis_type": "power_flow",
    "priority": "normal"
})
```

## Sample Networks & Cases

### Available Sample Networks
- **IEEE 9-Bus** - Simple transmission system
- **IEEE 14-Bus** - Standard test system
- **IEEE 30-Bus** - Complex transmission network
- **Simple Microgrid** - Renewable energy and storage
- **Industrial Plant** - Motor loads and backup systems
- **Distribution Feeder** - Radial distribution with DG
- **Renewable Grid** - High renewable penetration
- **DC Microgrid** - Direct current distribution

### Study Plans Available
- **Transmission Planning** - Load flow, contingency, stability analysis
- **Distribution Modernization** - Smart grid and renewable integration
- **Microgrid Design** - Islanding, control, and energy management
- **Renewable Integration** - Variability, hosting capacity, storage
- **Industrial Power Study** - Motor starting, power quality, protection
- **System Reliability** - N-1/N-2 contingency and stability assessment
- **Power Quality Assessment** - Harmonics, flicker, and mitigation
- **Grid Modernization** - PMU placement, state estimation, markets

## Configuration

### Environment Variables
```bash
# Required for AI features
OPENAI_API_KEY=your-openai-api-key

# Server configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
WORKERS=1

# Database (optional)
DATABASE_URL=sqlite:///opengrid.db
```

### Custom Networks
```python
from opengrid.modeling import PowerNetwork

# Create custom network
network = PowerNetwork("My Custom Grid")

# Add components
bus1 = network.add_bus(vn_kv=138.0, name="Main Bus")
bus2 = network.add_bus(vn_kv=138.0, name="Load Bus")

line = network.add_line(
    from_bus=bus1, to_bus=bus2, 
    length_km=50.0, std_type="NAYY 4x150 SE"
)

gen = network.add_generator(bus=bus1, p_mw=100.0, vm_pu=1.0)
load = network.add_load(bus=bus2, p_mw=80.0, q_mvar=20.0)
```

## AI Integration

The AI analysis provides:

- **Executive Summaries** - High-level system assessment
- **Technical Insights** - Detailed engineering analysis
- **Risk Warnings** - Critical issues and vulnerabilities
- **Action Items** - Prioritized recommendations
- **Compliance Assessment** - IEEE/IEC standards verification

Example AI output:
```
AI Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Summary:
The load flow analysis shows a well-balanced system with good voltage 
regulation. All bus voltages are within acceptable limits (0.95-1.05 pu).

Key Insights:
1. Maximum loading on Line 7-9 at 87% - monitor for thermal limits
2. Reactive power margin at Bus 14 is low - consider capacitor addition
3. System losses at 3.2% are within normal range for this network size

Recommendations:
1. Install 15 MVAR capacitor bank at Bus 14 for voltage support
2. Consider parallel line for Line 7-9 to improve reliability
3. Schedule maintenance for transformer 4-7 showing higher losses
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

# Production deployment
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "main.py"]
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opengrid --cov-report=html

# Run specific test categories
pytest tests/test_analysis.py
pytest tests/test_ai.py
pytest tests/test_api.py
```

## Documentation

- **API Documentation**: Available at `/docs` when server is running
- **Code Documentation**: Auto-generated from docstrings
- **Case Study Guide**: Comprehensive examples and tutorials
- **Developer Guide**: Architecture and extension documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Nik Jois** - *Lead Developer*
- Email: nikjois@llamasearch.ai
- GitHub: [@nikjois](https://github.com/nikjois)
- Organization: [LlamaSearch AI](https://github.com/llamasearchai)

## Acknowledgments

- **pandapower** - Power system modeling framework
- **PyPSA** - Open source energy system modeling
- **OpenAI** - AI analysis capabilities
- **FastAPI** - High-performance web framework
- **Power Engineering Community** - Standards and best practices

## Links

- **Repository**: https://github.com/llamasearchai/OpenGrid
- **Documentation**: https://llamasearchai.github.io/OpenGrid
- **Issues**: https://github.com/llamasearchai/OpenGrid/issues
- **Discussions**: https://github.com/llamasearchai/OpenGrid/discussions

---

*Built for the future of power systems analysis* 