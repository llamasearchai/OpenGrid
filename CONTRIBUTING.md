# Contributing to OpenGrid

We love your input! We want to make contributing to OpenGrid as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Docker for containerized development

### Setting up the Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/OpenGrid.git
cd OpenGrid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opengrid --cov-report=html

# Run specific test files
pytest tests/test_analysis.py
pytest tests/test_ai.py
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black opengrid tests

# Sort imports
isort opengrid tests

# Lint code
ruff check opengrid tests

# Type checking
mypy opengrid
```

### Running the Application

```bash
# Start API server
python main.py

# Use CLI interface
python main.py network list-samples
# or after installation:
opengrid --help
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Use type hints for all functions and methods

### Code Organization

```
opengrid/
├── analysis/       # Power system analysis modules
├── ai/            # AI integration and analysis
├── api/           # FastAPI application and models
├── data/          # Sample networks and test cases
├── modeling/      # Power network modeling
├── visualization/ # (Future) Plotting and dashboards
└── cli.py         # Command line interface
```

### Documentation

- Use clear, descriptive docstrings for all public functions and classes
- Follow [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Update README.md for any user-facing changes
- Include examples in docstrings where helpful

Example docstring:
```python
def analyze_power_flow(self, network: PowerNetwork, 
                      algorithm: str = "newton_raphson") -> Dict[str, Any]:
    """
    Perform power flow analysis on the given network.
    
    Args:
        network: The power network to analyze
        algorithm: The solution algorithm to use. Options are:
            - "newton_raphson": Newton-Raphson method (default)
            - "fast_decoupled": Fast-decoupled method
            - "dc_power_flow": DC approximation
    
    Returns:
        Dictionary containing analysis results with keys:
            - converged: Whether the analysis converged
            - bus_voltages: Bus voltage magnitudes in pu
            - total_losses_mw: Total system losses in MW
            - iteration_count: Number of iterations required
    
    Raises:
        AnalysisError: If the analysis fails to converge
        
    Example:
        >>> network = PowerNetwork("IEEE 14-bus")
        >>> analyzer = LoadFlowAnalyzer(network)
        >>> results = analyzer.analyze_power_flow(network)
        >>> print(f"Converged: {results['converged']}")
    """
```

### Testing

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Include both unit tests and integration tests
- Test edge cases and error conditions

Example test:
```python
def test_load_flow_newton_raphson_convergence():
    """Test that Newton-Raphson load flow converges for IEEE 14-bus."""
    network = sample_networks.get_network("ieee_14_bus")
    analyzer = LoadFlowAnalyzer(network)
    
    results = analyzer.run_newton_raphson(tolerance_mva=1e-6)
    
    assert results["converged"] is True
    assert results["iteration_count"] < 10
    assert all(0.9 <= v <= 1.1 for v in results["bus_voltages"].values())
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear title** - Brief description of the issue
2. **Environment details**:
   - OpenGrid version
   - Python version
   - Operating system
   - Relevant package versions
3. **Steps to reproduce** - Minimal code example
4. **Expected behavior** - What should happen
5. **Actual behavior** - What actually happens
6. **Additional context** - Screenshots, logs, etc.

### Feature Requests

For feature requests, please include:

1. **Problem description** - What problem does this solve?
2. **Proposed solution** - How should it work?
3. **Alternatives considered** - Other approaches you've thought of
4. **Use cases** - Real-world scenarios where this would be useful

## Contribution Areas

We welcome contributions in several areas:

### **Analysis Algorithms**
- New power system analysis methods
- Performance optimizations
- Algorithm improvements
- IEEE/IEC standard implementations

### **AI Integration**
- New prompt templates
- Analysis interpretation improvements
- Custom AI models for power systems
- Explanation quality enhancements

### **Data and Cases**
- New sample networks
- Additional case studies
- Real-world scenarios
- Educational content

### **API and CLI**
- New endpoints and features
- Performance improvements
- User experience enhancements
- Documentation improvements

### **Testing and Quality**
- Test coverage improvements
- Performance benchmarks
- Code quality tools
- CI/CD enhancements

### **Documentation**
- User guides and tutorials
- API documentation
- Educational content
- Video tutorials

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## Getting Help

- Email: nikjois@llamasearch.ai
- GitHub Discussions: [OpenGrid Discussions](https://github.com/llamasearchai/OpenGrid/discussions)
- Issues: [GitHub Issues](https://github.com/llamasearchai/OpenGrid/issues)
- Documentation: Available at `/docs` when running the API server

## Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes for significant contributions
- Project documentation
- Special recognition for major contributions

Thank you for helping make OpenGrid better! 