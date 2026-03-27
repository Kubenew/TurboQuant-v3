# Contributing to TurboQuant-v3

Thank you for considering contributing to TurboQuant-v3! 🎉

## How to Contribute

1. **Fork & Clone**
   ```bash
   git clone https://github.com/Kubenew/TurboQuant-v3.git
   cd TurboQuant-v3
   git checkout -b feature/your-feature

Install Development DependenciesBashpip install -e ".[dev]"
pre-commit install
Development Workflow
Follow PEP 8 / Black formatting
Add type hints where possible
Write tests for new features
Update README and docstrings

Commit Messages
Use conventional commits:
feat: add SVD correction module
fix: resolve NaN in protected channel scaling
docs: improve quickstart example

Pull Request Guidelines
Reference any related issues
Include before/after benchmarks when possible
Keep PRs focused (one feature/fix per PR)


What We Welcome

New quantization techniques or improvements
Better CUDA kernels / fused operations
vLLM / llama.cpp / Hugging Face integration
Additional model benchmarks (Llama-3, Mistral, Gemma-2, etc.)
Documentation & example improvements
Bug fixes

Code of Conduct
We follow the Contributor Covenant. Be kind and respectful.
Questions? Open an issue or reach out on GitHub Discussions.
text### 3. `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Lint with Ruff
      run: ruff check src/ tests/

    - name: Format check with Black
      run: black --check src/ tests/

    - name: Type check with mypy
      run: mypy src/

    - name: Run tests
      run: pytest tests/ -v --cov=src/turboquant --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      if: matrix.python-version == '3.11'
