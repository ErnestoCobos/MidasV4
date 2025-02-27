# MidasScalpingv4 Project Guide

## Commands
- **Run**: `python -m midasscalpingv4` or `python hello.py`
- **Test**: `pytest` or `pytest tests/test_specific.py::test_function`
- **Lint**: `ruff check .` or `ruff check path/to/file.py`
- **Type check**: `mypy .` or `mypy path/to/file.py`
- **Format code**: `black .` or `black path/to/file.py`
- **Install dependencies**: `pip install -e .` or `pip install -r requirements.txt`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Formatting**: Follow PEP 8, use Black for auto-formatting
- **Types**: Use type hints for function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Functions**: Limit to single responsibility, use descriptive names
- **Error handling**: Use try/except with specific exceptions, avoid bare except clauses
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Testing**: Write unit tests for all new functionality

Remember to run tests and type checking before committing changes.