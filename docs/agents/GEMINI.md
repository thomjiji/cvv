# AGENTS.md - Development Guide for Agentic Coding

## Build/Lint/Test Commands

### Running Tests
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_pfndispatchcopy.py -v

# Run tests with coverage
python3 -m pytest tests/ --cov=src --cov-report=html

# Run a specific test class or method
python3 -m pytest tests/test_pfndispatchcopy.py::TestProgressTracker -v
python3 -m pytest tests/test_pfndispatchcopy.py::TestProgressTracker::test_progress_tracker_initialization -v
```

### Linting and Formatting
```bash
# Run Ruff linter
ruff check .

# Run Ruff formatter
ruff format .

# Run both linter and formatter
ruff check . --fix && ruff format .
```

### Type Checking
```bash
# Run pyright type checking
pyright src/
```

### Development Installation
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with xxhash support
pip install -e ".[xxhash]"

# Install with all optional dependencies
pip install -e ".[all]"
```

## Code Style Guidelines

### Imports
1. Standard library imports first, sorted alphabetically
2. Third-party imports second, sorted alphabetically
3. Local application imports last, sorted alphabetically
4. Use explicit imports (avoid `import *`)
5. Group imports by category with blank lines between groups

### Formatting
1. Follow PEP 8 style guide
2. Line length: 88 characters (configured for Black/Ruff)
3. Use 4 spaces for indentation (no tabs)
4. Use Black formatter for consistent code style
5. Use isort for import organization

### Types
1. Use type hints for all function parameters and return values
2. Use dataclasses for simple data containers
3. Use Enum for constants and options
4. Use Path objects from pathlib for file paths
5. Use typing module for complex types (Union, Optional, etc.)

### Naming Conventions
1. Use snake_case for functions, variables, and module names
2. Use PascalCase for class names
3. Use UPPER_CASE for constants
4. Use descriptive names that explain the purpose
5. Use verbs for functions (calculate_hash, verify_source)
6. Use nouns for variables (file_path, buffer_size)

### Error Handling
1. Use specific exception types rather than generic exceptions
2. Provide meaningful error messages
3. Log errors appropriately with context
4. Use try/except blocks for expected error conditions
5. Let unexpected errors propagate up the call stack

### Documentation
1. Use NumPy-style docstrings for all functions and classes
2. Include parameter and return value documentation
3. Include examples in docstrings when helpful
4. Write clear, concise comments for complex logic
5. Keep README.md updated with usage examples

### Logging
1. Use Python's logging module for all logging
2. Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
3. Include context information in log messages
4. Use consistent log message formatting
5. Avoid printing directly to stdout/stderr in library code

### Performance Considerations
1. Use appropriate buffer sizes for I/O operations
2. Minimize memory usage with streaming operations
3. Use threading carefully and with proper synchronization
4. Close file handles and resources promptly
5. Use efficient algorithms for hash calculation and verification

### Testing
1. Write unit tests for all public functions and classes
2. Use descriptive test method names
3. Test both success and failure cases
4. Use temporary directories and files for test isolation
5. Mock external dependencies when appropriate
