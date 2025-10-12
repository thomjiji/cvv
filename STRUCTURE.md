# Project Structure - offload-ai

This document describes the organized structure of the offload-ai project, a professional file copying tool for DIT workflows.

## 📁 Directory Layout

```
offload-ai/
├── src/                              # Source code package
│   ├── __init__.py                   # Package initialization and exports
│   └── pfndispatchcopy.py           # Main implementation (529 lines)
│
├── tests/                           # Test suite
│   ├── __init__.py                  # Test package initialization
│   └── test_pfndispatchcopy.py     # Unit tests and integration tests
│
├── examples/                        # Example scripts and demonstrations
│   ├── example_usage.py            # Comprehensive usage examples
│   └── quick_demo.py               # Quick demonstration script
│
├── ref/                            # Reference materials
│   ├── A013_pfncopy_AF1ADD.log    # Original Offload Manager log (analyzed)
│   ├── A013_pfnverify_3BBB37.log  # Additional reference logs
│   ├── pfncopy_E0A1FF.log         # More reference logs
│   └── pfnverify_061CFA.log       # Verification logs
│
├── pfndispatchcopy                 # Command-line entry point (executable)
├── setup.py                       # Legacy setuptools configuration
├── pyproject.toml                 # Modern Python packaging configuration
├── Makefile                       # Development and build automation
├── requirements.txt               # Python dependencies
│
├── README.md                      # Main project documentation
├── QUICKSTART.md                  # Quick start guide
├── STRUCTURE.md                   # This file - project structure documentation
├── LICENSE                        # MIT License
│
├── .gitignore                     # Git ignore rules
└── .venv/                        # Virtual environment (created by user)
```

## 📦 Core Components

### `src/` - Source Code Package

**Purpose**: Contains the main implementation following Python packaging best practices.

- **`__init__.py`**: Package initialization file that exports public APIs
  - Exports main classes: `HashCalculator`, `ProgressTracker`
  - Exports main functions: `copy_with_multiple_destinations`, `main`
  - Version information and package metadata

- **`pfndispatchcopy.py`**: Main implementation (529 lines)
  - Professional file copying with multi-destination support
  - Progress tracking with real-time updates
  - Hash verification (XXH64BE, MD5, SHA1, SHA256)
  - Atomic file operations
  - Command-line argument parsing
  - Comprehensive error handling

### `tests/` - Test Suite

**Purpose**: Comprehensive testing infrastructure for reliability.

- **`test_pfndispatchcopy.py`**: Complete test suite
  - Unit tests for all major components
  - Integration tests for file operations
  - Command-line argument parsing tests
  - Error handling verification
  - Mock-based testing for edge cases

### `examples/` - Demonstrations and Examples

**Purpose**: Show real-world usage patterns and provide learning resources.

- **`quick_demo.py`**: Interactive demonstration script
  - Basic copy operations
  - Performance comparisons with different buffer sizes
  - Hash algorithm demonstrations
  - Error handling examples

- **`example_usage.py`**: Comprehensive usage examples
  - Professional DIT workflow scenarios
  - Command-line interface examples
  - Integration with shell scripts
  - Large file handling demonstrations

### `ref/` - Reference Materials

**Purpose**: Original log files from Offload Manager that inspired this implementation.

- Contains actual log files from professional DIT operations
- Used for understanding original behavior patterns
- Reference for output format matching
- Performance benchmarking baselines

## 🔧 Configuration Files

### Python Packaging

- **`pyproject.toml`**: Modern Python packaging standard
  - Build system configuration
  - Dependencies and optional extras
  - Development tool configurations (black, mypy, pytest)
  - Project metadata and entry points

- **`setup.py`**: Legacy setuptools support for compatibility
  - Backwards compatibility with older pip versions
  - Console script entry point definition

- **`requirements.txt`**: Runtime dependencies
  - Core dependencies (mostly stdlib)
  - Optional xxhash for performance

### Development Tools

- **`Makefile`**: Automation for common development tasks
  - `make setup`: Initialize development environment
  - `make test`: Run test suite
  - `make demo`: Run demonstrations
  - `make lint`: Code quality checks
  - `make clean`: Clean build artifacts

- **`.gitignore`**: Comprehensive exclusion rules
  - Python artifacts (__pycache__, *.pyc)
  - Build and distribution files
  - IDE and editor files
  - OS-specific files (macOS, Windows, Linux)
  - Test outputs and temporary files

## 🚀 Entry Points

### Command-Line Interface

- **`pfndispatchcopy`**: Main executable script
  - Direct command-line access without Python prefix
  - Handles path resolution and module loading
  - Provides professional CLI experience

### Import Interface

```python
# Import from package
from pfndispatchcopy import copy_with_multiple_destinations, HashCalculator

# Or use main function directly
from pfndispatchcopy import main as pfndispatchcopy_main
```

## 🎯 Design Principles

### 1. **Professional Structure**
- Clear separation of concerns
- Standard Python packaging practices
- Comprehensive testing and documentation

### 2. **User-Friendly Interface**
- Multiple ways to run the tool
- Clear command-line interface
- Extensive examples and documentation

### 3. **Developer-Friendly**
- Well-organized source code
- Comprehensive test suite
- Automated development workflows
- Clear documentation

### 4. **Production-Ready**
- Proper error handling
- Logging and monitoring
- Performance optimization
- Cross-platform compatibility

## 📊 File Statistics

```
Total Lines of Code:
├── src/pfndispatchcopy.py:     529 lines
├── tests/test_pfndispatchcopy.py: 369 lines
├── examples/quick_demo.py:     236 lines
├── examples/example_usage.py:  351 lines
├── Documentation:              ~800 lines
└── Configuration:              ~400 lines
```

## 🔄 Development Workflow

### Quick Development Cycle
```bash
# Set up environment
make setup
source .venv/bin/activate

# Development cycle
make format    # Format code
make lint      # Check code quality
make test      # Run tests
make demo      # Test functionality
```

### File Organization Benefits

1. **Maintainability**: Clear structure makes code easy to navigate
2. **Testing**: Isolated test suite with comprehensive coverage
3. **Documentation**: Multiple levels of documentation for different users
4. **Distribution**: Ready for PyPI publication
5. **Development**: Streamlined development workflow with automation

## 📈 Future Expansion

The structure supports easy expansion:

- **`src/`**: Add new modules (e.g., `pfnverify.py`, `pfncompare.py`)
- **`tests/`**: Add corresponding test modules
- **`examples/`**: Add use-case specific examples
- **`docs/`**: Future comprehensive documentation
- **`scripts/`**: Additional utility scripts

This structure follows modern Python best practices and provides a solid foundation for professional DIT workflow tools.
