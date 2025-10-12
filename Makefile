# Makefile for offload-ai project
# Professional file copying tools for DIT workflows

.PHONY: help install install-dev test test-cov clean lint format check demo examples setup venv

# Default target
help:
	@echo "offload-ai - Professional File Copy Tools"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  help          Show this help message"
	@echo "  setup         Set up development environment"
	@echo "  venv          Create virtual environment"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  test          Run unit tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run code linting (flake8, mypy)"
	@echo "  format        Format code with black and isort"
	@echo "  check         Run all code quality checks"
	@echo "  demo          Run quick demonstration"
	@echo "  examples      Run comprehensive examples"
	@echo "  clean         Clean build artifacts and cache files"
	@echo "  build         Build distribution packages"
	@echo "  upload-test   Upload to test PyPI"
	@echo "  upload        Upload to PyPI"
	@echo ""

# Set up development environment
setup: venv install-dev
	@echo "✅ Development environment set up successfully"
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

# Create virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv .venv
	@echo "✅ Virtual environment created in .venv/"

# Install package in development mode
install:
	@echo "📦 Installing package in development mode..."
	pip install -e .

# Install package with development dependencies
install-dev:
	@echo "📦 Installing package with development dependencies..."
	pip install -e ".[dev,xxhash]"

# Run unit tests
test:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/ -v

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/"

# Run linting
lint:
	@echo "🔍 Running code linting..."
	@echo "  → flake8..."
	flake8 src/ tests/ examples/
	@echo "  → mypy..."
	mypy src/
	@echo "✅ Linting passed"

# Format code
format:
	@echo "🎨 Formatting code..."
	@echo "  → black..."
	black src/ tests/ examples/
	@echo "  → isort..."
	isort src/ tests/ examples/
	@echo "✅ Code formatted"

# Run all code quality checks
check: lint test
	@echo "✅ All code quality checks passed"

# Run quick demonstration
demo:
	@echo "🎬 Running quick demonstration..."
	python examples/quick_demo.py

# Run comprehensive examples
examples:
	@echo "📚 Running comprehensive examples..."
	python examples/example_usage.py

# Clean build artifacts and cache files
clean:
	@echo "🧹 Cleaning build artifacts and cache files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleaned up"

# Build distribution packages
build: clean
	@echo "📦 Building distribution packages..."
	python -m build
	@echo "✅ Distribution packages built in dist/"

# Upload to test PyPI
upload-test: build
	@echo "🚀 Uploading to test PyPI..."
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI
upload: build
	@echo "🚀 Uploading to PyPI..."
	python -m twine upload dist/*

# Run pfndispatchcopy with arguments (for development testing)
run:
	@echo "🏃 Running pfndispatchcopy..."
	python src/pfndispatchcopy.py $(ARGS)

# Show project structure
tree:
	@echo "📁 Project structure:"
	@tree -I '__pycache__|*.pyc|.venv|.git|*.egg-info|build|dist|htmlcov' .

# Development workflow shortcuts
dev-setup: setup
	@echo "🚀 Development setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. source .venv/bin/activate"
	@echo "2. make demo"
	@echo "3. make test"

# Quick development cycle
dev-cycle: format lint test
	@echo "🔄 Development cycle complete"

# Pre-commit hook simulation
pre-commit: format lint test
	@echo "✅ Pre-commit checks passed"
