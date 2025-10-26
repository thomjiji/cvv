# cvv - Professional File Copy Tools for DIT Workflows

A Python implementation of professional file copying tools inspired by Offload Manager's `pfndispatchcopy`, designed for reliable file copying with integrity verification in professional DIT (Digital Imaging Technician) workflows.

## 🎯 Features

- **Multi-destination copying**: Copy one source file to multiple destinations simultaneously
- **Integrity verification**: Support for multiple hash algorithms (XXH64BE, MD5, SHA1, SHA256)
- **Progress monitoring**: Real-time progress updates with transfer speed calculation
- **Atomic operations**: Uses temporary files and atomic moves to prevent corruption
- **Selective flush control**: Control file system flushing per destination for performance optimization
- **Buffer size control**: Configurable buffer sizes for optimal performance
- **File size verification**: Pre-flight and post-copy file size validation
- **Professional logging**: Detailed logging similar to commercial DIT tools

## 📁 Project Structure

```
cvv/
├── src/
│   ├── __init__.py              # Package initialization
│   └── pfndispatchcopy.py       # Main implementation
├── tests/
│   ├── __init__.py              # Test package initialization
│   └── test_pfndispatchcopy.py  # Unit tests
├── examples/
│   ├── example_usage.py         # Comprehensive examples
│   └── quick_demo.py            # Quick demonstration script
├── ref/                         # Reference log files from Offload Manager
│   ├── A013_pfncopy_AF1ADD.log
│   └── ...
├── pfndispatchcopy              # Command-line entry point
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Option 1: Development Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd cvv
   ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   ```

### Option 2: Regular Installation

```bash
pip install -r requirements.txt
```

### Option 3: Direct Usage

You can run the tools directly without installation:

```bash
# Using the command-line entry point
./pfndispatchcopy [OPTIONS] SOURCE DEST1 [DEST2 ...]

#
 Or directly with Python
python3 src/pfndispatchcopy.py [OPTIONS] SOURCE DEST1 [DEST2 ...]
```

## 📦 Dependencies

- **Python 3.8+** (uses walrus operator `:=`)
- **xxhash** (optional, for XXH64BE hash support):
  ```bash
  pip install xxhash
  ```

All other dependencies are part of Python's standard library.

## 🔧 Usage

### Basic Syntax

```bash
pfndispatchcopy [OPTIONS] SOURCE DEST1 [DEST2 ...]
```

### Command Line Options

- `-v, --verbose`: Enable verbose output
- `-t, --hash {xxh64be,md5,sha1,sha256}`: Hash algorithm for verification
- `-k, --keep`: Keep source file after copying (default: True)
- `-f_size, --file-size SIZE`: Expected file size in bytes for verification
- `-b, --buffer-size SIZE`: Buffer size in bytes (default: 8MB)
- `-noflush_dest PATH`: Skip file system flush for specific destination (can be used multiple times)

### 📚 Examples

#### 1. Basic Copy to Multiple Destinations

```bash
pfndispatchcopy -v source_video.mov backup1/video.mov backup2/video.mov
```

#### 2. Professional DIT Workflow (Matching Original Log)

```bash
pfndispatchcopy \
  -v \
  -noflush_dest "/Volumes/Pegasus32 R8/project/video.mov" \
  -noflush_dest "/Volumes/GSJ-promise_r6_80TB/project/video.mov" \
  -t xxh64be \
  -k \
  -f_size 6792544938 \
  -b 8388608 \
  "/Volumes/DJI/A013_VW5X15/A013C0002_251011_VW5X15.MOV" \
  "/Volumes/Pegasus32 R8/25082-project/A013C0002_251011_VW5X15.MOV" \
  "/Volumes/GSJ-promise_r6_80TB/25082-project/A013C0002_251011_VW5X15.MOV"
```

#### 3. High-Performance Copy with Selective Flushing

```bash
pfndispatchcopy \
  -v \
  -noflush_dest "/fast/ssd/destination.mov" \
  -t sha256 \
  -b 16777216 \
  "source.mov" \
  "/fast/ssd/destination.mov" \
  "/slow/hdd/destination.mov"
```

## 🧪 Testing and Examples

### Run Quick Demo

```bash
python3 examples/quick_demo.py
```

### Run Comprehensive Examples

```bash
python3 examples/example_usage.py
```

### Run Unit Tests

```bash
python3 -m pytest tests/ -v
```

Or run tests directly:

```bash
cd tests/
python3 test_pfndispatchcopy.py
```

## 📊 Implementation Details

### Architecture

The implementation closely follows the behavior observed in the original Offload Manager logs:

1. **Command Parsing**: Uses `argparse` for robust command-line argument handling
2. **Progress Tracking**: Multi-threaded progress reporting with configurable intervals
3. **Hash Calculation**: Pluggable hash algorithms with streaming calculation
4. **Atomic File Operations**: Temporary files with atomic moves to prevent corruption
5. **Multi-destination Logic**: Single read, multiple write pattern for efficiency

### Key Classes

- **`ProgressTracker`**: Handles progress reporting with thread-safe updates
- **`HashCalculator`**: Manages hash calculation with support for multiple algorithms
- **`copy_with_multiple_destinations()`**: Core copying logic with simultaneous multi-destination support

### Performance Characteristics

- **Memory Usage**: Configurable buffer size (default 8MB) keeps memory usage predictable
- **I/O Pattern**: Sequential reads with parallel writes for optimal disk usage
- **Progress Updates**: Non-blocking progress updates every 500ms (configurable)
- **Hash Calculation**: Streaming hash calculation to avoid memory overhead

## 🔍 Log Output Comparison

### Original Offload Manager Log

```
I 2025-10-11 17:55:32.958+0800 <AF1ADD> [43197129] pfndispatchcopy (97241): INFO: copying /Volumes/DJI/A013_VW5X15...
I 2025-10-11 17:55:33.461+0800 <AF1ADD> [43197973] pfndispatchcopy (97241): INFO: copied 218103809 of 6792544938 bytes
I 2025-10-11 17:55:41.749+0800 <AF1ADD> [43197967] pfndispatchcopy (97241): hash XXH64BE:a607a84d34b1350a
I 2025-10-11 17:55:41.749+0800 <AF1ADD> [43197967] pfndispatchcopy (97241): INFO: copy speed 6792544938 bytes in 8.79026 sec (772.7 MB/sec)
```

### Python Implementation Output

```
INFO: copying /Volumes/DJI/A013_VW5X15/A013C0002_251011_VW5X15.MOV...
INFO: copied 218103809 of 6792544938 bytes
INFO: hash XXH64BE:a607a84d34b1350a
INFO: copy speed 6792544938 bytes in 8.79026 sec (772.7 MB/sec)
INFO: Copy completed successfully for destination file with file size check (/Volumes/Pegasus32 R8/...)
INFO: moving files in place..
INFO: flushing files
INFO: done.
```

## ⚡ Performance Tips

1. **Buffer Size**: Increase buffer size (`-b`) for large files and fast storage
2. **Hash Algorithm**: XXH64BE is fastest, SHA256 most secure
3. **Flush Control**: Use `-noflush_dest` for SSDs to improve performance
4. **File Size**: Pre-specify file size (`-f_size`) for better progress tracking

## 🔧 Development

### Project Setup

```bash
# Clone the repository
git clone <repository-url>
cd cvv

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with extras
pip install -e ".[dev,xxhash]"
```

### Code Style

The project follows Python best practices:

- **PEP 8**: Standard Python style guide
- **Type hints**: Full type annotation coverage
- **Docstrings**: NumPy-style documentation
- **Error handling**: Comprehensive exception handling
- **Logging**: Professional-grade logging throughout
- **Code formatting**: Uses Ruff, Black, and isort for consistent code style
- **Linting**: Uses Ruff, flake8, and mypy for code quality checks

### Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python3 tests/test_pfndispatchcopy.py
```

## 🚧 Limitations

- **Threading**: Current implementation uses single-threaded I/O (can be enhanced)
- **Network Storage**: May need tuning for network-attached storage
- **Platform**: Tested primarily on macOS/Linux (Windows compatibility may vary)

## 📋 Changelog

### Version 1.0.0

- Initial implementation with core functionality
- Multi-destination copying support
- Hash verification (XXH64BE, MD5, SHA1, SHA256)
- Progress monitoring and performance statistics
- Atomic file operations
- Selective flush control
- Professional project structure with src/, tests/, examples/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow PEP 8 coding standards
4. Add type hints for all functions
5. Include comprehensive docstrings
6. Add appropriate error handling
7. Update tests for new features
8. Commit your changes (`git commit -m
'Add some amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## 📄 License

This project is provided as-is for educational and professional use.

## 🙏 Acknowledgments

- Inspired by Offload Manager's professional DIT workflow tools
- Designed for the film and video production community
- Built with Python's robust standard library and ecosystem
