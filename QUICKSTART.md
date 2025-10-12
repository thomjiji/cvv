# Quick Start Guide - offload-ai

Get up and running with offload-ai in minutes! This guide will walk you through installation and basic usage.

## 🚀 Quick Installation

### Option 1: Ready to Use (No Installation Required)

```bash
# Clone the repository
git clone <repository-url>
cd offload-ai

# Run directly
./pfndispatchcopy --help
```

### Option 2: Development Setup

```bash
# Clone and set up development environment
git clone <repository-url>
cd offload-ai
make setup
source .venv/bin/activate
```

## 🎯 Basic Usage

### Simple Copy to Multiple Destinations

```bash
# Copy one file to two backup locations
./pfndispatchcopy -v source_video.mov backup1/video.mov backup2/video.mov
```

### Professional DIT Workflow

```bash
# Copy with hash verification and performance optimization
./pfndispatchcopy \
  -v \
  -t xxh64be \
  -b 8388608 \
  -noflush_dest /fast/ssd/backup.mov \
  source_camera_file.mov \
  /fast/ssd/backup.mov \
  /slow/archive/backup.mov
```

## 📚 Quick Examples

### Run the Demo

```bash
# See the tool in action with sample files
python3 examples/quick_demo.py
```

Example output:
```
🎬 pfndispatchcopy - Professional File Copy Tool Demo
============================================================

==================================================
🚀 DEMO: Basic Multi-Destination Copy
==================================================
📁 Created demo file: camera_footage.mov (878,145 bytes)
INFO: copying camera_footage.mov...
INFO: copied 65536 of 878145 bytes
✅ Copy completed in 0.00 seconds
⚡ Speed: 496.4 MB/sec
🔐 MD5 Hash: ba6b264733b4fa1e...
```

### Test Real Files

```bash
# Create a test file
echo "Test content for copy demo" > test_source.txt

# Copy to multiple destinations
./pfndispatchcopy -v -t md5 test_source.txt dest1.txt dest2.txt

# Expected output:
# INFO: copying test_source.txt...
# INFO: copy speed 26 bytes in 0.00050 sec (0.0 MB/sec)
# INFO: hash MD5:abc123...
# INFO: done.
```

## 🛠️ Command Reference

### Essential Options

| Option | Description | Example |
|--------|-------------|---------|
| `-v` | Verbose output | `-v` |
| `-t HASH` | Hash verification | `-t xxh64be`, `-t md5` |
| `-b SIZE` | Buffer size | `-b 8388608` (8MB) |
| `-f_size SIZE` | Expected file size | `-f_size 1000000` |
| `-noflush_dest PATH` | Skip flush for destination | `-noflush_dest /fast/ssd/dest.mov` |

### Hash Algorithms

- `xxh64be` - Fastest (requires: `pip install xxhash`)
- `md5` - Fast, widely compatible
- `sha1` - Good balance
- `sha256` - Most secure

### Buffer Sizes

- Small files: `-b 65536` (64KB)
- Large files: `-b 8388608` (8MB)
- Very large files: `-b 16777216` (16MB)

## 📋 Common Use Cases

### 1. Camera Card Backup

```bash
# Backup camera card to two drives
./pfndispatchcopy -v -t xxh64be \
  /Volumes/Camera_Card/CLIP001.MOV \
  /Volumes/Backup_Drive_1/Project/CLIP001.MOV \
  /Volumes/Backup_Drive_2/Project/CLIP001.MOV
```

### 2. Fast SSD + Archive HDD

```bash
# Copy to fast SSD (no flush) and archive drive
./pfndispatchcopy -v \
  -noflush_dest /fast/working/file.mov \
  -t sha256 \
  original.mov \
  /fast/working/file.mov \
  /archive/storage/file.mov
```

### 3. Verify File Size Before Copy

```bash
# Get file size first
ls -l source.mov
# -> 6792544938 bytes

# Copy with size verification
./pfndispatchcopy -v -f_size 6792544938 -t md5 \
  source.mov dest1.mov dest2.mov
```

## 🧪 Testing Your Setup

### Run All Tests

```bash
# Quick functionality test
make demo

# Comprehensive test suite
make test

# With coverage report
make test-cov
```

### Manual Verification

```bash
# Create test files
mkdir -p test_run
echo "Hello, offload-ai!" > test_run/source.txt

# Test copy
./pfndispatchcopy -v -t md5 \
  test_run/source.txt \
  test_run/dest1.txt \
  test_run/dest2.txt

# Verify results
diff test_run/source.txt test_run/dest1.txt
diff test_run/source.txt test_run/dest2.txt
echo "✅ Files match!"
```

## ⚡ Performance Tips

1. **Use appropriate buffer sizes**: Larger buffers for larger files
2. **Skip flush for SSDs**: Use `-noflush_dest` for solid-state drives
3. **Choose fast hash algorithms**: `xxh64be` > `md5` > `sha1` > `sha256`
4. **Pre-specify file size**: Use `-f_size` for better progress tracking

## 🆘 Troubleshooting

### Common Issues

**Issue**: `xxhash not found`
```bash
# Solution: Install xxhash
pip install xxhash
```

**Issue**: Permission denied
```bash
# Solution: Check file permissions
chmod +x pfndispatchcopy
```

**Issue**: Import errors in examples
```bash
# Solution: Run from project root directory
cd offload-ai
python3 examples/quick_demo.py
```

### Getting Help

```bash
# Show help
./pfndispatchcopy --help

# Run diagnostics
python3 -c "import sys; print(f'Python: {sys.version}')"
python3 -c "from src.pfndispatchcopy import main; print('✅ Import successful')"
```

## 📞 Support

- 📖 Full documentation: [README.md](README.md)
- 🧪 Examples: Run `python3 examples/example_usage.py`
- 🐛 Issues: Check logs with `-v` flag for detailed output

---

**Ready to get started?** Run `make demo` to see the tool in action! 🚀
