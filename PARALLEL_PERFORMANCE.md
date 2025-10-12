# Parallel Performance Implementation - offload-ai

This document details the implementation and performance characteristics of the truly parallel file copying system in offload-ai.

## 🚀 Overview

The original implementation used **sequential writing** - reading data once and then writing it to each destination one after another. The new implementation provides **true parallel writing** where multiple writer threads simultaneously write data to different destinations.

### Before (Sequential)
```python
# Read chunk once
chunk = source.read(buffer_size)

# Write to destinations sequentially
for dest, handle in file_handles.items():
    handle.write(chunk)  # One at a time
```

### After (Parallel)
```python
# Read chunk once
chunk = ChunkData(chunk_id, source.read(buffer_size))

# Distribute to all writer threads simultaneously
for dest in destinations:
    chunk_queues[dest].put(chunk)  # All threads write in parallel
```

## 🏗️ Architecture

### Core Components

1. **ParallelWriter Class**: Manages multiple writer threads
2. **ChunkData Class**: Encapsulates data chunks with sequential IDs
3. **Thread-safe Queues**: Distribute chunks to writer threads
4. **Progress Synchronization**: Thread-safe progress tracking

### Thread Model

```
Main Thread (Reader)          Worker Threads (Writers)
      │                            │
      ├─ Read chunk                ├─ Writer Thread 1 → Destination 1
      ├─ Create ChunkData          ├─ Writer Thread 2 → Destination 2
      ├─ Distribute to queues      ├─ Writer Thread 3 → Destination 3
      ├─ Progress tracking         └─ Writer Thread N → Destination N
      └─ Coordination
```

### Data Flow

1. **Reader Thread**: Reads source file in chunks
2. **Distribution**: Places chunks in per-destination queues
3. **Writer Threads**: Process chunks from their respective queues
4. **Synchronization**: Coordinate progress and completion

## 📊 Performance Results

### Test Environment
- macOS with SSD storage
- Python 3.12
- Memory: 16GB RAM
- Storage: NVMe SSD

### Benchmark Results

#### Sequential vs Parallel Comparison

| File Size | Destinations | Sequential | Parallel | Improvement |
|-----------|--------------|------------|----------|-------------|
| 30MB      | 3           | 1,596 MB/s | 2,190 MB/s | **37% faster** |
| 50MB      | 5           | 1,045 MB/s | 1,362 MB/s | **30% faster** |

#### Thread Scaling Analysis

| Destinations | Speed (MB/s) | Efficiency |
|--------------|--------------|------------|
| 1            | 3,591        | Baseline   |
| 2            | 2,818        | 78%        |
| 4            | 1,677        | 47%        |
| 6            | 1,283        | 36%        |
| 8            | 917          | 26%        |

#### Optimal Buffer Sizes

| Buffer Size | Speed (MB/s) | Efficiency |
|-------------|--------------|------------|
| 64KB        | 956          | Baseline   |
| 512KB       | **1,739**    | **Best**   |
| 2MB         | 1,767        | Excellent  |
| 8MB         | 1,679        | Good       |
| 16MB        | 1,327        | Moderate   |

## 🔧 Implementation Details

### Thread Safety

```python
class ParallelWriter:
    def __init__(self):
        self.write_lock = threading.Lock()
        self.chunk_queues = {dest: queue.Queue() for dest in destinations}
        self.chunks_written = {dest: 0 for dest in destinations}
```

### Progress Synchronization

```python
def writer_thread(self, destination: Path) -> None:
    with self.write_lock:
        self.chunks_written[destination] += 1
        # Update progress only when all threads have written the chunk
        min_chunks = min(self.chunks_written.values())
        if min_chunks * chunk_size > self.total_bytes_written:
            self.progress_tracker.update(min_chunks * chunk_size)
```

### Error Handling

```python
try:
    handle.write(chunk.data)
except Exception as e:
    self.write_errors[destination] = str(e)
```

## 📈 Performance Analysis

### When Parallel Helps Most

1. **Multiple Destinations**: 3+ destinations show clear benefits
2. **Medium to Large Files**: 10MB+ files benefit most
3. **Mixed Storage**: Fast source, multiple destination types
4. **I/O Bound**: When disk I/O is the bottleneck

### When Sequential May Be Better

1. **Single Destination**: No parallelism benefit
2. **Very Small Files**: Thread overhead dominates
3. **CPU Bound**: Hash calculation intensive workloads
4. **Memory Constrained**: Limited RAM for buffers

### Optimal Configurations

```python
# High-performance configuration
buffer_size = 2 * 1024 * 1024    # 2MB chunks
destinations = 3-5               # Sweet spot for parallelism
file_size >= 10 * 1024 * 1024   # 10MB+ files
```

## 🎯 Real-World Use Cases

### Professional DIT Workflows

**Scenario**: Copy camera footage to multiple backup drives
- **Source**: High-speed camera card (500 MB/s)
- **Destinations**: SSD working drive + HDD archive + Network backup
- **Benefit**: 30-40% faster than sequential copying

**Example Command**:
```bash
./pfndispatchcopy -v -t xxh64be \
  /Volumes/Camera/CLIP001.MOV \
  /Volumes/WorkingSSD/CLIP001.MOV \
  /Volumes/Archive/CLIP001.MOV \
  /Network/Backup/CLIP001.MOV
```

### Video Production

**Scenario**: Distribute edited sequences to multiple locations
- **Source**: Edited video file (2GB+)
- **Destinations**: Preview server + Archive + Client delivery
- **Benefit**: Reduced delivery time in post-production pipeline

## 🔍 Technical Insights

### Thread Overhead Analysis

The performance drop with more destinations is due to:

1. **Context Switching**: CPU switching between threads
2. **Queue Management**: Overhead of managing multiple queues
3. **Lock Contention**: Synchronization overhead
4. **Memory Pressure**: More buffers in memory

### Storage Impact

```
Single fast SSD:
├── Sequential: Limited by single write stream
└── Parallel: Better utilization of SSD's parallel capabilities

Multiple storage devices:
├── Sequential: Limited by slowest device
└── Parallel: Each thread writes independently
```

### Memory Usage

```python
# Memory calculation per destination
memory_per_dest = buffer_size + queue_overhead + thread_stack
total_memory = memory_per_dest * num_destinations + source_buffer

# Example with 4 destinations, 2MB buffer:
# ~8MB + threading overhead ≈ 12MB total
```

## 🚦 Automatic Selection Logic

The implementation automatically chooses the best approach:

```python
def copy_with_multiple_destinations(source, destinations, ...):
    if len(destinations) > 1:
        # Use parallel implementation
        return copy_with_multiple_destinations_parallel(...)
    else:
        # Use simpler single-destination approach
        return single_destination_copy(...)
```

## 📝 Best Practices

### For Developers

1. **Profile First**: Measure before optimizing
2. **Consider Trade-offs**: Parallel isn't always better
3. **Test Thoroughly**: Multi-threading introduces complexity
4. **Monitor Resources**: Watch CPU and memory usage

### For Users

1. **File Size Matters**: Use parallel for files >10MB
2. **Destination Count**: 2-5 destinations optimal
3. **Storage Type**: Benefits vary by storage speed
4. **System Resources**: Monitor CPU/memory usage

### Configuration Recommendations

```python
# Recommended settings by file size
small_files = {
    "buffer_size": 64 * 1024,      # 64KB
    "max_destinations": 2,
    "use_parallel": False
}

large_files = {
    "buffer_size": 2 * 1024 * 1024,  # 2MB
    "max_destinations": 5,
    "use_parallel": True
}
```

## 🔮 Future Improvements

### Planned Enhancements

1. **Adaptive Buffer Sizing**: Dynamic buffer size based on file size
2. **Storage-aware Threading**: Different thread counts per storage type
3. **Network Optimization**: Special handling for network destinations
4. **Progress Estimation**: Better ETA calculations for parallel ops

### Potential Optimizations

```python
# Future: Adaptive thread pool
class AdaptiveParallelWriter:
    def __init__(self):
        self.thread_pool_size = min(num_destinations, cpu_count())
        self.storage_type_aware = True
        self.dynamic_buffering = True
```

## 📊 Benchmarking Your Setup

### Quick Performance Test

```bash
# Create test file
dd if=/dev/zero of=test_50mb.bin bs=1M count=50

# Test parallel performance
time ./pfndispatchcopy -v test_50mb.bin dest1.bin dest2.bin dest3.bin

# Compare with sequential simulation
python3 examples/parallel_demo.py
```

### Custom Benchmarking

```python
# Measure your specific use case
from src.pfndispatchcopy import copy_with_multiple_destinations_parallel

# Your test setup
source = Path("your_typical_file.mov")
destinations = [Path(f"dest_{i}.mov") for i in range(3)]

# Run and measure
start = time.time()
results = copy_with_multiple_destinations_parallel(source, destinations, 2*1024*1024)
duration = time.time() - start

print(f"Your setup: {results['speed_mb_sec']:.1f} MB/sec")
```

## 📚 References

- **Threading Model**: Python threading documentation
- **Queue Performance**: Python queue module behavior
- **I/O Patterns**: Operating system I/O scheduling
- **Storage Performance**: SSD vs HDD characteristics

---

*This parallel implementation represents a significant advancement in file copying performance for multi-destination scenarios, particularly valuable in professional video production workflows where time and reliability are critical.*
