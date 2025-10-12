# Batch Processing for PopSift

This document explains the batch processing functionality in PopSift for efficiently extracting features from multiple images and matching multiple image pairs.

## Overview

The batch processing functions allow you to:
1. **Extract features** from multiple images using a single PopSift instance
2. **Match multiple image pairs** using a single PopSift instance

This is significantly more efficient than processing images one at a time.

## Key Benefits for Batch Extraction

### 🚀 Performance Improvements

1. **Single Initialization**: PopSift is initialized once instead of repeatedly
2. **Pipeline Efficiency**: GPU pipeline stays active, processing images as they're enqueued
3. **Reduced Overhead**: Minimizes CPU-GPU synchronization overhead
4. **Better GPU Utilization**: Images are uploaded and processed concurrently through the pipeline

### ⚡ Expected Speedup

- **2-5x faster** for batches of 5-10 images
- **3-10x faster** for larger batches (10+ images)
- Speedup increases with batch size due to better pipeline utilization

## API Functions - Feature Extraction

### From NumPy Arrays

#### Basic Batch Extraction
```python
import popsift_extract
import numpy as np

# Prepare your images
images = [img1, img2, img3]  # List of numpy arrays (H, W, uint8)

# Batch process all images
results = popsift_extract.extract_multiple_from_arrays(
    images,
    verbose=True,
    print_time_info=True
)

# results is a list of SiftResult objects
for i, result in enumerate(results):
    print(f"Image {i}: {result.num_features} features")
    print(f"  Keypoints: {result.keypoints_x}, {result.keypoints_y}")
    print(f"  Descriptors: {len(result.descriptors)} descriptors")
```

#### With Custom Configuration
```python
from popsift_config import SiftConfig

# Create custom configuration
config = SiftConfig()
config.octaves = 4
config.levels = 3
config.threshold = 0.01

# Batch process with custom config
results = popsift_extract.extract_multiple_from_arrays_with_config(
    images,
    config,
    verbose=True,
    print_time_info=True
)
```

### From Image Files

#### Basic Batch Extraction
```python
# Prepare file paths
image_files = ["img1.png", "img2.png", "img3.png"]

# Batch process all files
results = popsift_extract.extract_multiple_from_files(
    image_files,
    verbose=True,
    print_time_info=True
)
```

#### With Custom Configuration
```python
results = popsift_extract.extract_multiple_from_files_with_config(
    image_files,
    config,
    verbose=True,
    print_time_info=True
)
```

## Key Benefits for Batch Matching

### 🚀 Performance Improvements

1. **Single Initialization**: PopSift is initialized once instead of repeatedly
2. **Pipeline Efficiency**: GPU pipeline stays active, processing images as they're enqueued
3. **Reduced Overhead**: Minimizes CPU-GPU synchronization overhead
4. **Better GPU Utilization**: Images are uploaded and processed concurrently through the pipeline

### ⚡ Expected Speedup

- **2-5x faster** for batches of 5-10 pairs
- **3-10x faster** for larger batches (10+ pairs)
- Speedup increases with batch size due to better pipeline utilization

## API Functions

### From NumPy Arrays

#### Basic Batch Matching
```python
import popsift_match
import numpy as np

# Prepare your image pairs
left_images = [img1, img2, img3]  # List of numpy arrays (H, W, uint8)
right_images = [img1, img2, img3]

# Batch process all pairs
results = popsift_match.match_multiple_pairs_from_arrays(
    left_images,
    right_images,
    verbose=True,
    print_time_info=True
)

# results is a list of MatchResult objects
for i, result in enumerate(results):
    print(f"Pair {i}: {result.num_matches} matches")
    print(f"  Match indices (left): {result.matches_left_idx}")
    print(f"  Match indices (right): {result.matches_right_idx}")
    print(f"  Match distances: {result.match_distances}")
```

#### With Custom Configuration
```python
from popsift_config import SiftConfig

# Create custom configuration
config = SiftConfig()
config.octaves = 4
config.levels = 3
config.threshold = 0.01

# Batch process with custom config
results = popsift_match.match_multiple_pairs_from_arrays_with_config(
    left_images,
    right_images,
    config,
    verbose=True,
    print_time_info=True
)
```

### From Image Files

#### Basic Batch Matching
```python
# Prepare file paths
left_files = ["img1_left.png", "img2_left.png", "img3_left.png"]
right_files = ["img1_right.png", "img2_right.png", "img3_right.png"]

# Batch process all file pairs
results = popsift_match.match_multiple_pairs_from_files(
    left_files,
    right_files,
    verbose=True,
    print_time_info=True
)
```

#### With Custom Configuration
```python
results = popsift_match.match_multiple_pairs_from_files_with_config(
    left_files,
    right_files,
    config,
    verbose=True,
    print_time_info=True
)
```

## MatchResult Structure

Each result in the returned list contains:

```python
class MatchResult:
    matches_left_idx: list[int]      # Indices of matched features in left image
    matches_right_idx: list[int]     # Indices of matched features in right image
    match_distances: list[float]     # Match distances (lower = better)
    match_time_ms: float             # GPU matching time in milliseconds
    left_gpu_time_ms: float          # Left image GPU processing time
    right_gpu_time_ms: float         # Right image GPU processing time
    num_matches: int                 # Number of accepted matches
    num_total_matches: int           # Total matches before filtering
```

## Implementation Details

### How It Works

1. **Initialization Phase**
   - Creates a single `PopSift` object in matching mode
   - Sets up the processing pipeline (upload thread + processing thread)

2. **Enqueue Phase**
   - All images from all pairs are enqueued into the pipeline
   - Images are uploaded to GPU and processed asynchronously
   - The pipeline processes images while new ones are being enqueued

3. **Matching Phase**
   - Features are retrieved from each job (blocks if processing not complete)
   - Pairs are matched one by one
   - Results are collected

4. **Cleanup Phase**
   - All features are cleaned up
   - PopSift is uninitialized once at the end

### Pipeline Architecture

```
Image 1 → [Upload Queue] → [Upload Thread] → [Process Queue] → [Process Thread] → Features 1
Image 2 → [Upload Queue] → [Upload Thread] → [Process Queue] → [Process Thread] → Features 2
Image 3 → [Upload Queue] → [Upload Thread] → [Process Queue] → [Process Thread] → Features 3
...
```

The pipeline allows overlapping of:
- Image upload (CPU → GPU memory)
- Feature extraction (GPU processing)
- Feature retrieval (GPU → CPU memory)

## Migration Guide

### Before (Processing One Pair at a Time)

```python
results = []
for left, right in zip(left_images, right_images):
    result = popsift_match.match_features_from_arrays(left, right)
    results.append(result)
```

**Issues:**
- PopSift is initialized/uninitialized for each pair
- GPU pipeline is restarted for each pair
- Significant overhead from repeated initialization

### After (Batch Processing)

```python
results = popsift_match.match_multiple_pairs_from_arrays(
    left_images,
    right_images
)
```

**Benefits:**
- Single PopSift initialization
- Pipeline stays active
- Much faster for multiple pairs

## Performance Tips

1. **Batch Size**: Larger batches = better efficiency (but more memory)
2. **Image Size**: GPU memory limits how many images can be in flight
3. **Memory Management**: Ensure you have enough GPU memory for batch size
4. **Pre-loading**: Have all images loaded into memory before batch processing

## Error Handling

```python
try:
    results = popsift_match.match_multiple_pairs_from_arrays(
        left_images,
        right_images,
        verbose=True
    )
except RuntimeError as e:
    print(f"Batch processing failed: {e}")
    # Common errors:
    # - "Number of left and right images must match"
    # - "No image pairs provided"
    # - "All image arrays must be 2D (grayscale)"
    # - "Failed to process pair X"
```

## Complete Example

See `batch_matching_example.py` for complete working examples including:
- Basic batch processing from arrays
- Basic batch processing from files
- Custom configuration usage
- Performance comparison (single vs batch)

## When to Use

### Use Batch Processing When:
- You have **3 or more** image pairs to process
- Images are ready to be processed together
- You want maximum throughput

### Use Single-Pair Processing When:
- Processing only **1-2 pairs**
- Images arrive one at a time (streaming)
- You need results immediately for each pair

## Technical Notes

- **Thread Safety**: Each PopSift instance is NOT thread-safe. Use one instance per thread.
- **GPU Memory**: All enqueued images use GPU memory. Monitor usage for large batches.
- **Image Format**: All images must be grayscale (2D numpy arrays of uint8)
- **Error Recovery**: If one pair fails, the entire batch fails (by design for consistency)

## Questions?

For more information, see:
- `src/application/py_match.cpp` - Implementation
- `batch_matching_example.py` - Usage examples
- PopSift documentation - General SIFT documentation

