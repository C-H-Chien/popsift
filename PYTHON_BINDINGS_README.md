# PopSift Python Bindings

This directory contains Python bindings for the PopSift library, allowing you to use GPU-accelerated SIFT feature extraction and matching from Python.

## Files Overview

- `src/application/py_main.cpp` - Python bindings for SIFT feature extraction
- `src/application/py_match.cpp` - Python bindings for SIFT feature matching
- `popsift_python.py` - Main Python interface script
- `setup.py` - Python package setup script
- `build_python_bindings.sh` - Automated build script
- `diagnose_build.py` - Diagnostic tool for troubleshooting build issues
- `src/application/pybind_CMakeLists.txt` - CMake configuration for Python bindings

## Prerequisites

1. **PopSift Library**: The main PopSift library must be built first
2. **Python 3.6+**: Python 3.6 or higher
3. **pybind11**: `pip install pybind11`
4. **CUDA Toolkit**: CUDA compiler and libraries
5. **CMake 3.24+**: For building (matches main project requirement)
6. **Optional dependencies**: `numpy`, `opencv-python`, `matplotlib`

## Quick Start

### 1. Build the Main PopSift Library

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

### 2. Build Python Bindings

Three options:

**Option 1: Using the automated script (recommended)**
```bash
./build_python_bindings.sh cmake
```

**Option 2: Using CMake directly**
```bash
mkdir -p build_python
cd build_python
cmake -DPopSift_BUILD_PYTHON_BINDINGS=ON -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc)
cd ..
```

**Option 3: Using setuptools**
```bash
python setup.py build_ext --inplace
```

### 3. Test the Installation

```bash
python popsift_python.py --help
```

## Usage Examples

### Command Line Interface

```bash
# Extract SIFT features from an image
python popsift_python.py --verbose --timing extract image.jpg

# Match features between two images
python popsift_python.py --verbose --timing match left.jpg right.jpg --visualize

# Interactive mode
python popsift_python.py interactive
```

### Python API

```python
import popsift_extract
import popsift_match
import numpy as np
import cv2

# Load images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Extract features
features1 = popsift_extract.extract_features_from_array(img1)
features2 = popsift_extract.extract_features_from_array(img2)

# Match features
matches = popsift_match.match_features_from_arrays(img1, img2)

print(f"Found {matches.num_matches} matches")
print(f"Processing time: {matches.match_time_ms:.2f} ms")
```

## Troubleshooting

### Common Issues

1. **"Failed to build Python modules"**
   - Run the diagnostic script: `python diagnose_build.py`
   - Ensure the main PopSift library is built first
   - Check that all dependencies are installed

2. **"Could not import PopSift modules"**
   - Make sure the `.so` files are in your Python path
   - Try copying them to the current directory or adding the build directory to PYTHONPATH

3. **CUDA-related errors**
   - Ensure CUDA toolkit is properly installed
   - Check that `nvcc` is in your PATH
   - Verify GPU compatibility

4. **CMake errors**
   - Ensure CMake version is 3.24 or higher
   - Check that pybind11 is installed: `pip install pybind11`

### Diagnostic Tool

Run the diagnostic script to identify issues:

```bash
python diagnose_build.py
```

This will check:
- Required dependencies (CMake, Python, CUDA, pybind11)
- Project structure
- PopSift library availability
- Python environment
- CUDA environment
- CMake configuration

### Build Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| `build_python_bindings.sh` | Automated, error checking, diagnostics | Custom script | Development, testing |
| CMake directly | Standard, integrated with main build | Manual steps | Production builds |
| setuptools | Python packaging standard | May need manual library linking | Distribution |

## API Reference

### Feature Extraction (`popsift_extract`)

- `extract_features_from_file(filename, verbose=False, print_time_info=False)` - Extract from image file
- `extract_features_from_array(image_array, verbose=False, print_time_info=False)` - Extract from numpy array
- `get_version()` - Get PopSift version

### Feature Matching (`popsift_match`)

- `match_features_from_files(left_file, right_file, verbose=False, print_time_info=False)` - Match from files
- `match_features_from_arrays(left_array, right_array, verbose=False, print_time_info=False)` - Match from arrays
- `get_version()` - Get PopSift version

### Data Structures

- `SiftResult`: Contains keypoints (x, y, scale, orientation) and descriptors
- `MatchResult`: Contains match indices and distances between features

## Performance Notes

- GPU memory usage scales with image size
- Processing time depends on GPU compute capability
- Large images may require significant GPU memory
- Consider downsampling very large images for faster processing

## License

Mozilla Public License 2.0 (Same as PopSift library)
