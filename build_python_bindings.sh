#!/bin/bash

# Build script for PopSift Python bindings
# This script helps build the Python extensions using either CMake or setuptools

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "Please run this script from the PopSift root directory"
    exit 1
fi

# Check for required dependencies
print_status "Checking dependencies..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

# Check for pybind11
if ! python3 -c "import pybind11" &> /dev/null; then
    print_error "pybind11 is not installed. Install with: pip install pybind11"
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    print_warning "CUDA compiler (nvcc) not found in PATH. Make sure CUDA is properly installed."
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake is not installed or not in PATH"
    exit 1
fi

print_status "All dependencies found!"

# Method selection
METHOD=${1:-cmake}
BUILD_DIR="build_python"

print_status "Building Python bindings using method: $METHOD"

if [ "$METHOD" = "cmake" ]; then
    # CMake method
    print_status "Using CMake to build Python bindings..."
    
    # Create build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure with CMake
    print_status "Configuring with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DPopSift_BUILD_PYTHON_BINDINGS=ON \
          -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_INSTALL_PREFIX="/gpfs/data/bkimia/cchien3/popsift/install_python_bin/" \
          -DPython3_EXECUTABLE=$(which python3) \
          ..
    
    # Build
    print_status "Building..."
    make -j$(nproc)
    
    # Check if modules were built (they're in Linux-x86_64 subdirectory)
    if [ -d "Linux-x86_64" ] && ls Linux-x86_64/popsift_extract*.so 1> /dev/null 2>&1 && ls Linux-x86_64/popsift_match*.so 1> /dev/null 2>&1; then
        print_status "Python modules built successfully!"
        print_status "Modules location: $(pwd)/Linux-x86_64/"
        
        # Copy modules to parent directory for easy access
        cp Linux-x86_64/popsift_extract*.so ../
        cp Linux-x86_64/popsift_match*.so ../
        print_status "Modules copied to root directory"
        
        # Show the actual module names
        echo "Built modules:"
        ls -la Linux-x86_64/popsift_*.so
    else
        print_error "Failed to build Python modules"
        print_error "Looking for popsift_extract*.so and popsift_match*.so in $(pwd)/Linux-x86_64/"
        print_error "Available files in Linux-x86_64/:"
        if [ -d "Linux-x86_64" ]; then
            ls -la Linux-x86_64/*.so 2>/dev/null || echo "No .so files found in Linux-x86_64/"
        else
            echo "Linux-x86_64 directory not found"
        fi
        exit 1
    fi
    
elif [ "$METHOD" = "setup" ]; then
    # setuptools method
    print_status "Using setuptools to build Python bindings..."
    
    # First ensure the main PopSift library is built
    if [ ! -f "build/libpopsift.so" ]; then
        print_status "Building main PopSift library first..."
        mkdir -p build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)
        cd ..
    fi
    
    # Build Python extensions
    python3 setup.py build_ext --inplace
    
    # Check if modules were built
    if ls popsift_extract*.so 1> /dev/null 2>&1 && ls popsift_match*.so 1> /dev/null 2>&1; then
        print_status "Python modules built successfully!"
        print_status "Modules location: $(pwd)"
        echo "Built modules:"
        ls -la popsift_*.so
    else
        print_error "Failed to build Python modules"
        print_error "Looking for popsift_extract*.so and popsift_match*.so in $(pwd)"
        print_error "Available files:"
        ls -la *.so 2>/dev/null || echo "No .so files found"
        exit 1
    fi
    
else
    print_error "Unknown method: $METHOD"
    print_error "Use 'cmake' or 'setup'"
    exit 1
fi

# Test the modules
print_status "Testing Python modules..."

# Set up environment for testing
export LD_LIBRARY_PATH="$(pwd)/Linux-x86_64:$LD_LIBRARY_PATH"

# Try to load CUDA module if available
if command -v module &> /dev/null; then
    module load cuda/12.1.1 2>/dev/null || true
fi

# Test import from the copied modules in parent directory
cd ..
if python3 -c "import popsift_extract, popsift_match; print('Modules imported successfully!')" 2>/dev/null; then
    print_status "Python modules are working correctly!"
    
    # Show version info
    python3 -c "
import popsift_extract, popsift_match
print('PopSift Extract Version:', popsift_extract.get_version())
print('PopSift Match Version:', popsift_match.get_version())
"
else
    print_error "Failed to import Python modules"
    print_error "Check the build output above for errors"
    print_error "Trying to import from: $(pwd)"
    print_error "Available modules:"
    ls -la popsift_*.so 2>/dev/null || echo "No popsift modules found"
    exit 1
fi

print_status "Build completed successfully!"
print_status "You can now use the Python interface:"
print_status "  python3 popsift_python.py --help"
print_status "  python3 popsift_python.py interactive"
