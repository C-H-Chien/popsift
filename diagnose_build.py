#!/usr/bin/env python3
"""
Diagnostic script for PopSift Python bindings build issues.

This script helps identify common problems that prevent the Python modules
from building correctly.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_command(cmd, description):
    """Check if a command is available."""
    if shutil.which(cmd):
        print(f"✓ {description}: {cmd} found")
        return True
    else:
        print(f"✗ {description}: {cmd} not found")
        return False

def check_python_package(package, description):
    """Check if a Python package is available."""
    try:
        __import__(package)
        print(f"✓ {description}: {package} available")
        return True
    except ImportError:
        print(f"✗ {description}: {package} not available")
        return False

def check_file(file_path, description):
    """Check if a file exists."""
    if os.path.exists(file_path):
        print(f"✓ {description}: {file_path} exists")
        return True
    else:
        print(f"✗ {description}: {file_path} not found")
        return False

def check_directory(dir_path, description):
    """Check if a directory exists."""
    if os.path.isdir(dir_path):
        print(f"✓ {description}: {dir_path} exists")
        return True
    else:
        print(f"✗ {description}: {dir_path} not found")
        return False

def run_command(cmd, description):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description}: Success")
            return True
        else:
            print(f"✗ {description}: Failed")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description}: Exception - {e}")
        return False

def main():
    print("PopSift Python Bindings Diagnostic Tool")
    print("=" * 50)
    
    # Check basic requirements
    print("\n1. Checking Basic Requirements:")
    cmake_ok = check_command("cmake", "CMake")
    python_ok = check_command("python3", "Python 3")
    nvcc_ok = check_command("nvcc", "CUDA Compiler")
    
    # Check Python packages
    print("\n2. Checking Python Packages:")
    pybind11_ok = check_python_package("pybind11", "pybind11")
    numpy_ok = check_python_package("numpy", "NumPy")
    
    # Check project structure
    print("\n3. Checking Project Structure:")
    project_root = Path(__file__).parent
    cmake_ok = check_file(project_root / "CMakeLists.txt", "Main CMakeLists.txt")
    py_main_ok = check_file(project_root / "src" / "application" / "py_main.cpp", "py_main.cpp")
    py_match_ok = check_file(project_root / "src" / "application" / "py_match.cpp", "py_match.cpp")
    pybind_cmake_ok = check_file(project_root / "src" / "application" / "pybind_CMakeLists.txt", "pybind_CMakeLists.txt")
    
    # Check if main library is built
    print("\n4. Checking PopSift Library:")
    build_dir = project_root / "build"
    lib_ok = check_file(build_dir / "Linux-x86_64" / "libpopsift.so", "libpopsift.so")
    
    if not lib_ok:
        print("  Note: Main PopSift library may need to be built first")
        print("  Run: mkdir -p build && cd build && cmake .. && make")
    
    # Check Python version and pybind11 version
    print("\n5. Checking Python Environment:")
    try:
        import pybind11
        print(f"✓ pybind11 version: {pybind11.__version__}")
        print(f"✓ pybind11 include path: {pybind11.get_cmake_dir()}")
    except ImportError:
        print("✗ pybind11 not available")
    
    try:
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not available")
    
    # Check CUDA environment
    print("\n6. Checking CUDA Environment:")
    if nvcc_ok:
        try:
            result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ CUDA version info:")
                for line in result.stdout.split('\n')[:3]:
                    if line.strip():
                        print(f"  {line}")
        except:
            print("✗ Could not get CUDA version")
    
    # Test CMake configuration
    print("\n7. Testing CMake Configuration:")
    if cmake_ok and python_ok and pybind11_ok:
        test_build_dir = project_root / "test_build"
        if test_build_dir.exists():
            shutil.rmtree(test_build_dir)
        
        os.makedirs(test_build_dir)
        os.chdir(test_build_dir)
        
        cmake_cmd = f"cmake -DPopSift_BUILD_PYTHON_BINDINGS=ON -DBUILD_SHARED_LIBS=ON -DPython3_EXECUTABLE={sys.executable} .."
        cmake_config_ok = run_command(cmake_cmd, "CMake configuration")
        
        if cmake_config_ok:
            print("✓ CMake configuration successful")
            print("  You can now run: make")
            
            # Check if Python targets were created
            result = subprocess.run("cmake --build . --dry-run 2>/dev/null | grep -E 'popsift_extract|popsift_match' || true", 
                                  shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                print("✓ Python binding targets found in build configuration")
            else:
                print("⚠ Python binding targets not found - check CMake output")
        else:
            print("✗ CMake configuration failed")
        
        os.chdir(project_root)
        if test_build_dir.exists():
            shutil.rmtree(test_build_dir)
    
    # Summary and recommendations
    print("\n8. Summary and Recommendations:")
    
    if not (cmake_ok and python_ok and nvcc_ok and pybind11_ok):
        print("✗ Missing required dependencies:")
        if not cmake_ok:
            print("  - Install CMake")
        if not python_ok:
            print("  - Install Python 3")
        if not nvcc_ok:
            print("  - Install CUDA toolkit")
        if not pybind11_ok:
            print("  - Install pybind11: pip install pybind11")
    
    if not lib_ok:
        print("✗ Main PopSift library not built:")
        print("  - Build main library first: mkdir build && cd build && cmake .. && make")
    
    if cmake_ok and python_ok and nvcc_ok and pybind11_ok and lib_ok:
        print("✓ All dependencies available!")
        print("  Try building with:")
        print("  ./build_python_bindings.sh cmake")
        print("  or")
        print("  python setup.py build_ext --inplace")

if __name__ == "__main__":
    main()
