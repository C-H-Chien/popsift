#!/usr/bin/env python3
"""
Setup script for building PopSift Python bindings with pybind11.

This setup script provides an alternative to CMake for building the Python extensions.
It requires pybind11 to be installed.

Usage:
    python setup.py build_ext --inplace
    python setup.py install

Requirements:
    - pybind11
    - CUDA toolkit
    - PopSift library (built)
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# Get the project directory
project_dir = Path(__file__).parent.absolute()
src_dir = project_dir / "src"
app_dir = src_dir / "application"
generated_dir = src_dir / "generated"


class CMakeExtension(Extension):
    """Extension that uses CMake to build."""
    
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command for CMake-based extensions."""
    
    def build_extensions(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for CMake to find the generated files
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DCMAKE_SOURCE_DIR={project_dir}',
            f'-DCMAKE_INCLUDE_PATH={src_dir}',
            f'-DCMAKE_INCLUDE_PATH={generated_dir}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if sys.platform.startswith("darwin"):
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14']
            if 'ARCHFLAGS' in os.environ:
                cmake_args += ['-DCMAKE_OSX_ARCHITECTURES={}'.format(os.environ['ARCHFLAGS'])]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', str(app_dir)] + cmake_args,
                            cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=self.build_temp)


# Simple pybind11 extension for direct compilation
def create_pybind11_extension(name, sources):
    """Create a pybind11 extension with proper includes and libraries."""
    return Extension(
        name,
        sources,
        include_dirs=[
            str(src_dir),
            str(generated_dir),
            pybind11.get_cmake_dir() + "/../../../include",
            # Add other include directories as needed
        ],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
        extra_link_args=[],
    )


# Define extensions
extensions = [
    CMakeExtension('popsift_extract'),
    CMakeExtension('popsift_match'),
]

# Alternative: Direct pybind11 extensions (uncomment if CMake approach doesn't work)
# extensions = [
#     create_pybind11_extension('popsift_extract', [str(app_dir / 'py_main.cpp')]),
#     create_pybind11_extension('popsift_match', [str(app_dir / 'py_match.cpp')]),
# ]


setup(
    name='popsift-python',
    version='1.0.0',
    author='PopSift Contributors',
    author_email='',
    description='Python bindings for PopSift SIFT feature extraction and matching',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=extensions,
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
