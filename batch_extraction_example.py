#!/usr/bin/env python3
"""
Example script demonstrating batch feature extraction using PopSift.

This shows how to efficiently extract SIFT features from multiple images
using the new batch functions that reuse a single PopSift instance.
"""

import os
import time
import numpy as np
import popsift_extract
from popsift_config import SiftConfig
try:
    import cv2
except ImportError:
    print("Error: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

def load_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img

def extract_features_from_file(image_path, print_timing=False, sift_config=None):
       
    if sift_config is not None:
        result = popsift_extract.extract_features_from_file_with_config(
            image_path, sift_config,
            verbose=False, 
            print_time_info=print_timing
        )
    else:
        result = popsift_extract.extract_features_from_file(
            image_path, 
            verbose=False,
            print_time_info=print_timing
        )
    
    return result

def example_batch_from_files(image_files):
    """Example: Batch extract features from multiple image files."""
    print("=" * 60)
    print("Example: Batch Feature Extraction from Files")
    print("=" * 60)
    
    # Process all images at once (efficient!)
    results = popsift_extract.extract_multiple_from_files(
        image_files,
        verbose=True,
        print_time_info=True
    )
    
    # Process results
    total_time = 0
    for i, result in enumerate(results):
        print(f"\nImage {i} Results:")
        print(f"  Features: {result.num_features}")
        print(f"  Descriptors: {result.num_descriptors}")
        print(f"  GPU time: {result.gpu_time_ms:.2f} ms")
        if result.num_features > 0:
            print(f"  First keypoint: ({result.keypoints_x[0]:.2f}, {result.keypoints_y[0]:.2f})")
        total_time += result.gpu_time_ms
    print(f"Total time with batch: {total_time:.2f} ms")
    
    total_time = 0
    for i in range(len(image_files)):   
        features = extract_features_from_file(image_files[i], print_timing=True)
        total_time += features.gpu_time_ms
        print(f"\nResults:")
        print(f"  Features: {features.num_features}")
        print(f"  Descriptors: {features.num_descriptors}")
        print(f"  Processing time: {features.gpu_time_ms:.2f} ms")
    print(f"Total time without batch: {total_time:.2f} ms")


def example_batch_from_arrays(image_files):
    """Example: Batch extract features from multiple numpy arrays."""
    print("\n" + "=" * 60)
    print("Example: Batch Feature Extraction from NumPy Arrays")
    print("=" * 60)
    
    # Create some dummy images (replace with actual image loading)
    images = [
        load_img(image_files[0]),
        load_img(image_files[1]),
        load_img(image_files[2]),
        load_img(image_files[3])
    ]
    
    # Process all images at once (efficient!)
    start = time.time()
    results = popsift_extract.extract_multiple_from_arrays(
        images,
        verbose=True,
        print_time_info=True
    )
    time_batch = time.time() - start
    print(f"Batch processing time: {time_batch:.2f} seconds")
    
    # Process results
    # for i, result in enumerate(results):
    #     print(f"\nImage {i} Results:")
    #     print(f"  Features: {result.num_features}")
    #     print(f"  Descriptors: {result.num_descriptors}")
    #     print(f"  GPU time: {result.gpu_time_ms:.2f} ms")

    start = time.time()
    for img in images:
        result = popsift_extract.extract_features_from_array(img, verbose=False, print_time_info=False)
    time_single = time.time() - start
    print(f"Single processing time: {time_single:.2f} seconds")


def example_batch_with_custom_config():
    """Example: Batch extract with custom SIFT configuration."""
    print("\n" + "=" * 60)
    print("Example: Batch Extraction with Custom Configuration")
    print("=" * 60)
    
    # Create custom SIFT configuration
    config = SiftConfig()
    config.octaves = 4
    config.levels = 3
    config.threshold = 0.01
    config.filter_sort = "up"  # Deterministic filtering
    
    # Create dummy images
    images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    # Process with custom config
    results = popsift_extract.extract_multiple_from_arrays_with_config(
        images,
        config,
        verbose=True,
        print_time_info=True
    )
    
    print(f"\nProcessed {len(results)} images with custom configuration")
    for i, result in enumerate(results):
        print(f"Image {i}: {result.num_features} features, {result.num_descriptors} descriptors")


def comparison_single_vs_batch():
    """Compare performance: processing images one-by-one vs batch."""
    print("\n" + "=" * 60)
    print("Performance Comparison: Single vs Batch Extraction")
    print("=" * 60)
    
    # Create test images
    num_images = 10
    images = [np.random.randint(0, 256, (480, 640), dtype=np.uint8) for _ in range(num_images)]
    
    # Method 1: Process one image at a time
    print(f"\nMethod 1: Processing {num_images} images one at a time...")
    start = time.time()
    results_single = []
    for img in images:
        result = popsift_extract.extract_features_from_array(img, verbose=False)
        results_single.append(result)
    time_single = time.time() - start
    
    # Method 2: Batch process all images
    print(f"Method 2: Batch processing {num_images} images...")
    start = time.time()
    results_batch = popsift_extract.extract_multiple_from_arrays(images, verbose=False)
    time_batch = time.time() - start
    
    # Compare
    print(f"\nResults:")
    print(f"  Single processing: {time_single:.3f} seconds")
    print(f"  Batch processing:  {time_batch:.3f} seconds")
    print(f"  Speedup: {time_single/time_batch:.2f}x faster")
    print(f"\n  Benefits:")
    print(f"  - Reusing single PopSift instance (no repeated initialization)")
    print(f"  - Pipeline keeps GPU busy while uploading next images")
    print(f"  - Reduced CPU-GPU synchronization overhead")


def main():
    """Run all examples."""
    print("PopSift Batch Extraction Examples")
    print(f"Version: {popsift_extract.get_version()}\n")  

    dataset_path = "/gpfs/data/bkimia/Datasets/LaMAR/CAB/raw/"
    instance = "hl_2021-06-02-11-31-59-805/raw_data/"
    
    image_files = [
        os.path.join(dataset_path, instance, "images/hetlf/326766299.jpg"),
        os.path.join(dataset_path, instance, "images/hetlf/327591274.jpg"),
        os.path.join(dataset_path, instance, "images/hetlf/329241225.jpg"),
        os.path.join(dataset_path, instance, "images/hetlf/331650154.jpg"),
    ]
    # example_batch_from_files(image_files)

    example_batch_from_arrays(image_files)
    # example_batch_with_custom_config()
    
    # Performance comparison (works with dummy data):
    # comparison_single_vs_batch()


if __name__ == "__main__":
    main()

