#!/usr/bin/env python3
"""
Example script demonstrating batch processing of multiple image pairs using PopSift.

This shows how to efficiently match multiple image pairs using the new batch functions
that reuse a single PopSift instance and enqueue all images at once.
"""

import os
import numpy as np
import popsift_match
from popsift_config import SiftConfig

def example_batch_from_files(left_files, right_files):
    """Example: Batch process multiple image pairs from files."""
    print("=" * 60)
    print("Example: Batch Processing from Files")
    print("=" * 60)
    
    # Process all pairs at once (efficient!)
    results = popsift_match.match_multiple_pairs_from_files(
        left_files,
        right_files,
        verbose=True,
        print_time_info=True
    )
    
    # Process results
    for i, result in enumerate(results):
        print(f"\nPair {i} Results:")
        print(f"  Number of Accepted Matches: {result.num_matches}")
        print(f"  Left GPU time: {result.left_gpu_time_ms:.2f} ms")
        print(f"  Right GPU time: {result.right_gpu_time_ms:.2f} ms")
        print(f"  Matching time: {result.match_time_ms:.2f} ms")
        if result.num_matches > 0:
            print(f"  Average match distance: {np.mean(result.match_distances):.2f}")


def example_batch_from_arrays():
    """Example: Batch process multiple image pairs from numpy arrays."""
    print("\n" + "=" * 60)
    print("Example: Batch Processing from NumPy Arrays")
    print("=" * 60)
    
    # Create some dummy images (replace with actual image loading)
    # In practice, you'd load these from files or a dataset
    left_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    right_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    # Process all pairs at once (efficient!)
    results = popsift_match.match_multiple_pairs_from_arrays(
        left_images,
        right_images,
        verbose=True,
        print_time_info=True
    )
    
    # Process results
    for i, result in enumerate(results):
        print(f"\nPair {i} Results:")
        print(f"  Matches: {result.num_matches}")
        print(f"  Total matches (before filtering): {result.num_total_matches}")
        print(f"  Left GPU time: {result.left_gpu_time_ms:.2f} ms")
        print(f"  Right GPU time: {result.right_gpu_time_ms:.2f} ms")
        print(f"  Matching time: {result.match_time_ms:.2f} ms")


def example_batch_with_custom_config():
    """Example: Batch process with custom SIFT configuration."""
    print("\n" + "=" * 60)
    print("Example: Batch Processing with Custom Configuration")
    print("=" * 60)
    
    # Create custom SIFT configuration
    config = SiftConfig()
    config.octaves = 4
    config.levels = 3
    config.threshold = 0.01
    
    # Create dummy images
    left_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    right_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),
    ]
    
    # Process with custom config
    results = popsift_match.match_multiple_pairs_from_arrays_with_config(
        left_images,
        right_images,
        config,
        verbose=True,
        print_time_info=True
    )
    
    print(f"\nProcessed {len(results)} pairs with custom configuration")
    for i, result in enumerate(results):
        print(f"Pair {i}: {result.num_matches} matches")


def verify_files_vs_arrays_consistency():
    """Verify that file-based and array-based matching give identical results."""
    print("\n" + "=" * 60)
    print("Verification: Files vs Arrays Consistency")
    print("=" * 60)
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping verification test")
        return
    
    # Test with actual files if available
    dataset_path = "/gpfs/data/bkimia/Datasets/LaMAR/CAB/raw/"
    instance = "hl_2021-06-02-11-31-59-805/raw_data/"
    
    left_file = os.path.join(dataset_path, instance, "images/hetlf/326766299.jpg")
    right_file = os.path.join(dataset_path, instance, "images/hetrf/326766299.jpg")
    
    if not os.path.exists(left_file) or not os.path.exists(right_file):
        print(f"Test files not found, skipping verification")
        return
    
    # Method 1: Match from files
    print("\nMethod 1: Matching from files...")
    result_files = popsift_match.match_features_from_files(
        left_file,
        right_file,
        verbose=False
    )
    
    # Method 2: Load images and match from arrays
    print("Method 2: Matching from arrays...")
    left_img = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)
    
    result_arrays = popsift_match.match_features_from_arrays(
        left_img,
        right_img,
        verbose=False
    )
    
    # Compare results
    print("\n" + "-" * 60)
    print("Comparison:")
    print(f"  File-based matches:  {result_files.num_matches}")
    print(f"  Array-based matches: {result_arrays.num_matches}")
    
    if result_files.num_matches == result_arrays.num_matches:
        print("  ✓ Number of matches: IDENTICAL")
    else:
        print("  ✗ Number of matches: DIFFERENT!")
    
    # Check if match indices are the same
    if (result_files.matches_left_idx == result_arrays.matches_left_idx and
        result_files.matches_right_idx == result_arrays.matches_right_idx):
        print("  ✓ Match indices: IDENTICAL")
    else:
        print("  ✗ Match indices: DIFFERENT!")
    
    # Check distances
    if result_files.match_distances == result_arrays.match_distances:
        print("  ✓ Match distances: IDENTICAL")
    else:
        # Allow for small floating point differences
        if len(result_files.match_distances) == len(result_arrays.match_distances):
            max_diff = max(abs(a - b) for a, b in zip(result_files.match_distances, result_arrays.match_distances))
            print(f"  ~ Match distances: Max difference = {max_diff:.6f}")
        else:
            print("  ✗ Match distances: DIFFERENT LENGTH!")
    
    print("\nConclusion: Both methods should produce identical results!")
    print("-" * 60)


def comparison_single_vs_batch():
    """Compare performance: processing pairs one-by-one vs batch."""
    print("\n" + "=" * 60)
    print("Performance Comparison: Single vs Batch Processing")
    print("=" * 60)
    
    import time
    
    # Create test images
    num_pairs = 5
    left_images = [np.random.randint(0, 256, (480, 640), dtype=np.uint8) for _ in range(num_pairs)]
    right_images = [np.random.randint(0, 256, (480, 640), dtype=np.uint8) for _ in range(num_pairs)]
    
    # Method 1: Process one pair at a time
    print(f"\nMethod 1: Processing {num_pairs} pairs one at a time...")
    start = time.time()
    results_single = []
    for left, right in zip(left_images, right_images):
        result = popsift_match.match_features_from_arrays(left, right, verbose=False)
        results_single.append(result)
    time_single = time.time() - start
    
    # Method 2: Batch process all pairs
    print(f"Method 2: Batch processing {num_pairs} pairs...")
    start = time.time()
    results_batch = popsift_match.match_multiple_pairs_from_arrays(
        left_images, 
        right_images, 
        verbose=False
    )
    time_batch = time.time() - start
    
    # Compare
    print(f"\nResults:")
    print(f"  Single processing: {time_single:.3f} seconds")
    print(f"  Batch processing:  {time_batch:.3f} seconds")
    print(f"  Speedup: {time_single/time_batch:.2f}x faster")
    print(f"\n  Note: Speedup comes from:")
    print(f"  - Reusing single PopSift instance (no repeated initialization)")
    print(f"  - Pipeline keeps GPU busy while uploading next images")
    print(f"  - Reduced CPU-GPU synchronization overhead")


def main():
    """Run all examples."""
    print("PopSift Batch Matching Examples")
    print(f"Version: {popsift_match.get_version()}\n")
    
    # Verify that files and arrays produce identical results
    print("Running verification test...")
    
    # Uncomment the examples you want to run:
    
    # example_batch_from_arrays()
    # example_batch_with_custom_config()
 
    # If you have actual image files, uncomment this:
    dataset_path = "/gpfs/data/bkimia/Datasets/LaMAR/CAB/raw/"
    instance = "hl_2021-06-02-11-31-59-805/raw_data/"
    
    left_files = [
        os.path.join(dataset_path, instance, "images/hetlf/326766299.jpg"),
        os.path.join(dataset_path, instance, "images/hetlf/327591274.jpg"),
        os.path.join(dataset_path, instance, "images/hetlf/329241225.jpg"),
        os.path.join(dataset_path, instance, "images/hetlf/331650154.jpg"),
    ]
    right_files = [
        os.path.join(dataset_path, instance, "images/hetrf/326766299.jpg"),
        os.path.join(dataset_path, instance, "images/hetrf/327591274.jpg"),
        os.path.join(dataset_path, instance, "images/hetrf/329241225.jpg"),
        os.path.join(dataset_path, instance, "images/hetrf/331650154.jpg"),
    ]

    # example_batch_from_files(left_files, right_files)
    
    comparison_single_vs_batch()


if __name__ == "__main__":
    main()

