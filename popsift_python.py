#!/usr/bin/env python3
"""
PopSift Python Interface

This script provides a high-level Python interface to the PopSift library
for SIFT feature extraction and matching using pybind11 bindings.

Usage:
    python popsift_python.py extract <image_file> [options]
    python popsift_python.py match <left_image> <right_image> [options]
    python popsift_python.py interactive

Requirements:
    - numpy
    - opencv-python (for image loading)
    - matplotlib (for visualization)
    - The compiled PopSift Python modules (popsift_extract and popsift_match)
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    print("Warning: matplotlib not available. Visualization features will be disabled.")
    plt = None

# Try to import the PopSift modules
try:
    import popsift_config
    import popsift_extract
    import popsift_match
    POPSIFT_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import PopSift modules: {e}")
    print("Make sure the modules are compiled and in your Python path.")
    POPSIFT_AVAILABLE = False


class PopSiftProcessor:
    """High-level interface to PopSift functionality."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        if not POPSIFT_AVAILABLE:
            raise RuntimeError("PopSift modules are not available")
    
    def load_image(self, image_path):
        """Load an image and convert to grayscale numpy array."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if self.verbose:
            print(f"Loaded image: {img.shape[0]} x {img.shape[1]} pixels")
        
        return img
    
    def extract_features_from_file(self, image_path, print_timing=False, sift_config=None):
        """Extract SIFT features from an image file."""
        if self.verbose:
            print(f"Extracting features from: {image_path}")
        
        if sift_config is not None:
            result = popsift_extract.extract_features_from_file_with_config(
                image_path, sift_config,
                verbose=self.verbose, 
                print_time_info=print_timing
            )
        else:
            result = popsift_extract.extract_features_from_file(
                image_path, 
                verbose=self.verbose, 
                print_time_info=print_timing
            )
        
        if self.verbose:
            print(f"Extracted {result.num_features} keypoints and {result.num_descriptors} descriptors")
            if print_timing:
                print(f"Processing time: {result.gpu_time_ms:.2f} ms")
        
        return result
    
    def extract_features_from_array(self, image_array, print_timing=False, sift_config=None):
        """Extract SIFT features from a numpy array."""
        if self.verbose:
            print(f"Extracting features from array: {image_array.shape[1]} x {image_array.shape[0]} pixels")
        
        if sift_config is not None:
            result = popsift_extract.extract_features_from_array_with_config(
                image_array, sift_config,
                verbose=self.verbose, 
                print_time_info=print_timing
            )
        else:
            result = popsift_extract.extract_features_from_array(
                image_array, 
                verbose=self.verbose, 
                print_time_info=print_timing
            )
        
        if self.verbose:
            print(f"Extracted {result.num_features} keypoints and {result.num_descriptors} descriptors")
            if print_timing:
                print(f"Processing time: {result.gpu_time_ms:.2f} ms")
        
        return result
    
    def match_features_from_files(self, left_path, right_path, print_timing=False, sift_config=None):
        """Match SIFT features between two image files."""
        if self.verbose:
            print(f"Matching features between: {left_path} <-> {right_path}")
        
        if sift_config is not None:
            result = popsift_match.match_features_from_files_with_config(
                left_path, right_path, sift_config,
                verbose=self.verbose,
                print_time_info=print_timing
            )
        else:
            result = popsift_match.match_features_from_files(
                left_path, right_path,
                verbose=self.verbose,
                print_time_info=print_timing
            )
        
        if self.verbose:
            print(f"Found {result.num_matches} matches")
            if print_timing:
                print(f"Matching time: {result.match_time_ms:.2f} ms")
        
        return result
    
    def match_features_from_arrays(self, left_array, right_array, print_timing=False, sift_config=None):
        """Match SIFT features between two numpy arrays."""
        if self.verbose:
            print(f"Matching features between arrays: {left_array.shape} <-> {right_array.shape}")
        
        if sift_config is not None:
            result = popsift_match.match_features_from_arrays_with_config(
                left_array, right_array, sift_config,
                verbose=self.verbose,
                print_time_info=print_timing
            )
        else:
            result = popsift_match.match_features_from_arrays(
                left_array, right_array,
                verbose=self.verbose,
                print_time_info=print_timing
            )
        
        if self.verbose:
            print(f"Found {result.num_matches} matches")
            if print_timing:
                print(f"Matching time: {result.match_time_ms:.2f} ms")

        return result
    
    def visualize_features(self, image_array, features, title="SIFT Features", max_features=100):
        """Visualize SIFT features on an image."""
        if plt is None:
            print("Warning: matplotlib not available for visualization")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_array, cmap='gray')
        
        # Plot keypoints
        num_to_show = min(max_features, len(features.keypoints_x))
        for i in range(num_to_show):
            x, y = features.keypoints_x[i], features.keypoints_y[i]
            scale = features.keypoints_scale[i]
            orientation = features.keypoints_orientation[i]
            
            # Draw circle for scale
            circle = patches.Circle((x, y), scale, fill=False, color='red', alpha=0.7)
            ax.add_patch(circle)
            
            # Draw orientation line
            end_x = x + scale * np.cos(orientation)
            end_y = y + scale * np.sin(orientation)
            ax.plot([x, end_x], [y, end_y], 'red', linewidth=1, alpha=0.7)
        
        ax.set_title(f"{title} ({num_to_show} of {len(features.keypoints_x)} features)")
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def visualize_matches(self, left_array, right_array, left_features, right_features, 
                         matches, title="SIFT Matches", max_matches=50):
        """Visualize SIFT feature matches between two images."""
        if plt is None:
            print("Warning: matplotlib not available for visualization")
            return
        
        if self.verbose:
            print(f"Visualizing matches: {len(matches.matches_left_idx)} accepted matches")
            print(f"Left features: {len(left_features.keypoints_x)} keypoints")
            print(f"Right features: {len(right_features.keypoints_x)} keypoints")
        
        h1, w1 = left_array.shape
        h2, w2 = right_array.shape
        
        # Create combined image
        combined = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
        combined[:h1, :w1] = left_array
        combined[:h2, w1:] = right_array
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(combined, cmap='gray')
        
        # Sort matches by distance (ascending order - best matches first)
        if len(matches.matches_left_idx) > 0 and len(matches.match_distances) > 0:
            # Create list of (distance, left_idx, right_idx) tuples
            match_tuples = list(zip(matches.match_distances, matches.matches_left_idx, matches.matches_right_idx))
            # Sort by distance (ascending)
            match_tuples.sort(key=lambda x: x[0])
            
            if self.verbose:
                print(f"Sorted matches by distance. Best match distance: {match_tuples[0][0]:.4f}, Worst match distance: {match_tuples[-1][0]:.4f}")
        else:
            match_tuples = []
        
        # Plot matches (now in distance-sorted order)
        num_to_show = min(max_matches, len(match_tuples))
        matches_plotted = 0
        
        for i in range(num_to_show):
            distance, left_idx, right_idx = match_tuples[i]
            
            # Bounds checking
            if (left_idx < len(left_features.keypoints_x) and 
                right_idx < len(right_features.keypoints_x) and
                left_idx >= 0 and right_idx >= 0):
                
                x1 = left_features.keypoints_x[left_idx]
                y1 = left_features.keypoints_y[left_idx]
                x2 = right_features.keypoints_x[right_idx] + w1  # Offset for right image
                y2 = right_features.keypoints_y[right_idx]
                
                # Additional bounds checking for image coordinates
                if (0 <= x1 < w1 and 0 <= y1 < h1 and 
                    0 <= x2 < w1 + w2 and 0 <= y2 < h2):
                    
                    # Draw match line with color based on distance (lower distance = better match)
                    # Since matches are now sorted by distance, we can use a ranking-based color
                    if len(match_tuples) > 1:
                        # Color based on rank (0 = best match, 1 = worst match)
                        rank_ratio = i / (len(match_tuples) - 1) if len(match_tuples) > 1 else 0
                        # Best matches are bright green, worst are red
                        if rank_ratio < 0.3:  # Top 30% - bright green
                            color = (0, 1.0, 0)  # Green
                        elif rank_ratio < 0.7:  # Middle 40% - yellow
                            color = (1.0, 1.0, 0)  # Yellow  
                        else:  # Bottom 30% - red
                            color = (1.0, 0.3, 0)  # Orange-red
                    else:
                        color = 'green'  # Single match
                    
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.7)
                    # Draw keypoints
                    ax.plot(x1, y1, 'ro', markersize=3)
                    ax.plot(x2, y2, 'ro', markersize=3)
                    matches_plotted += 1
                else:
                    if self.verbose:
                        print(f"Warning: Match {i} has out-of-bounds coordinates: ({x1}, {y1}) -> ({x2}, {y2})")
            else:
                if self.verbose:
                    print(f"Warning: Match {i} has invalid feature indices: left[{left_idx}] right[{right_idx}]")
        
        if self.verbose:
            print(f"Successfully plotted {matches_plotted} matches out of {num_to_show} attempted")
        
        acceptance_rate = (matches.num_matches / matches.num_total_matches * 100) if matches.num_total_matches > 0 else 0
        if len(match_tuples) > 0 and matches_plotted > 0:
            best_distance = match_tuples[0][0]
            worst_distance = match_tuples[min(matches_plotted-1, len(match_tuples)-1)][0]
            ax.set_title(f"{title} - Top {matches_plotted} matches by quality\n"
                        f"Best: {best_distance:.3f}, Worst shown: {worst_distance:.3f} "
                        f"({acceptance_rate:.1f}% acceptance rate)")
        else:
            ax.set_title(f"{title} ({matches_plotted} of {matches.num_matches} accepted matches plotted, {acceptance_rate:.1f}% acceptance rate)")
        ax.axis('off')
        plt.tight_layout()
        return fig


def create_sift_config(**kwargs):
    """Create a SiftConfig object with custom parameters.
    
    Available parameters:
    - octaves: Number of octaves (default: use PopSift default)
    - levels: Number of levels per octave (default: use PopSift default)
    - sigma: Initial sigma value (default: use PopSift default)
    - threshold: Contrast threshold (default: use PopSift default)
    - edge_threshold: Edge threshold (default: use PopSift default)
    - downsampling: Downscale factor (default: use PopSift default)
    - initial_blur: Initial blur value (default: use PopSift default)
    - gauss_mode: Gaussian mode ("VLFeat_Compute", "VLFeat_Relative", "VLFeat_Relative_All", "OpenCV_Compute", "Fixed9", "Fixed15")
    - desc_mode: Descriptor mode ("Loop", "ILoop", "RGrid", "IGrid", "VGrid", "VLFeat")
    - popsift_mode: Enable PopSift mode ("true" or "false")
    - vlfeat_mode: Enable VLFeat mode ("true" or "false")
    - opencv_mode: Enable OpenCV mode ("true" or "false")
    - direct_scaling: Enable direct scaling ("true" or "false")
    - norm_multi: Normalization multiplier (default: use PopSift default)
    - norm_mode: Normalization mode ("L2", "RootSift")
    - root_sift: Enable RootSift ("true" or "false")
    - filter_max_extrema: Maximum extrema for filtering (default: use PopSift default)
    - filter_grid: Grid size for filtering (default: use PopSift default)
    - filter_sort: Filter sorting ("random", "up", "down")
    - print_gauss_tables: Print Gaussian tables (True/False)
    
    Example:
    config = create_sift_config(threshold=0.04, octaves=4, levels=3, vlfeat_mode="true")
    
    Note: The returned SiftConfig works with both extraction and matching functions.
    """
    if not POPSIFT_AVAILABLE:
        raise ImportError("PopSift modules not available")
    
    # Use the SiftConfig from the dedicated config module
    config = popsift_config.SiftConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown SIFT parameter: {key}")
    
    return config

def save_features_to_file(features, output_path):
    """Save SIFT features to a text file."""
    with open(output_path, 'w') as f:
        f.write(f"# SIFT Features\n")
        f.write(f"# Number of features: {features.num_features}\n")
        f.write(f"# Number of descriptors: {features.num_descriptors}\n")
        f.write(f"# Processing time: {features.gpu_time_ms:.2f} ms\n")
        f.write(f"# Format: x y scale orientation descriptor[128]\n")
        
        for i in range(features.num_features):
            x = features.keypoints_x[i]
            y = features.keypoints_y[i]
            scale = features.keypoints_scale[i]
            orientation = features.keypoints_orientation[i]
            
            f.write(f"{x:.3f} {y:.3f} {scale:.3f} {orientation:.3f}")
            
            if i < len(features.descriptors):
                desc = features.descriptors[i]
                for val in desc:
                    f.write(f" {val:.6f}")
            f.write("\n")
    
    print(f"Features saved to: {output_path}")


def interactive_mode():
    """Interactive mode for exploring PopSift functionality."""
    if not POPSIFT_AVAILABLE:
        print("PopSift modules are not available. Cannot run interactive mode.")
        return
    
    processor = PopSiftProcessor(verbose=True)
    
    print("\n=== PopSift Interactive Mode ===")
    print("Available commands:")
    print("  1. Extract features from image file")
    print("  2. Match features between two image files")
    print("  3. Show version information")
    print("  4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                image_path = input("Enter image file path: ").strip()
                if os.path.exists(image_path):
                    features = processor.extract_features_from_file(image_path, print_timing=True)
                    print(f"\nResults:")
                    print(f"  Features: {features.num_features}")
                    print(f"  Descriptors: {features.num_descriptors}")
                    print(f"  Processing time: {features.gpu_time_ms:.2f} ms")
                    
                    # Visualize if matplotlib is available
                    if plt:
                        img = processor.load_image(image_path)
                        fig = processor.visualize_features(img, features)
                        plt.show()
                else:
                    print("File not found!")
            
            elif choice == '2':
                left_path = input("Enter left image file path: ").strip()
                right_path = input("Enter right image file path: ").strip()
                
                if os.path.exists(left_path) and os.path.exists(right_path):
                    matches = processor.match_features_from_files(left_path, right_path, print_timing=True)
                    print(f"\nResults:")
                    print(f"  Matches found: {matches.num_matches}")
                    print(f"  Matching time: {matches.match_time_ms:.2f} ms")
                    
                    # Visualize if matplotlib is available
                    if plt:
                        left_img = processor.load_image(left_path)
                        right_img = processor.load_image(right_path)
                        left_features = processor.extract_features_from_array(left_img)
                        right_features = processor.extract_features_from_array(right_img)
                        fig = processor.visualize_matches(left_img, right_img, 
                                                       left_features, right_features, matches)
                        plt.show()
                else:
                    print("One or both files not found!")
            
            elif choice == '3':
                print(f"PopSift Extract Version: {popsift_extract.get_version()}")
                print(f"PopSift Match Version: {popsift_match.get_version()}")
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='PopSift Python Interface')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--timing', '-t', action='store_true', help='Print timing information')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract SIFT features')
    extract_parser.add_argument('image', help='Input image file')
    extract_parser.add_argument('--output', '-o', help='Output file for features')
    extract_parser.add_argument('--visualize', action='store_true', help='Visualize features')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match SIFT features from image files')
    match_parser.add_argument('left', help='Left image file')
    match_parser.add_argument('right', help='Right image file')
    match_parser.add_argument('--output', '-o', help='Output file for matches')
    match_parser.add_argument('--visualize', action='store_true', help='Visualize matches')
    
    # Match images command
    match_images_parser = subparsers.add_parser('match-images', help='Match SIFT features from numpy arrays which could be images captured by a camera')
    match_images_parser.add_argument('left', help='Left image file')
    match_images_parser.add_argument('right', help='Right image file')
    match_images_parser.add_argument('--output', '-o', help='Output file for matches')
    match_images_parser.add_argument('--visualize', action='store_true', help='Visualize matches')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')
    
    args = parser.parse_args()

    #> some customized sift config
    # rootsift_config = create_sift_config(root_sift="true", norm_mode="RootSift")
    custom_config = create_sift_config(
        threshold=0.04, 
        octaves=4, 
        levels=3, 
        vlfeat_mode="true",
        filter_sort="up"  # Use deterministic sorting instead of random!
    )
    
    if not POPSIFT_AVAILABLE:
        print("Error: PopSift modules are not available.")
        print("Please compile the pybind11 modules first.")
        sys.exit(1)
    
    processor = PopSiftProcessor(verbose=args.verbose)
    
    if args.command == 'extract':
        try:
            features = processor.extract_features_from_file(args.image, print_timing=args.timing)
            
            if args.output:
                save_features_to_file(features, args.output)
            
            if args.visualize and plt:
                img = processor.load_image(args.image)
                fig = processor.visualize_features(img, features)
                plt.show()
            
            print(f"Successfully extracted {features.num_features} features from {args.image}")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'match':
        try:
            matches = processor.match_features_from_files(args.left, args.right, print_timing=args.timing, sift_config=custom_config)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(f"# SIFT Feature Matches\n")
                    f.write(f"# Left image: {args.left}\n")
                    f.write(f"# Right image: {args.right}\n")
                    f.write(f"# Number of matches: {matches.num_matches}\n")
                    f.write(f"# Matching time: {matches.match_time_ms:.2f} ms\n")
                    f.write(f"# Format: left_idx right_idx distance\n")
                    
                    for i in range(len(matches.matches_left_idx)):
                        f.write(f"{matches.matches_left_idx[i]} {matches.matches_right_idx[i]} {matches.match_distances[i]:.6f}\n")
                
                print(f"Matches saved to: {args.output}")
            
            if args.visualize and plt:
                left_img = processor.load_image(args.left)
                right_img = processor.load_image(args.right)
                left_features = processor.extract_features_from_array(left_img)
                right_features = processor.extract_features_from_array(right_img)
                fig = processor.visualize_matches(left_img, right_img, 
                                               left_features, right_features, matches)
                plt.show()
            
            acceptance_rate = (matches.num_matches / matches.num_total_matches * 100) if matches.num_total_matches > 0 else 0
            print(f"Successfully found {matches.num_matches} accepted matches out of {matches.num_total_matches} total ({acceptance_rate:.1f}% acceptance rate)")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'match-images':
        #> This is actually useful when interfacing with cameras (e.g., Arducam cameras)
        try:
            # Load images as arrays
            left_img = processor.load_image(args.left)
            right_img = processor.load_image(args.right)
            
            # Match features from arrays
            matches = processor.match_features_from_arrays(left_img, right_img, print_timing=args.timing, sift_config=custom_config)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(f"# SIFT Feature Matches (from arrays)\n")
                    f.write(f"# Left image: {args.left}\n")
                    f.write(f"# Right image: {args.right}\n")
                    f.write(f"# Number of accepted matches: {matches.num_matches}\n")
                    f.write(f"# Total potential matches: {matches.num_total_matches}\n")
                    f.write(f"# Matching time: {matches.match_time_ms:.2f} ms\n")
                    f.write(f"# Format: left_idx right_idx distance\n")
                    
                    for i in range(len(matches.matches_left_idx)):
                        f.write(f"{matches.matches_left_idx[i]} {matches.matches_right_idx[i]} {matches.match_distances[i]:.6f}\n")
                
                print(f"Matches saved to: {args.output}")
            
            if args.visualize and plt:
                # Extract features for visualization
                left_features = processor.extract_features_from_array(left_img)
                right_features = processor.extract_features_from_array(right_img)
                fig = processor.visualize_matches(left_img, right_img, 
                                               left_features, right_features, 
                                               matches)
                plt.show()
            
            acceptance_rate = (matches.num_matches / matches.num_total_matches * 100) if matches.num_total_matches > 0 else 0
            print(f"Successfully found {matches.num_matches} accepted matches out of {matches.num_total_matches} total ({acceptance_rate:.1f}% acceptance rate) between {args.left} and {args.right}")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'interactive':
        interactive_mode()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
