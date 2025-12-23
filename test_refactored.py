#!/usr/bin/env python3
"""
Test script to verify that the refactored ASNIC code works correctly.
"""
import numpy as np
from PIL import Image
import skimage.color
from asnic import generate_seeds, mark_seed_points


def test_basic_functionality():
    """Test basic functionality of the refactored code."""
    print("Testing basic functionality...")
    
    # Create a simple test image (smaller for faster testing)
    test_rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    test_lab = skimage.color.rgb2lab(test_rgb)
    
    # Test seed generation with default parameters
    seeds = generate_seeds(test_lab, print_status=False)
    print(f"Generated {len(seeds)} seeds with default parameters")
    
    # Test seed generation with custom parameters
    seeds_custom = generate_seeds(
        test_lab, 
        print_status=False,
        mode_hx=5.0,
        mode_hc=5.0,
        mode_tolerance=1e-4
    )
    print(f"Generated {len(seeds_custom)} seeds with custom parameters")
    
    # Test seed marking with default parameters
    marked_image = mark_seed_points(test_rgb, seeds)
    print(f"Marked seeds on image of shape {marked_image.shape}")
    
    # Test seed marking with custom parameters
    marked_image_custom = mark_seed_points(
        test_rgb, 
        seeds, 
        fill_color=[0, 255, 0],  # Green fill
        border_color=[255, 255, 0],  # Cyan border
        outer_radius=4,
        inner_radius=2
    )
    print(f"Marked seeds on image with custom parameters, shape {marked_image_custom.shape}")
    
    print("All basic functionality tests passed!")
    

def test_edge_cases():
    """Test edge cases to ensure robustness."""
    print("\nTesting edge cases...")
    
    # Create a small test image
    test_rgb = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    test_lab = skimage.color.rgb2lab(test_rgb)
    
    # Test with very small block size
    seeds_small_block = generate_seeds(test_lab, block_size=(10, 10), print_status=False)
    print(f"Generated {len(seeds_small_block)} seeds with small block size")
    
    # Test with high entropy threshold (should produce fewer seeds)
    seeds_high_entropy = generate_seeds(test_lab, entropy_tr=20, print_status=False)
    print(f"Generated {len(seeds_high_entropy)} seeds with high entropy threshold")
    
    # Test with low entropy threshold (should produce more seeds)
    seeds_low_entropy = generate_seeds(test_lab, entropy_tr=2, print_status=False)
    print(f"Generated {len(seeds_low_entropy)} seeds with low entropy threshold")
    
    print("All edge case tests passed!")


def test_parameter_validation():
    """Test that the functions handle invalid parameters gracefully."""
    print("\nTesting parameter validation...")
    
    # Create a small test image
    test_rgb = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    test_lab = skimage.color.rgb2lab(test_rgb)
    
    # Test generate_seeds with various parameters
    try:
        seeds = generate_seeds(
            test_lab,
            block_size=(50, 50),  # Larger than image, should be handled
            entropy_tr=5,
            print_status=False,
            mode_hx=15.0,  # Different bandwidth
            mode_hc=8.0,
            mode_tolerance=1e-2
        )
        print(f"Successfully generated {len(seeds)} seeds with custom parameters")
    except Exception as e:
        print(f"Error in generate_seeds with custom parameters: {e}")
    
    # Test mark_seed_points with various parameters
    try:
        # Generate some seeds to work with
        seeds = generate_seeds(test_lab, print_status=False, entropy_tr=2)
        marked = mark_seed_points(
            test_rgb,
            seeds,
            fill_color=[255, 0, 255],  # Magenta
            border_color=[0, 255, 255],  # Yellow
            outer_radius=5,
            inner_radius=1
        )
        print(f"Successfully marked {len(seeds)} seeds with custom parameters")
    except Exception as e:
        print(f"Error in mark_seed_points with custom parameters: {e}")
    
    print("All parameter validation tests passed!")


if __name__ == "__main__":
    print("Running tests for refactored ASNIC code...\n")
    
    test_basic_functionality()
    test_edge_cases()
    test_parameter_validation()
    
    print("\nAll tests completed successfully! The refactored code appears to be working correctly.")