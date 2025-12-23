#!/usr/bin/env python3
"""
Test script to verify that the refactored ASNIC code works correctly with an actual image.
"""
import numpy as np
from PIL import Image
import skimage.color
from asnic import generate_seeds, mark_seed_points


def test_with_real_image():
    """Test with the provided example image."""
    print("Testing with real image...")
    
    try:
        # Load the example image
        rgb_image = np.array(Image.open("images/G-MYUP - 1995 Letov LK-2M Sluka.jpg"))
        print(f"Loaded image with shape: {rgb_image.shape}")
        
        # Convert the image from RGB to CIELAB
        lab_image = skimage.color.rgb2lab(rgb_image)
        print(f"Converted to LAB with shape: {lab_image.shape}")
        
        # Generate seeds with default parameters
        print("Generating seeds with default parameters...")
        seeds = generate_seeds(lab_image, print_status=True)
        print(f"Generated {len(seeds)} seeds")
        
        # Show seed placement
        print("Marking seed points on image...")
        image_of_seed_placement = mark_seed_points(rgb_image, seeds)
        print(f"Created marked image with shape: {image_of_seed_placement.shape}")
        
        # Test with different parameters
        print("\nGenerating seeds with custom parameters...")
        seeds_custom = generate_seeds(
            lab_image,
            block_size=(50, 50),  # Smaller blocks
            entropy_tr=5,  # Lower threshold
            print_status=True,
            mode_hx=8.0,  # Different spatial bandwidth
            mode_hc=8.0,  # Different color bandwidth
            mode_tolerance=1e-4  # Tighter tolerance
        )
        print(f"Generated {len(seeds_custom)} seeds with custom parameters")
        
        # Test seed marking with custom parameters
        print("Marking seed points with custom parameters...")
        image_of_seed_placement_custom = mark_seed_points(
            rgb_image,
            seeds_custom,
            fill_color=[0, 255, 0],  # Green fill
            border_color=[255, 255, 0],  # Yellow border
            outer_radius=4,
            inner_radius=2
        )
        print(f"Created custom marked image with shape: {image_of_seed_placement_custom.shape}")
        
        print("All tests with real image completed successfully!")
        
    except FileNotFoundError:
        print("Example image not found, using a generated test image instead...")
        
        # Create a test image
        test_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_lab = skimage.color.rgb2lab(test_rgb)
        
        # Test seed generation
        seeds = generate_seeds(test_lab, print_status=True, entropy_tr=2)
        print(f"Generated {len(seeds)} seeds from test image")
        
        # Test seed marking
        marked = mark_seed_points(test_rgb, seeds)
        print(f"Marked {len(seeds)} seeds on test image")
        
        print("Tests with generated image completed successfully!")


if __name__ == "__main__":
    print("Running tests with real image for refactored ASNIC code...\n")
    test_with_real_image()
    print("\nTest completed!")