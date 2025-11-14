#!/usr/bin/env python3
"""
Create a test image for the Photo Filter Application.
This generates a sample image with various features (shapes, text, edges)
to test all the filters effectively.
"""

import cv2
import numpy as np

def create_test_image():
    """Create a test image with various features to test filters."""

    # Create a blank canvas
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Add colorful gradient background
    for i in range(480):
        color_val = int(200 * (i / 480))
        # Clip values to stay within uint8 range
        img[i, :] = [
            np.clip(220 - color_val, 0, 255),
            180,
            np.clip(150 + color_val, 0, 255)
        ]

    # Draw some geometric shapes with different colors
    # Rectangle (red)
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), -1)
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), 3)

    # Circle (green)
    cv2.circle(img, (450, 100), 60, (0, 255, 0), -1)
    cv2.circle(img, (450, 100), 60, (0, 0, 0), 3)

    # Triangle (blue)
    triangle = np.array([[320, 180], [260, 280], [380, 280]], np.int32)
    cv2.fillPoly(img, [triangle], (255, 0, 0))
    cv2.polylines(img, [triangle], True, (0, 0, 0), 3)

    # Add some text with different sizes
    cv2.putText(img, "PHOTO FILTER TEST", (100, 350),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "Test Image for Filters", (120, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)

    # Add some fine details and texture
    # Small circles for texture
    for i in range(10):
        x = np.random.randint(50, 590)
        y = np.random.randint(400, 470)
        cv2.circle(img, (x, y), 5, (100, 100, 100), -1)

    # Add a checkerboard pattern in corner
    square_size = 20
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                x1 = 500 + i * square_size
                y1 = 300 + j * square_size
                cv2.rectangle(img, (x1, y1),
                            (x1 + square_size, y1 + square_size),
                            (0, 0, 0), -1)

    # Add some noise for blur testing
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

def main():
    """Main function to create and save test image."""
    print("Creating test image for Photo Filter Application...")

    # Create test image
    test_img = create_test_image()

    # Save it
    output_path = "test_image.jpg"
    cv2.imwrite(output_path, test_img)
    print(f"Test image created: {output_path}")
    print("\nYou can now test the application with:")
    print(f"  python photo_filter_app.py {output_path}")
    print("\nThe image contains:")
    print("  - Geometric shapes (for edge detection)")
    print("  - Text (for sharpening effects)")
    print("  - Gradients (for blur visualization)")
    print("  - Fine details (for filter comparison)")
    print("  - Noise (for blur testing)")

    # Display the image
    print("\nDisplaying test image (press any key to close)...")
    cv2.imshow("Test Image", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
