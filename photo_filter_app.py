#!/usr/bin/env python3
"""
Real-Time Photo Filtering Application
A comprehensive image processing application with multiple filters and real-time controls.
"""

import cv2
import numpy as np
from datetime import datetime
import os

class PhotoFilterApp:
    """Main application class for real-time photo filtering."""

    def __init__(self):
        """Initialize the application with default parameters."""
        # Current filter mode
        self.filter_mode = 'original'

        # Box blur parameters
        self.box_kernel_size = 5  # Options: 5, 11

        # Gaussian blur parameters
        self.gaussian_kernel_size = 5  # Must be odd
        self.gaussian_sigma = 1.0

        # Sharpening parameters
        self.sharpen_amount = 1.0

        # Canny edge detection parameters
        self.canny_low = 50
        self.canny_high = 150

        # Sobel parameters
        self.sobel_ksize = 3

        # Video source
        self.cap = None
        self.current_frame = None
        self.processed_frame = None

        # Output directory for saved images
        self.output_dir = "filtered_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Window names
        self.main_window = "Photo Filter App - Press 'H' for Help"
        self.control_window = "Filter Controls"

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = cv2.getTickCount()

    def create_control_window(self):
        """Create a window with trackbars for real-time parameter adjustment."""
        cv2.namedWindow(self.control_window)

        # Box blur kernel size (using discrete values)
        cv2.createTrackbar('Box Kernel (0=5x5, 1=11x11)', self.control_window,
                          0, 1, self.on_box_kernel_change)

        # Gaussian blur kernel size (odd values only: 3, 5, 7, 9, 11, 13, 15)
        cv2.createTrackbar('Gaussian Kernel', self.control_window,
                          2, 7, self.on_gaussian_kernel_change)  # Index 2 = size 5

        # Gaussian sigma (0-50, divide by 10 to get actual value)
        cv2.createTrackbar('Gaussian Sigma x10', self.control_window,
                          10, 50, self.on_gaussian_sigma_change)

        # Sharpen amount (0-30, divide by 10 to get actual value)
        cv2.createTrackbar('Sharpen Amount x10', self.control_window,
                          10, 30, self.on_sharpen_change)

        # Canny low threshold (0-255)
        cv2.createTrackbar('Canny Low', self.control_window,
                          50, 255, self.on_canny_low_change)

        # Canny high threshold (0-255)
        cv2.createTrackbar('Canny High', self.control_window,
                          150, 255, self.on_canny_high_change)

    def on_box_kernel_change(self, val):
        """Callback for box kernel size trackbar."""
        self.box_kernel_size = 5 if val == 0 else 11

    def on_gaussian_kernel_change(self, val):
        """Callback for Gaussian kernel size trackbar."""
        # Map trackbar value to odd kernel sizes: 3, 5, 7, 9, 11, 13, 15, 17
        self.gaussian_kernel_size = (val * 2) + 3

    def on_gaussian_sigma_change(self, val):
        """Callback for Gaussian sigma trackbar."""
        self.gaussian_sigma = val / 10.0
        if self.gaussian_sigma == 0:
            self.gaussian_sigma = 0.1  # Avoid zero sigma

    def on_sharpen_change(self, val):
        """Callback for sharpen amount trackbar."""
        self.sharpen_amount = val / 10.0

    def on_canny_low_change(self, val):
        """Callback for Canny low threshold trackbar."""
        self.canny_low = val

    def on_canny_high_change(self, val):
        """Callback for Canny high threshold trackbar."""
        self.canny_high = val

    def apply_box_blur(self, image):
        """
        Apply box blur filter using averaging convolution.

        Args:
            image: Input image (BGR)

        Returns:
            Blurred image
        """
        # Create box kernel (all values equal to 1/kernel_area)
        kernel = np.ones((self.box_kernel_size, self.box_kernel_size), np.float32)
        kernel = kernel / (self.box_kernel_size * self.box_kernel_size)

        # Apply convolution using filter2D
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred

    def apply_gaussian_blur(self, image):
        """
        Apply Gaussian blur filter.

        Args:
            image: Input image (BGR)

        Returns:
            Gaussian blurred image
        """
        # Apply Gaussian blur with specified kernel size and sigma
        blurred = cv2.GaussianBlur(image,
                                   (self.gaussian_kernel_size, self.gaussian_kernel_size),
                                   self.gaussian_sigma)
        return blurred

    def apply_sharpen(self, image):
        """
        Apply sharpening filter to enhance edges.

        Args:
            image: Input image (BGR)

        Returns:
            Sharpened image
        """
        # Create sharpening kernel
        # The kernel enhances the center pixel while subtracting surrounding pixels
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        # Scale kernel based on sharpen amount
        kernel = kernel * self.sharpen_amount
        kernel[1, 1] = 1 + (4 * self.sharpen_amount)  # Adjust center

        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)

        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def apply_sobel(self, image):
        """
        Apply Sobel edge detection.

        Args:
            image: Input image (BGR)

        Returns:
            Edge map showing detected edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel in X and Y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)

        # Compute gradient magnitude
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to 0-255 range
        sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

        # Convert back to BGR for display
        sobel_bgr = cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2BGR)
        return sobel_bgr

    def apply_canny(self, image):
        """
        Apply Canny edge detection.

        Args:
            image: Input image (BGR)

        Returns:
            Edge map showing detected edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Convert back to BGR for display
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_bgr

    def apply_emboss(self, image):
        """
        Apply emboss filter for a 3D raised effect.

        Args:
            image: Input image (BGR)

        Returns:
            Embossed image
        """
        # Emboss kernel
        kernel = np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ], dtype=np.float32)

        # Apply emboss filter
        embossed = cv2.filter2D(image, -1, kernel)

        # Add gray offset to make the embossed effect visible
        embossed = embossed.astype(np.int16) + 128
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)

        return embossed

    def process_frame(self, frame):
        """
        Process a frame based on the current filter mode.

        Args:
            frame: Input frame (BGR)

        Returns:
            Processed frame
        """
        if self.filter_mode == 'original':
            return frame.copy()
        elif self.filter_mode == 'box_blur':
            return self.apply_box_blur(frame)
        elif self.filter_mode == 'gaussian_blur':
            return self.apply_gaussian_blur(frame)
        elif self.filter_mode == 'sharpen':
            return self.apply_sharpen(frame)
        elif self.filter_mode == 'sobel':
            return self.apply_sobel(frame)
        elif self.filter_mode == 'canny':
            return self.apply_canny(frame)
        elif self.filter_mode == 'emboss':
            return self.apply_emboss(frame)
        else:
            return frame.copy()

    def calculate_fps(self):
        """Calculate and update FPS."""
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = cv2.getTickCount()
            time_diff = (end_time - self.start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.start_time = cv2.getTickCount()

    def add_overlay_text(self, frame):
        """
        Add informative text overlay to the frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Frame with text overlay
        """
        # Create a copy to avoid modifying original
        display_frame = frame.copy()

        # Prepare text information
        filter_name = self.filter_mode.replace('_', ' ').title()
        info_text = [
            f"Filter: {filter_name}",
            f"FPS: {self.fps:.1f}",
        ]

        # Add filter-specific parameters
        if self.filter_mode == 'box_blur':
            info_text.append(f"Kernel: {self.box_kernel_size}x{self.box_kernel_size}")
        elif self.filter_mode == 'gaussian_blur':
            info_text.append(f"Kernel: {self.gaussian_kernel_size}x{self.gaussian_kernel_size}")
            info_text.append(f"Sigma: {self.gaussian_sigma:.1f}")
        elif self.filter_mode == 'sharpen':
            info_text.append(f"Amount: {self.sharpen_amount:.1f}")
        elif self.filter_mode == 'canny':
            info_text.append(f"Thresholds: {self.canny_low}/{self.canny_high}")

        # Draw semi-transparent background for text
        overlay = display_frame.copy()
        text_height = len(info_text) * 30 + 20
        cv2.rectangle(overlay, (10, 10), (400, text_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

        # Draw text
        y_offset = 35
        for text in info_text:
            cv2.putText(display_frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

        return display_frame

    def save_image(self):
        """Save the current processed frame to disk."""
        if self.processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/{self.filter_mode}_{timestamp}.jpg"
            cv2.imwrite(filename, self.processed_frame)
            print(f"Saved: {filename}")

    def print_help(self):
        """Print help information to console."""
        help_text = """
╔════════════════════════════════════════════════════════════════╗
║           PHOTO FILTER APP - KEYBOARD CONTROLS                 ║
╠════════════════════════════════════════════════════════════════╣
║  FILTER SELECTION:                                             ║
║    O - Original (no filter)                                    ║
║    B - Box Blur                                                ║
║    G - Gaussian Blur                                           ║
║    S - Sharpen                                                 ║
║    E - Sobel Edge Detection                                    ║
║    C - Canny Edge Detection                                    ║
║    M - Emboss (Creative Filter)                                ║
╠════════════════════════════════════════════════════════════════╣
║  ACTIONS:                                                      ║
║    SPACE - Save current filtered image                         ║
║    H - Show this help                                          ║
║    Q / ESC - Quit application                                  ║
╠════════════════════════════════════════════════════════════════╣
║  REAL-TIME CONTROLS:                                           ║
║    Use the 'Filter Controls' window sliders to adjust:        ║
║    - Box blur kernel size                                      ║
║    - Gaussian blur kernel size and sigma                       ║
║    - Sharpen amount                                            ║
║    - Canny edge detection thresholds                           ║
╚════════════════════════════════════════════════════════════════╝
"""
        print(help_text)

    def initialize_video(self, source=0):
        """
        Initialize video capture.

        Args:
            source: Video source (0 for webcam, or path to video/image file)

        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return False

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        return True

    def load_image(self, image_path):
        """
        Load an image from file and process it in a loop.

        Args:
            image_path: Path to image file
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return

        print(f"Loaded image: {image_path}")
        print("Processing image (press 'Q' to quit)...")

        # Create windows
        cv2.namedWindow(self.main_window)
        self.create_control_window()
        self.print_help()

        # Process image in a loop (allowing filter changes)
        while True:
            # Process frame
            self.processed_frame = self.process_frame(image)

            # Add overlay
            display_frame = self.add_overlay_text(self.processed_frame)

            # Display
            cv2.imshow(self.main_window, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('h'):
                self.print_help()
            elif key == ord('o'):
                self.filter_mode = 'original'
                print("Filter: Original")
            elif key == ord('b'):
                self.filter_mode = 'box_blur'
                print(f"Filter: Box Blur ({self.box_kernel_size}x{self.box_kernel_size})")
            elif key == ord('g'):
                self.filter_mode = 'gaussian_blur'
                print(f"Filter: Gaussian Blur (kernel={self.gaussian_kernel_size}, sigma={self.gaussian_sigma})")
            elif key == ord('s'):
                self.filter_mode = 'sharpen'
                print(f"Filter: Sharpen (amount={self.sharpen_amount})")
            elif key == ord('e'):
                self.filter_mode = 'sobel'
                print("Filter: Sobel Edge Detection")
            elif key == ord('c'):
                self.filter_mode = 'canny'
                print(f"Filter: Canny Edge Detection (thresholds={self.canny_low}/{self.canny_high})")
            elif key == ord('m'):
                self.filter_mode = 'emboss'
                print("Filter: Emboss")
            elif key == ord(' '):
                self.save_image()

        cv2.destroyAllWindows()

    def run(self, source=0):
        """
        Main application loop.

        Args:
            source: Video source (0 for webcam, or path to video file)
        """
        # Check if source is an image file
        if isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.load_image(source)
            return

        # Initialize video capture
        if not self.initialize_video(source):
            return

        # Create windows
        cv2.namedWindow(self.main_window)
        self.create_control_window()

        # Print help
        self.print_help()

        print("Starting video processing...")
        print("Press 'H' for help, 'Q' to quit")

        # Main loop
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream")
                    break

                self.current_frame = frame

                # Process frame
                self.processed_frame = self.process_frame(frame)

                # Calculate FPS
                self.calculate_fps()

                # Add overlay
                display_frame = self.add_overlay_text(self.processed_frame)

                # Display
                cv2.imshow(self.main_window, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('h'):
                    self.print_help()
                elif key == ord('o'):
                    self.filter_mode = 'original'
                    print("Filter: Original")
                elif key == ord('b'):
                    self.filter_mode = 'box_blur'
                    print(f"Filter: Box Blur ({self.box_kernel_size}x{self.box_kernel_size})")
                elif key == ord('g'):
                    self.filter_mode = 'gaussian_blur'
                    print(f"Filter: Gaussian Blur (kernel={self.gaussian_kernel_size}, sigma={self.gaussian_sigma})")
                elif key == ord('s'):
                    self.filter_mode = 'sharpen'
                    print(f"Filter: Sharpen (amount={self.sharpen_amount})")
                elif key == ord('e'):
                    self.filter_mode = 'sobel'
                    print("Filter: Sobel Edge Detection")
                elif key == ord('c'):
                    self.filter_mode = 'canny'
                    print(f"Filter: Canny Edge Detection (thresholds={self.canny_low}/{self.canny_high})")
                elif key == ord('m'):
                    self.filter_mode = 'emboss'
                    print("Filter: Emboss")
                elif key == ord(' '):
                    self.save_image()

        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\nApplication closed successfully")


def main():
    """Main entry point for the application."""
    import sys

    print("="*70)
    print("  REAL-TIME PHOTO FILTERING APPLICATION")
    print("="*70)
    print("\nStarting application...")
    print("- Press 'H' at any time for help")
    print("- Press 'Q' or ESC to quit")
    print("-"*70)

    # Create application instance
    app = PhotoFilterApp()

    # Check command line arguments for input source
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # Try to convert to int (webcam ID) or use as file path
        try:
            source = int(source)
        except ValueError:
            pass
        app.run(source)
    else:
        # Default to webcam
        app.run(0)


if __name__ == "__main__":
    main()
