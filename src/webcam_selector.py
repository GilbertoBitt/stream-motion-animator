"""
Webcam selector utility for choosing camera device.

Provides UI to list and select available webcams before starting the application.
"""

import cv2
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class WebcamSelector:
    """Utility for selecting webcam device."""

    @staticmethod
    def list_available_cameras(max_cameras: int = 10) -> List[Tuple[int, str]]:
        """
        List all available camera devices.

        Args:
            max_cameras: Maximum number of camera indices to check

        Returns:
            List of tuples (index, name/info)
        """
        available_cameras = []

        logger.info("Scanning for available cameras...")

        for index in range(max_cameras):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Try to get camera info
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Try to get backend name
                    backend = cap.getBackendName()

                    info = f"Camera {index} ({width}x{height} @ {fps:.0f}fps, {backend})"
                    available_cameras.append((index, info))
                    logger.info(f"Found: {info}")
                else:
                    # Camera exists but couldn't read frame
                    info = f"Camera {index} (available but not readable)"
                    available_cameras.append((index, info))
                    logger.warning(f"Found: {info}")

                cap.release()

        if not available_cameras:
            logger.warning("No cameras found!")
        else:
            logger.info(f"Found {len(available_cameras)} camera(s)")

        return available_cameras

    @staticmethod
    def select_camera_cli(available_cameras: List[Tuple[int, str]]) -> Optional[int]:
        """
        Command-line interface for selecting camera.

        Args:
            available_cameras: List of available cameras

        Returns:
            Selected camera index or None if cancelled
        """
        if not available_cameras:
            print("\n❌ No cameras found!")
            print("Please connect a camera and try again.")
            return None

        print("\n" + "="*60)
        print("WEBCAM SELECTION")
        print("="*60)
        print("\nAvailable cameras:")

        for index, info in available_cameras:
            print(f"  [{index}] {info}")

        print("\n" + "="*60)

        # If only one camera, use it automatically
        if len(available_cameras) == 1:
            selected_index = available_cameras[0][0]
            print(f"\n✓ Only one camera found, automatically selecting: Camera {selected_index}")
            return selected_index

        # Multiple cameras - ask user to choose
        while True:
            try:
                user_input = input(f"\nSelect camera [0-{len(available_cameras)-1}] or 'q' to quit: ").strip().lower()

                if user_input == 'q':
                    print("Camera selection cancelled.")
                    return None

                selected = int(user_input)

                # Validate selection
                valid_indices = [idx for idx, _ in available_cameras]
                if selected in valid_indices:
                    print(f"\n✓ Selected: Camera {selected}")
                    return selected
                else:
                    print(f"❌ Invalid selection. Please choose from: {valid_indices}")

            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q' to quit.")
            except KeyboardInterrupt:
                print("\n\nCamera selection cancelled.")
                return None

    @staticmethod
    def select_camera_gui(available_cameras: List[Tuple[int, str]]) -> Optional[int]:
        """
        GUI-based camera selection with preview.

        Args:
            available_cameras: List of available cameras

        Returns:
            Selected camera index or None if cancelled
        """
        if not available_cameras:
            print("\n❌ No cameras found!")
            return None

        # If only one camera, use it
        if len(available_cameras) == 1:
            return available_cameras[0][0]

        print("\n" + "="*60)
        print("WEBCAM PREVIEW")
        print("="*60)
        print("\nShowing preview for each camera...")
        print("Press SPACE to select current camera, or N for next camera, Q to quit")
        print("="*60 + "\n")

        current_idx = 0
        cap = None

        try:
            while True:
                # Close previous camera if open
                if cap is not None:
                    cap.release()

                # Open current camera
                camera_index, camera_info = available_cameras[current_idx]
                cap = cv2.VideoCapture(camera_index)

                if not cap.isOpened():
                    print(f"❌ Failed to open Camera {camera_index}")
                    current_idx = (current_idx + 1) % len(available_cameras)
                    continue

                print(f"\nPreviewing: {camera_info}")

                # Show preview
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame")
                        break

                    # Add overlay with info
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"Camera {camera_index}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, "SPACE: Select | N: Next | Q: Quit", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.imshow("Camera Selection", display_frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord(' '):  # Space - select this camera
                        cap.release()
                        cv2.destroyAllWindows()
                        print(f"\n✓ Selected: Camera {camera_index}")
                        return camera_index
                    elif key == ord('n') or key == ord('N'):  # Next camera
                        break
                    elif key == ord('q') or key == ord('Q'):  # Quit
                        cap.release()
                        cv2.destroyAllWindows()
                        print("\nCamera selection cancelled.")
                        return None

                # Move to next camera
                current_idx = (current_idx + 1) % len(available_cameras)

        except KeyboardInterrupt:
            print("\n\nCamera selection cancelled.")
            return None
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

    @staticmethod
    def select_camera(use_gui: bool = True) -> Optional[int]:
        """
        Select camera with automatic detection and UI.

        Args:
            use_gui: Whether to use GUI preview (True) or CLI selection (False)

        Returns:
            Selected camera index or None if cancelled
        """
        # Scan for cameras
        available_cameras = WebcamSelector.list_available_cameras()

        if not available_cameras:
            return None

        # If only one camera, auto-select it
        if len(available_cameras) == 1:
            selected = available_cameras[0][0]
            print(f"\n✓ Auto-selected Camera {selected} (only camera available)")
            return selected

        # Multiple cameras - let user choose
        if use_gui:
            return WebcamSelector.select_camera_gui(available_cameras)
        else:
            return WebcamSelector.select_camera_cli(available_cameras)


def test_webcam_selector():
    """Test function for webcam selector."""
    print("Testing Webcam Selector\n")

    # Test listing cameras
    cameras = WebcamSelector.list_available_cameras()

    if not cameras:
        print("No cameras found!")
        return

    # Test CLI selection
    print("\n--- Testing CLI Selection ---")
    selected = WebcamSelector.select_camera_cli(cameras)
    print(f"CLI Selection result: {selected}")

    # Test GUI selection (uncomment to test)
    # print("\n--- Testing GUI Selection ---")
    # selected = WebcamSelector.select_camera_gui(cameras)
    # print(f"GUI Selection result: {selected}")


if __name__ == "__main__":
    test_webcam_selector()

