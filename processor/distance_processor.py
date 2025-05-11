import numpy as np
import cv2
import sys
import os

# Get project root directory to ensure court_calibration modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from court_calibration.court_calibrator import CourtCalibrator, calculate_player_positions

# Centralized configuration - All adjustable parameters
DISTANCE_BAR_CONFIG = {
    # Bar display parameters
    "bar_height": 80,  # Bar height (pixels)
    "margin": 40,  # Bar margin (pixels)
    "name_margin": 15,  # Distance between name and bar (pixels)
    "pixels_per_meter": 18,  # Pixels per meter for bar (higher = longer bar)
    "right_offset": 200,  # Right side offset (moves whole bar left)

    # Text parameters
    "font_scale": 2.0,  # Font size
    "text_thickness": 4,  # Text thickness
    "bg_padding": 8,  # Text background padding (pixels)
    "text_left_margin": 10,  # Left margin for text

    # Color parameters (BGR format)
    "upper_player_color": (0, 255, 255),  # Upper player color (yellow)
    "lower_player_color": (255, 0, 255),  # Lower player color (magenta)
    "border_color": (0, 0, 0),  # Border color (black)
    "bg_color": (0, 0, 0),  # Text background color (black)
}


class DistanceTracker:
    def __init__(self, calibration_file=None):
        """Initialize the distance tracker with a calibration file

        Args:
            calibration_file: Path to the calibration file, default is None
        """
        # Pass calibration file path
        self.calibrator = CourtCalibrator(None, calibration_file)  # No video path needed for loading
        self.load_success = self.calibrator.load_calibration()

        if not self.load_success:
            if calibration_file:
                print(
                    f"Warning: Unable to load court calibration data '{calibration_file}'. Distance tracking will be disabled.")
            else:
                print("Warning: Unable to load court calibration data. Distance tracking will be disabled.")

    def is_calibrated(self):
        """Check if calibration is loaded

        Returns:
            bool: True if calibration is loaded, False otherwise
        """
        return self.load_success

    def reset_distances(self):
        """Reset all player distances to zero"""
        if self.load_success:
            self.calibrator.reset_distances()

    def update_player_positions(self, keypoints):
        """Update player positions and calculate distances

        Args:
            keypoints: Keypoints from YOLO model

        Returns:
            positions: Dictionary of player positions
            distances: Dictionary of current frame distances for each player
        """
        if not self.load_success or keypoints is None:
            return {}, {}

        # Calculate player positions from keypoints
        positions = calculate_player_positions(keypoints)

        # Update positions and calculate distances
        frame_distances = {}
        for player_id, pos in positions.items():
            distance = self.calibrator.update_player_position(player_id, pos)
            frame_distances[player_id] = distance

        return positions, frame_distances

    def get_total_distances(self):
        """Get total distances for all players

        Returns:
            distances: Dictionary of total distances for each player
        """
        if not self.load_success:
            return {}

        return dict(self.calibrator.player_distances)

    def draw_distance_bars(self, frame, player_aliases=None, pixels_per_meter=None, max_display_width=None,
                           player_display_order=None):
        """Draw distance bars on the frame with dynamic sizing and fixed display order

        Args:
            frame: Video frame to draw on
            player_aliases: Dictionary of player aliases
            pixels_per_meter: Number of pixels per meter for bar scaling (if not specified, uses configuration value)
            max_display_width: Maximum width for bars (if None, will use 3/4 of frame width)
            player_display_order: Player display order list, controls who's on top (top to bottom order)

        Returns:
            frame: Frame with distance bars drawn
        """
        if not self.load_success:
            return frame

        # If maximum width not specified, use three-quarters of video width
        if max_display_width is None:
            # Get video frame width
            frame_width = frame.shape[1]
            # Calculate three-quarters width
            max_display_width = int(frame_width * 0.75)

        # Call the custom drawing method
        return self._draw_dynamic_distance_bars(
            frame,
            player_aliases,
            pixels_per_meter,
            max_display_width,
            player_display_order
        )

    def _draw_dynamic_distance_bars(self, frame, player_aliases=None, pixels_per_meter=None, max_display_width=500,
                                    player_display_order=None):
        """Draw larger distance bars with clearly visible player names and fixed display order

        Args:
            frame: Video frame to draw on
            player_aliases: Dictionary mapping player IDs to display names
            pixels_per_meter: Number of pixels to represent one meter of distance (if not specified, uses configuration value)
            max_display_width: Maximum width in pixels for the bar display
            player_display_order: Player display order list, controls who's on top (top to bottom order)

        Returns:
            frame: Frame with distance bars drawn
        """
        # Get distance data
        distances = self.get_total_distances()
        if not distances:
            return frame

        # Read parameters from configuration
        cfg = DISTANCE_BAR_CONFIG
        bar_height = cfg["bar_height"]
        margin = cfg["margin"]
        name_margin = cfg["name_margin"]
        font_scale = cfg["font_scale"]
        text_thickness = cfg["text_thickness"]
        bg_padding = cfg["bg_padding"]
        text_left_margin = cfg["text_left_margin"]

        # If pixels_per_meter not specified, use configuration value
        if pixels_per_meter is None:
            pixels_per_meter = cfg["pixels_per_meter"]

        # Initial Y position
        start_y = margin

        # Use fixed order (if provided) or sort by distance (original behavior)
        if player_display_order:
            # Use specified order, only include players with distance data
            sorted_players = [(pid, distances[pid]) for pid in player_display_order if pid in distances]
        else:
            # Sort by distance (original behavior)
            sorted_players = sorted(distances.items(), key=lambda x: x[1], reverse=True)

        for player_id, distance in sorted_players:
            # Calculate bar width
            bar_width = int(distance * pixels_per_meter)
            bar_width = min(bar_width, max_display_width)

            # Calculate positions - apply right offset to move everything left
            end_x = frame.shape[1] - margin - cfg["right_offset"]
            start_x = end_x - bar_width

            # Determine color
            if player_id == "upper":
                color = cfg["upper_player_color"]
            else:
                color = cfg["lower_player_color"]

            # Draw bar
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (end_x, start_y + bar_height),
                          color,
                          -1)

            # Draw bar border
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (end_x, start_y + bar_height),
                          cfg["border_color"],
                          2)

            # Draw distance text (on left side of bar)
            distance_text = f"{distance:.1f}m"
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            text_x = start_x - text_size[0] - text_left_margin
            text_y = start_y + bar_height // 2 + text_size[1] // 2

            # Prevent distance text from going out of bounds
            if text_x < 10:
                text_x = 10

            # Draw distance text background
            cv2.rectangle(frame,
                          (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                          (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                          cfg["bg_color"],
                          -1)

            # Draw distance text
            cv2.putText(frame,
                        distance_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        text_thickness)

            # Draw player name (on right side of bar) - ensure visibility
            player_name = player_aliases.get(player_id, player_id) if player_aliases else player_id
            name_x = end_x + name_margin
            name_y = start_y + bar_height // 2 + text_size[1] // 2

            # Draw name background (black background for visibility) - ensure sufficient width
            name_size = cv2.getTextSize(player_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            cv2.rectangle(frame,
                          (name_x - bg_padding, name_y - name_size[1] - bg_padding),
                          (name_x + name_size[0] + bg_padding + 50, name_y + bg_padding),
                          # Add extra width to ensure full display
                          cfg["bg_color"],
                          -1)

            # Ensure name is drawn prominently
            cv2.putText(frame,
                        player_name,
                        (name_x, name_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,  # Use same color as bar
                        text_thickness)

            # Move to next bar position
            start_y += bar_height + margin

        return frame
