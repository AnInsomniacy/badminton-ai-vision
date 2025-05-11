import cv2
import numpy as np
from collections import deque
import sys
import os

# Get project root directory to ensure court_calibration modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from court_calibration.court_calibrator import CourtCalibrator, calculate_player_positions


class SpeedTracker:
    def __init__(self, calibration_file=None, fps=30, smoothing_window=5):
        """Initialize speed tracker with its own court calibration instance

        Args:
            calibration_file: Path to the calibration file, default is None
            fps: Frames per second of the video
            smoothing_window: Number of frames to use for speed smoothing
        """
        self.fps = fps
        self.time_delta = 1.0 / fps
        self.smoothing_window = smoothing_window
        self.player_speeds = {}  # Current speeds
        self.player_speed_history = {}  # History for smoothing
        self.player_positions = {}  # Last known court positions
        self.max_speed = 0  # Track maximum speed for scaling

        # Create own calibrator instance without modifying the distance tracker
        self.calibrator = CourtCalibrator(None, calibration_file)  # No video path needed, pass calibration file path
        self.load_success = self.calibrator.load_calibration()

        if not self.load_success:
            if calibration_file:
                print(
                    f"Warning: Unable to load court calibration data '{calibration_file}'. Speed tracking will be disabled.")
            else:
                print("Warning: Unable to load court calibration data. Speed tracking will be disabled.")

    def is_calibrated(self):
        """Check if calibration is loaded

        Returns:
            bool: True if calibration is loaded, False otherwise
        """
        return self.load_success

    def update_player_speeds(self, keypoints):
        """Calculate and update player speeds directly from keypoints

        Args:
            keypoints: Keypoints from YOLO model

        Returns:
            speeds: Dictionary of current speeds per player
        """
        if not self.load_success or keypoints is None:
            return {}

        # Calculate player positions from keypoints
        positions = calculate_player_positions(keypoints)
        speeds = {}

        for player_id, image_coords in positions.items():
            # Convert image coordinates to court coordinates
            court_coords = self.calibrator.image_to_court_coords(image_coords)

            # If this is a new player or first detection, just store position
            if player_id not in self.player_positions:
                self.player_positions[player_id] = court_coords
                continue

            # Get previous position
            prev_coords = self.player_positions[player_id]

            # Calculate distance moved
            dx = court_coords[0] - prev_coords[0]
            dy = court_coords[1] - prev_coords[1]
            distance = np.sqrt(dx * dx + dy * dy)

            # Only count reasonable distances to avoid jumps due to detection errors
            # 0.5 meters is a reasonable maximum distance for one frame
            if distance < 0.5:
                # Calculate instantaneous speed (m/s)
                speed = distance / self.time_delta

                # Initialize history if this is a new player
                if player_id not in self.player_speed_history:
                    self.player_speed_history[player_id] = deque(maxlen=self.smoothing_window)

                # Add to history
                self.player_speed_history[player_id].append(speed)

                # Calculate smoothed speed (moving average)
                smoothed_speed = sum(self.player_speed_history[player_id]) / len(self.player_speed_history[player_id])
                speeds[player_id] = smoothed_speed

                # Update player's current speed
                self.player_speeds[player_id] = smoothed_speed

                # Update max speed for scaling
                if smoothed_speed > self.max_speed:
                    self.max_speed = smoothed_speed

            # Update stored position
            self.player_positions[player_id] = court_coords

        return speeds

    def get_player_speeds(self):
        """Get current speeds for all players

        Returns:
            speeds: Dictionary of current speeds for each player
        """
        return self.player_speeds

    def draw_speed_bars(self, frame, player_aliases=None, pixels_per_speed=30, max_display_width=250,
                        player_display_order=None):
        """Draw speed bars on the frame below the distance bars with fixed display order

        Args:
            frame: Video frame to draw on
            player_aliases: Dictionary mapping player IDs to display names
            pixels_per_speed: Number of pixels per unit of speed (m/s)
            max_display_width: Maximum width for the speed bars
            player_display_order: Player display order list, controls who's on top (top to bottom order)

        Returns:
            frame: Frame with speed bars drawn
        """
        if not self.player_speeds:
            return frame

        # Set up parameters for the bars (matching style with distance bars)
        bar_height = 80
        margin = 40
        name_margin = 15
        text_margin = 10
        font_scale = 2.0
        text_thickness = 4
        bg_padding = 8
        right_offset = 200

        # Start position below where distance bars would be
        # For each player in speeds there would be a bar_height + margin vertical space
        num_players = len(self.player_speeds)
        start_y = margin + num_players * (bar_height + margin) + margin

        # Use fixed order (if provided) or sort by speed (original behavior)
        if player_display_order:
            # Use specified order, only include players with speed data
            sorted_players = [(pid, self.player_speeds[pid]) for pid in player_display_order if
                              pid in self.player_speeds]
        else:
            # Sort by speed (original behavior)
            sorted_players = sorted(self.player_speeds.items(), key=lambda x: x[1], reverse=True)

        # Determine max speed for scaling (ensure at least 1.0 to avoid division by zero)
        max_speed = max(self.max_speed, 1.0)

        for player_id, speed in sorted_players:
            # Calculate bar width based on speed
            bar_width = min(int(speed * pixels_per_speed), max_display_width)

            # Calculate positions
            end_x = frame.shape[1] - margin - right_offset
            start_x = end_x - bar_width

            # Determine color based on player_id
            if player_id == "upper":
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 0, 255)  # Magenta

            # Draw the speed bar
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (end_x, start_y + bar_height),
                          color,
                          -1)

            # Draw border
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (end_x, start_y + bar_height),
                          (0, 0, 0),  # Black border
                          2)

            # Draw speed text on the left
            speed_text = f"{speed:.1f} m/s"
            text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            text_x = start_x - text_size[0] - text_margin
            text_y = start_y + bar_height // 2 + text_size[1] // 2

            # Ensure text doesn't go off-screen
            if text_x < 10:
                text_x = 10

            # Draw text background
            cv2.rectangle(frame,
                          (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                          (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                          (0, 0, 0),  # Black background
                          -1)

            # Draw speed text
            cv2.putText(frame,
                        speed_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        text_thickness)

            # Draw player name on the right
            player_name = player_aliases.get(player_id, player_id) if player_aliases else player_id
            name_x = end_x + name_margin
            name_y = start_y + bar_height // 2 + text_size[1] // 2

            # Draw name background
            name_size = cv2.getTextSize(player_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            cv2.rectangle(frame,
                          (name_x - bg_padding, name_y - name_size[1] - bg_padding),
                          (name_x + name_size[0] + bg_padding + 50, name_y + bg_padding),
                          (0, 0, 0),  # Black background
                          -1)

            # Draw player name
            cv2.putText(frame,
                        player_name,
                        (name_x, name_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        text_thickness)

            # Move to next bar position
            start_y += bar_height + margin

        return frame

    def reset_speeds(self):
        """Reset all player speeds and positions to zero"""
        self.player_speeds = {}
        self.player_speed_history = {}
        self.player_positions = {}
        self.max_speed = 0
