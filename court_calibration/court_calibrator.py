import cv2
import numpy as np
import json
import os
import time
import pickle
from collections import defaultdict, deque

# Hard-coded configuration for court calibration
CONFIG = {
    # Input/Output paths
    "video_path": "/Users/sekiro/Projects/PyCharm/Badminton_AI_Analysis/raw_videos/OneMatch.mp4",  # Absolute path to video
    "calibration_file": "/Users/sekiro/Projects/PyCharm/Badminton_AI_Analysis/court_calibration/court_calibration.pkl",  # Absolute path to calibration file
    "force_calibrate": True,  # Force recalibration even if calibration file exists
}

# Standard badminton court dimensions in meters
COURT_WIDTH = 6.1  # meters (doubles court width)
COURT_LENGTH = 13.4  # meters (full court length)
NET_LINE_POSITION = 6.7  # meters (half court length)
SERVICE_SHORT_LINE = 1.98  # meters from net line
SERVICE_LONG_LINE = 0.76  # meters from back boundary
SINGLES_WIDTH = 5.18  # meters (singles court width)
CENTER_LINE_LENGTH = 3.96  # meters (from service line to back boundary)

# Define standard court points in real world coordinates (meters)
STANDARD_COURT_POINTS = {
    # Court corners (required)
    "left_net_corner": (0, NET_LINE_POSITION),
    "right_net_corner": (COURT_WIDTH, NET_LINE_POSITION),
    "right_back_corner": (COURT_WIDTH, COURT_LENGTH),
    "left_back_corner": (0, COURT_LENGTH),

    # Optional additional points
    "left_singles_net_corner": ((COURT_WIDTH - SINGLES_WIDTH) / 2, NET_LINE_POSITION),
    "right_singles_net_corner": ((COURT_WIDTH + SINGLES_WIDTH) / 2, NET_LINE_POSITION),
    "left_service_line_left": (0, NET_LINE_POSITION + SERVICE_SHORT_LINE),
    "right_service_line_left": (COURT_WIDTH, NET_LINE_POSITION + SERVICE_SHORT_LINE),
    "left_singles_back_corner": ((COURT_WIDTH - SINGLES_WIDTH) / 2, COURT_LENGTH),
    "right_singles_back_corner": ((COURT_WIDTH + SINGLES_WIDTH) / 2, COURT_LENGTH),
    "left_service_back_line_left": (0, COURT_LENGTH - SERVICE_LONG_LINE),
    "right_service_back_line_right": (COURT_WIDTH, COURT_LENGTH - SERVICE_LONG_LINE),
    "center_net": (COURT_WIDTH / 2, NET_LINE_POSITION),
    "center_service_line": (COURT_WIDTH / 2, NET_LINE_POSITION + SERVICE_SHORT_LINE),
    "center_back_line": (COURT_WIDTH / 2, COURT_LENGTH),
    "center_service_back_line": (COURT_WIDTH / 2, COURT_LENGTH - SERVICE_LONG_LINE)
}


class CourtCalibrator:
    def __init__(self, video_path, calibration_file=None):
        """Initialize the court calibrator with a video path

        Args:
            video_path: Path to the video file
            calibration_file: Path to the calibration file, default is None
        """
        self.video_path = video_path
        self.calibration_file = calibration_file  # Store the calibration file path
        self.calibration_points_image = {}  # Dictionary of labeled calibration points
        self.calibration_points_court = {}  # Dictionary of corresponding court coordinates
        self.required_points = ["left_net_corner", "right_net_corner", "right_back_corner", "left_back_corner"]
        self.current_point_label = None
        self.src_points = None
        self.dst_points = None
        self.transformation_matrix = None
        self.inverse_matrix = None
        self.calibration_complete = False
        self.player_tracks = {}
        self.player_distances = defaultdict(float)
        self.history_length = 10  # Number of frames to keep for smoothing
        self.player_positions_history = defaultdict(lambda: deque(maxlen=self.history_length))

        # Calibration point order and descriptions
        self.calibration_sequence = [
            ("left_net_corner", "Net left boundary point (court corner from your view at left front)"),
            ("right_net_corner", "Net right boundary point (court corner from your view at right front)"),
            ("right_back_corner", "Back line right boundary point (court corner from your view at right back)"),
            ("left_back_corner", "Back line left boundary point (court corner from your view at left back)"),
            ("center_net", "Net center point (where net meets the center line)"),
            ("center_back_line", "Back line center point (where back line meets the center line)"),
            ("left_service_line_left", "Front service line left endpoint (left end of short line near the net)"),
            ("right_service_line_left", "Front service line right endpoint (right end of short line near the net)"),
            ("center_service_line", "Front service line middle point (center of short line near the net)"),
            ("left_service_back_line_left", "Back service line left endpoint (left end of short line away from net)"),
            ("right_service_back_line_right", "Back service line right endpoint (right end of short line away from net)"),
            ("center_service_back_line", "Back service line middle point (center of short line away from net)"),
        ]

        # Additional explanations for calibration points
        self.point_explanations = {
            "left_net_corner": "This is the left side court corner near the net, serving as a boundary point for doubles and singles. From the spectator's view, it's the left corner at the front of the court.",
            "right_net_corner": "This is the right side court corner near the net, serving as a boundary point for doubles and singles. From the spectator's view, it's the right corner at the front of the court.",
            "right_back_corner": "This is the right side court corner away from the net, where the back line meets the right sideline. From the spectator's view, it's the right corner at the back of the court.",
            "left_back_corner": "This is the left side court corner away from the net, where the back line meets the left sideline. From the spectator's view, it's the left corner at the back of the court.",
            "center_net": "This is the center point of the net, where the center line meets the net. It corresponds to the front end of the center line.",
            "center_back_line": "This is the center point of the back line, where the center line meets the back line. It corresponds to the back end of the center line.",
            "left_service_line_left": "This is where the front service line (short line near the net) meets the left sideline. It's the front left corner of the service area.",
            "right_service_line_left": "This is where the front service line (short line near the net) meets the right sideline. It's the front right corner of the service area.",
            "center_service_line": "This is the midpoint of the front service line, where it intersects with the center line.",
            "left_service_back_line_left": "This is where the back service line (short line away from the net) meets the left sideline. It's the back left corner of the service area.",
            "right_service_back_line_right": "This is where the back service line (short line away from the net) meets the right sideline. It's the back right corner of the service area.",
            "center_service_back_line": "This is the midpoint of the back service line, where it intersects with the center line.",
        }

    def get_first_frame(self):
        """Extract the first frame from the video

        Returns:
            frame: First frame of the video
        """
        print(f"Attempting to open video at: {os.path.abspath(self.video_path)}")
        print(f"File exists: {os.path.exists(self.video_path)}")

        cap = cv2.VideoCapture(self.video_path)
        print(f"VideoCapture isOpened: {cap.isOpened()}")

        ret, frame = cap.read()
        print(f"Read frame result: {ret}")

        cap.release()

        if not ret:
            raise Exception(f"Could not read the first frame from {self.video_path}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for capturing court corner points

        Args:
            event: Mouse event type
            x, y: Coordinates of the mouse event
            flags: Additional flags
            param: Additional parameters (the image)
        """
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point_label:
            # Add new point with its label
            self.calibration_points_image[self.current_point_label] = (x, y)

            # Get the corresponding court coordinate
            self.calibration_points_court[self.current_point_label] = STANDARD_COURT_POINTS[self.current_point_label]

            print(f"Added point '{self.current_point_label}': ({x}, {y})")

            # Redraw the image
            image_copy = param.copy()
            self.draw_calibration_points(image_copy)
            cv2.imshow("Court Calibration", image_copy)

            # Move to the next point in the sequence if available
            self.move_to_next_point()

    def draw_calibration_points(self, image):
        """Draw calibration points and connections on the image

        Args:
            image: Image to draw on

        Returns:
            image: Image with drawn points and connections
        """
        # Draw points and labels
        for idx, (label, point) in enumerate(self.calibration_points_image.items()):
            cv2.circle(image, point, 5, (0, 0, 255), -1)
            cv2.putText(image, label, (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw court outline if we have the four corners
        if all(point in self.calibration_points_image for point in self.required_points):
            # Court outline
            corners = [
                self.calibration_points_image["left_net_corner"],
                self.calibration_points_image["right_net_corner"],
                self.calibration_points_image["right_back_corner"],
                self.calibration_points_image["left_back_corner"]
            ]

            # Draw the court boundary
            for i in range(len(corners)):
                cv2.line(image, corners[i], corners[(i + 1) % len(corners)], (255, 0, 0), 2)

        # Draw service lines if available
        service_line_points = ["left_service_line_left", "right_service_line_left", "center_service_line"]
        if all(point in self.calibration_points_image for point in service_line_points):
            cv2.line(image,
                     self.calibration_points_image["left_service_line_left"],
                     self.calibration_points_image["right_service_line_left"],
                     (0, 255, 0), 2)

        # Draw back service lines if available
        back_service_line_points = ["left_service_back_line_left", "right_service_back_line_right",
                                    "center_service_back_line"]
        if all(point in self.calibration_points_image for point in back_service_line_points):
            cv2.line(image,
                     self.calibration_points_image["left_service_back_line_left"],
                     self.calibration_points_image["right_service_back_line_right"],
                     (0, 255, 0), 2)

        # Draw center line if we have center points
        center_points = ["center_net", "center_service_line", "center_back_line"]
        if all(point in self.calibration_points_image for point in center_points):
            cv2.line(image,
                     self.calibration_points_image["center_net"],
                     self.calibration_points_image["center_back_line"],
                     (0, 255, 0), 2)

        # Highlight current point to be labeled
        if self.current_point_label and self.current_point_label not in self.calibration_points_image:
            cv2.putText(image, f"Please mark: {self.current_point_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add description of the current point
            if self.current_point_label in self.point_explanations:
                desc = self.point_explanations[self.current_point_label]
                cv2.putText(image, f"Description: {desc}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image

    def move_to_next_point(self):
        """Move to the next point in the calibration sequence"""
        # Find the index of the current point
        if not self.current_point_label:
            # Start with the first point
            self.current_point_label = self.calibration_sequence[0][0]
            return

        # Find current index
        current_idx = -1
        for idx, (label, _) in enumerate(self.calibration_sequence):
            if label == self.current_point_label:
                current_idx = idx
                break

        # Move to next point if available
        if current_idx >= 0 and current_idx < len(self.calibration_sequence) - 1:
            self.current_point_label = self.calibration_sequence[current_idx + 1][0]
            print(f"\nMarking: {self.calibration_sequence[current_idx + 1][1]}")
            print(f"Description: {self.point_explanations[self.current_point_label]}")
        else:
            # We've reached the end of the sequence
            self.current_point_label = None
            print("\nAll points have been marked! Press Enter to confirm calibration, or press 'r' to restart.")

    def print_calibration_instructions(self):
        """Print detailed calibration instructions to the console"""
        print("\n==== Badminton Court Calibration Tool ====")
        print("Calibration Instructions:")
        print("This tool will help you calibrate the badminton court to calculate player movement distances")
        print("\nStandard Badminton Court Dimensions:")
        print("- Doubles Court: 13.4m long × 6.1m wide")
        print("- Singles Court: 13.4m long × 5.18m wide")
        print("- Net Height: 1.55m at the ends, 1.524m in the middle")

        print("\nCalibration Steps:")
        print("1. First, you must mark the four corner points of the court (these are required)")
        print("2. Then mark the center line and service line intersections (optional but recommended for better accuracy)")
        print("3. After completing, press Enter; to reset press 'r', to skip the current point press 's'")

        print("\nOperation Instructions:")
        print("- Click on the corresponding position on the image to mark court points")
        print("- When marking, try to precisely click on the intersection of lines")
        print("- The system will automatically proceed to the next point after marking")
        print("- Required four corner points: left net corner, right net corner, right back corner, left back corner")

        print("\nCalibration sequence:")
        for idx, (label, desc) in enumerate(self.calibration_sequence):
            print(f"{idx + 1}. {desc}")
            print(f"   Description: {self.point_explanations[label]}")

        print("\n=== Begin calibrating the first point ===")
        print("Please mark: Net left boundary point (court corner from your view at left front)")
        print("Description: This is the left side court corner near the net, from the spectator's view it's the left corner at the front of the court\n")

    def calibrate_court(self):
        """Manually calibrate the court by selecting points

        Returns:
            bool: True if calibration was successful
        """
        # Get the first frame
        frame = self.get_first_frame()
        original_frame = frame.copy()

        # Print instructions to console
        self.print_calibration_instructions()

        # Display instructions on the image
        instruction_img = frame.copy()

        # Create a semi-transparent overlay for text background
        overlay = instruction_img.copy()
        cv2.rectangle(overlay, (5, 5), (750, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, instruction_img, 0.3, 0, instruction_img)

        instructions = [
            "Badminton Court Calibration Tool",
            "",
            "Please mark court points following console prompts:",
            "1. Required points: Four main court corners (left/right net corners and left/right back corners)",
            "2. Recommended points: Center line endpoints, service line intersections, etc.",
            "3. Press 'Enter' to confirm after marking is complete",
            "4. Press 'r' to restart calibration",
            "5. Press 's' to skip the current point",
            "6. Press 'ESC' to cancel calibration",
            "",
            "Please refer to the console for detailed instructions and the current point to mark"
        ]

        y_offset = 30
        # Draw header with different color and size
        cv2.putText(instruction_img, instructions[0], (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        y_offset += 40

        # Draw remaining instructions
        for inst in instructions[1:]:
            if inst == "":  # Add space for empty line
                y_offset += 15
                continue

            cv2.putText(instruction_img, inst, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        # Draw a badminton court diagram to illustrate the points
        diagram_start_x = 400
        diagram_start_y = 50
        diagram_width = 200
        diagram_height = 150

        # Draw court outline
        cv2.rectangle(instruction_img,
                      (diagram_start_x, diagram_start_y),
                      (diagram_start_x + diagram_width, diagram_start_y + diagram_height),
                      (255, 255, 255), 2)

        # Draw net line
        cv2.line(instruction_img,
                 (diagram_start_x, diagram_start_y + diagram_height // 2),
                 (diagram_start_x + diagram_width, diagram_start_y + diagram_height // 2),
                 (255, 255, 255), 1)

        # Draw center line
        cv2.line(instruction_img,
                 (diagram_start_x + diagram_width // 2, diagram_start_y),
                 (diagram_start_x + diagram_width // 2, diagram_start_y + diagram_height),
                 (255, 255, 255), 1)

        # Draw service lines
        service_line_y1 = diagram_start_y + diagram_height // 4
        service_line_y2 = diagram_start_y + 3 * diagram_height // 4
        cv2.line(instruction_img,
                 (diagram_start_x, service_line_y1),
                 (diagram_start_x + diagram_width, service_line_y1),
                 (255, 255, 255), 1)
        cv2.line(instruction_img,
                 (diagram_start_x, service_line_y2),
                 (diagram_start_x + diagram_width, service_line_y2),
                 (255, 255, 255), 1)

        # Mark and label the corner points
        corners = [
            (diagram_start_x, diagram_start_y + diagram_height // 2, "Left Net Corner"),
            (diagram_start_x + diagram_width, diagram_start_y + diagram_height // 2, "Right Net Corner"),
            (diagram_start_x, diagram_start_y + diagram_height, "Left Back Corner"),
            (diagram_start_x + diagram_width, diagram_start_y + diagram_height, "Right Back Corner")
        ]
        for x, y, label in corners:
            cv2.circle(instruction_img, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Court Calibration", instruction_img)
        cv2.waitKey(5000)  # Wait for 5 seconds to allow reading

        # Initialize the first point to mark
        self.current_point_label = self.calibration_sequence[0][0]

        # Show the frame and set up callback
        cv2.imshow("Court Calibration", frame)
        cv2.setMouseCallback("Court Calibration", self.mouse_callback, frame)

        # Wait for user to finish calibration
        while True:
            key = cv2.waitKey(1) & 0xFF

            # If Enter key is pressed and we have at least the required points
            if key == 13 and all(point in self.calibration_points_image for point in self.required_points):
                break

            # If 'r' is pressed, reset points
            if key == ord('r'):
                self.calibration_points_image = {}
                self.calibration_points_court = {}
                self.current_point_label = self.calibration_sequence[0][0]
                print("\nCalibration points have been reset! Starting over...")
                print(f"Please mark: {self.calibration_sequence[0][1]}")
                cv2.imshow("Court Calibration", frame)

            # If ESC is pressed, cancel calibration
            if key == 27:
                cv2.destroyAllWindows()
                return False

            # If 's' is pressed, skip current point
            if key == ord('s') and self.current_point_label:
                print(f"\nSkipping point: {self.current_point_label}")
                self.move_to_next_point()
                # Redraw the image
                image_copy = frame.copy()
                self.draw_calibration_points(image_copy)
                cv2.imshow("Court Calibration", image_copy)

        # Close the window
        cv2.destroyAllWindows()

        # Check if we have the minimum required points (four corners)
        if not all(point in self.calibration_points_image for point in self.required_points):
            print("Error: Not all necessary court corner points were marked!")
            return False

        # Prepare arrays for perspective transformation
        src_points = []
        dst_points = []

        # First add the required corner points (to ensure proper order)
        for label in self.required_points:
            src_points.append(self.calibration_points_image[label])
            dst_points.append(self.calibration_points_court[label])

        # Convert to numpy arrays
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)

        # Calculate the perspective transformation matrix using the four corners
        self.transformation_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # Save the calibration data
        self.save_calibration()

        # Mark calibration as complete
        self.calibration_complete = True

        return True

    def save_calibration(self):
        """Save the calibration data to a file"""
        calibration_data = {
            "calibration_points_image": self.calibration_points_image,
            "calibration_points_court": self.calibration_points_court,
            "src_points": self.src_points.tolist(),
            "dst_points": self.dst_points.tolist(),
            "transformation_matrix": self.transformation_matrix.tolist(),
            "inverse_matrix": self.inverse_matrix.tolist(),
            "timestamp": time.time()
        }

        # Determine save path
        calibration_file = self.calibration_file if self.calibration_file else "court_calibration.pkl"

        # Ensure directory exists
        calibration_dir = os.path.dirname(calibration_file)
        if calibration_dir and not os.path.exists(calibration_dir):
            os.makedirs(calibration_dir)

        # Save both JSON and pickle files
        json_file = os.path.splitext(calibration_file)[0] + ".json"

        # Save as JSON
        with open(json_file, "w") as f:
            json.dump(calibration_data, f, indent=4)

        # Save as pickle
        with open(calibration_file, "wb") as f:
            pickle.dump(calibration_data, f)

        print(f"Calibration data saved to {json_file} and {calibration_file}")

    def load_calibration(self):
        """Load the calibration data from file

        Returns:
            bool: True if loading was successful
        """
        # Determine the file path to load
        calibration_file = self.calibration_file if self.calibration_file else "court_calibration.pkl"

        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, "rb") as f:
                    calibration_data = pickle.load(f)

                self.calibration_points_image = calibration_data.get("calibration_points_image", {})
                self.calibration_points_court = calibration_data.get("calibration_points_court", {})
                self.src_points = np.array(calibration_data["src_points"])
                self.dst_points = np.array(calibration_data["dst_points"])
                self.transformation_matrix = np.array(calibration_data["transformation_matrix"])
                self.inverse_matrix = np.array(calibration_data["inverse_matrix"])
                self.calibration_complete = True

                print(f"Loaded calibration data from {calibration_file}")
                return True
            except Exception as e:
                print(f"Error loading calibration file: {e}")
                return False
        else:
            print(f"Calibration file not found: {calibration_file}")
            return False

    def image_to_court_coords(self, image_coords):
        """Convert image coordinates to court coordinates

        Args:
            image_coords: (x, y) in image coordinates

        Returns:
            court_coords: (x, y) in court coordinates (meters)
        """
        if not self.calibration_complete:
            raise Exception("Court calibration is not complete")

        # Convert to numpy array
        pts = np.array([[[image_coords[0], image_coords[1]]]], dtype=np.float32)

        # Transform the point
        transformed_pts = cv2.perspectiveTransform(pts, self.transformation_matrix)

        return (transformed_pts[0][0][0], transformed_pts[0][0][1])

    def court_to_image_coords(self, court_coords):
        """Convert court coordinates to image coordinates

        Args:
            court_coords: (x, y) in court coordinates (meters)

        Returns:
            image_coords: (x, y) in image coordinates
        """
        if not self.calibration_complete:
            raise Exception("Court calibration is not complete")

        # Convert to numpy array
        pts = np.array([[[court_coords[0], court_coords[1]]]], dtype=np.float32)

        # Transform the point
        transformed_pts = cv2.perspectiveTransform(pts, self.inverse_matrix)

        return (int(transformed_pts[0][0][0]), int(transformed_pts[0][0][1]))

    def update_player_position(self, player_id, image_coords):
        """Update a player's position and calculate distance traveled

        Args:
            player_id: Identifier for the player
            image_coords: (x, y) in image coordinates

        Returns:
            distance: Distance traveled since last update (meters)
        """
        if not self.calibration_complete:
            return 0

        # Convert to court coordinates
        court_coords = self.image_to_court_coords(image_coords)

        # Add to history for smoothing
        self.player_positions_history[player_id].append(court_coords)

        # If we have enough history, calculate smoothed position
        if len(self.player_positions_history[player_id]) >= 2:
            # Simple moving average smoothing
            recent_positions = list(self.player_positions_history[player_id])
            smoothed_current = recent_positions[-1]
            smoothed_previous = recent_positions[-2]

            # Calculate distance between the current and previous positions
            dx = smoothed_current[0] - smoothed_previous[0]
            dy = smoothed_current[1] - smoothed_previous[1]
            distance = np.sqrt(dx * dx + dy * dy)

            # Only count reasonable distances to avoid jumps due to detection errors
            # 0.5 meters is a reasonable maximum distance for one frame
            if distance < 0.5:
                self.player_distances[player_id] += distance
                return distance

        return 0

    def get_player_total_distance(self, player_id):
        """Get the total distance traveled by a player

        Args:
            player_id: Identifier for the player

        Returns:
            distance: Total distance traveled (meters)
        """
        return self.player_distances[player_id]

    def draw_distance_bars(self, frame, player_aliases=None):
        """Draw distance bars in the upper right corner of the frame

        Args:
            frame: Video frame to draw on
            player_aliases: Dictionary mapping player IDs to display names

        Returns:
            frame: Frame with distance bars drawn
        """
        if not self.player_distances:
            return frame

        # Set up parameters for the bars
        bar_height = 30
        max_bar_width = 250
        margin = 20
        text_margin = 10
        start_y = margin
        max_distance = max(self.player_distances.values()) if self.player_distances else 1
        max_distance = max(max_distance, 1)  # Avoid division by zero

        # Sort players by distance
        sorted_players = sorted(self.player_distances.items(), key=lambda x: x[1], reverse=True)

        for player_id, distance in sorted_players:
            # Calculate bar width based on distance
            bar_width = int((distance / max_distance) * max_bar_width)

            # Calculate positions
            start_x = frame.shape[1] - margin - max_bar_width
            end_x = start_x + bar_width

            # Draw background rectangle for the full potential bar
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (start_x + max_bar_width, start_y + bar_height),
                          (50, 50, 50),
                          -1)

            # Determine color based on player_id
            if player_id == "upper":
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 0, 255)  # Magenta

            # Draw the actual bar
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (end_x, start_y + bar_height),
                          color,
                          -1)

            # Draw distance text on the left end of the bar
            distance_text = f"{distance:.1f}m"
            cv2.putText(frame,
                        distance_text,
                        (start_x + text_margin, start_y + bar_height - text_margin),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

            # Draw player name on the right end of the bar
            player_name = player_aliases.get(player_id, player_id) if player_aliases else player_id
            text_size = cv2.getTextSize(player_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = frame.shape[1] - margin - text_size[0] - text_margin
            cv2.putText(frame,
                        player_name,
                        (text_x, start_y + bar_height - text_margin),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

            # Move to the next bar position
            start_y += bar_height + margin

        return frame

    def reset_distances(self):
        """Reset all player distances to zero"""
        self.player_distances = defaultdict(float)
        self.player_positions_history = defaultdict(lambda: deque(maxlen=self.history_length))


def calculate_player_positions(keypoints):
    """Calculate the position of each player based on keypoints

    Args:
        keypoints: Keypoints detected by YOLO model

    Returns:
        positions: Dictionary of player positions (upper, lower)
    """
    if keypoints is None or len(keypoints) == 0:
        return {}

    positions = {}

    # If there's only one person, classify as upper by default
    if len(keypoints) == 1:
        # Use ankles or hips as the position indicator
        kpts = keypoints[0]
        valid_foot_kpts = []

        # Check for valid ankle keypoints (indices 15 and 16)
        if kpts[15][0] > 0 and kpts[15][1] > 0:
            valid_foot_kpts.append(kpts[15])
        if kpts[16][0] > 0 and kpts[16][1] > 0:
            valid_foot_kpts.append(kpts[16])

        # If no valid ankle keypoints, try hips (indices 11 and 12)
        if not valid_foot_kpts:
            if kpts[11][0] > 0 and kpts[11][1] > 0:
                valid_foot_kpts.append(kpts[11])
            if kpts[12][0] > 0 and kpts[12][1] > 0:
                valid_foot_kpts.append(kpts[12])

        # Calculate average position if we have valid keypoints
        if valid_foot_kpts:
            x = sum(kpt[0] for kpt in valid_foot_kpts) / len(valid_foot_kpts)
            y = sum(kpt[1] for kpt in valid_foot_kpts) / len(valid_foot_kpts)
            positions["upper"] = (x, y)

        return positions

    # Get the average y-coordinate of each person's keypoints
    person_y_positions = []
    for person_idx, person_kpts in enumerate(keypoints):
        valid_foot_kpts = []

        # Check for valid ankle keypoints
        if person_kpts[15][0] > 0 and person_kpts[15][1] > 0:
            valid_foot_kpts.append(person_kpts[15])
        if person_kpts[16][0] > 0 and person_kpts[16][1] > 0:
            valid_foot_kpts.append(person_kpts[16])

        # If no valid ankle keypoints, try hips
        if not valid_foot_kpts:
            if person_kpts[11][0] > 0 and person_kpts[11][1] > 0:
                valid_foot_kpts.append(person_kpts[11])
            if person_kpts[12][0] > 0 and person_kpts[12][1] > 0:
                valid_foot_kpts.append(person_kpts[12])

        # Calculate average position if we have valid keypoints
        if valid_foot_kpts:
            x = sum(kpt[0] for kpt in valid_foot_kpts) / len(valid_foot_kpts)
            y = sum(kpt[1] for kpt in valid_foot_kpts) / len(valid_foot_kpts)

            # Store for classification
            person_y_positions.append((person_idx, y))

    # Sort people by their y-positions
    sorted_persons = sorted(person_y_positions, key=lambda x: x[1])

    # Assign positions
    for rank, (person_idx, _) in enumerate(sorted_persons):
        person_kpts = keypoints[person_idx]
        valid_foot_kpts = []

        # Get valid ankle or hip keypoints
        if person_kpts[15][0] > 0 and person_kpts[15][1] > 0:
            valid_foot_kpts.append(person_kpts[15])
        if person_kpts[16][0] > 0 and person_kpts[16][1] > 0:
            valid_foot_kpts.append(person_kpts[16])

        if not valid_foot_kpts:
            if person_kpts[11][0] > 0 and person_kpts[11][1] > 0:
                valid_foot_kpts.append(person_kpts[11])
            if person_kpts[12][0] > 0 and person_kpts[12][1] > 0:
                valid_foot_kpts.append(person_kpts[12])

        # Calculate average position
        if valid_foot_kpts:
            x = sum(kpt[0] for kpt in valid_foot_kpts) / len(valid_foot_kpts)
            y = sum(kpt[1] for kpt in valid_foot_kpts) / len(valid_foot_kpts)

            if rank == 0:  # Upper player (smaller y value)
                positions["upper"] = (x, y)
            else:  # Lower player (larger y value)
                positions["lower"] = (x, y)

    return positions


def main():
    """Main function to demonstrate court calibration and distance tracking"""
    # Use hard-coded CONFIG values instead of importing from main.py
    video_path = CONFIG["video_path"]
    calibration_file = CONFIG["calibration_file"]
    force_calibrate = CONFIG["force_calibrate"]

    print("\n===== Badminton Court Calibration and Distance Tracking Tool =====")
    print(f"Video file: {video_path}")
    print(f"Calibration file path: {calibration_file}")
    print("This tool is used to calibrate the badminton court and calculate player movement distances")
    print("After calibration, you can use this calibration data in main.py to process videos")
    print("============================================\n")

    # Create a calibrator with the video path and calibration file path
    calibrator = CourtCalibrator(video_path, calibration_file)

    # Try to load existing calibration
    if not force_calibrate and calibrator.load_calibration():
        print("Loaded existing calibration data")

        # Ask the user if they want to view the loaded calibration data
        user_input = input("Do you want to view the loaded calibration data? (y/n): ")
        if user_input.lower() == 'y':
            # Get the first frame to display calibration
            frame = calibrator.get_first_frame()

            # Draw calibration points on the frame
            calibrator.draw_calibration_points(frame)

            # Display the frame with calibration points
            cv2.imshow("Loaded Calibration Data", frame)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Ask the user if they want to recalibrate
        user_input = input("Do you want to recalibrate the court? (y/n): ")
        if user_input.lower() == 'y':
            force_calibrate = True
        else:
            print("Will use the loaded calibration data")

    # If needed, perform calibration
    if force_calibrate or not calibrator.calibration_complete:
        print(f"\nStarting court calibration for video: {video_path}")
        print("Please mark the key points on the court according to the prompts")

        if calibrator.calibrate_court():
            print("\n✓ Calibration successfully completed!")

            # Calculate and display actual court dimensions based on calibration
            try:
                # Calculate diagonal distance (meters)
                p1 = calibrator.calibration_points_court["left_net_corner"]
                p2 = calibrator.calibration_points_court["right_back_corner"]
                diagonal_distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

                # Calculate court width and length (meters)
                court_width = calibrator.calibration_points_court["right_net_corner"][0] - \
                              calibrator.calibration_points_court["left_net_corner"][0]
                court_length = calibrator.calibration_points_court["left_back_corner"][1] - \
                               calibrator.calibration_points_court["left_net_corner"][1]

                print(f"\nBased on calibration results, court dimensions are:")
                print(f"- Court width: {court_width:.2f} meters")
                print(f"- Court length: {court_length:.2f} meters")
                print(f"- Diagonal length: {diagonal_distance:.2f} meters")

                # Standard dimensions
                print(f"\nStandard badminton court dimensions are:")
                print(f"- Doubles court: 13.4m × 6.1m")
                print(f"- Singles court: 13.4m × 5.18m")
                print(f"- Standard diagonal length: 14.366m")
            except:
                # Skip if some points are missing
                pass
        else:
            print("\n✗ Calibration was cancelled or failed.")
            return

    print("\n✓ Calibration data is ready for use in the main processing pipeline")
    print(f"→ You can now run main.py using this calibration data ({calibration_file}) for video processing and distance tracking")


if __name__ == "__main__":
    main()