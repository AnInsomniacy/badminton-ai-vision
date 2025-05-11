# Configuration and main function for YOLO pose detection with distance tracking
import os

import cv2

from processor.image_processor import process_image
from processor.video_processor import process_video

# Configuration as hard-coded JSON-style dictionary
CONFIG = {
    # Model settings
    "model_path": "models/yolo11x-pose.mlpackage",
    "confidence_threshold": 0.25,

    # Input/Output settings
    "input_path": "raw_videos/OneMatch.mp4",  # Can be image or video

    # Output path settings
    "output_suffix": "_pose_detected",  # Suffix to add to output filename
    "output_dir": "processed_videos",  # Manually specified output directory
    "get_output_path": lambda config: os.path.join(
        config["output_dir"],
        os.path.splitext(os.path.basename(config["input_path"]))[0] +
        config["output_suffix"] +
        os.path.splitext(config["input_path"])[1]
    ),

    # ROI coordinates [x_min, y_min, x_max, y_max] in relative format
    "roi": [0.24, 0.31, 0.77, 0.97],

    # Processing settings
    "display_output": True,  # Whether to display output frames in window
    "save_output": True,  # Whether to save output image/video

    # Distance tracking settings
    "enable_distance_tracking": True,  # Enable distance tracking
    "distance_tracking_calibration": "court_calibration/court_calibration.pkl",  # Calibration file path
    "require_calibration": False,  # Whether to abort if calibration is not found

    # Player display order configuration, controls the order of distance and speed bars (from top to bottom)
    "player_display_order": ["upper", "lower"],  # Fixed display order: upper player on top, lower player below

    # Speed tracking settings
    "enable_speed_tracking": True,  # Enable speed tracking (works independently)
    "pixels_per_speed": 30,  # Pixels per m/s for speed bars
    "max_speed_bar_width": 250,  # Maximum width for speed bars in pixels
    "speed_smoothing_window": 5,  # Number of frames to use for speed smoothing

    # Visualization settings
    "show_bounding_box": False,
    "show_confidence_label": False,  # Don't show confidence score labels
    "box_color": (255, 0, 0),  # Red for bounding boxes
    "keypoint_color": (0, 255, 255),  # Yellow for keypoints
    "line_thickness": 8,  # Thicker lines for skeleton

    # Person classification settings
    "classify_persons": True,
    "person_aliases": {
        "upper": "TAGO",
        "lower": "LIN D."
    },
    "person_label_colors": {
        "upper": (0, 255, 255),  # Yellow (BGR format)
        "lower": (255, 0, 255)  # Magenta (BGR format)
    },
    "label_font_scale": 1.6,
    "label_thickness": 4,
    "label_background": True,
    "label_background_color": (0, 0, 0),  # Black background for labels
    "label_background_alpha": 0.5,  # Background opacity

    # Skeleton colors for different body parts
    "skeleton_colors": {
        "face_neck": (255, 0, 255),  # Magenta for face and neck
        "right_arm": (0, 0, 255),  # Blue for right arm
        "left_arm": (255, 0, 0),  # Red for left arm
        "torso": (0, 255, 0),  # Green for torso
        "right_leg": (255, 165, 0),  # Orange for right leg
        "left_leg": (255, 255, 0)  # Yellow for left leg
    },

    # COCO pose connections with body part grouping
    "pose_connections": {
        "face_neck": [[0, 1], [0, 2], [1, 3], [2, 4]],  # Face and neck
        "left_arm": [[5, 7], [7, 9]],  # Left arm
        "right_arm": [[6, 8], [8, 10]],  # Right arm
        "torso": [[5, 6], [5, 11], [6, 12], [11, 12]],  # Body
        "left_leg": [[11, 13], [13, 15]],  # Left leg
        "right_leg": [[12, 14], [14, 16]]  # Right leg
    }
}


def is_video_file(file_path):
    """Determine if the file is a video based on its extension or properties

    Args:
        file_path: Path to the file

    Returns:
        bool: True if file is a video, False otherwise
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    _, extension = os.path.splitext(file_path)

    # Check by extension
    if extension.lower() in video_extensions:
        return True

    # If extension check fails, try opening as video
    try:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            is_video = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1
            cap.release()
            return is_video
        return False
    except:
        return False


def main():
    """Main function to detect input type and process accordingly"""
    config = CONFIG.copy()

    print("\n===== YOLO Pose Detection and Distance Tracking System =====")
    print(f"Input file: {config['input_path']}")
    print(f"Model file: {config['model_path']}")

    if config["enable_distance_tracking"]:
        print("Distance tracking: Enabled")
        print(f"Calibration file: {config['distance_tracking_calibration']}")

        # Check if calibration file exists
        if not os.path.exists(config["distance_tracking_calibration"]):
            # Check if directory exists
            calibration_dir = os.path.dirname(config["distance_tracking_calibration"])
            if calibration_dir and not os.path.exists(calibration_dir):
                print(
                    f"Warning: Calibration file directory '{calibration_dir}' does not exist, will attempt to create it")
                try:
                    os.makedirs(calibration_dir)
                    print(f"Successfully created directory: {calibration_dir}")
                except Exception as e:
                    print(f"Directory creation failed: {e}")

            print(
                f"Warning: Calibration file '{config['distance_tracking_calibration']}' not found. To use distance tracking, please run court_calibration/court_calibrator.py first.")
            user_input = input("Continue processing video without distance tracking? (y/n): ")
            if user_input.lower() != 'y':
                print(
                    "Processing cancelled. Please run court_calibration/court_calibrator.py for court calibration first.")
                return
            config["enable_distance_tracking"] = False
    else:
        print("Distance tracking: Disabled")

    if config["enable_speed_tracking"]:
        print("Speed tracking: Enabled")
        print(f"Calibration file: {config['distance_tracking_calibration']}")

        # Check if calibration file exists
        if not os.path.exists(config["distance_tracking_calibration"]):
            print(
                f"Warning: Calibration file '{config['distance_tracking_calibration']}' not found. To use speed tracking, please run court_calibration/court_calibrator.py first.")
            user_input = input("Continue processing video without speed tracking? (y/n): ")
            if user_input.lower() != 'y':
                print(
                    "Processing cancelled. Please run court_calibration/court_calibrator.py for court calibration first.")
                return
            config["enable_speed_tracking"] = False
    else:
        print("Speed tracking: Disabled")

    print("============================================\n")

    # Determine if input is video or image
    if is_video_file(config["input_path"]):
        print(f"Processing video: {config['input_path']}")
        return process_video(config)
    else:
        print(f"Processing image: {config['input_path']}")
        print("Note: Distance tracking and speed tracking only support video processing")
        config["enable_distance_tracking"] = False
        config["enable_speed_tracking"] = False
        return process_image(config)


if __name__ == "__main__":
    main()
