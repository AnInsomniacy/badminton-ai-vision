import sys
import time

import cv2
from PIL import Image

from processor.mask_processor import create_mask, apply_mask_to_image
from processor.model_processor import load_yolo_model, detect_with_model, extract_detection_data
from renderer import render_boxes_on_image, render_keypoints_on_image
from processor.distance_processor import DistanceTracker
from processor.speed_processor import SpeedTracker


def progress_bar(iteration, total, length=50, fill='█', prefix='Progress:', suffix='', decimals=1,
                 elapsed_time=0, fps=0, eta=0):
    """
    Create a progress bar in the terminal

    Args:
        iteration: Current iteration
        total: Total iterations
        length: Bar length
        fill: Fill character
        prefix: Text before the bar
        suffix: Text after the bar
        decimals: Decimal places for percentage
        elapsed_time: Time elapsed in seconds
        fps: Frames per second
        eta: Estimated time remaining in seconds
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    # Format elapsed time as MM:SS
    mins, secs = divmod(int(elapsed_time), 60)
    elapsed_str = f"{mins:02d}:{secs:02d}"

    # Format ETA as MM:SS
    mins, secs = divmod(int(eta), 60)
    eta_str = f"{mins:02d}:{secs:02d}"

    # Create the progress line
    progress_line = f"\r{prefix} |{bar}| {percent}% {suffix} | FPS: {fps:.2f} | Time: {elapsed_str} | ETA: {eta_str}"

    # Clear previous line and print the new one
    sys.stdout.write('\r' + ' ' * len(progress_line))
    sys.stdout.write(progress_line)
    sys.stdout.flush()

    # Print a new line when complete
    if iteration == total:
        sys.stdout.write('\n')


def process_video(config):
    """Process video using YOLO pose detection with streaming mode and distance tracking

    Args:
        config: Configuration dictionary

    Returns:
        None, results_list: List of detection results for all frames
    """
    # Load model
    print("Loading model...")
    model = load_yolo_model(config["model_path"])

    # Initialize distance tracker if enabled
    distance_tracker = None
    if config.get("enable_distance_tracking", False):
        print("Initializing distance tracking...")
        distance_tracker = DistanceTracker(config.get("distance_tracking_calibration", "court_calibration.pkl"))

        if not distance_tracker.is_calibrated():
            print("Warning: Distance tracking is enabled but no calibration data found.")
            print("Please run court_calibrator.py for court calibration first.")
            if config.get("require_calibration", False):
                print("Processing aborted as configuration requires calibration data.")
                return None, None

    # Initialize speed tracker if enabled
    speed_tracker = None
    if config.get("enable_speed_tracking", False):
        print("Initializing speed tracking...")
        speed_tracker = SpeedTracker(
            calibration_file=config.get("distance_tracking_calibration", "court_calibration.pkl")
        )

        if not speed_tracker.is_calibrated():
            print("Warning: Speed tracking is enabled but no court calibration data found.")
            speed_tracker = None

    # Open video capture
    cap = cv2.VideoCapture(config["input_path"])

    if not cap.isOpened():
        print(f"Error: Unable to open video source {config['input_path']}")
        return None, None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} FPS, Total frames: {total_frames}")

    # Update speed tracker FPS if available
    if speed_tracker:
        speed_tracker.fps = fps
        speed_tracker.time_delta = 1.0 / fps

    # Create video writer if output path is specified
    video_writer = None
    if config["save_output"]:
        # Use the get_output_path function from config
        output_path = f"{config['get_output_path'](config)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        print(f"Output video will be saved to: {output_path}")

    # Process frames
    frame_count = 0
    processing_times = []
    results_list = []
    start_process_time = time.time()

    try:
        while True:
            frame_start_time = time.time()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create mask and apply it
            mask = create_mask(frame_rgb.shape, config["roi"])
            masked_frame = apply_mask_to_image(frame_rgb, mask)

            # Process with model in streaming mode
            results = detect_with_model(
                model,
                Image.fromarray(masked_frame).convert('RGB'),
                config.get("confidence_threshold", 0.25),
                stream=True  # Enable streaming mode
            )

            # Process results generator
            for result in results:
                results_list.append(result)

                # Extract detection data
                boxes, confs, keypoints = extract_detection_data([result])

                # Update distance tracking if enabled
                if distance_tracker and distance_tracker.is_calibrated() and keypoints is not None:
                    distance_tracker.update_player_positions(keypoints)

                # Update speed tracking if enabled (completely independent)
                if speed_tracker and speed_tracker.is_calibrated() and keypoints is not None:
                    speed_tracker.update_player_speeds(keypoints)

                # Render results on original frame
                output_frame = frame_rgb.copy()
                output_frame = render_boxes_on_image(
                    output_frame,
                    boxes,
                    confs,
                    config["box_color"],
                    config["show_bounding_box"],
                    config.get("show_confidence_label", False)
                )
                output_frame = render_keypoints_on_image(
                    output_frame,
                    keypoints,
                    config["keypoint_color"],
                    config["skeleton_colors"],
                    config["pose_connections"],
                    classify_persons=config.get("classify_persons", False),
                    person_aliases=config.get("person_aliases", None),
                    person_label_colors=config.get("person_label_colors", None),
                    label_font_scale=config.get("label_font_scale", 0.7),
                    label_thickness=config.get("label_thickness", 2),
                    label_color=config.get("label_color", (0, 255, 0)),
                    label_background=config.get("label_background", True),
                    label_background_color=config.get("label_background_color", (0, 0, 0)),
                    label_background_alpha=config.get("label_background_alpha", 0.5),
                    line_thickness=config.get("line_thickness", 2)
                )

                # Draw distance tracking bars (if enabled)
                if distance_tracker and distance_tracker.is_calibrated():
                    output_frame = distance_tracker.draw_distance_bars(
                        output_frame,
                        config.get("person_aliases", None),
                        player_display_order=config.get("player_display_order", None)
                        # Add player display order parameter
                    )

                # Draw speed bars (if speed tracking is enabled, draw after distance bars)
                if speed_tracker and speed_tracker.is_calibrated():
                    output_frame = speed_tracker.draw_speed_bars(
                        output_frame,
                        config.get("person_aliases", None),
                        pixels_per_speed=config.get("pixels_per_speed", 30),
                        max_display_width=config.get("max_speed_bar_width", 250),
                        player_display_order=config.get("player_display_order", None)
                        # Add player display order parameter
                    )

                # Convert RGB back to BGR for OpenCV
                output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

                # Calculate and update processing times
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                processing_times.append(processing_time)

                # Limit the size of processing_times to the last 30 frames
                if len(processing_times) > 30:
                    processing_times.pop(0)

                # Calculate current fps
                current_fps = 1.0 / (sum(processing_times) / len(processing_times))

                # Calculate elapsed time and ETA
                elapsed_time = frame_end_time - start_process_time
                if frame_count > 0:
                    eta = elapsed_time / frame_count * (total_frames - frame_count)
                else:
                    eta = 0

                # Update progress bar
                progress_bar(
                    frame_count + 1,
                    total_frames,
                    prefix='Processing:',
                    suffix=f'Frame {frame_count + 1}/{total_frames}',
                    elapsed_time=elapsed_time,
                    fps=current_fps,
                    eta=eta
                )

                # Add FPS info to the frame
                cv2.putText(
                    output_frame_bgr,
                    f"FPS: {current_fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Display frame
                if config["display_output"]:
                    cv2.imshow("YOLO Badminton Pose and Distance Tracker", output_frame_bgr)

                # Write frame to output video
                if video_writer:
                    video_writer.write(output_frame_bgr)

                frame_count += 1

                # Check for user exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"\nError processing video: {e}")

    finally:
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    total_elapsed = time.time() - start_process_time
    if frame_count > 0:
        print(f"\nProcessing complete! Processed {frame_count}/{total_frames} frames")
        print(f"Total time: {total_elapsed:.2f} seconds, Average speed: {frame_count / total_elapsed:.2f} FPS")

        # Print distance report if tracking was enabled
        if distance_tracker and distance_tracker.is_calibrated():
            distances = distance_tracker.get_total_distances()
            print("\nPlayer Movement Distance Report:")
            for player_id, distance in distances.items():
                player_name = config.get("person_aliases", {}).get(player_id, player_id)
                print(f"  {player_name}: {distance:.2f} meters")

        # Print speed report if tracking was enabled
        if speed_tracker and speed_tracker.is_calibrated():
            speeds = speed_tracker.get_player_speeds()
            print("\nPlayer Instantaneous Speed Report:")
            for player_id, speed in speeds.items():
                player_name = config.get("person_aliases", {}).get(player_id, player_id)
                print(f"  {player_name}: {speed:.2f} meters/second")

    return None, results_list
