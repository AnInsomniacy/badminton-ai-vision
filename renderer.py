import cv2
import matplotlib.pyplot as plt
import numpy as np


def render_boxes_on_image(image, boxes, confs, box_color, show_bounding_box=True, show_confidence_label=False):
    """Render bounding boxes on image

    Args:
        image: Original image
        boxes: Bounding boxes coordinates
        confs: Confidence scores
        box_color: Color for bounding boxes
        show_bounding_box: Whether to draw the bounding box rectangle
        show_confidence_label: Whether to show the confidence score label

    Returns:
        image: Image with drawn boxes
    """
    if boxes is None or confs is None:
        return image

    output_image = image.copy()

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box.astype(int)

        # Only draw the rectangle if show_bounding_box is True
        if show_bounding_box:
            cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 2)

        # Only show the confidence label if requested
        if show_confidence_label:
            label = f"Person: {conf:.2f}"
            cv2.putText(output_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    return output_image


def classify_people(keypoints):
    """Classify detected people as upper or lower based on their position

    Args:
        keypoints: Array of keypoints for each person

    Returns:
        person_classes: List of person classifications ("upper" or "lower")
    """
    if keypoints is None or len(keypoints) == 0:
        return []

    person_classes = []
    # If there is only one person, classify as upper by default
    if len(keypoints) == 1:
        person_classes.append("upper")
        return person_classes

    # Get the average y-coordinate of each person's keypoints
    person_y_positions = []
    for person_kpts in keypoints:
        valid_kpts = [(x, y) for x, y in person_kpts if x > 0 and y > 0]
        if valid_kpts:
            avg_y = sum(y for _, y in valid_kpts) / len(valid_kpts)
            person_y_positions.append(avg_y)
        else:
            # If no valid keypoints, use a default high value
            person_y_positions.append(float('inf'))

    # Sort people by their y-positions
    sorted_indices = np.argsort(person_y_positions)

    # Classify people based on their y-position rank
    for i in range(len(keypoints)):
        rank = np.where(sorted_indices == i)[0][0]

        if rank == 0:  # The person with the smallest y-coordinate (higher in the image)
            person_classes.append("upper")
        else:  # The person with the larger y-coordinate (lower in the image)
            person_classes.append("lower")

    return person_classes


def render_keypoints_on_image(image, keypoints, keypoint_color, skeleton_colors, pose_connections,
                              classify_persons=False, person_aliases=None, person_label_colors=None,
                              label_font_scale=0.7, label_thickness=2, label_color=(0, 255, 0),
                              label_background=True, label_background_color=(0, 0, 0),
                              label_background_alpha=0.5, line_thickness=2):
    """Render keypoints on image with different colors for different body parts

    Args:
        image: Image to draw on
        keypoints: Keypoints to draw
        keypoint_color: Color for keypoints
        skeleton_colors: Dictionary of colors for different body parts
        pose_connections: Dictionary of keypoint connections by body part
        classify_persons: Whether to classify and label people
        person_aliases: Dictionary of person aliases {"upper": "label", "lower": "label"}
        person_label_colors: Dictionary of label colors {"upper": (r,g,b), "lower": (r,g,b)}
        label_font_scale: Font scale for the label
        label_thickness: Thickness of the label text
        label_color: Default color for the label text (used if person_label_colors not specified)
        label_background: Whether to add background to the label
        label_background_color: Color for the label background
        label_background_alpha: Alpha value for label background
        line_thickness: Thickness for skeleton lines

    Returns:
        image: Image with drawn keypoints
    """
    if keypoints is None:
        return image

    output_image = image.copy()

    # Classify people if requested
    person_classes = []
    if classify_persons and person_aliases:
        person_classes = classify_people(keypoints)

    try:
        for idx, person_kpts in enumerate(keypoints):
            # Draw points
            for kpt_idx, (x, y) in enumerate(person_kpts):
                if x > 0 and y > 0:  # Valid keypoint
                    cv2.circle(output_image, (int(x), int(y)), 4, keypoint_color, -1)

            # Draw connection lines with different colors for each body part
            for body_part, connections in pose_connections.items():
                color = skeleton_colors[body_part]
                for p1_idx, p2_idx in connections:
                    if (p1_idx < len(person_kpts) and p2_idx < len(person_kpts)):
                        p1, p2 = person_kpts[p1_idx], person_kpts[p2_idx]
                        if min(p1[0], p1[1], p2[0], p2[1]) > 0:  # Both points are valid
                            cv2.line(output_image,
                                     (int(p1[0]), int(p1[1])),
                                     (int(p2[0]), int(p2[1])),
                                     color, line_thickness)

            # Add person label if classification is enabled
            if classify_persons and person_aliases and idx < len(person_classes):
                person_class = person_classes[idx]
                if person_class in person_aliases:
                    # Find the nose keypoint (usually index 0) for label placement
                    # If nose not visible, use the center of the bounding box formed by keypoints
                    valid_kpts = [(int(x), int(y)) for x, y in person_kpts if x > 0 and y > 0]

                    if len(valid_kpts) > 0:
                        if person_kpts[0][0] > 0 and person_kpts[0][1] > 0:  # Nose is visible
                            label_x, label_y = int(person_kpts[0][0]), int(person_kpts[0][1]) - 30
                        else:
                            # Calculate center of valid keypoints
                            label_x = sum(x for x, _ in valid_kpts) // len(valid_kpts)
                            label_y = min(y for _, y in valid_kpts) - 30

                        # Get the label text
                        label_text = person_aliases[person_class]

                        # Get text size for background rectangle
                        (text_width, text_height), _ = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness
                        )

                        # Determine label color - use custom color if available, otherwise default
                        current_label_color = label_color
                        if person_label_colors and person_class in person_label_colors:
                            current_label_color = person_label_colors[person_class]

                        # Draw label background if enabled
                        if label_background:
                            # Create a separate image for the semi-transparent background
                            if label_background_alpha < 1.0:
                                bg_img = output_image.copy()
                                cv2.rectangle(
                                    bg_img,
                                    (label_x - 5, label_y - text_height - 5),
                                    (label_x + text_width + 5, label_y + 5),
                                    label_background_color,
                                    -1
                                )
                                # Blend the background with the original image
                                cv2.addWeighted(
                                    bg_img, label_background_alpha,
                                    output_image, 1 - label_background_alpha,
                                    0, output_image
                                )
                            else:
                                cv2.rectangle(
                                    output_image,
                                    (label_x - 5, label_y - text_height - 5),
                                    (label_x + text_width + 5, label_y + 5),
                                    label_background_color,
                                    -1
                                )

                        # Draw the label text
                        cv2.putText(
                            output_image,
                            label_text,
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            label_font_scale,
                            current_label_color,
                            label_thickness
                        )

    except Exception as e:
        print(f"Error processing keypoints: {e}")

    return output_image


def save_output_image(image, output_path):
    """Save output image to specified path

    Args:
        image: Final output image
        output_path: Path to save the image
    """
    if output_path:
        # Save high quality image
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"High quality result saved to {output_path}")
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
