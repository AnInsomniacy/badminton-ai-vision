import numpy as np
from PIL import Image
from .model_processor import load_yolo_model, detect_with_model, extract_detection_data
from .mask_processor import create_mask, apply_mask_to_image
from renderer import render_boxes_on_image, render_keypoints_on_image, save_output_image


def process_image(config):
    """Process a single image with YOLO pose detection

    Args:
        config: Configuration dictionary

    Returns:
        output_image: Processed image
        results: Detection results
    """
    # Load original image
    original_image = np.array(Image.open(config["input_path"]))

    # 1. Generate mask and apply it
    mask = create_mask(original_image.shape, config["roi"])
    masked_image = apply_mask_to_image(original_image, mask)

    # 2. Model processing
    model = load_yolo_model(config["model_path"])
    results = detect_with_model(
        model,
        Image.fromarray(masked_image).convert('RGB'),
        config["confidence_threshold"]
    )
    boxes, confs, keypoints = extract_detection_data(results)

    # 3. Render results on original image
    output_image = original_image.copy()
    output_image = render_boxes_on_image(
        output_image,
        boxes,
        confs,
        config["box_color"],
        config["show_bounding_box"],
        config.get("show_confidence_label", False)
    )
    output_image = render_keypoints_on_image(
        output_image,
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

    # 4. Save final result
    if config["save_output"]:
        # Use the get_output_path function from config
        output_path = f"{config['get_output_path'](config)}.png"
        save_output_image(output_image, output_path)

    return output_image, results
