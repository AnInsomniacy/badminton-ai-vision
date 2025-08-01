# Badminton AI Vision

A computer vision system for analyzing badminton matches using AI-powered pose detection and court calibration to track player movements, distances, and speeds.

![Badminton Analysis](https://github.com/user-attachments/assets/8d020c5c-0c79-47ef-b818-3eea18f35f95)

## Features

- **Pose Detection**: High-precision skeletal tracking using YOLOv8/v11
- **Court Calibration**: Maps pixels to real-world coordinates for accurate measurements
- **Distance Tracking**: Calculates total movement distance for each player
- **Speed Analysis**: Real-time measurement of player instantaneous speed
- **ROI Processing**: Focus detection on relevant court areas
- **Player Identification**: Automatically distinguishes between players
- **Visualization**: Color-coded skeleton rendering with customizable display

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/AnInsomniacy/badminton-ai-vision.git
cd badminton-ai-vision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories for videos
mkdir -p raw_videos processed_videos
```

**Note**: You'll need to place a YOLOv8/v11 pose model in the `models/` directory.

## Usage

### 1. Court Calibration

Calibrate the court coordinates for accurate distance measurements:

```bash
python court_calibration/court_calibrator.py
```

### 2. Process Video

```bash
python main.py
```

### 3. Configuration

Edit the `CONFIG` dictionary in `main.py` to customize analysis settings:

```python
CONFIG = {
    "input_path": "raw_videos/YourMatch.mp4",
    "model_path": "models/yolo11x-pose.mlpackage",
    "enable_distance_tracking": True,
    "enable_speed_tracking": True,
    "person_aliases": {
        "upper": "PLAYER 1",
        "lower": "PLAYER 2"
    }
}
```

## Project Structure

```
badminton-ai-vision/
├── court_calibration/          # Court calibration tools
│   ├── court_calibrator.py
│   └── ROISelector.py
├── processor/                  # Core processing modules
│   ├── video_processor.py
│   ├── image_processor.py
│   ├── model_processor.py
│   ├── distance_processor.py
│   ├── speed_processor.py
│   └── mask_processor.py
├── models/                     # YOLO model storage
├── raw_videos/                 # Input videos
├── processed_videos/           # Output videos
├── main.py                     # Main application
├── renderer.py                 # Visualization utilities
└── requirements.txt
```

## Key Configuration Options

| Setting | Description |
|---------|-------------|
| `model_path` | Path to YOLO pose model |
| `confidence_threshold` | Detection confidence (0.0-1.0) |
| `input_path` | Path to input video file |
| `enable_distance_tracking` | Enable distance measurement |
| `enable_speed_tracking` | Enable speed analysis |
| `roi` | Region of interest coordinates |
| `person_aliases` | Custom player names |

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
