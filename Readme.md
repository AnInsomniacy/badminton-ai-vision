# 🏸 badminton-ai-vision

## 📊 Current & Future Features

**Currently Available:**
- ✅ Player movement distance tracking
- ✅ Player speed analysis

**Coming Soon:**
- ⏳ Shot detection and classification
- ⏳ Rally pattern analysis
- ⏳ Player positioning heatmaps
- ⏳ Performance metrics and statistics

---

## 🌟 Overview

Badminton-Rally-AI-Analyzer is a sophisticated computer vision system that uses AI to analyze badminton matches. The system employs YOLO pose detection to track players' movements, calculate distances traveled, and measure movement speeds with high precision through court calibration.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎯 Pose Detection** | High-precision skeletal tracking using YOLOv8 |
| **📏 Court Calibration** | Maps pixels to real-world coordinates for accurate measurements |
| **📈 Distance Tracking** | Calculates total movement distance for each player |
| **⚡ Speed Analysis** | Real-time measurement of player instantaneous speed |
| **🔍 Custom ROI** | Focus detection on relevant court areas |
| **👤 Player Identification** | Automatically distinguishes between players |
| **🎨 Visualization** | Color-coded skeleton rendering with customizable display |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv8
- PIL (Pillow)
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/badminton-rally-ai-analyzer.git
cd badminton-rally-ai-analyzer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models raw_videos processed_videos
```

> 💡 **Note:** You'll need to download a YOLOv8 pose model into the `models` directory.

---

## 📚 Usage Guide

### 1️⃣ Court Calibration

This step maps the video pixels to real-world court measurements:

```bash
python court_calibration/court_calibrator.py
```

### 2️⃣ Process Video

```bash
python main.py
```

The system will:
- Process the specified video
- Track players and their movements
- Calculate and visualize distances and speeds
- Save the analysis to an output video

### 3️⃣ Configuration

Edit the `CONFIG` dictionary in `main.py` to customize your analysis:

```python
CONFIG = {
    "input_path": "raw_videos/YourMatch.mp4",
    "enable_distance_tracking": True,
    "enable_speed_tracking": True,
    "person_aliases": {
        "upper": "PLAYER 1",
        "lower": "PLAYER 2"
    },
    # Additional settings...
}
```

---

## 🗂️ Project Structure

```
badminton-rally-ai-analyzer/
├── court_calibration/
│   └── court_calibrator.py     # Court calibration tool
├── processor/
│   ├── distance_processor.py   # Player distance tracking
│   ├── image_processor.py      # Static image processing
│   ├── mask_processor.py       # ROI masking utilities
│   ├── model_processor.py      # YOLO model handling
│   ├── speed_processor.py      # Player speed tracking
│   └── video_processor.py      # Video processing pipeline
├── main.py                     # Main application entry point
├── renderer.py                 # Visualization utilities 
├── utility.py                  # Helper functions
├── models/                     # YOLO model storage
├── raw_videos/                 # Input videos
└── processed_videos/           # Output videos
```

---

## ⚙️ Configuration Options

| Category | Option | Description |
|----------|--------|-------------|
| **Model** | `model_path` | Path to YOLO pose model |
| | `confidence_threshold` | Detection confidence (0.0-1.0) |
| **I/O** | `input_path` | Path to input video |
| | `output_dir` | Directory for processed videos |
| **Tracking** | `enable_distance_tracking` | Enable distance measurement |
| | `enable_speed_tracking` | Enable speed tracking |
| | `player_display_order` | Control order of player stats display |
| **Visual** | `show_bounding_box` | Show detection boxes |
| | `keypoint_color` | Color for keypoints |
| | `skeleton_colors` | Colors for different body parts |
| | `person_aliases` | Custom player names |

---

## 🔧 Troubleshooting

### Video File Access Issues

If you encounter problems accessing video files:

```bash
# Fix file permissions
chmod 644 raw_videos/YourVideo.mp4

# Convert video to compatible format
ffmpeg -i raw_videos/YourVideo.mp4 -c:v libx264 -preset slow -crf 22 raw_videos/converted.mp4
```

Or modify the `CONFIG` to use absolute paths:

```python
CONFIG = {
    "video_path": "/absolute/path/to/your/video.mp4",
    # Other settings...
}
```

### Calibration Issues

For court calibration problems:

- Ensure good lighting and court visibility
- Mark the four corners accurately
- Try a frame with clear court lines
- Verify calibration file paths

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV project

---

Created with ❤️ for badminton enthusiasts and coaches. Feel free to contribute to this project!