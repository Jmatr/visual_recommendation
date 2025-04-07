# Visual Recommendation System

## Purpose

This project enhances traditional recommendation systems by incorporating **visual content analysis** using computer vision. Platforms like TikTok let users mark videos as "not interested" based on metadata (e.g., author, tags, or keywords). However, videos may contain undesired elements (e.g., birds, insects, or fish) not reflected in those tags.

This program addresses that limitation by scanning the **actual video content**, detecting visible objects frame by frame using a YOLOv5 model, and aggregating them. The result is a list of detected objects that can be used to better filter videos and improve user satisfaction with content recommendations.

## Features

- Frame extraction from videos at a configurable frame rate
- Object detection using YOLOv5 (pretrained on the COCO dataset)
- Aggregated counts of detected objects across all sampled frames
- CLI interface to control frame rate and input file
- Swappable model size (`yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`) for speed vs. accuracy

## Requirements

- Python 3.7 or newer
- pip packages:
  - torch
  - torchvision
  - opencv-python
  - pandas
  - requests

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/visual_recommendation.git
cd visual_recommendation
```

2. **Create and Activate a Virtual Environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install torch torchvision opencv-python pandas requests
```

## Usage

To run the program:

```bash
python visual_recog.py path/to/video.mp4 --frame_rate 1.0
```

### Arguments

| Argument         | Description                                | Default |
|------------------|--------------------------------------------|---------|
| `video_path`     | Path to the video file (required)          | -       |
| `--frame_rate`   | Frames to process per second (optional)    | 1.0     |

### Sample Output

```
Loading YOLOv5 model...
Video FPS: 30.0, processing every 30 frame(s).

Aggregated detections:
bird: 112
person: 25
kite: 4
```

## Customizing the Model

You can switch to a more advanced YOLOv5 variant for better accuracy by modifying this line in `visual_recog.py`:

```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
```

Available options:
- `yolov5s`: Small, fast, less accurate
- `yolov5m`: Medium
- `yolov5l`: Large
- `yolov5x`: Most accurate, slowest

## Ignoring Large Files (GitHub)

To avoid pushing large files like `.mp4` videos to GitHub:

1. Add them to `.gitignore`:

```
*.mp4
*.avi
*.mov
```

2. Untrack any already-added large files:

```bash
git rm --cached path/to/large_video.mp4
```

## Troubleshooting

- **Missing `requests` error**  
  Install it with:
  ```bash
  pip install requests
  ```

- **YOLOv5 download warning**  
  If you see a warning about downloading an untrusted repository, use:
  ```python
  torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
  ```

- **Deprecation warning about autocast**  
  This is safe to ignore. It relates to PyTorch internal updates and won't affect output.

## Future Improvements

- Add object tracking across frames to reduce false positives
- Train a custom YOLO model for domain-specific objects (e.g., fish species)
- Add caption generation for scene understanding
- Integrate detection output with a real-time content filtering system

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for their powerful and open-source object detection models
- PyTorch and OpenCV for making real-time computer vision accessible
- The open-source community for the tools that made this project possible
