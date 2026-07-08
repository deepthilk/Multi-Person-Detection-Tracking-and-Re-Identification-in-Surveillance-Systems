## Multi-Person Detection, Tracking and Re-Identification in Surveillance Systems

A comprehensive system for detecting, tracking, and re-identifying persons across video frames and cameras using state-of-the-art deep learning models.

## System Architecture

```
Video Input
    ↓
[1] DETECTION (YOLOv8)    → Detects all persons in each frame
    ↓
[2] TRACKING (DeepSORT)   → Assigns unique IDs to persons across frames
    ↓
[3] RE-IDENTIFICATION (OSNet) → Matches persons and creates unified identities
    ↓
Output: Tracking + Re-ID Results
```

## Features

- **YOLOv8 Detection**: Real-time person detection with 35% confidence threshold
- **DeepSORT Tracking**: Multi-object tracking with Hungarian algorithm and Kalman filtering
- **OSNet Re-ID**: Person re-identification using deep metric learning
- **Re-ID Pipeline**: Works with Market-1501-style appearance embeddings and local runtime assets
- **Modular Design**: Easy to extend and customize each component
- **Visualization**: Draw tracking results and Re-ID matches on video

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)

### Required Assets

This repository is now source-first for GitHub sharing. To run the full pipeline, restore these assets locally:

- `models/yolov8s.pt` for YOLOv8 detection
- `input/*.mp4` for sample or test videos
- `reidentification/dataset/Market-1501/` for Re-ID training/reference data

The repo intentionally excludes generated outputs, uploads, logs, bytecode caches, and other runtime artifacts.

### Before You Run

If you share this repo with a friend, ask them to add these items before starting:

1. A video file inside `input/`.
2. The YOLOv8 weights at `models/yolov8s.pt`.
3. The Market-1501 dataset folder if they want the full Re-ID pipeline.
4. Python dependencies from `requirements.txt`.

### Setup

1. **Clone or download the project**
```bash
cd multi-person-tracking-ReId-system
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPU: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('✅ YOLOv8 OK')"
python -c "import torchreid; print('✅ TorchReid OK')"
```

## Directory Structure

```
multi-person-tracking-ReId-system/
├── detection/
│   └── detect_module.py          # Detection module
├── tracking/
│   └── track_module.py           # Tracking module
├── reidentification/
│   ├── reid_main.py              # Re-ID pipeline
│   ├── model/
│   │   └── reid_model.py         # Re-ID model definitions
├── main.py                       # Main orchestration script
├── config.py                     # Configuration settings
├── utils.py                      # Utility functions
└── requirements.txt              # Python dependencies
```

## Repository Notes

- The repo does not ship sample videos, downloaded model weights, or the Market-1501 dataset.
- Generated folders such as `outputs/`, `web/outputs/`, `web/uploads/`, and `__pycache__/` are intentionally excluded.
- If you want a fully runnable copy for your friends, add those assets back locally before running the pipeline.

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```bash
# Run detection + tracking + Re-ID on a video in input/
python main.py

# With GPU (default)
python main.py --device cuda

# With CPU
python main.py --device cpu
```

### Option 2: Run Individual Steps

```bash
# Step 1: Detection only
python main.py --step 1

# Step 2: Detection + Tracking
python main.py --step 2

# Step 3: Full pipeline
python main.py --step 3
```

### Option 3: Skip Steps (use existing results)

```bash
# Use existing detections, run tracking and re-id
python main.py --skip-detection

# Use existing tracking, run only re-id
python main.py --skip-detection --skip-tracking
```

### Option 4: Visualize Results

```bash
# Run full pipeline with visualization
python main.py --visualize

# Or visualize only
python -c "from utils import visualize_results; visualize_results('input/your_video.mp4', 'outputs/detections.json', 'outputs/tracking.json', mode='tracking')"
```

## Usage Examples

### Example 1: Process Different Video

```bash
python main.py --video input/video1.mp4
```

### Example 2: Adjust Detection Confidence

```bash
python main.py --conf-threshold 0.40
```

### Example 3: Use CPU (Faster on systems without good GPU)

```bash
python main.py --device cpu
```

### Example 4: Get Summary Report

```python
from utils import generate_summary_report

report = generate_summary_report(
    'outputs/detections.json',
    'outputs/tracking.json',
    'outputs/reid_results.json'
)
print(report)
```

## API Reference

### Detection Module

```python
from detection.detect_module import PersonDetector, run_detection

# Option 1: Use detector class
detector = PersonDetector(conf_threshold=0.35)
frame = cv2.imread('image.jpg')
detections = detector.detect(frame)

# Option 2: Run on entire video
detections = run_detection('video.mp4', 'detections.json')
```

### Tracking Module

```python
from tracking.track_module import PersonTracker, run_tracking

# Option 1: Use tracker class
tracker = PersonTracker()
tracks = tracker.update(frame, detections)

# Option 2: Run on entire video
tracks = run_tracking('video.mp4', 'detections.json', 'tracking.json')
```

### Re-ID Module

```python
from reidentification.reid_main import ReIDEngine, run_reid_pipeline

# Option 1: Use Re-ID engine
reid = ReIDEngine()
feature = reid.extract_feature(frame, bbox)
matches = reid.match_person(feature, top_k=3)

# Option 2: Run on entire video
reid, results = run_reid_pipeline('video.mp4', 'tracking.json', 'reid_results.json')
```

### Utilities

```python
from utils import visualize_results, generate_summary_report

# Visualize tracking results
visualize_results('video.mp4', 'detections.json', 'tracking.json', mode='tracking')

# Get summary statistics
report = generate_summary_report('detections.json', 'tracking.json', 'reid_results.json')
```

## Configuration

Edit `config.py` to customize system parameters:

```python
# Detection settings
DETECTION = {
    'conf_threshold': 0.35,  # Lower = more detections
    'imgsz': 960,             # YOLO input size
}

# Tracking settings
TRACKING = {
    'max_age': 3,             # Frames to keep lost tracks
    'n_init': 2,              # Frames before confirming track
    'max_cosine_distance': 0.3,  # Re-ID distance threshold
}

# Re-ID settings
REID = {
    'similarity_threshold': 0.5,  # Matching threshold
    'feature_dim': 512,
}
```

## Output Format

### Detections (`detections.json`)
```json
{
  "1": [
    [x1, y1, width, height, confidence],
    [x2, y2, width, height, confidence]
  ],
  "2": [...]
}
```

### Tracking (`tracking.json`)
```json
{
  "1": [
    {"id": 1, "bbox": [x1, y1, x2, y2]},
    {"id": 2, "bbox": [x1, y1, x2, y2]}
  ],
  "2": [...]
}
```

### Re-ID Results (`reid_results.json`)
```json
{
  "1": [
    {
      "id": 1,
      "bbox": [x1, y1, x2, y2],
      "matches": [
        {"person_id": 1, "similarity": 0.95},
        {"person_id": 5, "similarity": 0.78}
      ]
    }
  ],
  "2": [...]
}
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Use CPU or reduce image size
```bash
python main.py --device cpu
# Or modify config.py to reduce imgsz
```

### Issue: No models found
**Solution**: Restore `models/yolov8s.pt` before running the pipeline.

### Issue: Low detection rate
**Solution**: Adjust confidence threshold in config.py
```python
DETECTION['conf_threshold'] = 0.25  # Lower threshold
```

### Issue: Tracking ID jumps/switches
**Solution**: Adjust DeepSORT parameters in config.py
```python
TRACKING['max_cosine_distance'] = 0.5  # Higher = more tolerant matching
```

## Performance

Typical performance (on NVIDIA RTX 3080 with 2-minute video):

| Module | Time | FPS |
|--------|------|-----|
| Detection (YOLOv8) | ~60s | ~2 |
| Tracking (DeepSORT) | ~30s | ~4 |
| Re-ID (OSNet) | ~120s | ~1 |
| **Total** | **~210s** | **N/A** |

*Times vary based on video resolution and system specs*

## Citation

If you use this system in research, please cite:

```bibtex
@inproceedings{ge2018deep,
  title={Deep Tracking: Visual Tracking Using Deep Convolutional Networks},
  author={Ge, Zhun and Bewley, Alex},
  booktitle={IEEE TPAMI},
  year={2018}
}

@article{zhou2020osnet,
  title={OSNet: A Single-Stream Convolutional Network for Person Re-identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea},
  journal={ECCV},
  year={2020}
}
```

## License

This project uses open-source components. See individual module licenses for details.

## Support

For issues, questions, or contributions, refer to the module documentation or README files in each subdirectory.