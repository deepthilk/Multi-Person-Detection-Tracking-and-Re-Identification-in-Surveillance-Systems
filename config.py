"""
Configuration file for multi-person tracking and Re-ID system
Centralized settings for all modules
"""

# ==============================================================================
# DETECTION (YOLOv8) SETTINGS
# ==============================================================================

DETECTION = {
    'model_path': 'models/yolov8s.pt',
    'conf_threshold': 0.35,
    'imgsz': 960,
    'device': 'cuda',  # 'cuda' or 'cpu'
}

# ==============================================================================
# TRACKING (DeepSORT) SETTINGS
# ==============================================================================

TRACKING = {
    'max_age': 3,
    'n_init': 2,
    'max_iou_distance': 0.7,
    'max_cosine_distance': 0.3,
}

# ==============================================================================
# RE-IDENTIFICATION (OSNet) SETTINGS
# ==============================================================================

REID = {
    'model_name': 'osnet_x1_0',
    'device': 'cuda',  # 'cuda' or 'cpu'
    'similarity_threshold': 0.5,
    'feature_dim': 512,
}

# ==============================================================================
# VIDEO PATHS
# ==============================================================================

VIDEO = {
    'input_dir': 'input',
    'videos': ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4'],
    'default_video': 'input/video4.mp4',
}

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================

OUTPUT = {
    'output_dir': 'outputs',
    'detections_json': 'outputs/detections.json',
    'tracking_json': 'outputs/tracking.json',
    'reid_json': 'outputs/reid_results.json',
    'visualized_video': 'outputs/result_tracking.mp4',
}

# ==============================================================================
# DATASET SETTINGS (Market-1501)
# ==============================================================================

DATASET = {
    'name': 'Market-1501',
    'root_dir': 'reidentification/dataset/Market-1501',
    'query_dir': 'reidentification/dataset/Market-1501/query',
    'gallery_dir': 'reidentification/dataset/Market-1501/bounding_box_test',
    'train_dir': 'reidentification/dataset/Market-1501/bounding_box_train',
}

# ==============================================================================
# SYSTEM SETTINGS
# ==============================================================================

SYSTEM = {
    'log_level': 'INFO',
    'seed': 42,
    'num_workers': 4,
    'batch_size': 32,
}
