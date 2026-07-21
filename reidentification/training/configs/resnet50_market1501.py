import torch

CONFIG = {

    # Dataset
    "dataset": "market1501",
    "root": "reidentification/dataset",

    # Model
    "model_name": "resnet50",
    "pretrained": True,
    "feature_dim": 512,

    # Training
    "batch_size": 32,
    "num_workers": 4,
    "max_epoch": 30,

    # Optimizer
    "optimizer": "adam",
    "lr": 0.0003,
    "weight_decay": 5e-4,

    # Scheduler
    "stepsize": 20,
    "gamma": 0.1,

    # Loss
    "triplet_margin": 0.3,

    # Output
    "save_dir": "../weights",

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}