import os
import torchreid
from configs.resnet50_market1501 import CONFIG

# -----------------------------
# Dataset
# -----------------------------
datamanager = torchreid.data.ImageDataManager(
    root=CONFIG["root"],
    sources=CONFIG["dataset"],
    targets=CONFIG["dataset"],
    height=256,
    width=128,
    batch_size_train=CONFIG["batch_size"],
    batch_size_test=100,
    workers=CONFIG["num_workers"]
)

# -----------------------------
# Model
# -----------------------------
model = torchreid.models.build_model(
    name=CONFIG["model_name"],
    num_classes=datamanager.num_train_pids,
    loss="triplet",
    pretrained=CONFIG["pretrained"]
)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = torchreid.optim.build_optimizer(
    model,
    optim=CONFIG["optimizer"],
    lr=CONFIG["lr"]
)

# -----------------------------
# Scheduler
# -----------------------------
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=CONFIG["stepsize"]
)

# -----------------------------
# Engine
# -----------------------------
engine = torchreid.engine.ImageTripletEngine(
    datamanager,
    model,
    optimizer=optimizer,
    margin=CONFIG["triplet_margin"],
    scheduler=scheduler
)

# -----------------------------
# Train
# -----------------------------
engine.run(
    save_dir=CONFIG["save_dir"],
    max_epoch=CONFIG["max_epoch"],
    eval_freq=5,
    print_freq=20,
    test_only=False
)