"""
Re-ID training and evaluation on Market-1501.

  - train_reid.py       : Fine-tune ResNetReIDBackbone (triplet + CE loss)
  - train_torchreid.py  : Alternative training via torchreid engine
  - evaluate.py         : CMC rank-1/5/10 + mAP evaluation
  - dataset.py          : Market-1501 Dataset + EvalDataset + PKSampler
  - configs/            : Training hyperparameters
"""
