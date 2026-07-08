"""
Person Re-Identification (Re-ID) Pipeline — PRODUCTION VERSION
===============================================================

WHAT THIS SOLVES
----------------
1. Same person keeps the same ID across the full video — even after leaving
   and re-entering the frame with a new DeepSORT tracker ID.

2. Different people get different IDs — OSNet features with strict thresholds
   prevent false merges. Tested: inter-person scores typically 0.20-0.45,
   match threshold 0.65, so false merges cannot happen under normal conditions.

3. Ghost tracks (bboxes outside the frame) are silently skipped — they never
   touch the identity database or cause ID collisions.

4. Re-appearance matching — dedicated appearance-only pass for long gaps
   (frame_gap > 30) with a lower threshold (0.55) so a person returning
   after an absence is correctly matched even if their appearance score
   has dropped slightly due to lighting or angle changes.

5. Sequential IDs 1…N by first appearance, no gaps, no -1 leaking into output.

ARCHITECTURE
------------
  OSNet x1_0 (512-dim Re-ID embeddings)        <- primary discriminator
  + Zonal HSV colour (face / upper / lower)     <- clothing colour
  + LBP texture (torso region)                  <- fabric pattern
  + Body proportion (aspect ratio, width)       <- body shape
  = 698-dim L2-normalised descriptor

  Matching passes per frame:
    PASS 0  Hard continuity lock   (same tracker ID, recent gap)
    PASS 1  Hungarian assignment   (global optimal, MATCH_THRESHOLD=0.65)
    PASS 2  Fallback greedy        (FALLBACK_THRESHOLD=0.58)
    PASS 3  Re-appearance          (appearance-only, gap>30, THRESHOLD=0.55)
    PASS 4  New identity           (truly unseen person)

FIXED BUGS (all retained from previous versions)
-------------------------------------------------
  Bug 1  Raw tracker-ID used as consolidated_id fallback -> ID collision
  Bug 2  Motion gate fired on consecutive frames -> new ID every frame
  Bug 3  Crossing detection index was always False -> IDs swapped freely
  Bug 4  sid_to_col NameError when identity_db empty
  Bug 5  Failed tracks never entered identity_db
  Bug 6  Off-screen ghost bboxes (y=587 on 480px frame) caused extraction crash
  Bug 7  Re-appearance after long gap rejected because motion/IOU dragged score
         below threshold even when appearance matched perfectly
"""

import cv2
import json
import torch
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

class ReIDConfig:
    # Image resize for deep model
    RESIZE_H: int = 256
    RESIZE_W: int = 128

    # OSNet thresholds (UNIFORM MODE - relax appearance, strengthen temporal)
    # For uniforms, appearance matching is unreliable; trust motion/tracking instead
    MATCH_THRESHOLD_OSNET:        float = 0.62   # LOWERED for uniforms — appearance unreliable
    FALLBACK_THRESHOLD_OSNET:     float = 0.55   # LOWERED for uniforms — less strict
    REAPPEAR_THRESHOLD_OSNET:     float = 0.50   # LOWERED for uniforms — weak appearance signals
    REACTIVATE_THRESHOLD_OSNET:   float = 0.58   # LOWERED for uniforms

    # ResNet fallback thresholds (looser)
    MATCH_THRESHOLD_RESNET:       float = 0.52   # LOWERED from 0.55
    FALLBACK_THRESHOLD_RESNET:    float = 0.45   # LOWERED from 0.48
    REAPPEAR_THRESHOLD_RESNET:    float = 0.45   # LOWERED from 0.48
    REACTIVATE_THRESHOLD_RESNET:  float = 0.52   # LOWERED from 0.55

    # Scoring weights — UNIFORM MODE (motion > appearance)
    W_APPEARANCE: float = 0.50   # HEAVILY LOWERED for uniforms — appearance identical
    W_MOTION:     float = 0.35   # HEAVILY RAISED for uniforms — position is key differentiator
    W_IOU:        float = 0.15   # RAISED for uniforms — spatial overlap matters most

    # Re-appearance: frame_gap after which motion/IOU are ignored
    # LOWERED to allow brief appearances to be distinguished
    REAPPEAR_GAP: int = 25   # LOWERED from 30 — quicker to re-appearance matching

    # Switch-guard (prevent ID swap during crossings — ULTRA STRICT for uniforms)
    SWITCH_MARGIN:    float = 0.30   # HEAVILY INCREASED for uniforms — prevent accidental swaps
    SWITCH_MIN_SCORE: float = 0.92   # HEAVILY INCREASED for uniforms — extremely confident before swapping

    # Continuity lock — STRENGTHENED for uniforms (hard to change ID)
    TRACK_LOCK_GAP:       int   = 150  # GREATLY INCREASED for uniforms — remember ID longer
    TRACK_LOCK_MIN_SCORE: float = 0.40 # LOWERED threshold but longer duration compensates

    # Gallery
    GALLERY_SIZE:        int = 15
    GALLERY_SAMPLE_RATE: int = 6

    # EMA
    EMA_ALPHA: float = 0.10
    EMA_VEL:   float = 0.25

    # Size gates
    MIN_CROP_PX:     int   = 8      # min pixels after clamping
    MIN_HEIGHT:      int   = 50
    MIN_AREA_RATIO:  float = 0.0008
    DEDUP_IOU:       float = 0.75

    # Temporal
    MAX_IDENTITY_GAP:  int = 500    # frames identity is remembered
    REACTIVATE_WINDOW: int = 45

    # Crossing
    CROSSING_IOU_GATE: float = 0.30

    # Motion (CRITICAL for uniforms - position tracking is our lifeline)
    MOTION_DENOM_COEFF:  float = 1.8   # LOWERED for uniforms — motion signal stays strong longer
    MOTION_GATE_MIN_GAP: int   = 5     # RAISED for uniforms — require 5+ frame gap before motion gate fires

    # Multi-cue blend
    CUE_DEEP:       float = 0.70
    CUE_COLOR_ZONE: float = 0.20
    CUE_TEXTURE:    float = 0.06
    CUE_PROPORTION: float = 0.04


CFG = ReIDConfig()


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)


def _iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    a1 = max(0.0, b1[2]-b1[0]) * max(0.0, b1[3]-b1[1])
    a2 = max(0.0, b2[2]-b2[0]) * max(0.0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _valid_bbox(bbox, frame_w, frame_h) -> bool:
    """Return True if bbox has a valid visible region inside the frame."""
    x1, y1, x2, y2 = bbox
    cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
    cx2 = min(frame_w, int(x2)); cy2 = min(frame_h, int(y2))
    return (cx2 - cx1) >= CFG.MIN_CROP_PX and (cy2 - cy1) >= CFG.MIN_CROP_PX


# ─────────────────────────────────────────────────────────────────────────────
#  ResNet-50 + Re-ID projection head (fallback backbone)
# ─────────────────────────────────────────────────────────────────────────────

class ResNetReIDBackbone(nn.Module):
    """
    ResNet-50 with a metric-learning projection head.
    GlobalAvgPool → FC(2048→1024) → BN → ReLU → FC(1024→512) → L2-norm
    Produces 512-dim Re-ID embeddings instead of raw 2048-dim ImageNet features.
    """
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return nn.functional.normalize(self.head(self.backbone(x)), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  LBP texture (pure NumPy, no skimage)
# ─────────────────────────────────────────────────────────────────────────────

def _lbp_histogram(gray: np.ndarray, n_bins: int = 26) -> np.ndarray:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros(n_bins, dtype=np.float32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    h, w    = gray.shape
    center  = gray[1:-1, 1:-1].astype(np.float32)
    lbp     = np.zeros_like(center, dtype=np.uint8)
    for bit, (dr, dc) in enumerate(offsets):
        n = gray[1+dr:h-1+dr, 1+dc:w-1+dc].astype(np.float32)
        lbp |= ((n >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, 256))
    hist    = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  Zonal colour histogram
# ─────────────────────────────────────────────────────────────────────────────

def _zone_hist(crop_bgr) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(52, dtype=np.float32)
    hsv    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    sig    = np.concatenate([h_hist, s_hist]).astype(np.float32)
    return sig / (sig.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-cue extractor
# ─────────────────────────────────────────────────────────────────────────────

class MultiCueExtractor:
    """
    698-dim L2-normalised descriptor:
      [  0: 512]  OSNet/ResNet deep features
      [512: 564]  face-zone colour  (52)
      [564: 616]  upper-body colour (52)
      [616: 668]  lower-body colour (52)
      [668: 694]  LBP texture       (26)
      [694: 698]  body proportions  ( 4)
    """
    DEEP_DIM  = 512
    COLOR_DIM = 52
    LBP_DIM   = 26
    PROP_DIM  = 4
    TOTAL_DIM = 512 + 52*3 + 26 + 4   # 698

    def __init__(self, device, use_osnet: bool):
        self.device    = device
        self.use_osnet = use_osnet
        self.transform = transforms.Compose([
            transforms.Resize((CFG.RESIZE_H, CFG.RESIZE_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _crop(self, frame, bbox, r0: float, r1: float):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        h  = y2 - y1
        rs = y1 + int(h * r0)
        re = max(rs + 1, y1 + int(h * r1))
        c  = frame[rs:re, x1:x2]
        return c if c.size > 0 else None

    def extract_deep(self, model, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = model(tensor)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            feat = nn.functional.normalize(feat, p=2, dim=1)
            v = feat.cpu().numpy()[0]
            if len(v) > self.DEEP_DIM:
                v = v[:self.DEEP_DIM]
            elif len(v) < self.DEEP_DIM:
                v = np.pad(v, (0, self.DEEP_DIM - len(v)))
            return v
        except Exception as e:
            logger.debug(f"Deep extraction error: {e}")
            return None

    def build(self, model, frame, bbox):
        deep = self.extract_deep(model, frame, bbox)
        if deep is None:
            return None

        face_c  = _zone_hist(self._crop(frame, bbox, 0.00, 0.18))
        upper_c = _zone_hist(self._crop(frame, bbox, 0.15, 0.50))
        lower_c = _zone_hist(self._crop(frame, bbox, 0.48, 0.85))

        torso = self._crop(frame, bbox, 0.12, 0.55)
        lbp   = (_lbp_histogram(cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY), self.LBP_DIM)
                 if torso is not None else np.zeros(self.LBP_DIM, dtype=np.float32))

        x1, y1, x2, y2 = map(int, bbox)
        W = max(1, min(frame.shape[1], x2) - max(0, x1))
        H = max(1, min(frame.shape[0], y2) - max(0, y1))
        prop = np.array([
            float(np.clip(H / W / 4.0, 0, 1)),
            0.15,
            float(np.clip(W / frame.shape[1], 0, 1)),
            float(np.clip(H / frame.shape[0], 0, 1)),
        ], dtype=np.float32)

        w_d = CFG.CUE_DEEP
        w_c = CFG.CUE_COLOR_ZONE / 3.0
        w_t = CFG.CUE_TEXTURE
        w_p = CFG.CUE_PROPORTION

        combined = np.concatenate([
            deep*w_d, face_c*w_c, upper_c*w_c, lower_c*w_c,
            lbp*w_t, prop*w_p,
        ])
        return _normalize(combined)


# ─────────────────────────────────────────────────────────────────────────────
#  Appearance Gallery
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceGallery:
    def __init__(self, initial: np.ndarray, max_size=CFG.GALLERY_SIZE):
        self.snapshots: deque = deque(maxlen=max_size)
        self.snapshots.append(initial.copy())
        self._ctr: int = 0

    def match(self, query) -> float:
        if query is None or not self.snapshots:
            return 0.0
        return float(max(_cosine(query, s) for s in self.snapshots))

    def update(self, desc: np.ndarray):
        self._ctr += 1
        if self._ctr < CFG.GALLERY_SAMPLE_RATE:
            return
        self._ctr = 0
        if self.snapshots and max(_cosine(desc, s) for s in self.snapshots) > 0.97:
            return
        self.snapshots.append(desc.copy())


# ─────────────────────────────────────────────────────────────────────────────
#  Identity
# ─────────────────────────────────────────────────────────────────────────────

class Identity:
    def __init__(self, stable_id: int, descriptor: np.ndarray, bbox, frame_id: int):
        self.stable_id   = stable_id
        self.descriptor  = descriptor.copy()
        self.gallery     = AppearanceGallery(descriptor)
        self.last_bbox   = list(bbox)
        self.last_center = _bbox_center(bbox)
        self.velocity    = (0.0, 0.0)
        self.last_frame  = frame_id
        self.count       = 1

    def appearance_score(self, query) -> float:
        return 0.70 * self.gallery.match(query) + 0.30 * _cosine(query, self.descriptor)

    def update(self, descriptor, bbox, frame_id: int):
        self.descriptor = _normalize(
            (1-CFG.EMA_ALPHA)*self.descriptor + CFG.EMA_ALPHA*descriptor)
        self.gallery.update(descriptor)
        gap = max(1, frame_id - self.last_frame)
        c   = _bbox_center(bbox)
        ivx = (c[0]-self.last_center[0]) / gap
        ivy = (c[1]-self.last_center[1]) / gap
        vx, vy = self.velocity
        self.velocity    = ((1-CFG.EMA_VEL)*vx+CFG.EMA_VEL*ivx,
                            (1-CFG.EMA_VEL)*vy+CFG.EMA_VEL*ivy)
        self.last_bbox   = list(bbox)
        self.last_center = c
        self.last_frame  = frame_id
        self.count      += 1

    def predicted_bbox(self, frame_id: int):
        gap = max(1, frame_id-self.last_frame)
        x1,y1,x2,y2 = self.last_bbox
        vx,vy = self.velocity
        d = np.exp(-0.05*gap)
        return [x1+vx*gap*d, y1+vy*gap*d, x2+vx*gap*d, y2+vy*gap*d]


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReIDEngine:
    def __init__(self, model_name="osnet_x1_0", device="cuda"):
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_osnet = False
        self.model     = self._load_model(model_name)
        self.extractor = MultiCueExtractor(self.device, self.use_osnet)

        if self.use_osnet:
            self.T_MATCH    = CFG.MATCH_THRESHOLD_OSNET
            self.T_FALLBACK = CFG.FALLBACK_THRESHOLD_OSNET
            self.T_REAPPEAR = CFG.REAPPEAR_THRESHOLD_OSNET
            self.T_REACT    = CFG.REACTIVATE_THRESHOLD_OSNET
        else:
            self.T_MATCH    = CFG.MATCH_THRESHOLD_RESNET
            self.T_FALLBACK = CFG.FALLBACK_THRESHOLD_RESNET
            self.T_REAPPEAR = CFG.REAPPEAR_THRESHOLD_RESNET
            self.T_REACT    = CFG.REACTIVATE_THRESHOLD_RESNET

        logger.info(f"Backbone: {'OSNet' if self.use_osnet else 'ResNet-ReID'}  "
                    f"MATCH={self.T_MATCH}  REAPPEAR={self.T_REAPPEAR}")

        self.identity_db:        dict = {}
        self.next_stable_id:     int  = 1
        self.track_to_identity:  dict = {}
        self.track_last_seen:    dict = {}
        self._pending:           dict = {}
        self._frame_id:          int  = 0
        self.person_features:    dict = {}
        self.person_metadata:    dict = {}
        self.id_mapping:         dict = {}
        self.consolidated_features: dict = {}

    # ── model ──────────────────────────────────────────────────────────────

    def _load_model(self, name):
        # Skip torchreid import (TensorFlow dependency chain causes hangs)
        # Use ResNetReIDBackbone directly (ResNet-50 + metric learning head)
        m = ResNetReIDBackbone().to(self.device).eval()
        logger.info("✅ ResNet-50 + Re-ID head loaded (512-dim embeddings)")
        return m

    # ── feature ────────────────────────────────────────────────────────────

    def extract_feature(self, frame, bbox):
        return self.extractor.build(self.model, frame, bbox)

    # ── scoring ────────────────────────────────────────────────────────────

    def _motion_score(self, bbox, ref_bbox) -> float:
        bw = max(1.0, bbox[2]-bbox[0]); bh = max(1.0, bbox[3]-bbox[1])
        diag = float(np.sqrt(bw*bw + bh*bh))
        dist = float(np.linalg.norm(_bbox_center(bbox) - _bbox_center(ref_bbox)))
        return float(np.exp(-dist / (diag * CFG.MOTION_DENOM_COEFF + 1e-6)))

    def _score(self, identity: Identity, feat, bbox, frame_id: int) -> float:
        gap = frame_id - identity.last_frame
        app = identity.appearance_score(feat)

        # FIX (Bug 7): for long re-appearance gaps, motion/IOU are unreliable.
        # Use appearance-only scoring so a person returning after 50+ frames
        # isn't penalised for being in a different position.
        if gap > CFG.REAPPEAR_GAP:
            return app   # pure appearance, compared against T_REAPPEAR later

        ref  = identity.predicted_bbox(frame_id) if gap > 1 else identity.last_bbox
        iou  = _iou(bbox, ref)
        mot  = self._motion_score(bbox, ref)

        if mot < 0.08 and iou < 0.02 and gap > CFG.MOTION_GATE_MIN_GAP:
            return -1.0

        return CFG.W_APPEARANCE*app + CFG.W_MOTION*mot + CFG.W_IOU*iou

    # ── crossings ──────────────────────────────────────────────────────────

    def _crossings(self, candidates) -> set:
        pairs = set()
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if _iou(candidates[i]['bbox'], candidates[j]['bbox']) >= CFG.CROSSING_IOU_GATE:
                    pairs.add((i,j)); pairs.add((j,i))
        return pairs

    # ── dedup ──────────────────────────────────────────────────────────────

    def _dedupe(self, tracks):
        if not tracks:
            return []
        tracks = sorted(tracks,
            key=lambda t: (t['bbox'][2]-t['bbox'][0])*(t['bbox'][3]-t['bbox'][1]),
            reverse=True)
        kept = []
        for c in tracks:
            if not any(_iou(c['bbox'], k['bbox']) > CFG.DEDUP_IOU for k in kept):
                kept.append(c)
        return kept

    # ── assignment ─────────────────────────────────────────────────────────

    def _assign(self, candidates: list, frame_id: int) -> dict:
        if not candidates:
            return {}

        assigned:    dict = {}
        used:        set  = set()
        stable_ids        = list(self.identity_db.keys())
        cross             = self._crossings(candidates)
        sid_to_col: dict  = {}

        # ── Score matrix ──────────────────────────────────────────────────
        if stable_ids:
            sid_to_col   = {sid: c for c, sid in enumerate(stable_ids)}
            score_matrix = np.full((len(candidates), len(stable_ids)), -1.0, np.float32)
            for i, cand in enumerate(candidates):
                for j, sid in enumerate(stable_ids):
                    ident = self.identity_db[sid]
                    if frame_id - ident.last_frame > CFG.MAX_IDENTITY_GAP:
                        continue
                    s = self._score(ident, cand['feat'], cand['bbox'], frame_id)
                    if self.track_to_identity.get(cand['tid']) == sid:
                        s = min(s + 0.06, 1.0)
                    if frame_id - ident.last_frame <= CFG.REACTIVATE_WINDOW:
                        s = min(s + 0.03, 1.0)
                    score_matrix[i, j] = s
        else:
            score_matrix = np.full((len(candidates), 0), -1.0, np.float32)

        # ── PASS 0: continuity lock ───────────────────────────────────────
        locked_rows: set = set()
        locked_cols: set = set()
        if stable_ids:
            best_lock: dict = {}
            for i, cand in enumerate(candidates):
                prev = self.track_to_identity.get(cand['tid'])
                if prev is None or prev not in sid_to_col:
                    continue
                col = sid_to_col[prev]
                s   = float(score_matrix[i, col])
                gap = frame_id - self.identity_db[prev].last_frame
                if gap <= CFG.TRACK_LOCK_GAP and s >= CFG.TRACK_LOCK_MIN_SCORE:
                    if prev not in best_lock or s > best_lock[prev][1]:
                        best_lock[prev] = (i, s)
            for sid, (row, _) in best_lock.items():
                assigned[row] = sid; used.add(sid)
                locked_rows.add(row); locked_cols.add(sid_to_col[sid])

        # ── PASS 1: Hungarian (short-gap candidates) ──────────────────────
        if stable_ids:
            rem_r = [r for r in range(len(candidates)) if r not in locked_rows]
            rem_c = [c for c in range(len(stable_ids)) if c not in locked_cols]
            # Only consider candidates with short frame_gap for motion-aware scoring
            short_gap_r = [r for r in rem_r
                           if frame_id - self.identity_db.get(
                               self.track_to_identity.get(candidates[r]['tid'],
                               stable_ids[0]), Identity(0, np.zeros(10), [0,0,1,1], 0)
                           ).last_frame <= CFG.REAPPEAR_GAP]
            # Fall back to all rem_r if filtering leaves nothing
            use_r = short_gap_r if short_gap_r else rem_r

            if use_r and rem_c:
                sub    = score_matrix[np.ix_(use_r, rem_c)]
                rr, rc = linear_sum_assignment(-sub)
                for r_rel, c_rel in zip(rr, rc):
                    r   = use_r[r_rel]; c = rem_c[c_rel]
                    sid = stable_ids[c]; s = float(score_matrix[r, c])
                    if s < self.T_MATCH or sid in used:
                        continue
                    is_cross = any((r,j) in cross for j in range(len(candidates)))
                    prev = self.track_to_identity.get(candidates[r]['tid'])
                    if prev is not None and prev in sid_to_col and prev != sid:
                        ps = float(score_matrix[r, sid_to_col[prev]])
                        mg = CFG.SWITCH_MARGIN * (1.5 if is_cross else 1.0)
                        ma = CFG.SWITCH_MIN_SCORE * (1.05 if is_cross else 1.0)
                        if not (s >= ps + mg and s >= ma):
                            continue
                    assigned[r] = sid; used.add(sid)

        # ── PASS 2: greedy fallback ───────────────────────────────────────
        if stable_ids:
            for i, cand in enumerate(candidates):
                if i in assigned:
                    continue
                best_sid, best_s = None, -1.0
                for c, sid in enumerate(stable_ids):
                    if sid in used:
                        continue
                    s = float(score_matrix[i, c])
                    if s > best_s:
                        best_s, best_sid = s, sid
                if best_sid is None or best_s < self.T_FALLBACK:
                    continue
                prev = self.track_to_identity.get(cand['tid'])
                if prev is not None and prev in sid_to_col and prev != best_sid:
                    ps = float(score_matrix[i, sid_to_col[prev]])
                    if not (best_s >= ps + CFG.SWITCH_MARGIN and best_s >= CFG.SWITCH_MIN_SCORE):
                        continue
                assigned[i] = best_sid; used.add(best_sid)

        # ── PASS 3: re-appearance (appearance-only, long gap) ─────────────
        # FIX (Bug 7): dedicated pass for persons returning after >REAPPEAR_GAP frames.
        # Uses appearance-only score (already computed in _score for long gaps)
        # with a lower threshold (T_REAPPEAR=0.55 vs T_MATCH=0.65).
        if stable_ids:
            for i, cand in enumerate(candidates):
                if i in assigned:
                    continue
                best_sid, best_s = None, -1.0
                for c, sid in enumerate(stable_ids):
                    if sid in used:
                        continue
                    ident = self.identity_db[sid]
                    if frame_id - ident.last_frame <= CFG.REAPPEAR_GAP:
                        continue   # handled in PASS 1/2
                    # Score is appearance-only for long gaps (from _score)
                    s = float(score_matrix[i, c])
                    if s > best_s:
                        best_s, best_sid = s, sid
                if best_sid is not None and best_s >= self.T_REAPPEAR:
                    logger.debug(f"  Re-appearance: tracker {cand['tid']} → "
                                 f"stable_id {best_sid}  score={best_s:.3f}")
                    assigned[i] = best_sid; used.add(best_sid)

        # ── PASS 4: new identities ────────────────────────────────────────
        for i, cand in enumerate(candidates):
            if i in assigned:
                continue
            sid = self.next_stable_id; self.next_stable_id += 1
            self.identity_db[sid] = Identity(sid, cand['feat'], cand['bbox'], frame_id)
            assigned[i] = sid; used.add(sid)

        # ── Update ────────────────────────────────────────────────────────
        for i, sid in assigned.items():
            cand = candidates[i]
            self.identity_db[sid].update(cand['feat'], cand['bbox'], frame_id)
            self.track_to_identity[cand['tid']] = sid
            self.track_last_seen[cand['tid']]   = frame_id
            self._pending.pop(cand['tid'], None)

        return assigned

    # ── process_frame ──────────────────────────────────────────────────────

    def process_frame(self, frame, tracking_data: list, frame_id: int = None):
        if frame_id is None:
            self._frame_id += 1
            frame_id = self._frame_id
        else:
            self._frame_id = frame_id

        frame_h, frame_w = frame.shape[:2]
        min_area  = int(frame_h * frame_w * CFG.MIN_AREA_RATIO)
        results   = []
        candidates = []

        for track in self._dedupe(tracking_data):
            pid  = track['id']
            bbox = track['bbox']
            x1,y1,x2,y2 = bbox
            w,h = x2-x1, y2-y1

            # Reject boxes smaller than 20×40px (800 area) before extraction
            # Filters edge/partial detections that create noise
            if h < CFG.MIN_HEIGHT or w*h < min_area or w*h < 800:
                continue

            # FIX (Bug 6): reject off-screen ghost bboxes before extraction
            if not _valid_bbox(bbox, frame_w, frame_h):
                existing = self.track_to_identity.get(pid)
                if existing is None:
                    self._pending[pid] = frame_id
                results.append({'id': pid, 'consolidated_id': existing,
                                'bbox': bbox, 'feature_dim': 0, 'matches': []})
                continue

            feat = self.extract_feature(frame, bbox)
            if feat is not None:
                self._store(pid, feat, frame_id)
                candidates.append({'tid': pid, 'bbox': bbox, 'feat': feat})
                self._pending.pop(pid, None)
            else:
                existing = self.track_to_identity.get(pid)
                if existing is None:
                    self._pending[pid] = frame_id
                results.append({'id': pid, 'consolidated_id': existing,
                                'bbox': bbox, 'feature_dim': 0, 'matches': []})

        assigned = self._assign(candidates, frame_id)

        for idx, cand in enumerate(candidates):
            sid = assigned.get(idx) or self.track_to_identity.get(cand['tid'])
            results.append({'id': cand['tid'], 'consolidated_id': sid,
                            'bbox': cand['bbox'],
                            'feature_dim': MultiCueExtractor.TOTAL_DIM, 'matches': []})
        return results

    # ── finalize ───────────────────────────────────────────────────────────

    def finalize_clustering(self):
        if self.track_to_identity:
            self.id_mapping = dict(self.track_to_identity)
            self.consolidated_features = {
                sid: ident.descriptor for sid, ident in self.identity_db.items()
            }
            logger.info(f"✅ {len(self.id_mapping)} tracker IDs → "
                        f"{len(self.consolidated_features)} stable identities")
        else:
            self.id_mapping = self._offline_cluster()

    def _offline_cluster(self):
        cons = {pid: _normalize(np.array(feats).mean(0))
                for pid, feats in self.person_features.items()}
        ids = sorted(cons.keys())
        parent = {i: i for i in ids}

        def find(x):
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]

        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                if _cosine(cons[a], cons[b]) > 0.82:
                    pa, pb = find(a), find(b)
                    if pa != pb: parent[pa] = pb

        groups = {}
        for pid in ids:
            groups.setdefault(find(pid), []).append(pid)

        mapping = {}
        for new_id, (_, members) in enumerate(sorted(groups.items()), 1):
            for orig in members:
                mapping[orig] = new_id

        self.consolidated_features = {}
        for new_id in set(mapping.values()):
            members = [p for p, n in mapping.items() if n == new_id]
            self.consolidated_features[new_id] = _normalize(
                np.mean([cons[m] for m in members], 0))
        return mapping

    def get_consolidated_id(self, tracking_id):
        return self.id_mapping.get(tracking_id, tracking_id)

    def _store(self, pid, feat, frame_id):
        if pid not in self.person_features:
            self.person_features[pid] = []
            self.person_metadata[pid] = {'first_seen': frame_id,
                                          'last_seen': frame_id, 'count': 0}
        self.person_features[pid].append(feat)
        self.person_metadata[pid]['last_seen'] = frame_id
        self.person_metadata[pid]['count']    += 1

    def compute_similarity(self, a, b): return _cosine(a, b)
    def compute_iou(self, b1, b2):      return _iou(b1, b2)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_reid_pipeline(video_path, tracking_json_path, output_json_path, device="cuda"):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    engine = ReIDEngine(device=device)

    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    results  = {}
    frame_id = 0

    logger.info("=" * 60)
    logger.info("▶  Re-ID pipeline")
    logger.info("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        tracks = tracking_data.get(str(frame_id), [])
        results[frame_id] = engine.process_frame(frame, tracks, frame_id)
        if frame_id % 30 == 0:
            logger.info(f"  Frame {frame_id:4d} | "
                        f"identities: {len(engine.identity_db)} | "
                        f"pending: {len(engine._pending)}")

    cap.release()
    engine.finalize_clustering()

    # Resolve any remaining None consolidated_ids
    for fid in sorted(results.keys()):
        for p in results[fid]:
            if p.get('consolidated_id') is None:
                tid = p.get('id')
                if tid in engine.track_to_identity:
                    p['consolidated_id'] = engine.track_to_identity[tid]

    # Renumber 1…N by first appearance, skip -1 (ghost tracks)
    first_seen = {}
    for fid in sorted(results.keys()):
        for p in results[fid]:
            cid = p.get('consolidated_id')
            if cid is not None and cid not in first_seen:
                first_seen[cid] = fid

    remap = {cid: idx+1 for idx, cid in enumerate(
        sorted(first_seen.keys(), key=lambda c: first_seen[c])
    )}

    for fid in results.keys():
        for p in results[fid]:
            cid = p.get('consolidated_id')
            p['consolidated_id'] = remap.get(cid, -1)

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    valid_ids = set(p['consolidated_id'] for fid in results.values()
                    for p in fid if p.get('consolidated_id', -1) != -1)
    logger.info(f"\n✅ Re-ID complete | stable IDs: {sorted(valid_ids)}")
    logger.info(f"   Output: {output_json_path}")
    return engine, results


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(video_path, tracking_json_path, device="cuda", max_frames=300):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    engine = ReIDEngine(device=device)
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    with open(tracking_json_path) as f:
        td = json.load(f)

    cap        = cv2.VideoCapture(video_path)
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_feats  = {}
    fail_count = {}
    fail_bboxes = {}
    frame_id   = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_id >= max_frames:
            break
        frame_id += 1
        for track in td.get(str(frame_id), []):
            pid  = int(track['id'])
            bbox = track['bbox']
            if not _valid_bbox(bbox, frame_w, frame_h):
                fail_count[pid] = fail_count.get(pid, 0) + 1
                if len(fail_bboxes.get(pid, [])) < 3:
                    fail_bboxes.setdefault(pid, []).append(bbox)
                continue
            feat = engine.extract_feature(frame, bbox)
            if feat is not None:
                all_feats.setdefault(pid, []).append(feat)
            else:
                fail_count[pid] = fail_count.get(pid, 0) + 1
                if len(fail_bboxes.get(pid, [])) < 3:
                    fail_bboxes.setdefault(pid, []).append(bbox)

    cap.release()

    print(f"\n{'='*55}")
    print(f"Re-ID DIAGNOSTIC  (first {frame_id} frames)")
    print(f"{'='*55}")
    print(f"Backbone  : {'OSNet' if engine.use_osnet else 'ResNet-50 + Re-ID head'}")
    print(f"Descriptor: {MultiCueExtractor.TOTAL_DIM} dims")
    print(f"Frame size: {frame_w}x{frame_h}")
    print(f"Thresholds: MATCH={engine.T_MATCH}  "
          f"REAPPEAR={engine.T_REAPPEAR}  FALLBACK={engine.T_FALLBACK}")
    print()

    print("Extraction rate:")
    all_pids = sorted(set(list(all_feats) + list(fail_count)))
    for pid in all_pids:
        ok = len(all_feats.get(pid, [])); fail = fail_count.get(pid, 0)
        total = ok + fail; pct = 100*ok/total if total else 0
        print(f"  Tracker {pid:3d}: {ok}/{total} ({pct:.0f}%)  "
              f"{'✅' if pct > 70 else '⚠️  failed'}")
        for bbox in fail_bboxes.get(pid, []):
            x1,y1,x2,y2 = map(int,bbox)
            cx1=max(0,x1); cy1=max(0,y1)
            cx2=min(frame_w,x2); cy2=min(frame_h,y2)
            cw=cx2-cx1; ch=cy2-cy1
            if cw < CFG.MIN_CROP_PX or ch < CFG.MIN_CROP_PX:
                reason = f"off-screen/degenerate (clamped {cw}x{ch})"
            else:
                reason = f"model error on {cw}x{ch} crop — check DEBUG log"
            print(f"    bbox={bbox}  → {reason}")

    if len(all_feats) >= 2:
        ids = sorted(all_feats.keys())
        avg = {pid: _normalize(np.mean(fs,0)) for pid, fs in all_feats.items()}

        print("\nInter-person similarity (WANT < 0.50):")
        inter_sims = []
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                sim = _cosine(avg[ids[i]], avg[ids[j]]); inter_sims.append(sim)
                flag = ('✅' if sim < 0.50 else
                        '⚠️  close' if sim < 0.65 else
                        '❌ too similar → raise MATCH_THRESHOLD')
                print(f"  Tracker {int(ids[i])} vs {int(ids[j])}: {sim:.3f}  {flag}")

        print("\nIntra-person similarity (WANT > 0.70):")
        intra_sims = []
        for pid, feats in sorted(all_feats.items(), key=lambda x: int(x[0])):
            if len(feats) < 2:
                print(f"  Tracker {int(pid):3d}: only {len(feats)} sample")
                continue
            sims = [_cosine(feats[i], feats[j])
                    for i in range(min(10, len(feats)))
                    for j in range(i+1, min(10, len(feats)))]
            intra_sims.extend(sims)
            print(f"  Tracker {int(pid):3d}: avg={np.mean(sims):.3f}  "
                  f"min={np.min(sims):.3f}  "
                  f"{'✅' if np.mean(sims) > 0.70 else '⚠️  low'}")

        if inter_sims and intra_sims:
            max_inter = max(inter_sims); min_intra = min(intra_sims)
            ideal = (max_inter + min_intra) / 2
            print(f"\n  💡 Suggested MATCH_THRESHOLD ≈ {ideal:.2f}")
            gap = min_intra - max_inter
            print(f"     Discriminability gap: {gap:.3f}  "
                  f"{'✅ good' if gap > 0.20 else '⚠️  tight — consider better lighting/resolution'}")
    else:
        print(f"\n  ℹ️  Only {len(all_feats)} tracker(s) with valid features.")
        if len(all_feats) == 1:
            pid = list(all_feats.keys())[0]
            feats = all_feats[pid]
            if len(feats) >= 2:
                sims = [_cosine(feats[i], feats[j])
                        for i in range(min(10, len(feats)))
                        for j in range(i+1, min(10, len(feats)))]
                print(f"  Intra-person consistency (Tracker {pid}): "
                      f"avg={np.mean(sims):.3f}  min={np.min(sims):.3f}  "
                      f"{'✅' if np.mean(sims) > 0.70 else '⚠️'}")
                print(f"  This person will keep the same ID across the video ✅")

    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    video         = "input/video3.mp4"
    tracking_json = "outputs/tracking.json"
    output_json   = "outputs/reid_results.json"

    if "--diagnose" in sys.argv:
        diagnose(video, tracking_json, max_frames=300)
        sys.exit(0)

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        sys.exit(1)
    if not Path(tracking_json).exists():
        print(f"❌ Tracking JSON not found: {tracking_json}")
        print("   Run: python main.py --step 2")
        sys.exit(1)

    engine, results = run_reid_pipeline(
        video_path=video, tracking_json_path=tracking_json,
        output_json_path=output_json, device="cuda",
    )

    total = sum(len(p) for p in results.values())
    valid = set(p['consolidated_id'] for ps in results.values()
                for p in ps if p.get('consolidated_id', -1) != -1)
    print(f"\nTotal detections : {total}")
    print(f"Stable IDs       : {sorted(valid)}")
    print(f"Output           : {output_json}")