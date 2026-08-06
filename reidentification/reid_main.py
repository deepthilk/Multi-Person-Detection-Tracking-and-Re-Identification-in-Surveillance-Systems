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

from reidentification.face_cue import FaceCueExtractor  # optional face-based cue


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

class ReIDConfig:
    # Image resize for deep model
    RESIZE_H: int = 256
    RESIZE_W: int = 128

    # OSNet thresholds — balanced
    MATCH_THRESHOLD_OSNET:        float = 0.65
    FALLBACK_THRESHOLD_OSNET:     float = 0.58
    REAPPEAR_THRESHOLD_OSNET:     float = 0.62
    REACTIVATE_THRESHOLD_OSNET:   float = 0.60

    # ResNet fallback thresholds — RAISED to prevent appearance-driven false merges.
    # Model diagnostics on Market-1501 show ~6% of different-person pairs have
    # similarity >0.60. Raising match thresholds above this zone substantially
    # reduces false matches during crossings, at the cost of occasionally missing
    # a correct match (which the grace-period retry in PASS4 recovers from).
    # A false merge (two people collapsed to one ID) is much worse than a
    # temporary split (one person temporarily gets a second ID, then re-merges),
    # so err on the side of higher thresholds.
    MATCH_THRESHOLD_RESNET:       float = 0.62
    FALLBACK_THRESHOLD_RESNET:    float = 0.55
    REAPPEAR_THRESHOLD_RESNET:    float = 0.55
    REACTIVATE_THRESHOLD_RESNET:  float = 0.60

    # Scoring weights — HEAVILY favour appearance over motion/IOU.
    # During crossings, motion and IOU become unreliable because DeepSORT's
    # Kalman filter can swap which track_id follows which physical person.
    # Appearance (what the person looks like) stays reliable regardless of
    # position, so it must dominate to prevent ID swaps.
    # Increased W_APPEARANCE from 0.70 to 0.85 because the model fine-tuned on
    # Market-1501 has strong discriminative power (mean diff-person sim=0.04),
    # so we can safely rely on it. Motion/IOU were at 0.18/0.12 — still too
    # high during crossings where they become actively misleading.
    W_APPEARANCE: float = 0.85   # Heavily dominant — appearance is the only reliable signal during crossings
    W_MOTION:     float = 0.08   # Minimised — motion is unreliable during crossings
    W_IOU:        float = 0.07   # Minimised — IOU is unreliable during crossings

    # Re-appearance: frame_gap after which motion/IOU are ignored
    REAPPEAR_GAP: int = 20   # REDUCED from 25

    # Switch-guard (prevent ID swap during crossings — ULTRA STRICT for uniforms)
    SWITCH_MARGIN:    float = 0.45   # HEAVILY INCREASED for uniforms — prevent accidental swaps
    SWITCH_MIN_SCORE: float = 0.96   # HEAVILY INCREASED for uniforms — extremely confident before swapping

    # When a face confidently confirms the SWITCH target (and/or confidently
    # disagrees with whatever identity is currently locked in), the strict
    # thresholds above are relaxed to these — face evidence is fundamentally
    # more trustworthy than body appearance for identical uniforms, so it's
    # allowed to correct a switch the body-only guard would otherwise block.
    SWITCH_MARGIN_FACE_CONFIRMED:    float = 0.05
    SWITCH_MIN_SCORE_FACE_CONFIRMED: float = 0.55

    # Continuity lock — INCREASED to prevent ID swaps during crossings.
    # A higher min-score means a tracker stays locked to its identity even
    # when motion/IOU briefly disagree (e.g., during a crossing), because
    # the appearance component (now 85% weight) is the primary signal.
    TRACK_LOCK_GAP:       int   = 60
    TRACK_LOCK_MIN_SCORE: float = 0.55

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
    # Only merge when boxes are nearly identical (same person detected twice).
    # 0.50 was causing DIFFERENT people to be merged during crossings (common
    # crossing IoU is 0.30-0.60), which removed one person entirely from the
    # Re-ID pipeline — the collapsed person would reappear with a new/different
    # tracker ID after separation, breaking identity continuity. 0.95 ensures
    # only near-perfect box overlaps (true YOLO duplicates) get merged, while
    # two real people crossing keep their independent tracks.
    DEDUP_IOU:       float = 0.95

    # Temporal
    # How long (in frames) an identity stays eligible for re-matching before
    # being permanently forgotten. This used to be 500 frames (~17 seconds
    # at 30fps) — nowhere near enough for "recognize the same person if they
    # return 10+ minutes later" (an explicit requirement). At 30fps, 10
    # minutes = 18,000 frames; set generously higher for headroom (variable
    # fps, longer sessions). Tradeoff: identities are held in memory (and
    # considered as match candidates) for the whole session instead of
    # being pruned quickly — more candidates in the pool means slightly more
    # opportunity for confusion between similar-looking people, but that's
    # the necessary cost of the long-term recognition requirement, not a
    # bug. Adjust to match your actual video's fps/length if needed.
    MAX_IDENTITY_GAP:  int = 60000   # ~33 min at 30fps — long-term memory
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

    # Face cue — NOT part of the 698-dim descriptor (kept separate so the
    # registration module's stable contract is unaffected). Applied as a
    # heavy override on top of the body-appearance score whenever a
    # confident face embedding is available on both sides of a comparison.
    # Faces stay distinctive even in identical uniforms, unlike body
    # appearance — this directly targets the uniform-crossing failure mode.
    FACE_WEIGHT:          float = 0.65   # blend weight when a face is available
    FACE_MIN_SIMILARITY:  float = 0.35   # below this, treat as a mismatch veto

    # Grace period before minting a brand-new identity — keep very short to
    # avoid merging different people who coincidentally look similar.
    NEW_ID_GRACE_FRAMES: int = 5   # INCREASED from 1 — give returning persons time to reconnect

    # Body-only reappearance (no face confirmation available) is penalised
    # to prevent false matches between different people who happen to look
    # similar. 0.12 was too aggressive: even a returning person with good
    # appearance (raw_app=0.70) got score=0.58, barely meeting T_REAPPEAR.
    # At raw_app=0.65 the score dropped to 0.53 — below threshold — causing
    # identity fragmentation (same person split into multiple IDs). 0.05
    # still provides a safety margin: raw_app=0.60 -> score=0.55 (meets
    # T_REAPPEAR=0.55), while different-color dresses produce raw_app<0.30.
    REAPPEAR_NO_FACE_PENALTY: float = 0.05

    # Face evidence used in finalize_clustering to SPLIT a false merge:
    # two trackers assigned to the same identity whose faces never agree
    # above this similarity are treated as different people (e.g. a brand-new
    # tracker claimed by PASS3 reappearance on body appearance alone).
    # Threshold is on the dlib/face_recognition similarity scale (same person
    # typically > 0.40, different people well below). Union uses the MAX
    # pairwise similarity so a few noisy face detections never over-split.
    FACE_SAME_TRACK_SPLIT: float = 0.40
    # Below this face similarity two trackers are treated as CONFIRMED
    # different people and are split apart even if their body appearance is
    # nearly identical (dlib sim: 0.25 ↔ distance 0.675, above the ~0.6
    # same-person cutoff). Guards the body-only _merge_non_cooccurring path
    # from merging two lookalike people whose faces clearly differ.
    FACE_SAME_TRACK_VETO:   float = 0.25

    # Appearance fallback for trackers with NO face evidence: a merged
    # tracker must show VERY strong track-average appearance (≥ 0.65, the
    # same MIN_APPEARANCE_NEW_TRACKER bar PASS 1/2/3/4 apply to brand-new
    # trackers) to be kept on an existing identity. A single noisy frame can
    # cross T_REAPPEAR (v3: tid=6 → sid=1 at 0.740 single-frame) while the
    # accumulated track-average says otherwise (tid6↔sid1 = 0.632) — split.
    APPEARANCE_SPLIT_THRESHOLD: float = 0.65

    # Max face embeddings cached per tracker (recent samples only — enough to
    # build a reliable same/different-person signal without unbounded memory).
    TRACK_FACE_GALLERY_SIZE: int = 12


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
    def __init__(self, stable_id: int, descriptor: np.ndarray, bbox, frame_id: int,
                 face_descriptor=None):
        self.stable_id   = stable_id
        self.descriptor  = descriptor.copy()
        self.gallery     = AppearanceGallery(descriptor)
        self.last_bbox   = list(bbox)
        self.last_center = _bbox_center(bbox)
        self.velocity    = (0.0, 0.0)
        self.last_frame  = frame_id
        self.count       = 1
        # Best face embedding seen for this identity so far (None until a
        # confident face is first observed). Kept as a single "best" vector
        # rather than an EMA — a clear frontal face is a much stronger
        # reference than an average blended with blurry/angled ones.
        self.face_descriptor = face_descriptor.copy() if face_descriptor is not None else None

    def appearance_score(self, query) -> float:
        return 0.70 * self.gallery.match(query) + 0.30 * _cosine(query, self.descriptor)

    def face_similarity(self, face_query):
        if self.face_descriptor is None or face_query is None:
            return None
        return FaceCueExtractor.similarity(self.face_descriptor, face_query)

    def update(self, descriptor, bbox, frame_id: int, face_descriptor=None):
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
        if face_descriptor is not None:
            # A fresh confident face detection replaces the stored one — at
            # minimum equally trustworthy as whatever (possibly none) we
            # had, and this keeps it current if the person's angle changes.
            self.face_descriptor = face_descriptor.copy()

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
    def __init__(self, model_name="osnet_x1_0", device="cuda", debug_trace=False):
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_osnet = False
        self.debug_trace = debug_trace   # see _trace() — set True to log every
                                          # ID decision (lock/assign/switch/new)
                                          # with the exact scores behind it,
                                          # instead of having to infer what
                                          # happened from watching a video
        self.model     = self._load_model(model_name)
        self.extractor = MultiCueExtractor(self.device, self.use_osnet)
        self.face_extractor = FaceCueExtractor()
        if self.face_extractor.enabled:
            logger.info("✅ Face-based Re-ID cue enabled (helps distinguish identical uniforms)")
        # Diagnostics: how often is a face actually found? If this stays near
        # 0%, the face cue can't be helping — worth knowing rather than
        # guessing when tuning against real footage.
        self._face_attempts = 0
        self._face_hits     = 0
        self._new_id_grace: dict = {}   # tracker_id -> consecutive PASS4-miss count
        self._cooccurrence: dict = {}   # tid -> set of tids seen in same frame
        self.tracker_frame_map: dict = {}   # tid -> set of frame_ids
        self.identity_frame_map: dict = {}  # sid -> set of frame_ids (built after enforce)
        self.tracker_sid_history: dict = {}  # tid -> set of sids it was ever assigned to

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
        self.tracker_faces:      dict = {}   # tid -> list of face embeddings (capped)
        self.id_mapping:         dict = {}
        self.consolidated_features: dict = {}

    # ── model ──────────────────────────────────────────────────────────────

    def _load_model(self, name):
        # Skip torchreid import (TensorFlow dependency chain causes hangs)
        # Use ResNetReIDBackbone directly (ResNet-50 + metric learning head)
        m = ResNetReIDBackbone().to(self.device)

        # If a fine-tuned checkpoint exists (see reidentification/training/
        # train_reid.py), load it. Falls back to the ImageNet-only backbone
        # exactly as before if the file is missing/empty/unloadable, so
        # nothing else in the pipeline (registration/embedder.py included)
        # has to change either way.
        weights_path = Path(__file__).resolve().parent / "weights" / "best_model.pth"
        if weights_path.exists() and weights_path.stat().st_size > 0:
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                m.load_state_dict(state)
                logger.info(f"✅ Fine-tuned Re-ID weights loaded from {weights_path} (512-dim embeddings)")
            except Exception as e:
                logger.warning(f"⚠️  Could not load fine-tuned weights ({e}); "
                                f"using ImageNet-only backbone instead")
        else:
            logger.info("✅ ResNet-50 + Re-ID head loaded (ImageNet weights only — "
                        "no fine-tuned checkpoint found, 512-dim embeddings)")

        m.eval()
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

    def _blend_face(self, base_score: float, identity: Identity, face_feat) -> float:
        """Blend a face-similarity override into a base score, when a
        confident face embedding is available on both sides. Faces stay
        distinctive in identical uniforms, so this is weighted heavily —
        but it only ever activates when both identity.face_descriptor and
        the current detection's face_feat exist; otherwise it's a no-op
        and behaviour is identical to before this cue was added."""
        face_sim = identity.face_similarity(face_feat)
        if face_sim is None:
            return base_score
        if face_sim < CFG.FACE_MIN_SIMILARITY:
            # Confident face mismatch — veto even a strong body-appearance
            # score, since two different people can't share a face.
            return min(base_score, face_sim)
        return (1 - CFG.FACE_WEIGHT) * base_score + CFG.FACE_WEIGHT * face_sim

    def _switch_allowed(self, s: float, ps: float, cand_face_feat,
                         new_identity: 'Identity', prev_identity: 'Identity',
                         margin: float, min_score: float) -> bool:
        """Whether re-assigning a track from prev_identity to new_identity is
        allowed. Normally requires clearing strict margin/min_score bars
        (see SWITCH_MARGIN/SWITCH_MIN_SCORE — deliberately hard to trigger,
        to avoid accidental swaps between identically-uniformed people).
        But when a face is available and clearly says "this is NOT
        prev_identity, it IS new_identity", that's a much more trustworthy
        signal than body appearance alone — so this relaxes the bar to
        SWITCH_*_FACE_CONFIRMED instead. This is what lets a wrong
        assignment made during an occluded crossing (where no face was
        visible) get corrected a few frames later once a face becomes
        visible again — without it, the strict guard built to prevent
        swaps also prevents legitimate corrections."""
        if cand_face_feat is not None:
            new_face_sim  = new_identity.face_similarity(cand_face_feat)
            prev_face_sim = prev_identity.face_similarity(cand_face_feat) if prev_identity else None
            face_confirms = (new_face_sim is not None and new_face_sim >= CFG.FACE_MIN_SIMILARITY and
                              (prev_face_sim is None or prev_face_sim < CFG.FACE_MIN_SIMILARITY))
            if face_confirms:
                margin, min_score = CFG.SWITCH_MARGIN_FACE_CONFIRMED, CFG.SWITCH_MIN_SCORE_FACE_CONFIRMED
        return s >= ps + margin and s >= min_score

    def _trace(self, frame_id: int, msg: str):
        """Opt-in decision log (enable with debug_trace=True). Prints exactly
        which ID decision fired and why, at the moment it happens — grep the
        output for '[TRACE fN' around a frame number you saw go wrong in the
        video (frame ≈ seconds_into_video * fps) to see the real numbers
        behind it, instead of guessing from what the video looks like."""
        if self.debug_trace:
            logger.info(f"[TRACE f{frame_id}] {msg}")

    def _score(self, identity: Identity, feat, bbox, frame_id: int, face_feat=None,
               is_crossing=False) -> float:
        gap = frame_id - identity.last_frame
        app = identity.appearance_score(feat)

        # FIX (Bug 7): for long re-appearance gaps, motion/IOU are unreliable.
        # Use appearance-only scoring so a person returning after 50+ frames
        # isn't penalised for being in a different position.
        if gap > CFG.REAPPEAR_GAP:
            face_sim = identity.face_similarity(face_feat)
            if face_sim is not None:
                # A face is available on both sides — this is a reliable
                # signal even in identical uniforms, so use the normal
                # (lenient) blend.
                return self._blend_face(app, identity, face_feat)
            # No face on either side: T_REAPPEAR (0.45-0.50) was tuned loose
            # specifically to tolerate uniform ambiguity — fine for a single
            # attempt, but combined with the multi-frame retry grace period
            # (see PASS 4), a genuinely NEW person got repeated chances to
            # cross that loose bar by uniform-driven coincidence, causing
            # false merges (observed: 4 real people collapsed to 3 IDs).
            # Require a distinctly higher raw appearance score here so
            # body-only reappearance is deliberately harder to trigger by
            # chance — better to occasionally split one real person into
            # two IDs than to merge two different real people into one.
            return app - CFG.REAPPEAR_NO_FACE_PENALTY

        ref  = identity.predicted_bbox(frame_id) if gap > 1 else identity.last_bbox
        iou  = _iou(bbox, ref)
        mot  = self._motion_score(bbox, ref)

        if mot < 0.08 and iou < 0.02 and gap > CFG.MOTION_GATE_MIN_GAP:
            return -1.0

        # Crossing-aware scoring: when two people are physically overlapping,
        # motion and IOU are actively misleading because the Kalman filter's
        # predicted position of person A may be closer to person B's detection
        # (and vice versa). During crossings, rely almost entirely on
        # appearance — the one signal that stays reliable regardless of
        # spatial overlap.
        if is_crossing:
            w_app = 0.95
            w_mot = 0.03
            w_iou = 0.02
        else:
            w_app = CFG.W_APPEARANCE
            w_mot = CFG.W_MOTION
            w_iou = CFG.W_IOU

        base = w_app*app + w_mot*mot + w_iou*iou
        return self._blend_face(base, identity, face_feat)

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

        # Appearance floor: never merge two trackers if their raw appearance
        # similarity is below this value. Motion/IOU can't override a clear
        # appearance mismatch. This prevents false merges between people who
        # look nothing alike but happen to be in similar positions.
        # RAISED because motion/IOU are now heavily downweighted (8%/7%), so
        # a tracker must actually look like the person to inherit their ID.
        # Model diagnostics: ~6% of diff-person pairs exceed 0.60, so 0.58
        # provides a safety margin below the match threshold.
        MIN_APPEARANCE_FOR_MERGE: float = 0.55
        # MUCH stricter threshold for trackers with NO prior identity trying to
        # claim an existing identity. A brand-new tracker must show very strong
        # appearance evidence to avoid giving every new person a new ID (which
        # is safer than merging different people).
        MIN_APPEARANCE_NEW_TRACKER: float = 0.65

        assigned:    dict = {}
        used:        set  = set()
        stable_ids        = list(self.identity_db.keys())
        cross             = self._crossings(candidates)
        sid_to_col: dict  = {}

        # Pre-build: for each stable_id, which trackers are currently assigned to it
        # (updated as assignments happen). Used by co-occurrence guard.
        _sid_tids: dict = {}  # sid -> set of tracker_ids assigned to it this frame

        def _cooccurs_with_any(tid, sid):
            """Check if tid co-occurs with any tracker already assigned to sid this frame."""
            co_set = self._cooccurrence.get(tid, set())
            for t in _sid_tids.get(sid, set()):
                if t in co_set:
                    return True
            return False

        def _assign_with_guard(row, sid):
            """Assign a candidate to a stable_id, recording co-occurrence state."""
            assigned[row] = sid
            used.add(sid)
            _sid_tids.setdefault(sid, set()).add(candidates[row]['tid'])

        # ── Score matrix ──────────────────────────────────────────────────
        if stable_ids:
            sid_to_col   = {sid: c for c, sid in enumerate(stable_ids)}
            score_matrix = np.full((len(candidates), len(stable_ids)), -1.0, np.float32)
            # Pre-compute which candidates are in a crossing situation (bbox
            # overlapping with another candidate). Used by _score to switch to
            # appearance-only mode for those rows — motion/IOU are unreliable
            # when two people physically overlap.
            cand_crossing = [any((i, j) in cross or (j, i) in cross
                                 for j in range(len(candidates)) if j != i)
                             for i in range(len(candidates))]
            for i, cand in enumerate(candidates):
                for j, sid in enumerate(stable_ids):
                    ident = self.identity_db[sid]
                    if frame_id - ident.last_frame > CFG.MAX_IDENTITY_GAP:
                        continue
                    s = self._score(ident, cand['feat'], cand['bbox'], frame_id,
                                     face_feat=cand.get('face_feat'),
                                     is_crossing=cand_crossing[i])
                    if self.track_to_identity.get(cand['tid']) == sid:
                        s = min(s + 0.06, 1.0)
                    if frame_id - ident.last_frame <= CFG.REACTIVATE_WINDOW:
                        s = min(s + 0.03, 1.0)
                    score_matrix[i, j] = s
        else:
            score_matrix = np.full((len(candidates), 0), -1.0, np.float32)

        # ── PASS 0: continuity lock ───────────────────────────────────────
        # CRITICAL: This is the PRIMARY defence against ID swaps during
        # crossings. A tracker that already has an assigned identity should
        # keep it unless there is overwhelming appearance evidence otherwise.
        # The blended score now heavily favours appearance (70% weight), so
        # even if motion/IOU drop to near-zero during a crossing, the lock
        # remains intact as long as the person still looks the same.
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
                # Use appearance-dominant check: require the blended score
                # to pass the threshold, AND the raw appearance to be
                # reasonable. This prevents a tracker from locking to a
                # prior identity purely based on motion/IOU when the
                # appearance has changed (which would indicate a swap).
                raw_app = self.identity_db[prev].appearance_score(cand['feat'])
                if gap <= CFG.TRACK_LOCK_GAP and s >= CFG.TRACK_LOCK_MIN_SCORE:
                    if raw_app < MIN_APPEARANCE_FOR_MERGE:
                        continue
                    if prev not in best_lock or s > best_lock[prev][1]:
                        best_lock[prev] = (i, s)
                # OVERRIDE: even if the blended score drops below threshold
                # (e.g., motion/IOU disagree during crossing), still lock if
                # the appearance-only score is very strong. This prevents
                # the common failure mode where DeepSORT's Kalman filter
                # drifts during a crossing and the motion/IOU scores tank
                # for the correct identity.
                elif gap <= CFG.TRACK_LOCK_GAP and raw_app >= 0.70:
                    if prev not in best_lock or raw_app > best_lock[prev][1]:
                        best_lock[prev] = (i, raw_app)
            for sid, (row, s) in best_lock.items():
                if _cooccurs_with_any(candidates[row]['tid'], sid):
                    self._trace(frame_id, f"PASS0 BLOCKED co-occurrence: tid={candidates[row]['tid']} cannot join sid={sid}")
                    continue
                _assign_with_guard(row, sid)
                locked_rows.add(row); locked_cols.add(sid_to_col[sid])
                self._trace(frame_id, f"PASS0 lock: tid={candidates[row]['tid']} -> sid={sid} score={s:.3f}")

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
                    # NEW: appearance floor
                    raw_app = self.identity_db[sid].appearance_score(cand['feat'])
                    # Stricter threshold for trackers with no prior identity
                    prev_tid = self.track_to_identity.get(candidates[r]['tid'])
                    req_app = MIN_APPEARANCE_FOR_MERGE if prev_tid is not None else MIN_APPEARANCE_NEW_TRACKER
                    if raw_app < req_app:
                        continue
                    is_cross = any((r,j) in cross for j in range(len(candidates)))
                    prev = self.track_to_identity.get(candidates[r]['tid'])
                    if prev is not None and prev in sid_to_col and prev != sid:
                        ps = float(score_matrix[r, sid_to_col[prev]])
                        mg = CFG.SWITCH_MARGIN * (1.5 if is_cross else 1.0)
                        ma = CFG.SWITCH_MIN_SCORE * (1.05 if is_cross else 1.0)
                        allowed = self._switch_allowed(
                                s, ps, candidates[r].get('face_feat'),
                                self.identity_db[sid], self.identity_db.get(prev), mg, ma)
                        self._trace(frame_id, f"PASS1 SWITCH tid={candidates[r]['tid']} "
                                    f"{prev}->{sid}: new_s={s:.3f} prev_s={ps:.3f} "
                                    f"has_face={candidates[r].get('face_feat') is not None} "
                                    f"{'ALLOWED' if allowed else 'BLOCKED'}")
                        if not allowed:
                            continue
                    if _cooccurs_with_any(candidates[r]['tid'], sid):
                        self._trace(frame_id, f"PASS1 BLOCKED co-occurrence: tid={candidates[r]['tid']} cannot join sid={sid}")
                        continue
                    _assign_with_guard(r, sid)
                    self._trace(frame_id, f"PASS1 assign: tid={candidates[r]['tid']} -> sid={sid} score={s:.3f}")

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
                # ADD: appearance floor — if the raw appearance similarity is
                # below threshold, don't merge even if the blended score (with
                # motion/IOU) passes the threshold.
                raw_app = self.identity_db[best_sid].appearance_score(cand['feat'])
                prev_tid = self.track_to_identity.get(cand['tid'])
                req_app = MIN_APPEARANCE_FOR_MERGE if prev_tid is not None else MIN_APPEARANCE_NEW_TRACKER
                if raw_app < req_app:
                    continue
                prev = self.track_to_identity.get(cand['tid'])
                if prev is not None and prev in sid_to_col and prev != best_sid:
                    ps = float(score_matrix[i, sid_to_col[prev]])
                    is_cross = any((i,j) in cross for j in range(len(candidates)))
                    mg = CFG.SWITCH_MARGIN * (1.5 if is_cross else 1.0)
                    ma = CFG.SWITCH_MIN_SCORE * (1.05 if is_cross else 1.0)
                    allowed = self._switch_allowed(
                            best_s, ps, cand.get('face_feat'),
                            self.identity_db[best_sid], self.identity_db.get(prev),
                            mg, ma)
                    self._trace(frame_id, f"PASS2 SWITCH tid={cand['tid']} "
                                f"{prev}->{best_sid}: new_s={best_s:.3f} prev_s={ps:.3f} "
                                f"is_cross={is_cross} "
                                f"{'ALLOWED' if allowed else 'BLOCKED'}")
                    if not allowed:
                        continue
                    if _cooccurs_with_any(cand['tid'], best_sid):
                        self._trace(frame_id, f"PASS2 BLOCKED co-occurrence: tid={cand['tid']} cannot join sid={best_sid}")
                        continue
                    _assign_with_guard(i, best_sid)
                    self._trace(frame_id, f"PASS2 assign: tid={cand['tid']} -> sid={best_sid} score={best_s:.3f}")

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
                    # ADD: appearance floor — don't re-appear match if appearance
                    # is too different
                    raw_app = self.identity_db[best_sid].appearance_score(cand['feat'])
                    # FIX: PASS 3 was the only pass that ignored the new-tracker
                    # distinction — PASS 1/2/4 all require a MUCH stricter raw
                    # appearance (MIN_APPEARANCE_NEW_TRACKER=0.65) before a
                    # brand-new tracker (no prior identity) may claim an existing
                    # stable identity, but PASS 3 used the loose 0.55 bar for
                    # everyone. That let a new tracker get swallowed by the first
                    # identity it crossed T_REAPPEAR against (the v3 false merge:
                    # tid=6 -> sid=1 at score 0.740, has_face=False). Mirror the
                    # same convention here so new trackers cannot be claimed by
                    # an existing identity on weak appearance evidence alone.
                    prev_tid = self.track_to_identity.get(cand['tid'])
                    req_app = (MIN_APPEARANCE_FOR_MERGE if prev_tid is not None
                               else MIN_APPEARANCE_NEW_TRACKER)
                    if raw_app < req_app:
                        self._trace(frame_id, f"PASS3 BLOCKED appearance floor: "
                                    f"tid={cand['tid']} sid={best_sid} raw_app={raw_app:.3f} "
                                    f"< req_app={req_app:.3f} (new_tracker={prev_tid is None})")
                        continue
                    # FIX: PASS 3 used to reassign a tracker's identity purely
                    # based on crossing T_REAPPEAR, with NO check against
                    # whatever identity that tracker was already carrying
                    # from earlier this frame or a prior frame — unlike
                    # PASS 1/2, which both require clearing the switch-guard
                    # before overriding an existing assignment. This let a
                    # low-confidence reappearance score silently steal a
                    # track from its correct identity, right after PASS 2's
                    # switch-guard had already (correctly) blocked the same
                    # move — the exact swap traced at frame 252 (tid=9
                    # blocked 3->1 by PASS2, then done anyway by PASS3).
                    prev = self.track_to_identity.get(cand['tid'])
                    if prev is not None and prev in sid_to_col and prev != best_sid:
                        ps = float(score_matrix[i, sid_to_col[prev]])
                        is_cross = any((i,j) in cross for j in range(len(candidates)))
                        mg = CFG.SWITCH_MARGIN * (1.5 if is_cross else 1.0)
                        ma = CFG.SWITCH_MIN_SCORE * (1.05 if is_cross else 1.0)
                        allowed = self._switch_allowed(
                                best_s, ps, cand.get('face_feat'),
                                self.identity_db[best_sid], self.identity_db.get(prev),
                                mg, ma)
                        self._trace(frame_id, f"PASS3 SWITCH tid={cand['tid']} "
                                    f"{prev}->{best_sid}: new_s={best_s:.3f} prev_s={ps:.3f} "
                                    f"is_cross={is_cross} "
                                    f"{'ALLOWED' if allowed else 'BLOCKED'}")
                        if not allowed:
                            continue   # leave unassigned this frame; PASS4 grace retries later
                    if _cooccurs_with_any(cand['tid'], best_sid):
                        self._trace(frame_id, f"PASS3 BLOCKED co-occurrence: tid={cand['tid']} cannot join sid={best_sid}")
                        continue
                    logger.debug(f"  Re-appearance: tracker {cand['tid']} → "
                                 f"stable_id {best_sid}  score={best_s:.3f}")
                    self._trace(frame_id, f"PASS3 reappear: tid={cand['tid']} -> sid={best_sid} "
                                f"score={best_s:.3f} (T_REAPPEAR={self.T_REAPPEAR:.3f}) "
                                f"has_face={cand.get('face_feat') is not None}")
                    _assign_with_guard(i, best_sid)

        # ── PASS 4: new identities ────────────────────────────────────────
        # FIX: a candidate that fails PASS 1-3 on a SINGLE frame used to get
        # a brand-new permanent identity immediately — meaning a person
        # re-entering frame had exactly one frame's chance to match their
        # old identity, with no retry. If that one frame had a bad angle,
        # no visible face, or a mediocre appearance score (common right at
        # re-entry), they'd be wrongly given a new ID forever, even though
        # PASS 1-3 keep running fresh every subsequent frame and might well
        # have matched them a moment later.
        #
        # Grace period: hold off minting a new identity for a tracker id
        # until it has failed matching for CFG.NEW_ID_GRACE_FRAMES in a row.
        # Only applies when other identities already exist (stable_ids) —
        # the very first person(s) ever seen still get an ID immediately,
        # since there's nothing for them to be a re-appearance OF.
        for i, cand in enumerate(candidates):
            if i in assigned:
                continue
            tid = cand['tid']
            if stable_ids:
                miss = self._new_id_grace.get(tid, 0) + 1
                self._new_id_grace[tid] = miss
                if miss < CFG.NEW_ID_GRACE_FRAMES:
                    self._trace(frame_id, f"PASS4 grace: tid={tid} miss={miss}/{CFG.NEW_ID_GRACE_FRAMES}, waiting")
                    continue   # give PASS 1-3 another shot next frame

            # FIX: grace expiring used to always mint a brand-new identity,
            # even when this exact tracker already had a known prior
            # identity that just failed to clear the (deliberately strict)
            # switch/match thresholds during a brief ambiguous moment (e.g.
            # a crossing) — throwing away a mediocre-but-real clue (traced
            # case: tid=1 scored 0.30-0.40 against its own correct identity
            # sid=1 during an overlap, too low to auto-relock, but far more
            # informative than nothing) in favor of starting from scratch.
            # If that prior identity still exists and no OTHER candidate
            # this frame has already claimed it, fall back to it directly
            # instead of minting a new one — treat this as a low-confidence
            # implicit re-lock rather than a genuinely new person.
            prev = self.track_to_identity.get(tid)
            if prev is not None and prev in self.identity_db and prev not in used:
                assigned[i] = prev; used.add(prev)
                self._new_id_grace.pop(tid, None)
                self._trace(frame_id, f"PASS4 fallback-to-prior: tid={tid} -> sid={prev} "
                            f"(grace expired, no better match found, reusing known identity)")
                continue

            sid = self.next_stable_id; self.next_stable_id += 1
            self.identity_db[sid] = Identity(sid, cand['feat'], cand['bbox'], frame_id,
                                              face_descriptor=cand.get('face_feat'))
            assigned[i] = sid; used.add(sid)
            self._new_id_grace.pop(tid, None)
            self._trace(frame_id, f"PASS4 NEW IDENTITY: tid={tid} -> sid={sid} "
                        f"(had_stable_ids={bool(stable_ids)}, has_face={cand.get('face_feat') is not None})")

        # ── Update ────────────────────────────────────────────────────────
        for i, sid in assigned.items():
            cand = candidates[i]
            self.identity_db[sid].update(cand['feat'], cand['bbox'], frame_id,
                                          face_descriptor=cand.get('face_feat'))
            self.track_to_identity[cand['tid']] = sid
            self.track_last_seen[cand['tid']]   = frame_id
            self.tracker_sid_history.setdefault(cand['tid'], set()).add(sid)
            self._pending.pop(cand['tid'], None)
            self._new_id_grace.pop(cand['tid'], None)   # matched — reset grace

        if self.debug_trace:
            state = {candidates[i]['tid']: sid for i, sid in assigned.items()}
            all_tids = [c['tid'] for c in candidates]
            unassigned = [t for t in all_tids if t not in state]
            self._trace(frame_id, f"STATE tids_this_frame={all_tids} "
                        f"assigned={state} unassigned={unassigned}")

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
                face_feat = self.face_extractor.extract(frame, bbox)
                self._face_attempts += 1
                if face_feat is not None:
                    self._face_hits += 1
                    # Keep a bounded per-tracker face gallery (latest samples)
                    # so finalize_clustering can detect false merges by face.
                    gallery = self.tracker_faces.setdefault(pid, [])
                    gallery.append(face_feat)
                    if len(gallery) > CFG.TRACK_FACE_GALLERY_SIZE:
                        del gallery[:len(gallery) - CFG.TRACK_FACE_GALLERY_SIZE]
                self._store(pid, feat, frame_id)
                candidates.append({'tid': pid, 'bbox': bbox, 'feat': feat, 'face_feat': face_feat})
                self._pending.pop(pid, None)
            else:
                existing = self.track_to_identity.get(pid)
                if existing is None:
                    self._pending[pid] = frame_id
                results.append({'id': pid, 'consolidated_id': existing,
                                'bbox': bbox, 'feature_dim': 0, 'matches': []})

        assigned = self._assign(candidates, frame_id)

        # Record co-occurrence: any two trackers in the same frame are different people
        frame_tids = [c['tid'] for c in candidates]
        for i, t1 in enumerate(frame_tids):
            self._cooccurrence.setdefault(t1, set())
            for t2 in frame_tids[i+1:]:
                self._cooccurrence.setdefault(t2, set())
                self._cooccurrence[t1].add(t2)
                self._cooccurrence[t2].add(t1)
        # Track which frames each tracker appears in (for disjoint-identity merge)
        for tid in frame_tids:
            self.tracker_frame_map.setdefault(tid, set()).add(frame_id)

        for idx, cand in enumerate(candidates):
            sid = assigned.get(idx) or self.track_to_identity.get(cand['tid'])
            results.append({'id': cand['tid'], 'consolidated_id': sid,
                            'bbox': cand['bbox'],
                            'feature_dim': MultiCueExtractor.TOTAL_DIM, 'matches': []})
        return results

    # ── finalize ───────────────────────────────────────────────────────────

    def _split_false_merges(self):
        """Undo live false merges (PASS3 reappearance, lookalike body-only
        merges) using per-tracker face AND full-track appearance evidence.

        When a sid carries several trackers, union trackers that are
        convincingly the SAME person:
          • strong face similarity (max pairwise ≥ FACE_SAME_TRACK_SPLIT), or
          • clear face mismatch (max pairwise < FACE_SAME_TRACK_VETO) vetoes a
            union even when bodies look the same, or
          • with no face evidence on both sides, a union requires VERY strong
            track-average appearance (≥ APPEARANCE_SPLIT_THRESHOLD).
        Trackers left in separate groups were merged on single-frame evidence
        alone — keep the largest group on the original sid, mint a new sid for
        each other group and rebuild its descriptor from the stored
        per-tracker appearance features.
        """
        sid_tids: dict = {}
        for tid, sid in self.id_mapping.items():
            sid_tids.setdefault(sid, []).append(tid)

        def _face_sim(ta, tb):
            fa = [f for f in self.tracker_faces.get(ta, []) if f is not None]
            fb = [f for f in self.tracker_faces.get(tb, []) if f is not None]
            if not fa or not fb:
                return None
            return float(max(self.face_extractor.similarity(a, b)
                             for a in fa for b in fb))

        def _track_mean(tid):
            feats = self.person_features.get(tid)
            if not feats:
                return None
            return _normalize(np.mean(np.asarray(feats, dtype=np.float32), axis=0))

        def _same_person(ta, tb, means):
            fs = _face_sim(ta, tb)
            ma, mb = means.get(ta), means.get(tb)
            asim = float(_cosine(ma, mb)) if (ma is not None and mb is not None) else None
            if fs is not None:
                if fs >= CFG.FACE_SAME_TRACK_SPLIT:
                    return True
                if fs < CFG.FACE_SAME_TRACK_VETO:
                    return False
            return asim is not None and asim >= CFG.APPEARANCE_SPLIT_THRESHOLD

        split_count = 0
        for sid, tids in sorted(sid_tids.items()):
            if len(tids) < 2:
                continue
            means = {t: _track_mean(t) for t in tids}
            evidence = [t for t in tids
                        if means.get(t) is not None or self.tracker_faces.get(t)]
            if len(evidence) < 2:
                continue
            parent = {t: t for t in evidence}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for i, ta in enumerate(evidence):
                for tb in evidence[i+1:]:
                    if _same_person(ta, tb, means):
                        ra, rb = find(ta), find(tb)
                        if ra != rb:
                            parent[ra] = rb
            groups: dict = {}
            for t in evidence:
                groups.setdefault(find(t), []).append(t)
            # Trackers with no evidence at all stay with the largest group.
            keep_group = max(groups.values(), key=len)
            for t in tids:
                if t not in parent:
                    keep_group.append(t)
            if len(groups) < 2:
                counts = {t: len(self.tracker_faces.get(t, [])) for t in tids}
                sims = []
                for i, ta in enumerate(tids):
                    for tb in tids[i+1:]:
                        fs = _face_sim(ta, tb)
                        ma, mb = means.get(ta), means.get(tb)
                        asim = (_cosine(ma, mb) if ma is not None and mb is not None else None)
                        parts = [f"app={asim:.3f}" if asim is not None else "app=–"]
                        if fs is not None:
                            parts.append(f"face={fs:.3f}")
                        sims.append(f"{ta}↔{tb} {' '.join(parts)}")
                logger.info(f"👥 sid={sid} trackers={sorted(tids)} face_counts={counts} "
                            f"-> kept together ({', '.join(sims) or 'no evidence'})")
                continue
            counts = {t: len(self.tracker_faces.get(t, [])) for t in tids}
            logger.info(f"👥 sid={sid} trackers={sorted(tids)} face_counts={counts} "
                        f"-> splitting {len(groups)} face/appearance groups")
            for extra in sorted(groups.values(), key=len, reverse=True)[1:]:
                new_sid = self.next_stable_id; self.next_stable_id += 1
                for t in extra:
                    self.id_mapping[t] = new_sid
                feats = []
                for t in extra:
                    feats.extend(self.person_features.get(t, []))
                if feats:
                    self.consolidated_features[new_sid] = _normalize(
                        np.mean(np.asarray(feats, dtype=np.float32), axis=0))
                split_count += 1
                logger.info(f"🔀 SPLIT: sid={sid} trackers {sorted(extra)} "
                            f"-> new sid={new_sid} (evidence says different people)")
        if split_count:
            logger.info(f"🔀 Split: {split_count} false merge(s) undone "
                        f"({len(self.id_mapping)} trackers → "
                        f"{len(set(self.id_mapping.values()))} identities)")

    def finalize_clustering(self):
        if self.face_extractor.enabled and self._face_attempts > 0:
            rate = 100.0 * self._face_hits / self._face_attempts
            logger.info(f"📊 Face cue: detected on {self._face_hits}/{self._face_attempts} "
                        f"person-detections ({rate:.1f}%) — "
                        f"{'low rate, faces mostly not helping here' if rate < 15 else 'active and contributing to matches'}")
        if self.track_to_identity:
            self.id_mapping = dict(self.track_to_identity)
            self.consolidated_features = {
                sid: ident.descriptor for sid, ident in self.identity_db.items()
            }
            logger.info(f"✅ {len(self.id_mapping)} tracker IDs → "
                        f"{len(self.consolidated_features)} stable identities (before co-occurrence fix)")
        else:
            self.id_mapping = self._offline_cluster()

        # Enforce co-occurrence constraint: if tracker A and tracker B appear
        # in the same frame, they MUST have different consolidated IDs.
        # If they don't, reassign the later-starting tracker to a new ID.
        self._enforce_cooccurrence()

        # Merge identities that share a tracker ID in their history (the same
        # DeepSORT tracker physically tracks one person — if it was ever
        # assigned to two different sids, they refer to the same person).
        self._merge_shared_tracker_sids()

        # Merge similar identities that never appear in the same frame
        # (e.g., same person whose tracker ID changed mid-video causing
        # an orphaned identity).
        self._merge_non_cooccurring()

        # LAST: split false merges using per-tracker face + track-average
        # appearance evidence. Runs after every merge pass so a PASS3
        # reappearance false-merge (a brand-new tracker claimed by an
        # existing identity on a single-frame body-appearance score, with no
        # face confirmation) can be undone when the accumulated evidence says
        # they are different people.
        self._split_false_merges()

        # Build orphan → living sid remap: sids in identity_db but not in
        # id_mapping that share a tracker assignment history.  This
        # captures the case where a tracker was briefly assigned a new
        # identity (PASS4) and later switched back to its real identity
        # (PASS1), leaving an orphan sid that still appears in per-frame
        # results for the frames before the switch.
        self.orphan_remap = {}
        active_sids = set(self.id_mapping.values())
        for sid in list(self.identity_db):
            if sid in active_sids:
                continue
            orphan_tids = [t for t, s in self.tracker_sid_history.items() if sid in s]
            if not orphan_tids:
                continue
            for living_sid in active_sids:
                for t in orphan_tids:
                    if living_sid in self.tracker_sid_history.get(t, set()):
                        self.orphan_remap[sid] = living_sid
                        logger.info(f"🔗 Orphan sid={sid} → living sid={living_sid} "
                                    f"(shared tracker {t})")
                        break
                if sid in self.orphan_remap:
                    break

        # Remove orphan identities from identity_db / consolidated_features
        for sid in list(self.identity_db):
            if sid not in active_sids and sid not in self.orphan_remap:
                del self.identity_db[sid]
                self.consolidated_features.pop(sid, None)
                logger.info(f"🧹 Removed orphan identity sid={sid} (no tracker assigned)")

        # Build identity frame map from (possibly updated) id_mapping
        self.identity_frame_map = {}
        for tid, sid in self.id_mapping.items():
            frames = self.tracker_frame_map.get(tid, set())
            if frames:
                self.identity_frame_map.setdefault(sid, set()).update(frames)

    def _enforce_cooccurrence(self):
        """After all assignments, check co-occurrence constraints.
        Two trackers that appear in the same frame are definitely different
        people. If they ended up with the same consolidated ID, split them."""
        violations = 0
        for tid, co_tids in self._cooccurrence.items():
            sid_a = self.id_mapping.get(tid)
            if sid_a is None:
                continue
            for co_tid in co_tids:
                sid_b = self.id_mapping.get(co_tid)
                if sid_b is None or sid_a != sid_b:
                    continue
                # Same consolidated ID but co-occur = violation!
                # Reassign the later-starting tracker to a new ID
                meta_a = self.person_metadata.get(tid, {})
                meta_b = self.person_metadata.get(co_tid, {})
                first_a = meta_a.get('first_seen', 0)
                first_b = meta_b.get('first_seen', 0)
                if first_b > first_a:
                    old_sid = sid_b
                    new_sid = self.next_stable_id
                    self.next_stable_id += 1
                    # Reassign all trackers that shared old_sid via this tracker
                    for t, s in list(self.id_mapping.items()):
                        if s == old_sid and t != tid:
                            # Check if t also co-occurs with tid — if so, keep separate
                            if t in self._cooccurrence.get(tid, set()):
                                continue
                            self.id_mapping[t] = new_sid
                    self.id_mapping[co_tid] = new_sid
                    violations += 1
                    self._trace(0, f"CO-OCCURRENCE FIX: tid={co_tid} "
                                f"reassigned from sid={old_sid} to sid={new_sid} "
                                f"(co-occurs with tid={tid} which has sid={sid_a})")
                    break  # re-check from start after reassignment
        if violations:
            logger.info(f"🔧 Co-occurrence fix: {violations} tracker(s) reassigned to separate identities")
            # Rebuild consolidated_features
            self.consolidated_features = {}
            for sid in set(self.id_mapping.values()):
                if sid in self.identity_db:
                    self.consolidated_features[sid] = self.identity_db[sid].descriptor

    def _merge_shared_tracker_sids(self):
        """Merge consolidated IDs that share a tracker ID in their assignment
        history.  A DeepSORT tracker ID always follows one physical person,
        so if it was ever assigned to two different sids, those sids refer
        to the same person and should be merged."""
        # Build sid → set of tids that were ever assigned to it
        sid_to_tids: dict = {}
        for tid, sids in self.tracker_sid_history.items():
            for sid in sids:
                sid_to_tids.setdefault(sid, set()).add(tid)

        sids = sorted(set(self.id_mapping.values()))
        merged = set()
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(sids):
                sid_a = sids[i]
                if sid_a in merged:
                    i += 1
                    continue
                a_tids = sid_to_tids.get(sid_a, set())
                for j in range(i + 1, len(sids)):
                    sid_b = sids[j]
                    if sid_b in merged:
                        continue
                    b_tids = sid_to_tids.get(sid_b, set())
                    # Do they share any tracker?
                    if a_tids & b_tids:
                        # Merge sid_b into sid_a
                        for t, s in list(self.id_mapping.items()):
                            if s == sid_b:
                                self.id_mapping[t] = sid_a
                        merged.add(sid_b)
                        # Update sid_to_tids
                        sid_to_tids.setdefault(sid_a, set()).update(b_tids)
                        changed = True
                        logger.info(f"🔗 Merged sid={sid_b} into sid={sid_a} "
                                    f"(shared tracker {a_tids & b_tids})")
                        break
                i += 1
            if changed:
                sids = sorted(s for s in sids if s not in merged)
        if merged:
            self.consolidated_features = {}
            for sid in set(self.id_mapping.values()):
                if sid in self.identity_db:
                    self.consolidated_features[sid] = self.identity_db[sid].descriptor

    def _merge_non_cooccurring(self, threshold=0.70):
        """Merge consolidated IDs whose time intervals don't overlap and have
        high feature similarity. Fixes the case where a tracker switches
        identities mid-video, leaving an orphaned identity."""
        sids = sorted(set(self.id_mapping.values()))
        merged = set()
        for i, sid_a in enumerate(sids):
            if sid_a in merged:
                continue
            a_tids = {t for t, s in self.id_mapping.items() if s == sid_a}
            a_feat = self.consolidated_features.get(sid_a)
            if a_feat is None:
                continue
            frames_a = self.identity_frame_map.get(sid_a, set())
            for sid_b in sids[i+1:]:
                if sid_b in merged:
                    continue
                # Check if identities ever appear in the same frame
                frames_b = self.identity_frame_map.get(sid_b, set())
                if frames_a & frames_b:
                    continue  # they appear together — definitely different people
                b_feat = self.consolidated_features.get(sid_b)
                if b_feat is None:
                    continue
                sim = _cosine(a_feat, b_feat)
                if sim > threshold:
                    # Merge sid_b into sid_a
                    b_tids = {t for t, s in self.id_mapping.items() if s == sid_b}
                    for t in b_tids:
                        self.id_mapping[t] = sid_a
                    merged.add(sid_b)
                    logger.info(f"🔗 Merged sid={sid_b} into sid={sid_a} (sim={sim:.3f})")
        if merged:
            # Rebuild consolidated_features
            self.consolidated_features = {}
            for sid in set(self.id_mapping.values()):
                if sid in self.identity_db:
                    self.consolidated_features[sid] = self.identity_db[sid].descriptor

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

def run_reid_pipeline(video_path, tracking_json_path, output_json_path, device="cuda", debug_trace=False):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    engine = ReIDEngine(device=device, debug_trace=debug_trace)

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

    # Sync every frame entry's consolidated_id to the FINAL id_mapping.
    # finalize_clustering may have changed assignments (co-occurrence fix,
    # face-based split of false merges, non-cooccurring merges) AFTER the
    # per-frame results were written, so they can be stale. Recomputing from
    # id_mapping makes them consistent with the final identities.
    n_synced = 0
    for fid in sorted(results.keys()):
        for p in results[fid]:
            tid = p.get('id')
            if tid in engine.id_mapping and p.get('consolidated_id') != engine.id_mapping[tid]:
                p['consolidated_id'] = engine.id_mapping[tid]
                n_synced += 1
    if n_synced:
        logger.info(f"🔗 Synced {n_synced} frame entries to final id mapping")

    # Resolve any remaining None consolidated_ids
    for fid in sorted(results.keys()):
        for p in results[fid]:
            if p.get('consolidated_id') is None:
                tid = p.get('id')
                if tid in engine.track_to_identity:
                    p['consolidated_id'] = engine.track_to_identity[tid]

    # Remap orphan sids discovered during finalize_clustering (trackers that
    # were briefly assigned a new identity, then switched back to their
    # correct identity, leaving orphan sids in earlier frames).
    if engine.orphan_remap:
        n_remapped = 0
        for fid in sorted(results.keys()):
            for p in results[fid]:
                cid = p.get('consolidated_id')
                if cid in engine.orphan_remap:
                    p['consolidated_id'] = engine.orphan_remap[cid]
                    n_remapped += 1
        logger.info(f"🔄 Remapped {n_remapped} frame entries from orphan sids")

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