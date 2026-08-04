"""
Sighting / alert notifications
===============================

When a multi-camera session finishes, every tracked person that resolved to a
registered name becomes a "sighting": which camera, when, at what confidence,
plus all the stored data we have about that person.

Alerts are throttled per (name, camera): once a person is alerted on a given
camera, that camera won't alert for the same person again for `cooldown`
seconds. Without this, a person standing in front of a camera would fire one
alert per frame.

Alerts are kept in memory (capped) and mirrored to outputs/sightings.json so
the feed survives a server restart.
"""

import json
import time
import uuid
from pathlib import Path

from registration.identity_db import IdentityDatabase

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT_DIR / "outputs" / "sightings.json"

MAX_ALERTS = 200
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes between alerts for same person+camera

_alerts = []
_last_seen_at = {}  # (name, camera_label) -> time.monotonic() of last alert
_loaded = False


def _load():
    global _loaded
    if _loaded:
        return
    _loaded = True
    if OUT_FILE.exists():
        try:
            _alerts.extend(json.loads(OUT_FILE.read_text()))
        except Exception:
            pass


def _save():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(_alerts, indent=2))


def _photo_url(path_str: str) -> str:
    # image_paths are stored like "outputs/registration/images/<name>/<file>";
    # taking the last two parts is robust for relative/absolute paths.
    parts = Path(path_str).parts[-2:]
    return "/reg-photos/" + "/".join(parts)


def _person_payload(name: str) -> dict:
    """Everything the DB stores about a person, shaped for the alert."""
    db = IdentityDatabase()
    rec = db.get_person(name)
    if rec is None:
        return {"name": name, "photos": []}
    meta = rec.get("metadata", {})
    image_paths = meta.get("image_paths", [])
    return {
        "name": name,
        "num_images": meta.get("num_images", 0),
        "registered_at": meta.get("registered_at"),
        "last_updated": meta.get("last_updated"),
        "photos": [_photo_url(p) for p in image_paths],
    }


def get_alerts() -> list:
    _load()
    return list(_alerts)


def clear_alerts() -> int:
    _load()
    n = len(_alerts)
    _alerts.clear()
    _save()
    return n


def record_session_sightings(session_id: str, cameras: dict,
                             cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS) -> list:
    """Generate throttled alert entries from a finished session.

    `cameras` is the SESSIONS[session_id]["cameras"] dict — each value has
    `people` (list of {track_id, name, similarity, first_seen_sec, ...}),
    `camera_id` and `label`.

    Returns the list of newly recorded alerts.
    """
    _load()
    now = time.monotonic()
    new_alerts = []

    for cam in cameras.values():
        cam_id = cam.get("camera_id")
        label = cam.get("label", cam_id)
        for person in cam.get("people", []):
            name = person.get("name")
            if not name:
                continue

            key = (name, label)
            last = _last_seen_at.get(key)
            if last is not None and (now - last) < cooldown_seconds:
                continue
            _last_seen_at[key] = now

            alert = {
                "id": uuid.uuid4().hex[:10],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "session_id": session_id,
                "camera": cam_id,
                "camera_label": label,
                "similarity": person.get("similarity"),
                "first_seen_sec": person.get("first_seen_sec"),
                "last_seen_sec": person.get("last_seen_sec"),
                "person": _person_payload(name),
            }
            _alerts.append(alert)
            new_alerts.append(alert)

    if new_alerts:
        if len(_alerts) > MAX_ALERTS:
            del _alerts[: len(_alerts) - MAX_ALERTS]
        _save()

    return new_alerts
