"""
perception.gate_detector
Gate detection pipeline. Two backends, same output contract:

    ClassicalGateDetector   — HSV + contour, no GPU, works for Round 1
    ONNXGateDetector        — YOLO-family ONNX model, for Round 2

Both return List[GateDetection] from detect(frame).

Usage:
    from perception.gate_detector import make_detector, GateDetection

    detector = make_detector(backend="classical")
    detections = detector.detect(frame_bgr)
    if detections:
        best = detections[0]
        print(best.center_px, best.confidence, best.area_px)
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

import cv2
import numpy as np

from config.loader import cfg


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class GateDetection:
    center_px: Tuple[float, float]   # (u, v) image coordinates
    bbox_px: Tuple[int, int, int, int]  # (x, y, w, h)
    area_px: float
    confidence: float               # 0..1
    corners_px: Optional[np.ndarray] = None   # (4, 2) array if available
    distance_est_m: Optional[float] = None    # rough monocular estimate


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class GateDetector(ABC):

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> List[GateDetection]:
        """Return detections sorted by confidence descending."""
        ...

    def annotate(self, frame_bgr: np.ndarray,
                 detections: List[GateDetection]) -> np.ndarray:
        """Return a copy of frame with detections drawn on it."""
        out = frame_bgr.copy()
        for det in detections:
            x, y, w, h = det.bbox_px
            color = (0, 255, 0) if det.confidence >= cfg.perception.gate_confidence_threshold \
                else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cx, cy = int(det.center_px[0]), int(det.center_px[1])
            cv2.drawMarker(out, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            label = f"{det.confidence:.2f}"
            if det.distance_est_m is not None:
                label += f"  {det.distance_est_m:.1f}m"
            cv2.putText(out, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return out


# ---------------------------------------------------------------------------
# Classical detector — HSV + contour fitting
# ---------------------------------------------------------------------------

class ClassicalGateDetector(GateDetector):
    """
    Round 1 strategy:
      1. Convert to HSV
      2. Threshold for gate color (orange by default — tune via config)
      3. Find contours, filter by area and aspect ratio
      4. Fit bounding rect, score by compactness

    Very fast (~1ms on CPU), zero model overhead.
    Fragile against lighting changes → replaced by ONNX for Round 2.
    """

    def __init__(self) -> None:
        p = cfg.perception
        # Support multiple HSV ranges so we can match red gates (red wraps the
        # hue boundary, needing low + high ranges) alongside the orange range.
        # Primary range plus any configured gate_hsv_lower2/3 + gate_hsv_upper2/3.
        self._hsv_ranges = [(np.array(p.gate_hsv_lower, dtype=np.uint8),
                             np.array(p.gate_hsv_upper, dtype=np.uint8))]
        for i in (2, 3, 4):
            lo = getattr(p, f"gate_hsv_lower{i}", None)
            hi = getattr(p, f"gate_hsv_upper{i}", None)
            if lo is not None and hi is not None:
                self._hsv_ranges.append((np.array(lo, dtype=np.uint8),
                                         np.array(hi, dtype=np.uint8)))
        # Kept for backwards compatibility / introspection.
        self._hsv_lower = self._hsv_ranges[0][0]
        self._hsv_upper = self._hsv_ranges[0][1]
        self._min_area = p.gate_min_area_px
        self._max_area = p.gate_max_area_px
        self._conf_thresh = p.gate_confidence_threshold
        # Start-light rejection: gates render as solid red squares (same colour
        # as the start light), so the discriminator is the light's GREEN twin.
        self._reject_green_twin = getattr(p, "gate_reject_green_twin", True)
        self._green_lower = np.array(getattr(p, "green_hsv_lower", [40, 80, 80]),
                                     dtype=np.uint8)
        self._green_upper = np.array(getattr(p, "green_hsv_upper", [90, 255, 255]),
                                     dtype=np.uint8)
        # Real gate size for monocular distance: the red contour's bounding box
        # spans the OUTER gate frame, which VADR-TS-002 §3.7 fixes at 2.7 m
        # (inner flyable opening is 1.5 m). The old 1.2 m guess made every
        # distance estimate ~2.25x too short.
        self._gate_real_size_m = float(getattr(p, "gate_real_size_m", 2.7))
        self._fx = (p.image_width / 2) / np.tan(np.radians(p.camera_fov_deg / 2))

    def _has_green_twin(self, green_mask, x: int, y: int,
                        w: int, h: int) -> bool:
        """True if a comparable green blob sits just beside this red blob.

        The start light is a red+green pair, so a red candidate with a green
        companion roughly its own size, vertically aligned and within ~2 widths
        horizontally, is the light, not a gate.
        """
        if green_mask is None:
            return False
        # Search band: same rows (a little taller), extended left+right by ~2w.
        pad_x = max(int(2 * w), 20)
        x0 = max(0, x - pad_x)
        x1 = min(green_mask.shape[1], x + w + pad_x)
        y0 = max(0, y - int(0.3 * h))
        y1 = min(green_mask.shape[0], y + h + int(0.3 * h))
        band = green_mask[y0:y1, x0:x1]
        if band.size == 0:
            return False
        cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        red_area = float(w * h)
        for c in cnts:
            gx, gy, gw, gh = cv2.boundingRect(c)
            g_area = float(gw * gh)
            # Comparable size (within ~4x either way) => looks like the twin.
            if 0.25 * red_area <= g_area <= 4.0 * red_area:
                return True
        return False

    def detect(self, frame_bgr: np.ndarray) -> List[GateDetection]:
        # --- preprocessing ---
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, *self._hsv_ranges[0])
        for lo, hi in self._hsv_ranges[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Green mask — used only to spot the start light's green twin so we can
        # reject it (gates have no green companion). Cheap; skipped if disabled.
        if self._reject_green_twin:
            green_mask = cv2.inRange(hsv, self._green_lower, self._green_upper)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        else:
            green_mask = None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detections: List[GateDetection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area or area > self._max_area:
                continue

            # Bounding rect and aspect ratio filter
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if aspect < 0.3 or aspect > 3.5:
                continue

            cx = x + w / 2
            cy = y + h / 2

            # --- start-light rejection: the sim renders a red/green light pair
            # near the start. Gates render as solid red squares too, so colour
            # and fill don't separate them — but the start light has a GREEN
            # twin right beside the red blob, and a gate does not. Reject a red
            # candidate that has a comparable green blob adjacent to it. This
            # was the fly16 failure (detector chased the red light at conf 0.76).
            if self._has_green_twin(green_mask, x, y, w, h):
                continue

            # Confidence: penalize non-square and low fill ratio
            ideal_aspect_score = 1.0 - abs(1.0 - aspect) * 0.5
            fill_ratio = area / max(w * h, 1)
            confidence = float(np.clip(ideal_aspect_score * fill_ratio ** 0.5, 0, 1))

            # Rough monocular distance estimate using apparent width
            dist_m = (self._gate_real_size_m * self._fx) / max(w, 1) \
                     if w > 10 else None

            detections.append(GateDetection(
                center_px=(cx, cy),
                bbox_px=(x, y, w, h),
                area_px=area,
                confidence=confidence,
                distance_est_m=dist_m,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ---------------------------------------------------------------------------
# ONNX detector — Round 2 / realistic environments
# ---------------------------------------------------------------------------

class ONNXGateDetector(GateDetector):
    """
    Wraps a YOLO-family ONNX model via onnxruntime.
    Drop your exported model into perception/models/gate_detector.onnx
    and switch config.perception.detector_backend to "onnx".
    """

    def __init__(self) -> None:
        model_path = cfg.perception.onnx_model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path!r}. "
                "Place a YOLO gate detector at this path or switch to "
                "detector_backend='classical' in settings.yaml."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for the ONNX detector backend. "
                "Install it with `pip install onnxruntime`."
            ) from exc

        self._input_size = tuple(cfg.perception.onnx_input_size)  # (W, H)
        self._conf_thresh = cfg.perception.gate_confidence_threshold

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray) -> List[GateDetection]:
        blob = self._preprocess(frame_bgr)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._postprocess(outputs, frame_bgr.shape)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        W, H = self._input_size
        resized = cv2.resize(frame, (W, H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)

    def _postprocess(self, outputs, orig_shape) -> List[GateDetection]:
        # Standard YOLO output: (1, N, 6) → [x, y, w, h, conf, class_id]
        oh, ow = orig_shape[:2]
        W, H = self._input_size
        preds = outputs[0][0]  # (N, 6)
        detections = []
        for pred in preds:
            conf = float(pred[4])
            if conf < self._conf_thresh:
                continue
            cx_n, cy_n, w_n, h_n = pred[:4]
            cx = cx_n * ow / W
            cy = cy_n * oh / H
            bw = w_n * ow / W
            bh = h_n * oh / H
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            detections.append(GateDetection(
                center_px=(float(cx), float(cy)),
                bbox_px=(x, y, int(bw), int(bh)),
                area_px=bw * bh,
                confidence=conf,
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ---------------------------------------------------------------------------
# Temporal smoothing wrapper
# ---------------------------------------------------------------------------

class TemporalGateDetector(GateDetector):
    """Tracks a single gate across frames.

    Two behaviours, both learned the hard way against the live AI-GP sim where
    the course stacks several gates in view (especially after the hard right
    turn past gate 0):

    1. Target selection on acquisition follows the "next gate" principle used by
       Swift (feeds the next gate's corners) and MonoRace (reprojects the next
       gate from the map): pick the NEAREST gate, approximated onboard as the
       largest apparent blob, rather than the highest-confidence one (which can
       be a far gate down the line).
    2. Continuity tracking: once locked, keep following THAT gate by picking the
       raw blob nearest the last known position, instead of re-grabbing the
       highest-confidence blob every frame. This stops the mid-approach lurch to
       a far gate that caused the fly13 veer-off.
    """

    def __init__(self, backend: GateDetector,
                 history_frames: int,
                 max_missed_frames: int,
                 max_center_jump_px: float,
                 select: str = "nearest",
                 continuity: bool = True) -> None:
        self._backend = backend
        self._history_frames = max(1, history_frames)
        self._max_missed_frames = max(0, max_missed_frames)
        self._max_center_jump_px = max_center_jump_px
        self._select = select
        self._continuity = continuity
        self._history = deque(maxlen=self._history_frames)
        self._missed_frames = 0
        self._smoothed: Optional[GateDetection] = None

    def detect(self, frame_bgr: np.ndarray) -> List[GateDetection]:
        raw = self._backend.detect(frame_bgr)
        if raw:
            best = self._select_target(raw)
            if self._smoothed is None or self._is_consistent(best):
                self._history.append(best)
            else:
                self._history.clear()
                self._history.append(best)
            self._missed_frames = 0
            self._smoothed = self._compute_smoothed()
            return [self._smoothed]

        if self._history and self._missed_frames < self._max_missed_frames:
            self._missed_frames += 1
            return [self._smoothed] if self._smoothed is not None else []

        self._history.clear()
        self._smoothed = None
        return []

    def _select_target(self, raw: List[GateDetection]) -> GateDetection:
        """Pick which detection to track this frame from all candidates."""
        # Already locked onto a gate -> stay on the nearest blob to it.
        if self._continuity and self._smoothed is not None:
            sx, sy = self._smoothed.center_px
            # Allow the search window to grow while the gate is briefly missed.
            window = self._max_center_jump_px * (1 + self._missed_frames)
            near = [d for d in raw
                    if np.hypot(d.center_px[0] - sx, d.center_px[1] - sy) <= window]
            if near:
                return min(near, key=lambda d: np.hypot(
                    d.center_px[0] - sx, d.center_px[1] - sy))
            # No blob near the lock -> fall through to re-acquire.

        # Acquisition: "nearest" gate == largest apparent area; else highest conf.
        if self._select == "nearest":
            return max(raw, key=lambda d: d.area_px)
        return max(raw, key=lambda d: d.confidence)

    def reset(self) -> None:
        """Drop the current lock so the next gate is acquired fresh."""
        self._history.clear()
        self._missed_frames = 0
        self._smoothed = None

    def _is_consistent(self, detection: GateDetection) -> bool:
        if self._smoothed is None:
            return True
        dx = detection.center_px[0] - self._smoothed.center_px[0]
        dy = detection.center_px[1] - self._smoothed.center_px[1]
        return np.hypot(dx, dy) <= self._max_center_jump_px

    def _compute_smoothed(self) -> GateDetection:
        centers = np.array([det.center_px for det in self._history], dtype=float)
        bboxes = np.array([det.bbox_px for det in self._history], dtype=float)
        confidences = np.array([det.confidence for det in self._history], dtype=float)
        areas = np.array([det.area_px for det in self._history], dtype=float)
        distances = np.array([
            det.distance_est_m for det in self._history
            if det.distance_est_m is not None
        ], dtype=float)

        center = tuple(np.mean(centers, axis=0).tolist())
        bbox = tuple(np.round(np.mean(bboxes, axis=0)).astype(int).tolist())
        confidence = float(np.mean(confidences))
        area = float(np.mean(areas))
        distance_est_m = float(np.mean(distances)) if distances.size else None

        return GateDetection(
            center_px=center,
            bbox_px=bbox,
            area_px=area,
            confidence=confidence,
            distance_est_m=distance_est_m,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_detector(backend: Optional[str] = None) -> GateDetector:
    b = backend or cfg.perception.detector_backend
    if b == "classical":
        detector = ClassicalGateDetector()
    elif b == "onnx":
        detector = ONNXGateDetector()
    else:
        raise ValueError(f"Unknown detector backend: {b!r}. "
                         f"Use 'classical' or 'onnx'.")

    if getattr(cfg.perception, "temporal_smoothing_frames", 0) > 0:
        detector = TemporalGateDetector(
            backend=detector,
            history_frames=cfg.perception.temporal_smoothing_frames,
            max_missed_frames=cfg.perception.temporal_smoothing_miss_tolerance,
            max_center_jump_px=cfg.perception.detection_max_center_jump_px,
            select=getattr(cfg.perception, "gate_select", "nearest"),
            continuity=getattr(cfg.perception, "gate_track_continuity", True),
        )
    return detector
