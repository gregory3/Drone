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
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
        self._hsv_lower = np.array(p.gate_hsv_lower, dtype=np.uint8)
        self._hsv_upper = np.array(p.gate_hsv_upper, dtype=np.uint8)
        self._min_area = p.gate_min_area_px
        self._max_area = p.gate_max_area_px
        self._conf_thresh = p.gate_confidence_threshold
        self._gate_real_size_m = 1.2   # assume 1.2m gate for distance est.
        self._fx = (p.image_width / 2) / np.tan(np.radians(p.camera_fov_deg / 2))

    def detect(self, frame_bgr: np.ndarray) -> List[GateDetection]:
        # --- preprocessing ---
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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

            # Confidence: penalize non-square and low fill ratio
            ideal_aspect_score = 1.0 - abs(1.0 - aspect) * 0.5
            fill_ratio = area / max(w * h, 1)
            confidence = float(np.clip(ideal_aspect_score * fill_ratio ** 0.5, 0, 1))

            cx = x + w / 2
            cy = y + h / 2

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
        import onnxruntime as ort
        model_path = cfg.perception.onnx_model_path
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
# Factory
# ---------------------------------------------------------------------------

def make_detector(backend: Optional[str] = None) -> GateDetector:
    b = backend or cfg.perception.detector_backend
    if b == "classical":
        return ClassicalGateDetector()
    elif b == "onnx":
        return ONNXGateDetector()
    else:
        raise ValueError(f"Unknown detector backend: {b!r}. "
                         f"Use 'classical' or 'onnx'.")
