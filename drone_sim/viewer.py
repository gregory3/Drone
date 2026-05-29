"""
sim.viewer
Handles simulator camera view display and headless fallback recording.
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np


class SimViewer:
    def __init__(self, window_name: str, output_dir: Path) -> None:
        self.window_name = window_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._use_gui = True
        self._video_writer: cv2.VideoWriter | None = None
        self._frames_dir: Path | None = None
        self._frame_count = 0
        self._video_path = self.output_dir / "view.mp4"

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            self._use_gui = False
            print(f"[SimViewer] GUI unavailable, recording view to {self._video_path} ({exc})")

    def display(self, frame: np.ndarray) -> bool:
        self._frame_count += 1
        if self._use_gui:
            try:
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1)
                if key == 27:
                    return True
            except Exception as exc:
                self._use_gui = False
                print(f"[SimViewer] GUI display failed, switching to recording: {exc}")

        if not self._use_gui:
            self._record_frame(frame)
        return False

    def _record_frame(self, frame: np.ndarray) -> None:
        if self._video_writer is None and self._frames_dir is None:
            self._init_recording(frame)

        if self._video_writer is not None:
            self._video_writer.write(frame)
        elif self._frames_dir is not None:
            frame_path = self._frames_dir / f"frame_{self._frame_count:05d}.png"
            cv2.imwrite(str(frame_path), frame)

    def _init_recording(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._video_path), fourcc, 30.0,
                                 (width, height))
        if writer.isOpened():
            self._video_writer = writer
            print(f"[SimViewer] Recording view to {self._video_path}")
            return

        self._frames_dir = self.output_dir / "view_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SimViewer] Video writer unavailable, saving frames to {self._frames_dir}")

    def close(self) -> None:
        if self._use_gui:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

        if self._video_writer is not None:
            self._video_writer.release()
            print(f"[SimViewer] Saved recorded view to {self._video_path}")
        elif self._frames_dir is not None:
            print(f"[SimViewer] Saved {self._frame_count} frames to {self._frames_dir}")
