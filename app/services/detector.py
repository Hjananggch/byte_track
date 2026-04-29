from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = REPO_ROOT / "model"
WEIGHTS_DIR = REPO_ROOT / "weights"


@dataclass
class DetectionResult:
    detections: list[dict]
    annotated_image_bytes: bytes
    image_width: int
    image_height: int


class DetectorService:
    def __init__(self, model_repo: Path | None = None, weights_path: Path | None = None) -> None:
        self.model_repo = model_repo or MODEL_REPO
        self.weights_path = weights_path or self._resolve_weights_path()
        self._model = None

    def _resolve_weights_path(self) -> Path | None:
        configured = os.getenv("MODEL_WEIGHTS", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()

        if not WEIGHTS_DIR.exists():
            return None

        candidates = sorted(WEIGHTS_DIR.glob("*.pt"))
        return candidates[0] if candidates else None

    def status(self) -> dict:
        return {
            "ready": self.is_ready(),
            "model_repo": str(self.model_repo),
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "model_loaded": self._model is not None,
        }

    def is_ready(self) -> bool:
        return bool(self.weights_path and self.weights_path.exists() and self.model_repo.exists())

    def _ensure_model(self, conf_thres: float, iou_thres: float):
        if not self.weights_path or not self.weights_path.exists():
            raise FileNotFoundError(
                "未找到模型权重。请把 .pt 文件放到仓库的 weights/ 目录，或设置环境变量 MODEL_WEIGHTS。"
            )
        if not self.model_repo.exists():
            raise FileNotFoundError(f"未找到模型代码目录: {self.model_repo}")

        if self._model is None:
            self._model = torch.hub.load(
                str(self.model_repo),
                "custom",
                path=str(self.weights_path),
                source="local",
            )

        self._model.conf = conf_thres
        self._model.iou = iou_thres
        return self._model

    def detect_image(
        self,
        image_bytes: bytes,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        class_filters: Iterable[str] | None = None,
    ) -> DetectionResult:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析图片文件，请重新选择图片。")

        filters = {item.strip().lower() for item in (class_filters or []) if item.strip()}
        model = self._ensure_model(conf_thres, iou_thres)
        results = model(image)
        rows = results.xyxy[0].cpu().numpy()
        names = results.names

        annotated = image.copy()
        detections: list[dict] = []
        for row in rows:
            class_id = int(row[5])
            class_name = self._class_name(names, class_id)
            if filters and class_name.lower() not in filters:
                continue

            left, top, right, bottom = [int(value) for value in row[:4]]
            confidence = float(row[4])

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [left, top, right, bottom],
                }
            )
            self._draw_detection(annotated, left, top, right, bottom, class_name, confidence)

        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("图片结果编码失败。")

        height, width = annotated.shape[:2]
        return DetectionResult(
            detections=detections,
            annotated_image_bytes=encoded.tobytes(),
            image_width=width,
            image_height=height,
        )

    @staticmethod
    def _class_name(names, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _draw_detection(
        image: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
        class_name: str,
        confidence: float,
    ) -> None:
        color = (28, 184, 65)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        label = f"{class_name} {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_top = max(top - label_height - 12, 0)
        cv2.rectangle(
            image,
            (left, label_top),
            (left + label_width + 10, label_top + label_height + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (left + 5, label_top + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (14, 22, 34),
            2,
            cv2.LINE_AA,
        )
