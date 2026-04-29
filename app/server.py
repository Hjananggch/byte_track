from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.services.detector import DetectorService


BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "static" / "generated"


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    detector = DetectorService()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "ready": detector.is_ready()})

    @app.post("/api/detect/image")
    def detect_image():
        uploaded_file = request.files.get("image")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({"ok": False, "error": "请先选择一张图片。"}), 400

        conf_thres = _parse_float(request.form.get("conf"), default=0.25, minimum=0.01, maximum=0.99)
        iou_thres = _parse_float(request.form.get("iou"), default=0.45, minimum=0.01, maximum=0.99)

        try:
            result = detector.detect_image(
                image_bytes=uploaded_file.read(),
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "识别服务暂不可用，请联系管理员处理。"}), 503
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception:
            app.logger.exception("Image detection failed")
            return jsonify({"ok": False, "error": "识别失败，请稍后重试或联系管理员。"}), 500

        output_name = _build_output_name(uploaded_file.filename)
        output_path = GENERATED_DIR / output_name
        output_path.write_bytes(result.annotated_image_bytes)

        return jsonify(
            {
                "ok": True,
                "image_url": url_for("static", filename=f"generated/{output_name}"),
                "detections": result.detections,
                "meta": {
                    "image_width": result.image_width,
                    "image_height": result.image_height,
                    "count": len(result.detections),
                    "conf": conf_thres,
                    "iou": iou_thres,
                },
            }
        )

    return app


def _parse_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value) if value not in (None, "") else default
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _build_output_name(original_name: str) -> str:
    cleaned_name = secure_filename(original_name) or "capture.jpg"
    stem = Path(cleaned_name).stem or "capture"
    return f"{stem}-{uuid.uuid4().hex[:8]}.jpg"
