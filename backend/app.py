"""
app.py
======
Thin Flask layer around gear_counter.py.

Routes:
  GET  /                       -> serves the frontend (index.html)
  POST /api/upload             -> accepts a video file, stores it, returns a video_id
  POST /api/analyze/<video_id> -> runs the CV pipeline on a stored video, returns JSON
  GET  /api/video/<video_id>   -> streams the stored video back (for the <video> preview)

Keeping this file thin on purpose: it has no computer-vision logic in it at
all. Every route just validates input, calls into gear_counter.py, and
serializes the result. If you swap the CV pipeline for a YOLO-based one
later, this file shouldn't need to change.
"""

import dataclasses
import os
import uuid

from flask import Flask, jsonify, request, send_from_directory, abort

import gear_counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB upload cap

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _video_path(video_id: str) -> str:
    """Resolve a video_id to a file path, guarding against path traversal."""
    safe_id = os.path.basename(video_id)
    matches = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(safe_id + ".")]
    if not matches:
        abort(404, description="Video not found")
    return os.path.join(UPLOAD_DIR, matches[0])


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file field named 'video' in request"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'"}), 400

    video_id = uuid.uuid4().hex
    dest = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")
    file.save(dest)

    return jsonify({"video_id": video_id, "url": f"/api/video/{video_id}"})


@app.route("/api/video/<video_id>")
def get_video(video_id):
    path = _video_path(video_id)
    directory, filename = os.path.split(path)
    return send_from_directory(directory, filename)


@app.route("/api/analyze/<video_id>", methods=["POST"])
def analyze(video_id):
    path = _video_path(video_id)
    
    start_time_sec = 0.0
    if request.is_json:
        start_time_sec = float(request.json.get("start_time_sec", 0.0))

    try:
        result = gear_counter.analyze_video(path, start_time_sec=start_time_sec)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    payload = {
        "frames_sampled": result.frames_sampled,
        "frames_with_detection": result.frames_with_detection,
        "annotated_frame_b64": result.annotated_frame_b64,
        "annotated_timestamp_sec": result.annotated_timestamp_sec,
        "gears": [dataclasses.asdict(g) for g in result.gears],
    }
    return jsonify(payload)


if __name__ == "__main__":
    # debug=True is fine for local development; turn it off in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
