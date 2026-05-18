import threading
import time

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

import train_model
from attendance_core import (
    AttendanceRecognizer,
    Camera,
    clear_attendance,
    create_student,
    encode_jpeg,
    list_students,
    read_attendance,
    save_student_frame,
)


app = Flask(__name__, static_folder="static", static_url_path="")
camera = Camera()
recognizer = AttendanceRecognizer()

attendance_running = False
camera_enabled = True
attendance_lock = threading.Lock()
training_state = {"running": False, "message": "Idle"}
RECOGNITION_INTERVAL_SECONDS = 0.5
STREAM_FRAME_DELAY_SECONDS = 0.08
latest_frame = None
latest_results = []
latest_lock = threading.Lock()
recognition_thread = None


def api_error(message, status=400):
    return jsonify({"ok": False, "error": str(message)}), status


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    return jsonify(
        {
            "ok": True,
            "attendanceRunning": attendance_running,
            "cameraRunning": camera_enabled and camera.is_open(),
            "cameraEnabled": camera_enabled,
            "training": training_state,
            "students": list_students(),
            "attendanceCount": len(read_attendance()),
        }
    )


@app.get("/api/debug/camera")
def debug_camera():
    return jsonify({"ok": True, "camera": camera.debug_info()})


@app.get("/api/attendance")
def attendance():
    return jsonify({"ok": True, "rows": read_attendance()})


@app.delete("/api/attendance")
def clear_attendance_log():
    clear_attendance()
    return jsonify({"ok": True, "rows": []})


@app.post("/api/students")
def add_student():
    payload = request.get_json(silent=True) or {}
    try:
        usn = create_student(payload.get("usn", ""))
        return jsonify({"ok": True, "student": {"usn": usn, "images": 0}})
    except FileExistsError as exc:
        return api_error(exc, 409)
    except Exception as exc:
        return api_error(exc)


@app.get("/api/students")
def students():
    return jsonify({"ok": True, "students": list_students()})


@app.post("/api/students/<usn>/capture")
def capture_student(usn):
    global camera_enabled
    try:
        camera_enabled = True
        ret, frame = camera.read()
        if not ret or frame is None:
            return api_error("Could not read a camera frame.", 500)
        image_path = save_student_frame(usn, frame)
        return jsonify({"ok": True, "image": image_path, "students": list_students()})
    except Exception as exc:
        app.logger.exception("Failed to capture student photo")
        return api_error(exc)


def run_training():
    training_state["running"] = True
    training_state["message"] = "Training..."
    try:
        train_model.train_photo()
        recognizer.reload()
        training_state["message"] = "Training complete"
    except Exception as exc:
        training_state["message"] = f"Training failed: {exc}"
    finally:
        training_state["running"] = False


@app.post("/api/train")
def train():
    if training_state["running"]:
        return api_error("Training is already running.", 409)
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return jsonify({"ok": True, "training": training_state})


@app.post("/api/attendance/start")
def start_attendance():
    global attendance_running, camera_enabled, recognition_thread, latest_frame, latest_results
    try:
        camera_enabled = True
        recognizer.reload()
        camera.open()
    except Exception as exc:
        return api_error(exc, 500)

    with attendance_lock:
        attendance_running = True
    with latest_lock:
        latest_frame = None
        latest_results = []

    if recognition_thread is None or not recognition_thread.is_alive():
        recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
        recognition_thread.start()

    return jsonify(
        {
            "ok": True,
            "attendanceRunning": attendance_running,
            "cameraRunning": camera.is_open(),
            "cameraEnabled": camera_enabled,
        }
    )


@app.post("/api/attendance/stop")
def stop_attendance():
    global attendance_running, latest_frame, latest_results
    with attendance_lock:
        attendance_running = False
    with latest_lock:
        latest_frame = None
        latest_results = []
    return jsonify(
        {
            "ok": True,
            "attendanceRunning": attendance_running,
            "cameraRunning": camera.is_open(),
            "cameraEnabled": camera_enabled,
        }
    )


@app.post("/api/camera/start")
def start_camera():
    global camera_enabled
    try:
        camera_enabled = True
        camera.open()
        return jsonify({"ok": True, "cameraRunning": camera.is_open(), "cameraEnabled": camera_enabled})
    except Exception as exc:
        return api_error(exc, 500)


@app.post("/api/camera/stop")
def stop_camera():
    global attendance_running, camera_enabled, latest_frame, latest_results
    with attendance_lock:
        attendance_running = False
        camera_enabled = False
    with latest_lock:
        latest_frame = None
        latest_results = []
    camera.close()
    return jsonify({"ok": True, "attendanceRunning": attendance_running, "cameraRunning": False, "cameraEnabled": False})


def recognition_loop():
    global latest_results

    while True:
        with attendance_lock:
            should_log = attendance_running
        if not should_log:
            break

        with latest_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is not None:
            try:
                results = recognizer.process_frame(frame, log_matches=True)
                with latest_lock:
                    latest_results = results
            except Exception:
                with latest_lock:
                    latest_results = []

        time.sleep(RECOGNITION_INTERVAL_SECONDS)


def stream_frames():
    global latest_frame

    while True:
        try:
            if not camera_enabled:
                time.sleep(0.2)
                continue

            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            with attendance_lock:
                should_log = attendance_running

            if should_log:
                with latest_lock:
                    latest_frame = frame.copy()
                    results = list(latest_results)
                recognizer.draw_results(frame, results)

            jpg = encode_jpeg(frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(STREAM_FRAME_DELAY_SECONDS)
        except GeneratorExit:
            break
        except Exception:
            time.sleep(0.2)


@app.get("/api/video")
def video():
    return Response(stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
