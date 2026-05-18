import csv
import os
import pickle
import threading
from datetime import datetime

import cv2
import face_recognition
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_IMPORT_ERROR = None
except Exception as exc:
    Picamera2 = None
    PICAMERA2_IMPORT_ERROR = str(exc)

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PICAMERA_IMPORT_ERROR = None
except Exception as exc:
    PiCamera = None
    PiRGBArray = None
    PICAMERA_IMPORT_ERROR = str(exc)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")
ENTRY_LOG_PATH = os.path.join(BASE_DIR, "entry_log.csv")

MATCH_TOLERANCE = float(os.getenv("FACE_MATCH_TOLERANCE", "0.48"))
MATCH_MARGIN = float(os.getenv("FACE_MATCH_MARGIN", "0.06"))
COOLDOWN_SECONDS = int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", "10"))
REQUIRED_CONFIRMATIONS = int(os.getenv("FACE_REQUIRED_CONFIRMATIONS", "2"))


def normalize_frame(frame, source_format="bgr"):
    if frame is None:
        raise ValueError("Camera returned an empty frame.")

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        if source_format == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        if source_format == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported camera frame shape: {frame.shape}")

    return np.ascontiguousarray(frame)


def bgr_to_face_rgb(frame):
    frame = normalize_frame(frame, source_format="bgr")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.copy(), dtype=np.uint8)


def describe_array(frame):
    if frame is None:
        return "None"
    return (
        f"shape={getattr(frame, 'shape', None)}, "
        f"dtype={getattr(frame, 'dtype', None)}, "
        f"contiguous={frame.flags['C_CONTIGUOUS'] if hasattr(frame, 'flags') else None}"
    )


def face_locations(rgb, context):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if not (len(rgb.shape) == 3 and rgb.shape[2] == 3):
        raise ValueError(f"{context}: face image is not RGB: {describe_array(rgb)}")
    try:
        return face_recognition.face_locations(rgb, model="hog")
    except Exception as exc:
        raise RuntimeError(f"{context}: {exc}. Face input: {describe_array(rgb)}") from exc


def face_encodings(rgb, boxes, context):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    try:
        return face_recognition.face_encodings(rgb, boxes)
    except Exception as exc:
        raise RuntimeError(f"{context}: {exc}. Face input: {describe_array(rgb)}") from exc


def ensure_log_file():
    if not os.path.exists(ENTRY_LOG_PATH):
        open(ENTRY_LOG_PATH, "a", newline="").close()


def read_attendance():
    ensure_log_file()
    rows = []
    with open(ENTRY_LOG_PATH, "r", newline="") as file:
        for row in csv.reader(file):
            if len(row) >= 2:
                rows.append({"usn": row[0], "time": row[1]})
    return rows


def write_attendance(usn, captured_at=None):
    ensure_log_file()
    captured_at = captured_at or datetime.now()
    with open(ENTRY_LOG_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([usn, captured_at.isoformat(sep=" ", timespec="seconds")])
        file.flush()


def clear_attendance():
    with open(ENTRY_LOG_PATH, "w", newline=""):
        pass


def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError("encodings.pickle is missing. Train photos first.")
    with open(ENCODINGS_PATH, "rb") as file:
        data = pickle.loads(file.read())
    if not data.get("encodings") or not data.get("names"):
        raise ValueError("encodings.pickle has no usable face data. Capture and train photos first.")
    return data


def choose_identity(known_encodings, known_names, encoding):
    distances = face_recognition.face_distance(known_encodings, encoding)
    if len(distances) == 0:
        return "Unknown", None, "no_training_data"

    by_name = {}
    for name, distance in zip(known_names, distances):
        by_name.setdefault(name, []).append(float(distance))

    ranked = sorted(
        ((name, min(name_distances)) for name, name_distances in by_name.items()),
        key=lambda item: item[1],
    )
    best_name, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else None

    if best_distance > MATCH_TOLERANCE:
        return "Unknown", best_distance, "below_threshold"
    if second_distance is not None and second_distance - best_distance < MATCH_MARGIN:
        return "Unknown", best_distance, "ambiguous"
    return best_name, best_distance, "matched"


class AttendanceRecognizer:
    def __init__(self):
        self.data = None
        self.last_seen = {}
        self.confirmations = {}
        self.lock = threading.Lock()

    def reload(self):
        with self.lock:
            self.data = load_encodings()
            self.last_seen = {}
            self.confirmations = {}

    def _data(self):
        if self.data is None:
            self.reload()
        return self.data

    def process_frame(self, frame, log_matches=False):
        data = self._data()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = bgr_to_face_rgb(small_frame)

        boxes = face_locations(rgb_small_frame, "attendance recognition")
        encodings = face_encodings(rgb_small_frame, boxes, "attendance recognition")
        results = []
        now = datetime.now()

        for box, encoding in zip(boxes, encodings):
            name, distance, reason = choose_identity(data["encodings"], data["names"], encoding)
            logged = False

            if log_matches and name != "Unknown":
                self.confirmations[name] = self.confirmations.get(name, 0) + 1
                if self.confirmations[name] >= REQUIRED_CONFIRMATIONS:
                    last = self.last_seen.get(name)
                    if last is None or (now - last).total_seconds() > COOLDOWN_SECONDS:
                        write_attendance(name, now)
                        self.last_seen[name] = now
                        logged = True

            top, right, bottom, left = box
            results.append(
                {
                    "name": name,
                    "distance": distance,
                    "reason": reason,
                    "logged": logged,
                    "box": {
                        "top": top * 4,
                        "right": right * 4,
                        "bottom": bottom * 4,
                        "left": left * 4,
                    },
                }
            )

        return results

    def draw_results(self, frame, results):
        for result in results:
            box = result["box"]
            name = result["name"]
            distance = result["distance"]
            label = name
            if distance is not None and name != "Unknown":
                label = f"{name} ({distance:.2f})"

            left = box["left"]
            top = box["top"]
            right = box["right"]
            bottom = box["bottom"]
            color = (0, 180, 80) if name != "Unknown" else (0, 180, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = top - 12 if top - 12 > 12 else top + 20
            cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame


class Camera:
    def __init__(self):
        self.lock = threading.RLock()
        self.camera_type = None
        self.camera = None
        self.raw_capture = None

    def open(self):
        with self.lock:
            if self.camera is not None:
                return

            use_pi_camera = os.getenv("USE_PI_CAMERA", "0") == "1"
            use_legacy_pi_camera = os.getenv("USE_LEGACY_PI_CAMERA", "0") == "1"
            self.camera_type = None
            self.raw_capture = None
            if use_pi_camera and Picamera2 is not None:
                try:
                    camera = Picamera2()
                except Exception as exc:
                    raise RuntimeError(f"Picamera2 initialization failed: {exc}") from exc

                try:
                    config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
                    camera.configure(config)
                    camera.start()
                except Exception as exc:
                    try:
                        camera.close()
                    except Exception:
                        pass
                    raise RuntimeError(f"Picamera2 configuration/start failed: {exc}") from exc

                self.camera_type = "picamera2"
                self.camera = camera
                return
            if use_pi_camera and Picamera2 is None:
                raise RuntimeError("USE_PI_CAMERA=1 but picamera2 is not installed/importable.")

            if use_legacy_pi_camera and PiCamera is not None:
                camera = PiCamera()
                camera.resolution = (640, 480)
                camera.framerate = 10
                self.raw_capture = PiRGBArray(camera, size=(640, 480))
                self.camera_type = "picamera"
                self.camera = camera
                return
            if use_legacy_pi_camera and PiCamera is None:
                raise RuntimeError(f"USE_LEGACY_PI_CAMERA=1 but picamera is not installed/importable: {PICAMERA_IMPORT_ERROR}")

            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Camera is not accessible.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera_type = "webcam"
            self.camera = camera

    def read(self):
        with self.lock:
            if self.camera is None:
                self.open()
            if self.camera_type == "picamera2":
                frame = self.camera.capture_array()
                return True, normalize_frame(frame, source_format="rgb")
            if self.camera_type == "picamera":
                self.raw_capture.truncate(0)
                self.camera.capture(self.raw_capture, format="bgr", use_video_port=True)
                frame = self.raw_capture.array
                self.raw_capture.truncate(0)
                return True, normalize_frame(frame, source_format="bgr")
            ret, frame = self.camera.read()
            if ret:
                frame = normalize_frame(frame, source_format="bgr")
            return ret, frame

    def close(self):
        with self.lock:
            if self.camera is None:
                return
            if self.camera_type == "picamera2":
                self.camera.close()
            elif self.camera_type == "picamera":
                self.camera.close()
            else:
                self.camera.release()
            self.camera = None
            self.camera_type = None
            self.raw_capture = None

    def is_open(self):
        with self.lock:
            return self.camera is not None

    def debug_info(self, open_camera=False, close_camera=False):
        with self.lock:
            if close_camera:
                self.close()

            info = {
                "cameraType": self.camera_type,
                "isOpen": self.camera is not None,
                "usePiCamera": os.getenv("USE_PI_CAMERA", "0") == "1",
                "useLegacyPiCamera": os.getenv("USE_LEGACY_PI_CAMERA", "0") == "1",
                "picamera2Importable": Picamera2 is not None,
                "picamera2ImportError": PICAMERA2_IMPORT_ERROR,
                "picameraImportable": PiCamera is not None,
                "picameraImportError": PICAMERA_IMPORT_ERROR,
            }
            if open_camera and self.camera is None:
                try:
                    self.open()
                except Exception as exc:
                    info["openError"] = str(exc)
                info["cameraType"] = self.camera_type
                info["isOpen"] = self.camera is not None

            if self.camera is not None:
                try:
                    ret, frame = self.read()
                    info["readOk"] = ret
                    info["frame"] = describe_array(frame)
                    info["faceRgb"] = describe_array(bgr_to_face_rgb(frame)) if ret else None
                except Exception as exc:
                    info["readError"] = str(exc)
            return info


def encode_jpeg(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Unable to encode frame.")
    return buffer.tobytes()


def create_student(usn):
    clean_usn = usn.strip()
    if not clean_usn:
        raise ValueError("USN is required.")
    if any(part in clean_usn for part in ("/", "\\", "..")):
        raise ValueError("USN cannot contain path characters.")

    os.makedirs(DATASET_DIR, exist_ok=True)
    student_dir = os.path.join(DATASET_DIR, clean_usn)
    if os.path.exists(student_dir):
        raise FileExistsError("USN already exists.")
    os.mkdir(student_dir)
    return clean_usn


def list_students():
    if not os.path.exists(DATASET_DIR):
        return []
    students = []
    for name in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, name)
        if os.path.isdir(path):
            images = [file for file in os.listdir(path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
            students.append({"usn": name, "images": len(images)})
    return students


def save_student_frame(usn, frame):
    student_dir = os.path.join(DATASET_DIR, usn)
    if not os.path.isdir(student_dir):
        raise FileNotFoundError("Create the student before capturing photos.")

    frame = normalize_frame(frame, source_format="bgr")
    rgb = bgr_to_face_rgb(frame)
    boxes = face_locations(rgb, "student photo capture")
    if len(boxes) != 1:
        raise ValueError(f"Expected exactly one face in the frame, found {len(boxes)}.")

    existing = [file for file in os.listdir(student_dir) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_path = os.path.join(student_dir, f"image_{len(existing)}.jpg")
    cv2.imwrite(image_path, frame)
    return image_path
