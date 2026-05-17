#! /usr/bin/python

import cv2

from attendance_core import AttendanceRecognizer, Camera, ENCODINGS_PATH


run_video = True
recognizer = AttendanceRecognizer()
camera = Camera()


def stop_capture():
    global run_video
    run_video = False


def capture_internal(frame, data=None):
    if data is not None:
        recognizer.data = data
    results = recognizer.process_frame(frame, log_matches=True)
    recognizer.draw_results(frame, results)
    cv2.imshow("Facial Recognition is Running", frame)
    return results


def capture_attendance():
    global run_video

    print("[INFO] loading encodings...")
    try:
        recognizer.reload()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print(f"[INFO] Expected encodings at: {ENCODINGS_PATH}")
        return

    try:
        camera.open()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return

    cv2.namedWindow("Facial Recognition is Running", cv2.WINDOW_NORMAL)
    print("[INFO] Starting camera... Press 'q' or ESC to quit")

    frame_index = 0
    while run_video:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("[ERROR] Frame not captured")
            break

        if frame_index % 2 == 0:
            capture_internal(frame)
        else:
            cv2.imshow("Facial Recognition is Running", frame)
        frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            run_video = False
            break

    camera.close()
    cv2.destroyAllWindows()
    run_video = True
