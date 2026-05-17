# Real-Time Facial Attendance Logger

A camera-based attendance system that captures student face photos, trains local face encodings, and logs recognized students to `entry_log.csv`.

The frontend is now a React web page served by Flask instead of Tkinter.

## Built With

- Python 3
- Flask
- React
- OpenCV
- face_recognition
- Picamera2 on Raspberry Pi, or a standard webcam
- CSV attendance logs

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Raspberry Pi, set this before running if you want Picamera2 instead of the default webcam:

```bash
set USE_PI_CAMERA=1
```

## Run The Web App

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

Main actions:

- Add a student USN.
- Select the student and capture multiple clear single-face photos.
- Train photos.
- Start attendance capture.
- View attendance entries in the web table or directly in `entry_log.csv`.
- Clear the attendance register from the Attendance panel when you want a fresh CSV.

The older OpenCV window flow still exists for command-line use:

```bash
python rec_attendance.py
```

## Project Structure

- `app.py` - Flask API and React page server
- `static/index.html` - React frontend
- `attendance_core.py` - shared camera, matching, capture, and CSV logic
- `train_model.py` - training pipeline for `dataset/<USN>/image_*.jpg`
- `rec_attendance.py` - command-line attendance capture
- `dataset/` - captured student images
- `encodings.pickle` - generated training data
- `entry_log.csv` - attendance log

## Accuracy Notes

The previous attendance logic could miswrite a student because it accepted every face under the default tolerance and then chose the name with the most matching saved images. That can favor the wrong student when faces are similar or one student has more training photos.

The updated logic uses:

- Best face-distance per student instead of raw majority vote.
- A stricter default tolerance: `FACE_MATCH_TOLERANCE=0.48`.
- A margin check: `FACE_MATCH_MARGIN=0.06`, so close second-place matches are treated as unknown.
- Repeated confirmation before logging: `FACE_REQUIRED_CONFIRMATIONS=2`.
- A cooldown per student: `ATTENDANCE_COOLDOWN_SECONDS=10`.
- Training skips images with zero faces or multiple faces.
- Recognition runs on a background interval while the video stream stays responsive.

You can tune those values with environment variables. If false positives still happen, lower `FACE_MATCH_TOLERANCE` gradually, for example `0.45`.

## Dataset Tips

- Keep one student per folder: `dataset/<USN>/image_0.jpg`.
- Use clear, well-lit, single-face images.
- Avoid masks, heavy side angles, group photos, and photos with another face in the background.
- Capture several photos per student from slightly different angles.
- Re-run training after adding or deleting student photos.

## License

MIT
