#! /usr/bin/python

import os
import pickle

import cv2
import face_recognition
from imutils import paths


def train_photo():
    print("[INFO] start processing faces...")
    image_paths = list(paths.list_images("dataset"))

    known_encodings = []
    known_names = []
    skipped = []

    for index, image_path in enumerate(image_paths):
        print("[INFO] processing image {}/{}".format(index + 1, len(image_paths)))
        name = image_path.split(os.path.sep)[-2]

        image = cv2.imread(image_path)
        if image is None:
            skipped.append((image_path, "image_not_readable"))
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")

        if len(boxes) != 1:
            skipped.append((image_path, f"faces_found_{len(boxes)}"))
            continue

        encodings = face_recognition.face_encodings(rgb, boxes)
        if not encodings:
            skipped.append((image_path, "encoding_failed"))
            continue

        known_encodings.append(encodings[0])
        known_names.append(name)

    if not known_encodings:
        raise RuntimeError("No valid training images found. Capture clear single-face photos first.")

    print("[INFO] serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    print(f"[INFO] trained {len(known_encodings)} images for {len(set(known_names))} students")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} images:")
        for image_path, reason in skipped:
            print(f"  - {image_path}: {reason}")
