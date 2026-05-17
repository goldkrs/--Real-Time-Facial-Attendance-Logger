import tkinter as tk
from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
import csv
import os
import pickle
import cv2
import face_recognition
import train_model as tm
from datetime import datetime
from threading import Thread

try:
    import rec_attendance as rc
except Exception:
    class WebcamAttendance:
        def __init__(self):
            self.run_video = True

        def stop_capture(self):
            self.run_video = False

        def rec_entry(self, usn, cur_date_time):
            with open("entry_log.csv", "a", newline="") as file:
                file_write = csv.writer(file)
                file_write.writerow([usn, cur_date_time])
                file.flush()

        def capture_internal(self, frame, data):
            boxes = face_recognition.face_locations(frame)
            encodings = face_recognition.face_encodings(frame, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                if True in matches:
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matched_idxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)
                    self.rec_entry(name, datetime.now())
                    print(name)
                    names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Facial Recognition is Running", frame)

        def capture_attendance(self):
            encodings_path = "encodings.pickle"
            print("[INFO] loading encodings + face detector...")
            data = pickle.loads(open(encodings_path, "rb").read())

            cam = cv2.VideoCapture(0)

            cv2.namedWindow("Facial Recognition is Running", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Facial Recognition is Running", 500, 300)
            cv2.setWindowProperty("Facial Recognition is Running", cv2.WND_PROP_TOPMOST, 1)

            while self.run_video:
                ret, frame = cam.read()
                if not ret:
                    break

                self.capture_internal(frame, data)
                cv2.waitKey(1)

            cam.release()
            cv2.destroyAllWindows()
            self.run_video = True

    rc = WebcamAttendance()


data_dir_name = "./dataset/"


def custom_quit():
    rc.stop_capture()
    quit()


def rec_attendance():
    if r5["text"] == "Capture Attendance":
        r5.configure(text="Stop Capture")
        thread = Thread(target=rc.capture_attendance)
        thread.start()
    else:
        rc.stop_capture()
        r5.configure(text="Capture Attendance")


def student_frame():
    root1 = Toplevel(window)
    root1.geometry("750x250")
    root1.title("Student Entry")
    width = 500
    height = 400
    screen_width = root1.winfo_screenwidth()
    screen_height = root1.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root1.geometry("%dx%d+%d+%d" % (width, height, x, y))
    T = Text(root1, height=1, width=20)
    l = Label(root1, text="USN number")
    l.pack()
    T.pack()
    r2 = tk.Button(
        root1,
        text="Capture",
        command=lambda: capture(T),
        bd=10,
        font=("times new roman", 16),
        bg="black",
        fg="blue",
        height=2,
        width=17,
    )
    r2.pack()


def capture(text_box):
    x = text_box.get("1.0", "end-1c")
    if len(x.strip()) == 0:
        messagebox.showinfo("Warning", "Please Enter USN")
    else:
        os.makedirs(data_dir_name, exist_ok=True)
        if os.path.exists(data_dir_name + x):
            messagebox.showinfo("Warning", "USN already exists")
        else:
            os.mkdir(data_dir_name + x)
            open_camera(data_dir_name, x)


def open_camera(dir_name, name):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            img_name = dir_name + name + "/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


window = Tk()
window.title("Face recognizer")
window.geometry("1280x720")
window.configure(background="black")


def view_attendance():
    new = Toplevel(window)
    new.geometry("750x250")
    new.title("Attendance")
    width = 500
    height = 400
    screen_width = new.winfo_screenwidth()
    screen_height = new.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    new.geometry("%dx%d+%d+%d" % (width, height, x, y))
    new.resizable(0, 0)
    tree = ttk.Treeview(new, show="headings")
    with open("entry_log.csv", "r", newline="") as file:
        csv_reader = csv.reader(file)
        header = ["usn", "time"]
        tree.delete(*tree.get_children())
        tree["columns"] = header

        for col in header:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for row in csv_reader:
            tree.insert("", "end", values=row)

    tree.pack(padx=20, pady=20, fill="both", expand=True)
    status_label = tk.Label(new, text="Attendance", padx=20, pady=10)
    status_label.pack()


a = tk.Label(
    window,
    text="Face Recognition",
    bg="black",
    fg="blue",
    bd=10,
    font=("arial", 35),
)
a.pack()


r2 = tk.Button(
    window,
    text="Capture Student",
    command=student_frame,
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="blue",
    height=2,
    width=17,
)
r2.place(x=600, y=520)

r1 = tk.Button(
    window,
    text="View Attendance",
    command=view_attendance,
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="blue",
    height=2,
    width=17,
)
r1.place(x=1000, y=520)

r4 = tk.Button(
    window,
    text="train photos",
    command=tm.train_photo,
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="blue",
    height=2,
    width=17,
)
r4.place(x=300, y=520)

r3 = tk.Button(
    window,
    text="EXIT",
    bd=10,
    command=custom_quit,
    font=("times new roman", 16),
    bg="black",
    fg="blue",
    height=2,
    width=17,
)
r3.place(x=600, y=660)

r5 = tk.Button(
    window,
    text="Capture Attendance",
    command=rec_attendance,
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="blue",
    height=2,
    width=17,
)
r5.place(x=600, y=320)

window.mainloop()