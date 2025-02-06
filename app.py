import cv2
import numpy as np
import pymysql
import threading
from datetime import datetime
from deepface import DeepFace
from gtts import gTTS
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Database Config
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'attendance_system',
    'port': 3307,
    'cursorclass': pymysql.cursors.DictCursor
}

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lightweight Attendance System")
        self.root.geometry("400x300")
        
        self.cap = None
        self.running = False
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Video label
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(pady=5)

        # Name entry
        self.name_entry = ttk.Entry(main_frame, width=25)
        self.name_entry.insert(0, "Enter name")
        self.name_entry.pack(pady=5)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Add Employee", 
                 command=lambda: self.thread_wrapper('add')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Take Attendance", 
                 command=lambda: self.thread_wrapper('recognize')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Exit", 
                 command=self.cleanup).pack(side=tk.LEFT, padx=2)

    def thread_wrapper(self, mode):
        threading.Thread(target=self.capture_face, args=(mode,), daemon=True).start()

    def capture_face(self, mode):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 320)  # Reduce resolution
            self.cap.set(4, 240)
        
        ret, frame = self.cap.read()
        if ret:
            self.show_frame(frame)
            
            if mode == 'add':
                name = self.name_entry.get()
                if name and name != "Enter name":
                    self.add_employee(name, frame)
            elif mode == 'recognize':
                self.recognize_employee(frame)

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def add_employee(self, name, frame):
        embedding = self.get_face_embedding(frame)
        if embedding is not None:
            try:
                conn = pymysql.connect(**DB_CONFIG)
                with conn.cursor() as cursor:
                    cursor.execute("INSERT INTO employees (name) VALUES (%s)", (name,))
                    employee_id = cursor.lastrowid
                    cursor.execute(
                        "INSERT INTO face_encodings (employee_id, encoding) VALUES (%s, %s)",
                        (employee_id, embedding.tobytes())
                    )
                conn.commit()
                messagebox.showinfo("Success", f"{name} added successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Database error: {e}")
            finally:
                conn.close()

    def recognize_employee(self, frame):
        embedding = self.get_face_embedding(frame)
        if embedding is None:
            return

        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("SELECT e.id, e.name, f.encoding FROM employees e JOIN face_encodings f ON e.id = f.employee_id")
                for row in cursor.fetchall():
                    known_embedding = np.frombuffer(row['encoding'], dtype=np.float32)
                    if known_embedding.shape == embedding.shape:
                        distance = np.linalg.norm(known_embedding - embedding)
                        if distance < 0.8:
                            self.record_attendance(row['id'], row['name'])
                            return
        finally:
            conn.close()

    def get_face_embedding(self, img):
        try:
            return DeepFace.represent(img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def record_attendance(self, employee_id, name):
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO attendance (employee_id, type) VALUES (%s, %s)",
                             (employee_id, 'entry' if 5 <= datetime.now().hour < 17 else 'exit'))
            conn.commit()
            self.play_greeting(name)
        except Exception as e:
            messagebox.showerror("Error", f"Attendance error: {e}")
        finally:
            conn.close()

    def play_greeting(self, name):
        greeting = f"Welcome {name}"
        tts = gTTS(greeting, lang='en')
        tts.save("greeting.mp3")
        self.root.after(100, lambda: os.system("greeting.mp3"))

    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()