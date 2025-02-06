import cv2
import numpy as np
import pymysql
import time
from datetime import datetime
from deepface import DeepFace
from gtts import gTTS
from playsound import playsound
import os
import tkinter as tk
from tkinter import messagebox
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

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def add_employee(name, frame):
    embedding = get_face_embedding(frame)
    if embedding is not None:
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO employees (name) VALUES (%s)", (name,))
                employee_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO face_encodings (employee_id, encoding) VALUES (%s, %s)",
                    (employee_id, embedding.tobytes())
                )
            conn.commit()
            messagebox.showinfo("Success", f"Employee {name} added successfully!")
        except Exception as e:
            conn.rollback()
            messagebox.showerror("Error", f"Error adding employee: {e}")
        finally:
            conn.close()

def recognize_face(frame):
    embedding = get_face_embedding(frame)
    if embedding is None:
        return None, None
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT e.id, e.name, f.encoding FROM employees e JOIN face_encodings f ON e.id = f.employee_id")
            results = cursor.fetchall()
            for row in results:
                known_embedding = np.frombuffer(row['encoding'], dtype=np.float32)
                if known_embedding.shape == embedding.shape:
                    distance = np.linalg.norm(known_embedding - embedding)
                    if distance < 0.8:
                        return row['id'], row['name']
            return None, None
    finally:
        conn.close()

def get_face_embedding(img):
    try:
        cv2.imwrite("temp.jpg", img)
        embeddings = DeepFace.represent(img_path="temp.jpg", model_name="VGG-Face", enforce_detection=False)
        os.remove("temp.jpg")
        if embeddings and isinstance(embeddings, list) and "embedding" in embeddings[0]:
            return np.array(embeddings[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"Error getting face embedding: {e}")
    return None

def record_attendance(employee_id, name):
    now = datetime.now()
    is_morning = 5 <= now.hour < 17
    attendance_type = 'entry' if is_morning else 'exit'
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO attendance (employee_id, type) VALUES (%s, %s)", (employee_id, attendance_type))
        conn.commit()
        play_greeting(name, is_morning)
    finally:
        conn.close()

def play_greeting(name, is_morning):
    greeting = f"Good {'morning' if is_morning else 'evening'} {name}"
    tts = gTTS(greeting, lang='en')
    tts.save("greeting.mp3")
    playsound("greeting.mp3")
    os.remove("greeting.mp3")

def capture_face(mode):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)
        label_video.update()
        
        if mode == 'add':
            name = entry_name.get()
            if name:
                add_employee(name, frame)
                break
        elif mode == 'recognize':
            employee_id, name = recognize_face(frame)
            if employee_id:
                record_attendance(employee_id, name)
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# GUI
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("600x400")

title = tk.Label(root, text="Face Recognition Attendance", font=("Arial", 16))
title.pack()

frame_top = tk.Frame(root)
frame_top.pack()

label_video = tk.Label(frame_top)
label_video.pack()

frame_bottom = tk.Frame(root)
frame_bottom.pack()

entry_name = tk.Entry(frame_bottom)
entry_name.pack()
btn_add = tk.Button(frame_bottom, text="Add Employee", command=lambda: capture_face('add'))
btn_add.pack()

btn_recognize = tk.Button(frame_bottom, text="Start Attendance", command=lambda: capture_face('recognize'))
btn_recognize.pack()

root.mainloop()
