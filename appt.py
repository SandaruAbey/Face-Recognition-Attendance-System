import cv2
import numpy as np
from deepface import DeepFace
import pymysql
import time
from datetime import datetime
from gtts import gTTS
from playsound import playsound
import os

# MySQL Configuration
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

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id INT,
                    encoding BLOB,
                    FOREIGN KEY (employee_id) REFERENCES employees(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    type ENUM('entry', 'exit'),
                    FOREIGN KEY (employee_id) REFERENCES employees(id)
                )
            ''')
        conn.commit()
    finally:
        conn.close()

def get_face_embedding(img):
    try:
        cv2.imwrite("temp.jpg", img)
        embeddings = DeepFace.represent(
            img_path="temp.jpg",
            model_name="VGG-Face",  # Ensure consistency
            enforce_detection=False
        )
        os.remove("temp.jpg")
        
        if embeddings and isinstance(embeddings, list) and "embedding" in embeddings[0]:
            embedding_array = np.array(embeddings[0]["embedding"], dtype=np.float32)
            return embedding_array
        else:
            print("No face detected.")
            return None
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        return None

def add_new_employee():
    name = input("Enter employee name: ")
    print("Position face in front of the camera and press 'c' to capture (press 'q' to quit)")
    
    cap = cv2.VideoCapture(0)
    embedding = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        cv2.imshow('Register Face', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Capturing face...")
            embedding = get_face_embedding(frame)
            if embedding is not None:
                print("Face captured successfully!")
                break
            else:
                print("No face detected, try again")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if embedding is not None:
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO employees (name) VALUES (%s)", (name,))
                employee_id = cursor.lastrowid
                
                cursor.execute(
                    "INSERT INTO face_encodings (employee_id, encoding) VALUES (%s, %s)",
                    (employee_id, embedding.tobytes())  # Store as BLOB
                )
            conn.commit()
            print(f"Employee {name} added successfully!")
        except Exception as e:
            conn.rollback()
            print(f"Error adding employee: {e}")
        finally:
            conn.close()

def recognize_face(unknown_embedding):
    if unknown_embedding is None:
        return None, None

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT e.id, e.name, f.encoding 
                FROM employees e 
                JOIN face_encodings f ON e.id = f.employee_id
            """)
            results = cursor.fetchall()
            
            for row in results:
                known_embedding = np.frombuffer(row['encoding'], dtype=np.float32)
                
                if known_embedding.shape == unknown_embedding.shape:
                    distance = np.linalg.norm(known_embedding - unknown_embedding)
                    print(f"Distance: {distance}")  # Debugging
                    if distance < 0.8:  # Adjusted threshold
                        return row['id'], row['name']
                else:
                    print("Embedding shape mismatch, skipping comparison.")
                    
            return None, None
    finally:
        conn.close()

def play_greeting(name, is_morning):
    greeting = f"Good {'morning' if is_morning else 'evening'} {name}"
    print(greeting)
    tts = gTTS(greeting, lang='en')
    tts.save("greeting.mp3")
    playsound("greeting.mp3")
    os.remove("greeting.mp3")

def record_attendance(employee_id, is_morning):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE employee_id = %s 
                AND timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR)
            """, (employee_id,))
            
            if cursor.fetchone() is None:
                attendance_type = 'entry' if is_morning else 'exit'
                cursor.execute(
                    "INSERT INTO attendance (employee_id, type) VALUES (%s, %s)",
                    (employee_id, attendance_type)
                )
                conn.commit()
                return True
            return False
    except Exception as e:
        conn.rollback()
        print(f"Error recording attendance: {e}")
        return False
    finally:
        conn.close()

def main():
    print("Initializing database...")
    init_db()
    
    while True:
        print("\n1. Add new employee")
        print("2. Start attendance system")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            add_new_employee()
        elif choice == '2':
            print("Starting attendance system... (press 'q' to quit)")
            cap = cv2.VideoCapture(0)
            last_recognition_time = 0
            recognition_cooldown = 5  # Seconds between recognitions
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                cv2.imshow('Attendance System', frame)
                
                current_time = time.time()
                if current_time - last_recognition_time >= recognition_cooldown:
                    embedding = get_face_embedding(frame)
                    if embedding is not None:
                        employee_id, name = recognize_face(embedding)
                        if employee_id:
                            now = datetime.now()
                            
                            is_morning = 5 <= now.hour < 17
                            
                            if record_attendance(employee_id, is_morning):
                                play_greeting(name, is_morning)
                            
                            last_recognition_time = current_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
