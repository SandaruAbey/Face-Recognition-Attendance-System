# Face-Recognition-Attendance-System
A modern desktop application for managing employee attendance using facial recognition.

##Features

Face recognition-based attendance tracking
Employee registration system
Voice greetings
Real-time attendance logging
MySQL database integration

##Tech Stack

Python 3.8+
OpenCV
DeepFace
CustomTkinter
MySQL
PyMySQL
gTTS (Google Text-to-Speech)

Installation

Install dependencies:

bashCopypip install -r requirements.txt

Set up MySQL database:

sqlCopyCREATE DATABASE attendance_system;

CREATE TABLE employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE face_encodings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT,
    encoding BLOB NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT,
    type ENUM('entry', 'exit') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

Update database configuration in config.py:

pythonCopyDB_CONFIG = {
    'host': 'your_host',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'attendance_system',
    'port': 3307
}
