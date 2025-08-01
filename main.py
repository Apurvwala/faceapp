import os
import glob
import json
import random
import smtplib
import threading
import time
import base64
import asyncio
import io
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import pandas as pd
import edge_tts
import pygame
from fpdf import FPDF
from apscheduler.schedulers.background import BackgroundScheduler

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.clock import Clock, mainthread
from kivy.properties import StringProperty, BooleanProperty
from kivy.graphics.texture import Texture

# --- Configuration Constants ---
SAMPLES_PER_USER: int = 10  # This constant is not used, but kept for consistency
FRAME_REDUCE_FACTOR: float = 0.5
RECOGNITION_INTERVAL: int = 3 * 60  # 3 minutes cooldown

# Path to Haar cascade in data/
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")

# Replace this with android.storage.app_storage_path() when on Android
try:
    from android.storage import app_storage_path
    BASE_DIR = app_storage_path()
except ImportError:
    BASE_DIR = os.getcwd()

# Ensure known_faces folder is under app storage
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Google Sheet URLs and form URLs (update with your actual URLs)
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1ghgpG1z4ugXpu4cfOZjkPfPh-7oAQvOZhVRz2XHfot0/export?format=csv"
)
GOOGLE_FORM_VIEW_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
GOOGLE_FORM_POST_URL: str = "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"

FORM_FIELDS: Dict[str, str] = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

EMAIL_ADDRESS: str = os.environ.get("FACEAPP_EMAIL", "faceapp0011@gmail.com")
EMAIL_PASSWORD: str = os.environ.get("FACEAPP_PASS", "ytup bjrd pupf tuuj")
SMTP_SERVER: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ADMIN_EMAIL_ADDRESS: str = os.environ.get(
    "FACEAPP_ADMIN_EMAIL", "projects@archtechautomation.com"
)


def Logger(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def python_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _crop_and_resize_for_passport(
    cv_image: np.ndarray, target_size: Tuple[int, int] = (240, 320)
) -> np.ndarray:
    """Crop and resize an image to a target aspect ratio and size."""
    h, w = cv_image.shape[:2]
    target_width, target_height = target_size

    target_aspect_ratio = target_width / target_height
    current_aspect_ratio = w / h

    if current_aspect_ratio > target_aspect_ratio:
        new_width = int(h * target_aspect_ratio)
        x_start = (w - new_width) // 2
        cropped_image = cv_image[:, x_start : x_start + new_width]
    elif current_aspect_ratio < target_aspect_ratio:
        new_height = int(w / target_aspect_ratio)
        y_start = (h - new_height) // 2
        cropped_image = cv_image[y_start : y_start + new_height, :]
    else:
        cropped_image = cv_image

    return cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)


class ComplimentGenerator:
    def __init__(self):
        self.user_compliment_history: Dict[str, Dict[str, List[str]]] = {}  # emp_id -> history
        self.compliment_template = [
            "ThankYou!, {name}!",
            "{name}, your positivity is contagious!",
            "{name}, your smile brightens the office!",
            "Great to see you, {name}—looking professional as always!",
            "{name}, you're making a great impression today!",
            "Impressive look, {name}. Keep it up!",
            "{name}, your energy sets a fantastic example!",
        ]

    def get_daily_compliment(self, emp_id: str, name: str) -> str:
        today = date.today().isoformat()
        if emp_id not in self.user_compliment_history:
            self.user_compliment_history[emp_id] = {"dates": [], "compliments": []}

        # Remove compliments from previous days
        idx_to_remove = [
            i
            for i, d in enumerate(self.user_compliment_history[emp_id]["dates"])
            if d != today
        ]
        for i in reversed(idx_to_remove):
            self.user_compliment_history[emp_id]["dates"].pop(i)
            self.user_compliment_history[emp_id]["compliments"].pop(i)

        # Exclude compliments already used today
        available = [
            c
            for c in self.compliment_template
            if c not in self.user_compliment_history[emp_id]["compliments"]
        ]

        if not available:
            available = self.compliment_template.copy()
            self.user_compliment_history[emp_id]["compliments"].clear()

        compliment = random.choice(available)
        self.user_compliment_history[emp_id]["dates"].append(today)
        self.user_compliment_history[emp_id]["compliments"].append(compliment)
        return compliment.format(name=name)


class EdgeTTSHelper:
    """Helper class for instant Edge-TTS with a slightly slower cute female voice"""

    def __init__(self):
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=256)
        pygame.mixer.init()
        self.selected_voice = "en-IN-NeerjaNeural"
        Logger(f"[INFO] EdgeTTS initialized with voice: {self.selected_voice}")

    def speak(self, text: str) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_speak(text))
            loop.close()
            Logger(f"[INFO] INSTANTLY spoke: {text}")
        except Exception as e:
            Logger(f"[ERROR] Instant TTS error: {e}")

    async def _async_speak(self, text: str) -> None:
        try:
            communicate = edge_tts.Communicate(text=text, voice=self.selected_voice, rate="+8%", pitch="+10Hz")
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if audio_data:
                audio_buffer = io.BytesIO(audio_data)
                pygame.mixer.music.load(audio_buffer, "mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                Logger("[INFO] Audio playing INSTANTLY")
        except Exception as e:
            Logger(f"[ERROR] Edge-TTS async speak error: {e}")


class FaceAppBackend:
    def __init__(self, ui_instance=None):
        self.ui_instance = ui_instance
        self.known_faces_dir: str = KNOWN_FACES_DIR
        ensure_dir(self.known_faces_dir)
        Logger(f"[INFO] Known faces directory set to: {self.known_faces_dir}")

        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            Logger(f"[WARN] Failed to load Haar cascade from '{HAAR_CASCADE_PATH}'. Attempting fallback.")
            fallback_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(fallback_path)
            if self.face_cascade.empty():
                raise RuntimeError("Cannot load Haar cascade classifier.")
            Logger(f"[INFO] Successfully loaded Haar cascade from fallback path: '{fallback_path}'.")

        self.recognizer = None
        self.label_map = {}
        # Thread-safe dictionaries for concurrent access
        self.last_seen_time: Dict[str, float] = {}
        self.daily_first_recognition: Dict[str, str] = {}  # emp_id -> date string
        self.otp_storage: Dict[str, str] = {}
        self.pending_names: Dict[str, Optional[str]] = {}
        self.user_emails: Dict[str, str] = {}
        self.daily_attendance_status: Dict[str, str] = {}
        self.last_recognized_info: Dict[str, Any] = {}
        
        self.capture_mode: bool = False
        self.capture_target_count: int = 0
        self.capture_collected_count: int = 0
        self.capture_name: Optional[str] = None
        self.capture_emp_id: Optional[str] = None
        self.capture_start_index: int = 0
        self.capture_lock = threading.Lock() # Good for capture state

        self.tts_helper = EdgeTTSHelper()
        self.compliment_gen = ComplimentGenerator()
        self.user_gender: Dict[str, str] = self._load_user_genders()
        self.last_unknown_greeting_time: float = 0.0 # New attribute to track last unknown greeting

        self._train_recognizer_and_load_emails()
        self.daily_attendance_status = self._load_daily_attendance_status()

    def _load_user_genders(self) -> Dict[str, str]:
        gender_map = {}
        try:
            gender_file = Path(self.known_faces_dir) / "user_genders.json"
            if gender_file.is_file():
                with gender_file.open("r", encoding="utf-8") as f:
                    gender_map = json.load(f)
        except Exception as e:
            Logger(f"[WARN] Could not load user_genders.json: {e}")
        return gender_map

    def _save_user_genders(self) -> None:
        try:
            with (Path(self.known_faces_dir) / "user_genders.json").open("w", encoding="utf-8") as f:
                json.dump(self.user_gender, f, indent=2)
            Logger(f"[INFO] Saved user_genders.json with {len(self.user_gender)} entries.")
        except IOError as exc:
            Logger(f"[ERROR] Could not save user_genders.json: {exc}")

    def save_user_gender(self, emp_id: str, gender: str) -> None:
        self.user_gender[emp_id] = gender
        self._save_user_genders()

    def get_honorific(self, emp_id: str) -> str:
        gender = self.user_gender.get(emp_id, None)
        if gender == "female":
            return "ma'am"
        else:
            return "sir"

    def _train_recognizer_and_load_emails(self):
        # Ensure recognizer and label_map are updated after training
        self.recognizer, self.label_map = self._train_recognizer()
        self.user_emails = self._load_emails()

    def _train_recognizer(self):
        images: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[int, Tuple[str, str]] = {}
        label_id = 0

        ensure_dir(self.known_faces_dir)

        files = sorted(os.listdir(self.known_faces_dir))
        for file in files:
            if not file.lower().endswith((".jpg", ".png")):
                continue
            try:
                # Assuming filename format is "Name_EMP_ID_NNN.jpg"
                file_name_no_ext = file[:-4]
                parts = file_name_no_ext.split("_")
                if len(parts) < 3: # Handle cases where filename format might be off
                    Logger(f"[WARN] Skipping unrecognized filename format: {file}")
                    continue
                # Join parts for name in case name contains underscores
                name_parts = parts[:-2]
                name = " ".join(name_parts)
                emp_id = parts[-2].upper()
            except Exception as e:
                Logger(f"[WARN] Skipping unrecognized filename format: {file} due to error: {e}")
                continue

            img_path = Path(self.known_faces_dir) / file
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                Logger(f"[WARN] Could not read image: {img_path}")
                continue

            # Apply Histogram Equalization for better contrast and lighting robustness
            # This is a key improvement for face recognition performance
            img_eq = cv2.equalizeHist(img_gray)
            img_resized = cv2.resize(img_eq, (200, 200)) # Consistent size for training

            current_identity = (name.lower(), emp_id)

            # Assign new label_id if identity is new
            found_existing = False
            for k, v in label_map.items():
                if v == current_identity:
                    labels.append(k)
                    found_existing = True
                    break
            if not found_existing:
                label_map[label_id] = current_identity
                labels.append(label_id)
                label_id += 1
            
            images.append(img_resized)

        recognizer = cv2.face.LBPHFaceRecognizer_create() # Always attempt to create
        if images:
            try:
                recognizer.train(images, np.array(labels))
                Logger(f"[INFO] Trained recognizer on {len(images)} images across {len(label_map)} identities.")
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV training error: {e}. Recognizer might be partially trained or invalid.")
                recognizer = None # Invalidate recognizer if training fails
        else:
            Logger("[INFO] No images found – recognizer disabled.")
            recognizer = None

        final_label_map = {idx: identity for idx, identity in enumerate(sorted(label_map.values(), key=lambda x: x[1]))}

        return recognizer, label_map # Return the label_map as constructed

    def _load_emails(self) -> Dict[str, str]:
        emails_file = Path(self.known_faces_dir) / "user_emails.json"
        if emails_file.is_file():
            try:
                with emails_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                Logger(f"[ERROR] Could not read user_emails.json: {exc}")
        return {}

    def _save_email(self, emp_id: str, email: str) -> None:
        self.user_emails[emp_id] = email
        try:
            with (Path(self.known_faces_dir) / "user_emails.json").open("w", encoding="utf-8") as f:
                json.dump(self.user_emails, f, indent=2)
            Logger(f"[INFO] Saved user_emails.json with {len(self.user_emails)} entries.")
        except IOError as exc:
            Logger(f"[ERROR] Could not save user_emails.json: {exc}")

    def _load_daily_attendance_status(self) -> Dict[str, str]:
        attendance_file = Path(self.known_faces_dir) / "daily_attendance.json"
        if attendance_file.is_file():
            try:
                with attendance_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                Logger(f"[ERROR] Could not read daily_attendance.json: {exc}")
        return {}

    def _save_daily_attendance_status(self) -> None:
        try:
            with (Path(self.known_faces_dir) / "daily_attendance.json").open("w", encoding="utf-8") as f:
                json.dump(self.daily_attendance_status, f, indent=2)
            Logger(f"[INFO] Saved daily_attendance.json with {len(self.daily_attendance_status)} entries.")
        except IOError as exc:
            Logger(f"[ERROR] Could not save daily_attendance.json: {exc}")

    def _generate_otp(self) -> str:
        return str(random.randint(100000, 999999))

    def _send_email(
        self,
        recipient_email: str,
        subject: str,
        body_html: str,
        image_data: Optional[bytes] = None,
        image_cid: Optional[str] = None,
        pdf_data: Optional[bytes] = None,
        pdf_filename: Optional[str] = None,
    ) -> bool:
        msg = MIMEMultipart("related")
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body_html, "html"))

        if image_data and image_cid:
            image = MIMEImage(image_data, "jpeg")
            image.add_header("Content-ID", f"<{image_cid}>")
            image.add_header("Content-Disposition", "inline", filename="face.jpg")
            msg.attach(image)

        if pdf_data and pdf_filename:
            pdf_part = MIMEApplication(pdf_data, _subtype="pdf")
            pdf_part.add_header("Content-Disposition", "attachment", filename=pdf_filename)
            msg.attach(pdf_part)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            Logger(f"[INFO] Email successfully sent to {recipient_email}")
            return True
        except Exception as exc:
            Logger(f"[ERROR] SMTP email send error: {exc}")
            return False

    def _send_otp_email(self, email: str, otp: str, name: str, emp_id: str, is_admin_email: bool = False) -> bool:
        if is_admin_email:
            subject = f"FaceApp Notification: Person Details - {name.title()} ({emp_id})"
            body_html = f"""
                Name: {name.title()}<br/>
                Employee ID: {emp_id}<br/>
                This is an admin notification for OTP generation.
            """
        else:
            subject = f"FaceApp: Your OTP for {name.title()} ({emp_id})"
            body_html = f"""
                Employee ID: {emp_id}<br/>
                Email: {email}<br/>
                Generated OTP: {otp}<br/><br/>
                Dear {name.title()},<br/>
                Your OTP is <b>{otp}</b>. It is valid for 10 minutes.<br/>
                Please use this OTP to proceed with your photo update or registration.
            """

        return self._send_email(email, subject, body_html)

    def _send_attendance_email(
        self,
        email: str,
        name: str,
        emp_id: str,
        detection_time: str,
        email_type: str,
        face_image_b64: str,
    ) -> None:
        current_date_display = datetime.now().strftime("%B %d, %Y")
        image_data = base64.b64decode(face_image_b64)
        image_cid = "face_detection_image"
        image_html = f'<br/><img src="cid:{image_cid}" alt="Face Image"/>'

        compliment = self.compliment_gen.get_daily_compliment(emp_id, name)

        if email_type == "in":
            subject = f"FaceApp Attendance: In-Time Recorded for {name.title()} ({emp_id})"
            body_html = f"""
                In-time recorded successfully. Welcome!{image_html}<br/>
                Date: {current_date_display}<br/>
                In-Time: {detection_time}<br/>
                {compliment}<br/>
                Thank you for being part of our team!
            """
        elif email_type == "out":
            subject = f"FaceApp Attendance: Out-Time Recorded for {name.title()} ({emp_id})"
            body_html = f"""
                Out-time recorded. Take care!{image_html}<br/>
                Date: {current_date_display}<br/>
                Out-Time: {detection_time}<br/>
                Thank you for your support today. Have a great evening!
            """
        else:
            Logger(f"[ERROR] Invalid email_type '{email_type}'")
            return

        sent = self._send_email(email, subject, body_html, image_data, image_cid)
        if sent:
            Logger(f"[INFO] Attendance email sent to {email} ({email_type})")
        else:
            Logger(f"[ERROR] Failed to send attendance email to {email}")

    def _submit_to_google_form(self, name: str, emp_id: str) -> None:
        payload = {
            FORM_FIELDS["name"]: name.title(),
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: datetime.now().strftime("%d/%m/%Y"),
            FORM_FIELDS["time"]: datetime.now().strftime("%H:%M:%S"),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceApp Attendance Bot)",
            "Referer": GOOGLE_FORM_VIEW_URL,
        }
        try:
            with requests.Session() as session:
                resp = session.post(GOOGLE_FORM_POST_URL, data=payload, headers=headers, timeout=10)
                if resp.status_code in (200, 302):
                    Logger("[INFO] Attendance submitted successfully to Google Form.")
                else:
                    Logger(f"[WARN] Google Form submission returned status {resp.status_code}")
        except requests.exceptions.RequestException as exc:
            Logger(f"[ERROR] Google Form submission error: {exc}")

    def _handle_successful_recognition(self, name: str, emp_id: str, face_roi_color: np.ndarray) -> None:
        Logger(f"[INFO] Handling attendance for {name} ({emp_id})")
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time_str = datetime.now().strftime("%H:%M:%S")
        user_email = self.user_emails.get(emp_id)
        honorific = self.get_honorific(emp_id)
        processed_face_image = _crop_and_resize_for_passport(face_roi_color, (240, 320))
        _, buffer = cv2.imencode(".jpg", processed_face_image)
        face_image_b64 = base64.b64encode(buffer).decode("utf-8")

        with threading.Lock(): 
            first_time_today = self.daily_first_recognition.get(emp_id, None)
            is_in_time = first_time_today != current_date

            if is_in_time:
                self.daily_first_recognition[emp_id] = current_date

        greetings = [
            f"Oho! Good morning {name} {honorific}",
            f"Welcome, {name} {honorific}",
            f"All set, {name} {honorific}?",
            f"Nice to see you, {name} {honorific}",
            f"Hello, {name} {honorific}",
            f"Hey!, {name} {honorific}",
            f"Morning, {name} {honorific}",
            f"Ready to go, {name} {honorific}?",
            f"Good to have you, {name} {honorific}",
            f"Checked in, {name} {honorific}",
        ]

        farewells = [
            f"Bye bye {name} {honorific}",
            f"Take care, {name} {honorific}",
            f"See you, {name} {honorific}",
            f"Goodbye, {name} {honorific}",
            f"See you soon, {name} {honorific}",
            f"Catch you later, {name} {honorific}",
            f"All done, {name} {honorific}",
            f"Peace out, {name} {honorific}",
            f"Till next time, {name} {honorific}",
            f"Logging out, {name} {honorific}",
        ]

        greeting_text = random.choice(greetings if is_in_time else farewells)
        threading.Thread(target=self.tts_helper.speak, args=(greeting_text,), daemon=True).start()
        Logger(f"[INFO] Played greeting sound: {greeting_text}")

        if user_email:
            if is_in_time:
                with threading.Lock():
                    self.daily_attendance_status[emp_id] = current_date
                    self._save_daily_attendance_status()
                threading.Thread(
                    target=self._send_attendance_email,
                    args=(user_email, name, emp_id, current_time_str, "in", face_image_b64),
                    daemon=True,
                ).start()
            else:
                threading.Thread(
                    target=self._send_attendance_email,
                    args=(user_email, name, emp_id, current_time_str, "out", face_image_b64),
                    daemon=True,
                ).start()
        else:
            Logger(f"[WARN] No email found for {name} ({emp_id})")

        threading.Thread(target=self._submit_to_google_form, args=(name, emp_id), daemon=True).start()

        with threading.Lock():
            self.last_recognized_info = {
                "name": name.title(),
                "emp_id": emp_id,
                "time": current_time_str,
                "image": face_image_b64,
                "greeting": greeting_text,
            }
        
        # Update UI with recognition info
        if self.ui_instance:
            self.ui_instance.update_recognition_info(self.last_recognized_info)

    def process_frame(self, frame_data: np.ndarray) -> Dict[str, Any]:
        try:
            frame = frame_data
            if frame is None:
                return {"status": "error", "message": "Invalid image data"}

            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (int(w * FRAME_REDUCE_FACTOR), int(h * FRAME_REDUCE_FACTOR)))
            gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
            results = []

            known_face_detected_in_frame = False

            for (x, y, w_s, h_s) in faces:
                x_full, y_full, w_full, h_full = [int(v / FRAME_REDUCE_FACTOR) for v in (x, y, w_s, h_s)]
                # Expanding ROI slightly for better recognition context
                expansion_factor = 1.8
                exp_w = int(w_full * expansion_factor)
                exp_h = int(h_full * expansion_factor)
                center_x = x_full + w_full // 2
                center_y = y_full + h_full // 2
                exp_x = max(0, center_x - exp_w // 2)
                exp_y = max(0, center_y - exp_h // 2)
                frame_h, frame_w = frame.shape[:2]

                # Ensure expanded coordinates stay within frame boundaries
                exp_x = min(exp_x, frame_w - exp_w)
                exp_y = min(exp_y, frame_h - exp_h)
                exp_w = min(exp_w, frame_w - exp_x)
                exp_h = min(exp_h, frame_h - exp_y)

                if exp_w <= 0 or exp_h <= 0:
                    continue

                color_face_roi = frame[exp_y : exp_y + exp_h, exp_x : exp_x + exp_w].copy()
                grayscale_face_roi = cv2.cvtColor(color_face_roi, cv2.COLOR_BGR2GRAY)
                
                # Apply Histogram Equalization to the grayscale ROI for better recognition
                grayscale_face_roi_eq = cv2.equalizeHist(grayscale_face_roi)


                name = "unknown"
                emp_id = ""
                conf = 1000 # Initialize with a high confidence for unknown

                if self.recognizer:
                    try:
                        # Resize for prediction to match training size
                        face_for_prediction = cv2.resize(grayscale_face_roi_eq, (200, 200))
                        label, conf = self.recognizer.predict(face_for_prediction)
                        # Check if the predicted label exists in label_map
                        if label in self.label_map:
                            name, emp_id = self.label_map[label]
                        else:
                            Logger(f"[WARN] Predicted label {label} not found in label_map. Treating as unknown.")
                            name = "unknown"
                            emp_id = ""
                            conf = 1000 # Set confidence high for unknown if label not found
                    except Exception as e:
                        Logger(f"[ERROR] Recognizer prediction failed: {e}. Treating as unknown.")
                        name = "unknown"
                        emp_id = ""
                        conf = 1000 # If prediction fails, treat as unknown

                face_info = {
                    "box": [x_full, y_full, w_full, h_full],
                    "name": name.title(),
                    "emp_id": emp_id,
                    "confidence": float(conf),
                    "status": "unknown",
                }

                # Handle capture mode
                if self.capture_mode:
                    with self.capture_lock:
                        if self.capture_collected_count < self.capture_target_count:
                            # Use the equalized grayscale ROI for capturing
                            face_img_resized = cv2.resize(grayscale_face_roi_eq, (200, 200))
                            filename = f"{self.capture_name}_{self.capture_emp_id}_{self.capture_start_index + self.capture_collected_count:03d}.jpg"
                            cv2.imwrite(str(Path(self.known_faces_dir) / filename), face_img_resized)
                            self.capture_collected_count += 1
                            Logger(f"[INFO] Captured sample {self.capture_collected_count}/{self.capture_target_count}")
                            face_info["capture_progress"] = f"{self.capture_collected_count}/{self.capture_target_count}"
                            face_info["status"] = "capturing"
                            if self.ui_instance:
                                self.ui_instance.update_capture_status(self.capture_collected_count, self.capture_target_count)
                            time.sleep(0.5)  # Delay to prevent camera auto-off

                        if self.capture_collected_count >= self.capture_target_count:
                            Logger("[INFO] Capture complete – retraining recognizer.")
                            self.capture_mode = False
                            threading.Thread(target=self._retrain_after_capture, daemon=True).start()
                            threading.Thread(target=self.tts_helper.speak, args=("Registration process is done",), daemon=True).start()
                            face_info["status"] = "capture_complete"
                            if self.ui_instance:
                                self.ui_instance.on_capture_complete()
                        results.append(face_info)
                        continue # Skip recognition for faces actively being captured

                # Recognition logic for non-capture mode
                if conf < 60: # Threshold for known faces (adjust this value if needed)
                    known_face_detected_in_frame = True
                    now = time.time()
                    with threading.Lock():
                        last_seen = self.last_seen_time.get(emp_id, 0)
                        if now - last_seen > RECOGNITION_INTERVAL:
                            self.last_seen_time[emp_id] = now
                            face_info["status"] = "recognized_new"
                            threading.Thread(
                                target=self._handle_successful_recognition,
                                args=(name, emp_id, color_face_roi),
                                daemon=True,
                            ).start()
                        else:
                            face_info["status"] = "recognized_recent"
                else:
                    face_info["status"] = "unknown"
                    
                results.append(face_info)
            
            if not known_face_detected_in_frame and len(faces) > 0: 
                now = time.time()
                with threading.Lock(): 
                    if now - self.last_unknown_greeting_time > RECOGNITION_INTERVAL: 
                        self.last_unknown_greeting_time = now
                        greeting_unknown_text = "Hello! Welcome To Arc htech, I am Nova voice assistant, how can I help you?"
                        threading.Thread(target=self.tts_helper.speak, args=(greeting_unknown_text,), daemon=True).start()
                        Logger(f"[INFO] Played greeting sound for unknown person: {greeting_unknown_text}")

            return {"status": "success", "faces": results}
        except Exception as e:
            Logger(f"[ERROR] Error processing frame: {e}")
            return {"status": "error", "message": str(e)}

    def _retrain_after_capture(self):
        self.recognizer, self.label_map = self._train_recognizer()
        Logger("[INFO] Recognizer retraining finished.")
        if self.ui_instance:
            self.ui_instance.on_retraining_complete()

    def start_capture_samples(self, name: str, emp_id: str, updating=False, sample_count=None) -> Dict[str, Any]:
        with self.capture_lock:
            if self.capture_mode:
                return {"status": "error", "message": "Already in capture mode."}

            resolved_name = name

            if updating:
                if not resolved_name:
                    found_name = next(
                        (nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id),
                        None,
                    )
                    if found_name:
                        resolved_name = found_name
                    else:
                        return {"status": "error", "message": "No existing face found for this ID for update. Name required for new registration."}
            else: # New registration
                if not resolved_name:
                    return {"status": "error", "message": "Name is required for new registration."}


            self.capture_name = resolved_name
            self.capture_emp_id = emp_id
            self.capture_target_count = sample_count if sample_count is not None else 5
            self.capture_collected_count = 0

            pattern = str(Path(self.known_faces_dir) / f"{self.capture_name}_{self.capture_emp_id}_*.jpg")
            existing_files = glob.glob(pattern)
            self.capture_start_index = len(existing_files)

            self.capture_mode = True
            Logger(f"[INFO] Starting sample capture for {emp_id} (Name: {resolved_name}) – target {self.capture_target_count} faces starting from index {self.capture_start_index}")
            return {"status": "success", "message": "Capture mode initiated."}

    def stop_capture_samples(self) -> Dict[str, Any]:
        with self.capture_lock:
            if not self.capture_mode:
                return {"status": "error", "message": "Not in capture mode."}
            self.capture_mode = False
            Logger("[INFO] Sample capture stopped.")
            return {"status": "success", "message": "Capture mode stopped."}

    def get_user_email(self, emp_id: str) -> Dict[str, Any]:
        email = self.user_emails.get(emp_id)
        name = next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), None)
        return {"status": "success", "email": email, "name": name}

    def send_otp_flow(self, emp_id: str, email: str, name: Optional[str] = None) -> Dict[str, Any]:
        resolved_name = (
            name or next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), "Unknown User")
        )

        otp = self._generate_otp()
        with threading.Lock():
            self.otp_storage[emp_id] = otp
            self.pending_names[emp_id] = resolved_name

        def _send_thread():
            user_mail_ok = self._send_otp_email(email, otp, resolved_name, emp_id, False)
            admin_mail_ok = self._send_otp_email(ADMIN_EMAIL_ADDRESS, otp, resolved_name, emp_id, True)
            if not user_mail_ok:
                Logger(f"[WARN] Failed to send OTP email to user {email}")
            if not admin_mail_ok:
                Logger(f"[WARN] Failed to send admin notification email")

        threading.Thread(target=_send_thread, daemon=True).start()
        return {"status": "success", "message": "OTP sending initiated."}

    def verify_otp(self, emp_id: str, otp_entered: str) -> Dict[str, Any]:
        with threading.Lock():
            if self.otp_storage.get(emp_id) == otp_entered:
                del self.otp_storage[emp_id]
                return {"status": "success", "message": "OTP verified successfully."}
            else:
                return {"status": "error", "message": "Incorrect OTP."}

    def register_user_email(self, emp_id: str, email: str) -> Dict[str, Any]:
        self._save_email(emp_id, email)
        return {"status": "success", "message": "Email registered."}

    def get_last_recognized_info(self) -> Dict[str, Any]:
        with threading.Lock():
            info = self.last_recognized_info
            if info:
                self.last_recognized_info = {}
                return {"status": "success", "info": info}
            return {"status": "no_new_info"}

    def generate_and_send_monthly_reports(self):
        Logger("[INFO] Starting monthly attendance report generation and emailing...")
        try:
            # 1. Download the Google Sheet CSV
            resp = requests.get(GOOGLE_SHEET_CSV_URL, timeout=30)
            if resp.status_code != 200:
                Logger(f"[ERROR] Failed to download attendance sheet CSV: HTTP {resp.status_code}")
                return

            csv_data = resp.content.decode("utf-8")
            Logger("[INFO] CSV data downloaded successfully.")

            # 2. Read CSV into DataFrame
            df = pd.read_csv(io.StringIO(csv_data))
            Logger(f"[INFO] DataFrame loaded with {len(df)} rows.")

            # Expected columns - adapt as per your sheet's columns
            required_cols = ["Name", "Employee Id", "Date", "Time"]
            for col in required_cols:
                if col not in df.columns:
                    Logger(f"[ERROR] Attendance CSV missing required column: {col}. Available columns: {df.columns.tolist()}")
                    return

            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
            Logger(f"[INFO] Dates after initial parsing (first 5 rows of Date column):\n{df['Date'].head()}")

            def safe_parse_time(t):
                try:
                    return pd.to_datetime(str(t), format="%H:%M:%S", errors="coerce").time()
                except Exception:
                    return None

            df["Time"] = df["Time"].apply(safe_parse_time)
            Logger(f"[INFO] Times after initial parsing (first 5 rows of Time column):\n{df['Time'].head()}")

            initial_rows = len(df)
            df = df.dropna(subset=["Date", "Time"])
            Logger(f"[INFO] DataFrame after dropping invalid dates/times: {len(df)} rows (dropped {initial_rows - len(df)} rows).")

            # 3. Determine previous month's date range
            today = datetime.now().date()
            first_day_this_month = today.replace(day=1)
            last_day_last_month = first_day_this_month - timedelta(days=1)
            first_day_last_month = last_day_last_month.replace(day=1)

            df_prev_month = df[
                (df["Date"].dt.date >= first_day_last_month) & (df["Date"].dt.date <= last_day_last_month)
            ].copy()
            Logger(f"[INFO] Filtered previous month's data: {len(df_prev_month)} rows.")

            if df_prev_month.empty:
                Logger("[WARN] No attendance data found for the previous month. Skipping report generation.")
                return

            df_prev_month = df_prev_month.sort_values(by=["Employee Id", "Date", "Time"])
            Logger("[INFO] Previous month's data sorted.")

            attendance_summary = df_prev_month.groupby(['Employee Id', df_prev_month['Date'].dt.date])['Time'].agg(
                in_time=('min'),
                out_time=('max')
            ).reset_index()

            attendance_summary = attendance_summary.rename(columns={'Date': 'AttendanceDate'})
            attendance_summary['Employee Id'] = attendance_summary['Employee Id'].astype(str)
            Logger(f"[INFO] Attendance summary created with {len(attendance_summary)} entries.")

            full_dates = pd.date_range(start=first_day_last_month, end=last_day_last_month).date

            all_emp_ids = attendance_summary['Employee Id'].unique()
            Logger(f"[INFO] Found {len(all_emp_ids)} unique employees for reporting.")

            for emp_id in all_emp_ids:
                name = next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), None)

                if not name:
                    df_names = df_prev_month[df_prev_month["Employee Id"] == emp_id]["Name"].unique()
                    if len(df_names) > 0 and pd.notnull(df_names[0]):
                        name = df_names[0]
                    else:
                        name = "Unknown User"

                Logger(f"[INFO] Processing report for employee {emp_id} (Name: {name}, Email: {self.user_emails.get(emp_id, 'N/A')}).")

                email = self.user_emails.get(emp_id)
                if not email:
                    Logger(f"[WARN] No email found for {name} ({emp_id}), skipping report email.")
                    continue

                report_rows = []
                user_attendance_summary = attendance_summary[attendance_summary['Employee Id'] == emp_id]
                total_present_days = 0

                for single_date in full_dates:
                    daily_entry = user_attendance_summary[user_attendance_summary['AttendanceDate'] == single_date]
                    
                    in_time_str = ""
                    out_time_str = ""

                    if not daily_entry.empty:
                        total_present_days += 1
                        in_time_val = daily_entry['in_time'].iloc[0]
                        out_time_val = daily_entry['out_time'].iloc[0]
                        
                        if pd.notnull(in_time_val):
                            in_time_str = in_time_val.strftime('%H:%M:%S')
                        
                        if pd.notnull(out_time_val) and in_time_val != out_time_val:
                            out_time_str = out_time_val.strftime('%H:%M:%S')
                        
                    report_rows.append(
                        {
                            "Date": single_date.strftime("%Y-%m-%d"),
                            "In Time": in_time_str,
                            "Out Time": out_time_str,
                        }
                    )

                pdf = FPDF()
                pdf.add_page()
                logo_left_path = Path(self.known_faces_dir) / "nextgen.png"
                logo_right_path = Path(self.known_faces_dir) / "logo.jpg"
                logo_w = 26
                logo_h = 11.87

                if logo_left_path.is_file():
                    pdf.image(str(logo_left_path), x=11, y=8, w=logo_w, h=logo_h)
                else:
                    Logger(f"[WARN] Left logo not found at {logo_left_path}")

                if logo_right_path.is_file():
                    pdf.image(str(logo_right_path), x=pdf.w - logo_w - 10, y=8, w=logo_w, h=logo_h)
                else:
                    Logger(f"[WARN] Right logo not found at {logo_right_path}")

                pdf.set_font("Arial", "B", 18)
                pdf.set_text_color(0, 0, 128)
                pdf.cell(0, 12, f"Official Attendance Log {name}", ln=True, align="C")
                pdf.ln(4)
                pdf.set_font("Arial", "", 12)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 8, f"Employee ID: {emp_id}", ln=True, align="C")
                pdf.cell(0, 8, f"Period: {first_day_last_month.strftime('%B %Y')}", ln=True, align="C")
                pdf.ln(12)

                pdf.set_fill_color(200, 220, 255)
                pdf.set_font("Arial", "B", 12)
                col_width_date = 63.34
                col_width_in = 63.33
                col_width_out = 50

                pdf.cell(col_width_date, 10, "Date", border=1, fill=True, align="C")
                pdf.cell(col_width_in, 10, " Check-In Time", border=1, fill=True, align="C")
                pdf.cell(col_width_out, 10, "Check-Out Time", border=1, fill=True, align="C")
                pdf.ln()

                pdf.set_font("Arial", "", 12)
                for row in report_rows:
                    pdf.cell(col_width_date, 10, row["Date"], border=1)
                    pdf.cell(col_width_in, 10, row["In Time"], border=1)
                    pdf.cell(col_width_out, 10, row["Out Time"], border=1)
                    pdf.ln()

                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Attendance Count: {total_present_days}", ln=True, align="L")

                pdf_output = pdf.output(dest="S").encode("latin1")
                Logger(f"[INFO] PDF generated for {name} ({emp_id}). Attempting to send email.")

                subject = f"Attendance Report for {name} - {first_day_last_month.strftime('%B %Y')}"
                body_html = f"""
                    Dear {name},<br/><br/>
                    Please find attached your attendance report for {first_day_last_month.strftime('%B %Y')}.<br/><br/>
                    Best regards,<br/>
                    FaceApp Attendance System
                """

                send_ok = self._send_email(
                    email,
                    subject,
                    body_html,
                    pdf_data=pdf_output,
                    pdf_filename=f"Attendance_Report_{first_day_last_month.strftime('%Y_%m')}.pdf",
                )
                if send_ok:
                    Logger(f"[INFO] Attendance report emailed successfully to {email}")
                else:
                    Logger(f"[ERROR] Failed to email attendance report to {email}")

        except Exception as e:
            Logger(f"[ERROR] Exception during monthly report generation: {e}")


class FaceAppUI(BoxLayout):
    camera_is_active = BooleanProperty(True)
    recognition_status = StringProperty("Scanning for faces...")
    capture_status = StringProperty("")
    last_recognized_info = StringProperty("No recent recognition.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.backend = FaceAppBackend(ui_instance=self)
        self.scheduler = BackgroundScheduler()
        self._setup_scheduler()
        self.camera = None # Initialize to None
        self._setup_ui()
        self.capture_thread = None
        self.process_event = None
        
        # Start camera processing loop AFTER UI setup
        self.process_event = Clock.schedule_interval(self.process_frame, 1.0 / 30.0)


    def _setup_scheduler(self):
        self.scheduler.add_job(
            self.backend.generate_and_send_monthly_reports,
            "date",
            run_date=datetime.now(),
        )
        self.scheduler.add_job(
            self.backend.generate_and_send_monthly_reports,
            "cron",
            day="1",
            hour=1,
            minute=0,
        )
        self.scheduler.start()

    def _setup_ui(self):
        # Main UI layout
        main_layout = BoxLayout(orientation='horizontal', padding=10, spacing=10)
        self.add_widget(main_layout)

        # Left panel: Camera and Status
        camera_panel = BoxLayout(orientation='vertical', size_hint_x=0.7)
        self.camera = Camera(play=True, index=1, resolution=(640, 480))
        camera_panel.add_widget(self.camera)
        
        self.status_label = Label(text=self.recognition_status, size_hint_y=0.1, font_size='20sp')
        camera_panel.add_widget(self.status_label)
        
        self.last_recognized_label = Label(text=self.last_recognized_info, size_hint_y=0.2, halign='center', valign='middle')
        camera_panel.add_widget(self.last_recognized_label)
        
        main_layout.add_widget(camera_panel)

        # Right panel: Controls and registration form
        control_panel = BoxLayout(orientation='vertical', size_hint_x=0.3, padding=10, spacing=10)
        main_layout.add_widget(control_panel)

        # Registration form
        form_layout = GridLayout(cols=1, spacing=10, size_hint_y=0.6)
        
        form_layout.add_widget(Label(text="New User Registration", font_size='20sp'))
        form_layout.add_widget(Label(text="Name:"))
        self.name_input = TextInput(multiline=False)
        form_layout.add_widget(self.name_input)
        
        form_layout.add_widget(Label(text="Employee ID:"))
        self.emp_id_input = TextInput(multiline=False)
        form_layout.add_widget(self.emp_id_input)
        
        form_layout.add_widget(Label(text="Email:"))
        self.email_input = TextInput(multiline=False)
        form_layout.add_widget(self.email_input)
        
        form_layout.add_widget(Label(text="Gender ('male' or 'female'):"))
        self.gender_input = TextInput(multiline=False)
        form_layout.add_widget(self.gender_input)
        
        self.register_button = Button(text="Register User", on_press=self.start_registration)
        form_layout.add_widget(self.register_button)

        self.update_button = Button(text="Update User Photos", on_press=self.start_update)
        form_layout.add_widget(self.update_button)

        self.capture_progress_label = Label(text="", size_hint_y=0.1)
        form_layout.add_widget(self.capture_progress_label)
        
        control_panel.add_widget(form_layout)
        
        self.status_message_label = Label(text="", color=[1,0,0,1], size_hint_y=0.1)
        control_panel.add_widget(self.status_message_label)
        
    def start_registration(self, instance):
        name = self.name_input.text.strip()
        emp_id = self.emp_id_input.text.strip()
        email = self.email_input.text.strip()
        gender = self.gender_input.text.strip().lower()

        if not all([name, emp_id, email, gender]):
            self.status_message_label.text = "Please fill all registration fields."
            return

        if gender not in ("male", "female"):
            self.status_message_label.text = "Invalid gender, must be 'male' or 'female'."
            return
        
        if "@" not in email:
            self.status_message_label.text = "Invalid email format."
            return

        # Start capture logic in a separate thread
        def registration_thread():
            self.backend.register_user_email(emp_id, email)
            self.backend.save_user_gender(emp_id, gender)
            self.backend.tts_helper.speak("Hi I am Nova Voice assistant")
            time.sleep(0.005)
            self.backend.tts_helper.speak("I am clicking your photo... 3.... 2.... 1...Smaaaaaail")
            result = self.backend.start_capture_samples(name, emp_id, updating=False, sample_count=5)
            if result['status'] == 'error':
                Clock.schedule_once(lambda dt: self.update_status_message(result['message']), 0)

        threading.Thread(target=registration_thread, daemon=True).start()
        self.status_message_label.text = "Registration initiated. Look at the camera and smile."
        self.recognition_status = "Capturing Photos for Registration..."

    def start_update(self, instance):
        emp_id = self.emp_id_input.text.strip()
        name = self.name_input.text.strip() # Name can be optional for update
        
        if not emp_id:
            self.status_message_label.text = "Please provide an Employee ID to update."
            return
            
        def update_thread():
            result = self.backend.start_capture_samples(name, emp_id, updating=True, sample_count=5)
            if result['status'] == 'error':
                Clock.schedule_once(lambda dt: self.update_status_message(result['message']), 0)
            else:
                self.backend.tts_helper.speak("Starting photo update... Please look at the camera")
        
        threading.Thread(target=update_thread, daemon=True).start()
        self.status_message_label.text = "Update initiated. Look at the camera."
        self.recognition_status = "Capturing Photos for Update..."


    @mainthread
    def update_status_message(self, message):
        self.status_message_label.text = message
        
    @mainthread
    def update_recognition_info(self, info):
        self.last_recognized_label.text = (
            f"Last Recognized:\n"
            f"Name: {info['name']}\n"
            f"Time: {info['time']}\n"
            f"Greeting: {info['greeting']}"
        )
        self.status_label.text = f"Welcome, {info['name']}!"

    @mainthread
    def update_capture_status(self, current, total):
        self.capture_progress_label.text = f"Capturing: {current}/{total}"

    @mainthread
    def on_capture_complete(self):
        self.capture_progress_label.text = "Capture Complete. Retraining..."
        self.status_message_label.text = "Retraining in progress..."

    @mainthread
    def on_retraining_complete(self):
        self.capture_progress_label.text = "Registration and Training Complete!"
        self.status_message_label.text = "System is ready for new faces."
        # Reset form fields
        self.name_input.text = ""
        self.emp_id_input.text = ""
        self.email_input.text = ""
        self.gender_input.text = ""


    def process_frame(self, dt):
        if self.camera is None:
            # If the camera widget is not yet initialized, exit.
            # This handles the case where the clock event fires before _setup_ui completes.
            return

        if not self.camera.play:
            return
            
        # Get frame from camera texture
        texture = self.camera.texture
        if not texture:
            return

        w, h = texture.size
        pixels = texture.pixels
        
        # Convert Kivy texture pixels to OpenCV format
        frame = np.frombuffer(pixels, np.uint8).reshape((h, w, 4))
        frame = cv2.flip(frame, 1) # Flip vertically as Kivy texture is flipped
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) # Convert to BGR

        # Process frame with backend logic
        result = self.backend.process_frame(frame)
        
        # Draw bounding boxes and text on the frame
        if result["status"] == "success":
            for face in result["faces"]:
                x, y, w, h = face["box"]
                name = face["name"]
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Check the status of the face and display appropriate text
                if face["status"] == "recognized_new":
                    text = f"Hello, {name}!"
                    # Let the backend handle the greeting, the UI just reflects it
                elif face["status"] == "recognized_recent":
                    text = f"{name} (Seen)"
                elif face["status"] == "capturing":
                    text = f"Capturing {face['capture_progress']}"
                else:
                    text = "Unknown"
                    
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the processed frame back to a Kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera.texture = image_texture
        
        # Check for new recognition info from the backend and update UI
        recognized_info = self.backend.get_last_recognized_info()
        if recognized_info['status'] == 'success':
            self.update_recognition_info(recognized_info['info'])
    
    def on_stop(self):
        # Safely shut down the camera and scheduler
        if self.process_event:
            self.process_event.cancel()
        if self.camera:
            self.camera.play = False
        self.scheduler.shutdown()
        Logger("[INFO] Application stopped, scheduler and camera shut down.")


class FaceApp(App):
    def build(self):
        return FaceAppUI()

if __name__ == "__main__":
    from kivy import Config
    Config.set("graphics", "multisamples", "0")
    # Ensure necessary data directories exist before starting
    ensure_dir(KNOWN_FACES_DIR)
    FaceApp().run()
