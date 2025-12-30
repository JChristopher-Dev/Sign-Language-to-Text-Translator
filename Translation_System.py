import cv2
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
import mediapipe as mp
import tensorflow as tf

# LOADS TRAINED CNN MODEL WITH APPROPRIATE WEIGHTS
model = tf.keras.models.load_model('action_new_5.h5')

# LOADS ACTION LABELS
actions = np.load('C:/Final Design/Phase 3/Demo Project/Sign Language to Text Translator/actions.npy', allow_pickle=True)

threshold = 0.5

# MEDIAPIPE SETUP AND DETECTION CODE
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, holistic):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    # KEY POINT EXTRATION FOR FACE, BODY, LEFT AND RIGHT HAND
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# COLORS FOR CONFIDENCE BARS
confidence_colors = [(255, 100, 100), (200, 150, 255), (150, 200, 255)]

def prob_viz(res, actions, image):
    # SHOW TOP 3 PREDICTIONS
    top_indices = np.argsort(res)[-3:][::-1]
    for i, idx in enumerate(top_indices):
        action_text = f"{actions[idx]}: {res[idx]:.2f}"
        bar_x = 10
        bar_y = (i * 30) + 10
        text_y = bar_y + 20
        bar_width = int(res[idx] * 200)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), confidence_colors[i], -1)
        cv2.putText(image, action_text, (bar_x, text_y - 5), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image

class start_translation:
    def __init__(self, root):
        # MAIN WINDOW
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.attributes("-fullscreen", True)
        self.root.geometry("1519x703")  
        self.root.configure(bg="#f0f0f0")

        # SYSTEM THEME
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("dark-blue")

        # NAVBAR
        self.create_navbar()

        # MAIN FRAME
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=20)
        self.main_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # PREDICTION PANEL
        self.prediction_frame = ctk.CTkFrame(self.main_frame, corner_radius=20, fg_color="#1abc9c")
        self.prediction_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # WEBCAM PANEL
        self.webcam_frame = ctk.CTkFrame(self.main_frame, corner_radius=20)
        self.webcam_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # WEBCAM SETUP
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Error: Could not access the webcam.")
            self.root.quit()
        self.webcam_label = ctk.CTkLabel(self.webcam_frame)
        self.webcam_label.pack(fill="both", expand=True)

        # CURRENT GESTURE BAR
        self.create_current_gesture_bar()
        # LIVE SENTENCE BAR
        self.create_live_sentence_bar()
        self.root.after(100, self.show_feed)

    def create_navbar(self):
        # TOP NAVBAR WITH LOGO
        top_frame = ctk.CTkFrame(self.root, height=150, corner_radius=20, fg_color="#1abc9c")
        top_frame.pack(side="top", fill="x", padx=10, pady=10)

        logo = Image.open("communication.png")
        logo = logo.resize((200, 200), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(logo)

        logo_label = ctk.CTkLabel(top_frame, image=logo, text="")
        logo_label.image = logo
        logo_label.pack(side="left", padx=30)

        heading_label = ctk.CTkLabel(top_frame, text="Sign Language Translator", font=("Helvetica", 36, "bold"), text_color="white")
        heading_label.pack(side="left", padx=20)

    def create_current_gesture_bar(self):
        # CURRENT GESTURE INFO
        current_gesture_frame = ctk.CTkFrame(self.prediction_frame, height=120, corner_radius=20, fg_color="#1abc9c")
        current_gesture_frame.pack(side="top", fill="x", pady=10)

        current_gesture_label = ctk.CTkLabel(current_gesture_frame, text="Current Gesture:", font=("Helvetica", 18, "bold"))
        current_gesture_label.pack(side="top", padx=20, pady=5)

        self.current_gesture_text = ctk.CTkLabel(current_gesture_frame, text="Processing...", font=("Helvetica", 16), fg_color="#16a085", corner_radius=10)
        self.current_gesture_text.pack(side="top", padx=10, pady=5)

        additional_info = ctk.CTkLabel(current_gesture_frame, text="Waiting for gesture...", font=("Helvetica", 14), fg_color="#16a085")
        additional_info.pack(side="top", padx=10, pady=5)

    def create_live_sentence_bar(self):
        # LIVE SENTENCE INFO
        sentence_frame = ctk.CTkFrame(self.prediction_frame, height=80, corner_radius=20, fg_color="#1abc9c")
        sentence_frame.pack(side="bottom", fill="x", pady=10)

        sentence_label = ctk.CTkLabel(sentence_frame, text="Live Sentence:", font=("Helvetica", 18, "bold"))
        sentence_label.pack(side="left", padx=20)

        self.sentence_text = ctk.CTkLabel(sentence_frame, text="Processing...", font=("Helvetica", 16), fg_color="#16a085", corner_radius=10)
        self.sentence_text.pack(side="left", padx=10, pady=5)

    def show_feed(self):
        DISPLAY_WIDTH = 940
        DISPLAY_HEIGHT = 680

        # VARIABLES
        sequence = []
        sentence = []
        predictions = []
        translation_started = False
        threshold = 0.5

        # MEDIAPIPE + MODEL
        mp_holistic = mp.solutions.holistic
        model = tf.keras.models.load_model('action_new_5.h5')
        actions = np.load('C:/Final Design/Phase 3/Demo Project/Sign Language to Text Translator/actions.npy', allow_pickle=True)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # MEDIAPIPE FRAME PROCESS
                image, results = mediapipe_detection(frame, holistic)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    sequence_array = np.array(sequence)
                    if sequence_array.shape[1] == (468 * 3 + 21 * 3 + 21 * 3 + 33 * 4):
                        res = model.predict(np.expand_dims(sequence_array, axis=0))[0]
                        predictions.append(np.argmax(res))

                        if np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # SHOW PROBABILITIES
                    image = prob_viz(res, actions, image)

                # RESIZE FRAME
                image = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                # TKINTER IMAGE UPDATE
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_image)
                img_tk = ImageTk.PhotoImage(img)

                self.webcam_label.configure(image=img_tk)
                self.webcam_label.image = img_tk

                # UPDATE SENTENCE
                self.sentence_text.configure(text=" ".join(sentence))

                self.root.update()

    def start(self):
        # START APP
        self.root.mainloop()

    def close(self):
        # RELEASE CAMERA
        self.cap.release()

if __name__ == "__main__":
    root = ctk.CTk()
    start_app = start_translation(root)
    root.mainloop()
