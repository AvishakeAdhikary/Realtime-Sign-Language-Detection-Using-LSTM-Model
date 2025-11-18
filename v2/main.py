#!/usr/bin/env python3
# ---------------------------------------------------------------
# SIGN LANGUAGE DETECTION — MODERN MODULAR SINGLE-FILE VERSION
# ---------------------------------------------------------------

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import mediapipe as mp


# ===============================================================
# CONFIG CLASS (modifies application behavior)
# ===============================================================
@dataclass
class AppConfig:
    data_path: str = "MP_Data"
    model_path: str = "model.h5"
    model_weights_path: str = "model_weights.h5"

    # Full Mediapipe Holistic feature vector: 1662 values
    feature_vector_length: int = 1662

    # Default values (user can override interactively)
    sequence_length: int = 30
    sequences_per_sign: int = 30
    training_epochs: int = 2000

    # Drawing palette (Material Design)
    palette: dict = None

    def __post_init__(self):
        self.palette = {
            "face": (66, 133, 244),
            "pose": (52, 168, 83),
            "left_hand": (251, 188, 5),
            "right_hand": (234, 67, 53),
            "prob_bg": (33, 150, 243),
        }


# ===============================================================
# MEDIAPIPE HANDLER
# ===============================================================
class MediapipeHandler:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, image, model):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = model.process(rgb)
        rgb.flags.writeable = True
        out_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return out_image, results

    def draw_landmarks(self, image, results, palette):
        draw = self.mp_drawing
        holistic = self.mp_holistic

        # Face
        if results.face_landmarks:
            draw.draw_landmarks(
                image, results.face_landmarks,
                holistic.FACEMESH_TESSELATION,
                draw.DrawingSpec(color=palette["face"], thickness=1, circle_radius=1),
            )

        # Pose
        if results.pose_landmarks:
            draw.draw_landmarks(
                image, results.pose_landmarks,
                holistic.POSE_CONNECTIONS,
                draw.DrawingSpec(color=palette["pose"], thickness=2, circle_radius=2),
            )

        # Left Hand
        if results.left_hand_landmarks:
            draw.draw_landmarks(
                image, results.left_hand_landmarks,
                holistic.HAND_CONNECTIONS,
                draw.DrawingSpec(color=palette["left_hand"], thickness=2, circle_radius=2),
            )

        # Right Hand
        if results.right_hand_landmarks:
            draw.draw_landmarks(
                image, results.right_hand_landmarks,
                holistic.HAND_CONNECTIONS,
                draw.DrawingSpec(color=palette["right_hand"], thickness=2, circle_radius=2),
            )

    @staticmethod
    def extract_keypoints(results):
        # Pose (33 × 4)
        pose = np.array([[p.x, p.y, p.z, p.visibility]
                         for p in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 4)

        # Face (468 × 3)
        face = np.array([[f.x, f.y, f.z]
                         for f in results.face_landmarks.landmark]).flatten() \
            if results.face_landmarks else np.zeros(468 * 3)

        # Hands (21 × 3 each)
        lh = np.array([[h.x, h.y, h.z]
                       for h in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)

        rh = np.array([[h.x, h.y, h.z]
                       for h in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)

        return np.concatenate([pose, face, lh, rh])


# ===============================================================
# DATASET HANDLING
# ===============================================================
class DatasetManager:
    def __init__(self, cfg: AppConfig, signs: List[str]):
        self.cfg = cfg
        self.signs = signs
        self.mp_handler = MediapipeHandler()

        if not os.path.exists(cfg.data_path):
            os.mkdir(cfg.data_path)

    def prepare_folders(self):
        for sign in self.signs:
            sign_dir = os.path.join(self.cfg.data_path, sign)
            os.makedirs(sign_dir, exist_ok=True)

            for seq in range(self.cfg.sequences_per_sign):
                os.makedirs(os.path.join(sign_dir, str(seq)), exist_ok=True)

    def collect(self):
        self.prepare_folders()

        cap = cv2.VideoCapture(0)
        holistic = self.mp_handler.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        with holistic as model:
            for sign in self.signs:
                for seq in range(self.cfg.sequences_per_sign):
                    for frame_num in range(self.cfg.sequence_length):

                        ret, frame = cap.read()
                        if not ret:
                            continue

                        image, results = self.mp_handler.detect(frame, model)
                        self.mp_handler.draw_landmarks(image, results, self.cfg.palette)

                        if frame_num == 0:
                            cv2.putText(image, f"Start: {sign} (seq {seq})",
                                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 3)
                            cv2.imshow("Collecting Data", image)
                            cv2.waitKey(1000)

                        keypoints = self.mp_handler.extract_keypoints(results)
                        np.save(os.path.join(self.cfg.data_path, sign, str(seq), f"{frame_num}.npy"), keypoints)

                        cv2.imshow("Collecting Data", image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            return

        cap.release()
        cv2.destroyAllWindows()

    def load_dataset(self):
        sequences, labels = [], []
        label_map = {label: i for i, label in enumerate(self.signs)}

        for sign in self.signs:
            sign_dir = os.path.join(self.cfg.data_path, sign)
            sequences_in_sign = sorted(os.listdir(sign_dir))

            for seq in sequences_in_sign:
                window = []
                for f in range(self.cfg.sequence_length):
                    window.append(np.load(os.path.join(sign_dir, seq, f"{f}.npy")))
                sequences.append(window)
                labels.append(label_map[sign])

        return np.array(sequences), to_categorical(labels).astype(int)


# ===============================================================
# MODEL CREATION + TRAINING
# ===============================================================
class ModelHandler:
    def __init__(self, cfg: AppConfig, num_classes: int):
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = None

    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True,
                       activation='relu',
                       input_shape=(self.cfg.sequence_length, self.cfg.feature_vector_length)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        self.model = model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05
        )
        self.model.fit(X_train, y_train, epochs=self.cfg.training_epochs)
        predictions = np.argmax(self.model.predict(X_test), axis=1)
        real = np.argmax(y_test, axis=1)
        print("Accuracy:", accuracy_score(real, predictions))

    def save(self):
        self.model.save(self.cfg.model_path)
        self.model.save_weights(self.cfg.model_weights_path)

    def load(self):
        self.model = tf.keras.models.load_model(self.cfg.model_path)
        self.model.load_weights(self.cfg.model_weights_path)


# ===============================================================
# INFERENCE (webcam or video)
# ===============================================================
class InferenceEngine:
    def __init__(self, cfg: AppConfig, signs: List[str], model):
        self.cfg = cfg
        self.signs = signs
        self.model = model
        self.mp_handler = MediapipeHandler()

    def visualize_probabilities(self, frame, probs):
        bar_h = 25
        spacing = 35

        for i, p in enumerate(probs):
            cv2.rectangle(frame, (10, 10 + i * spacing),
                          (int(10 + p * 300), 10 + i * spacing + bar_h),
                          self.cfg.palette["prob_bg"], -1)
            cv2.putText(frame, f"{self.signs[i]}: {p:.2f}",
                        (15, 10 + i * spacing + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        holistic = self.mp_handler.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        sequence = []

        with holistic as model:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = self.mp_handler.detect(frame, model)
                self.mp_handler.draw_landmarks(image, results, self.cfg.palette)

                keypoints = self.mp_handler.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-self.cfg.sequence_length:]

                if len(sequence) == self.cfg.sequence_length:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    image = self.visualize_probabilities(image, res)

                cv2.imshow("Live Detection", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


# ===============================================================
# GPU SELECTION
# ===============================================================
def configure_gpu():
    choice = input("Use GPU? (y/n): ").strip().lower()
    if choice == "n":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPU disabled. Running on CPU.")
    else:
        print("GPU enabled (if available).")


# ===============================================================
# MAIN CONSOLE APPLICATION
# ===============================================================
def main():
    print("\n=== SIGN LANGUAGE DETECTION SYSTEM ===\n")

    # GPU Config
    configure_gpu()

    # User selects signs
    signs = input("Enter signs separated by commas: ").split(",")
    signs = [s.strip() for s in signs if s.strip()]

    # User config overrides
    cfg = AppConfig()
    cfg.sequence_length = int(input("Frames per sequence (default 30): ") or 30)
    cfg.sequences_per_sign = int(input("Sequences per sign (default 30): ") or 30)
    cfg.training_epochs = int(input("Training epochs (default 2000): ") or 2000)

    dataset = DatasetManager(cfg, signs)
    model_handler = ModelHandler(cfg, len(signs))

    print("\nSELECT MODE:")
    print("1. Collect Dataset")
    print("2. Train Model")
    print("3. Live Detection (Webcam)")
    print("4. Video File Detection")
    choice = input("Your choice: ").strip()

    if choice == "1":
        dataset.collect()

    elif choice == "2":
        X, y = dataset.load_dataset()
        model_handler.build_lstm_model()
        model_handler.train(X, y)
        model_handler.save()

    elif choice == "3":
        model_handler.load()
        engine = InferenceEngine(cfg, signs, model_handler.model)
        engine.run(0)

    elif choice == "4":
        path = input("Video file path: ")
        model_handler.load()
        engine = InferenceEngine(cfg, signs, model_handler.model)
        engine.run(path)

    else:
        print("Invalid choice.")


# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
