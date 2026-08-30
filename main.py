# import cv2

# CAM_SOURCE = 0;
# # 
# #  "http://192.168.1.9:8080/video"  #

# if isinstance(CAM_SOURCE, int):
#     cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
# else:
#     cap = cv2.VideoCapture(CAM_SOURCE)

# if not cap.isOpened():
#     raise RuntimeError("Could not open camera. Check CAM_SOURCE / camera permissions.")

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break

#     frame = cv2.resize(frame, (940, 480))
#     cv2.imshow("Hello, Camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




"""
Purple Glitter Hand Tracker
----------------------------
Tracks your hand via webcam or IP camera and spawns a purple glitter trail
that follows your fingertip. Different hand gestures change the glitter
behavior:

  - Open palm   -> gentle floating sparkle trail (default)
  - Pinch       -> glitter bursts / concentrates at the pinch point
  - Fist        -> glitter explodes outward (a "shape" burst)
  - Peace sign  -> glitter forms a swirling ring around the hand

Controls:
  q  -> quit
  c  -> clear all particles on screen

Requirements:
    pip install opencv-python mediapipe numpy
"""

import cv2
import numpy as np
import random
import math
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CAM_SOURCE = 0  # 0 = local webcam, or "http://<phone-ip>:8080/video" for IoT cam
MAX_PARTICLES = 600
GLITTER_COLORS = [
    (255, 0, 255),   # magenta
    (200, 0, 255),   # purple
    (150, 0, 255),   # violet
    (255, 100, 255), # light pink-purple
    (180, 50, 220),  # orchid
]

# ----------------------------------------------------------------------
# PARTICLE SYSTEM
# ----------------------------------------------------------------------
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    def __init__(self, x, y, vx, vy, life, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.06          # gentle gravity
        self.vx *= 0.98          # air drag
        self.vy *= 0.98
        self.life -= 1
        return self.life > 0

    def draw(self, frame, glow_layer):
        alpha = self.life / self.max_life
        r = max(1, int(self.size * alpha))
        color = tuple(int(c * alpha + 255 * (1 - alpha) * 0.15) for c in self.color)

        # soft glow on a separate layer (blurred later)
        cv2.circle(glow_layer, (int(self.x), int(self.y)), r * 3, color, -1)
        # sharp sparkle core
        cv2.circle(frame, (int(self.x), int(self.y)), max(1, r // 2), (255, 255, 255), -1)
        cv2.circle(frame, (int(self.x), int(self.y)), r, color, -1)


class ParticleSystem:
    def __init__(self, max_particles=MAX_PARTICLES):
        self.particles = []
        self.max_particles = max_particles

    def spawn(self, x, y, count=6, speed=2.5, life_range=(20, 45), size_range=(2, 6), spread=1.0):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            angle = random.uniform(0, 2 * math.pi)
            spd = random.uniform(0.2, speed) * spread
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd - random.uniform(0.5, 1.5)  # slight upward bias
            life = random.randint(*life_range)
            size = random.uniform(*size_range)
            color = random.choice(GLITTER_COLORS)
            self.particles.append(Particle(x, y, vx, vy, life, size, color))

    def spawn_ring(self, cx, cy, radius=70, count=24, speed=0.5):
        for i in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            angle = (2 * math.pi / count) * i
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            vx = math.cos(angle + math.pi / 2) * speed
            vy = math.sin(angle + math.pi / 2) * speed
            color = random.choice(GLITTER_COLORS)
            self.particles.append(Particle(x, y, vx, vy, random.randint(15, 25), random.uniform(2, 4), color))

    def update_and_draw(self, frame):
        glow_layer = np.zeros_like(frame)
        alive = []
        for p in self.particles:
            if p.update():
                p.draw(frame, glow_layer)
                alive.append(p)
        self.particles = alive

        # blur the glow layer and blend it under the sharp particles for a soft sparkle look
        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=6, sigmaY=6)
        cv2.add(frame, glow_layer, dst=frame)

    def clear(self):
        self.particles = []


# ----------------------------------------------------------------------
# HAND LANDMARKER SETUP (current MediaPipe Tasks API)
# ----------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    """Download the hand landmark model file once, if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~10MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def create_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGER_PIPS = [3, 6, 10, 14, 18]   # joints below each tip


def get_landmark_px(landmarks, idx, w, h):
    """landmarks: a list of NormalizedLandmark objects (from the Tasks API)."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def count_extended_fingers(landmarks, w, h, handedness_label):
    """Rough heuristic: a finger is 'extended' if its tip is farther from the
    wrist than its pip joint (for fingers), and thumb uses an x-based check.
    landmarks: list of NormalizedLandmark (Tasks API)."""
    extended = []

    # Thumb: compare x position of tip vs pip relative to handedness
    thumb_tip = get_landmark_px(landmarks, 4, w, h)
    thumb_ip = get_landmark_px(landmarks, 3, w, h)
    if handedness_label == "Right":
        extended.append(thumb_tip[0] < thumb_ip[0])
    else:
        extended.append(thumb_tip[0] > thumb_ip[0])

    # Other 4 fingers: tip above pip (smaller y = higher on screen) => extended
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip = get_landmark_px(landmarks, tip_idx, w, h)
        pip = get_landmark_px(landmarks, pip_idx, w, h)
        extended.append(tip[1] < pip[1])

    return extended  # [thumb, index, middle, ring, pinky] booleans


def classify_gesture(extended, landmarks, w, h):
    thumb, index, middle, ring, pinky = extended
    count = sum(extended)

    # Pinch: thumb tip close to index tip
    thumb_tip = get_landmark_px(landmarks, 4, w, h)
    index_tip = get_landmark_px(landmarks, 8, w, h)
    dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    if dist < 35:
        return "pinch"

    if count == 0:
        return "fist"
    if index and middle and not ring and not pinky and count <= 3:
        return "peace"
    if count >= 4:
        return "open_palm"
    return "point"


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def draw_skeleton(frame, landmarks_px):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], (80, 0, 80), 1, cv2.LINE_AA)
    for pt in landmarks_px:
        cv2.circle(frame, pt, 2, (120, 0, 120), -1, cv2.LINE_AA)


def main():
    ensure_model()

    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAM_SOURCE)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAM_SOURCE / camera permissions.")

    particles = ParticleSystem()
    landmarker = create_landmarker()
    prev_index_pos = {}  # per-hand smoothing memory
    frame_timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (960, 720))
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33  # approx frame interval; must be increasing
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                handedness_label = "Right"
                if result.handedness and len(result.handedness) > hand_idx:
                    handedness_label = result.handedness[hand_idx][0].category_name

                extended = count_extended_fingers(hand_landmarks, w, h, handedness_label)
                gesture = classify_gesture(extended, hand_landmarks, w, h)

                index_tip = get_landmark_px(hand_landmarks, 8, w, h)

                # Smooth the fingertip position a bit to reduce jitter
                key = hand_idx
                if key in prev_index_pos:
                    px, py = prev_index_pos[key]
                    sx = int(px * 0.5 + index_tip[0] * 0.5)
                    sy = int(py * 0.5 + index_tip[1] * 0.5)
                else:
                    sx, sy = index_tip
                prev_index_pos[key] = (sx, sy)

                # Gesture-driven glitter behavior
                if gesture == "pinch":
                    thumb_tip = get_landmark_px(hand_landmarks, 4, w, h)
                    mid_x = (thumb_tip[0] + sx) // 2
                    mid_y = (thumb_tip[1] + sy) // 2
                    particles.spawn(mid_x, mid_y, count=10, speed=1.0, spread=0.3,
                                    life_range=(15, 25), size_range=(3, 7))
                    label = "Pinch - sparkle burst"

                elif gesture == "fist":
                    wrist = get_landmark_px(hand_landmarks, 0, w, h)
                    particles.spawn(wrist[0], wrist[1], count=18, speed=5.0, spread=2.0,
                                    life_range=(20, 35), size_range=(3, 8))
                    label = "Fist - explosion"

                elif gesture == "peace":
                    wrist = get_landmark_px(hand_landmarks, 0, w, h)
                    particles.spawn_ring(wrist[0], wrist[1], radius=90, count=3, speed=0.8)
                    label = "Peace - glitter ring"

                else:  # open_palm / point -> flowing trail
                    particles.spawn(sx, sy, count=5, speed=1.8, spread=1.0,
                                     life_range=(25, 45), size_range=(2, 5))
                    label = f"{gesture} - trail"

                # draw the skeleton faintly for feedback
                landmarks_px = [get_landmark_px(hand_landmarks, i, w, h) for i in range(21)]
                draw_skeleton(frame, landmarks_px)

                cv2.putText(frame, label, (sx + 15, sy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        particles.update_and_draw(frame)

        cv2.putText(frame, "q: quit   c: clear glitter", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Purple Glitter Hand Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            particles.clear()

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()