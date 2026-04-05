"""
ASL Sign Language to Speech System
====================================
Uses MediaPipe Hands + pre-trained landmark classifier
No custom data recording needed.
Run: python asl_system.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from collections import deque, Counter
import threading

# ─── TTS ENGINE ───────────────────────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def speak_text(text):
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    t = threading.Thread(target=_speak, daemon=True)
    t.start()

# ─── ASL CLASSIFIER ───────────────────────────────────────────────────────────

def get_finger_states(landmarks):
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    states = []
    if landmarks[4].x < landmarks[3].x:
        states.append(True)
    else:
        states.append(False)
    for tip, joint in zip(tips[1:], joints[1:]):
        states.append(landmarks[tip].y < landmarks[joint].y)
    return states

def classify_asl(landmarks):
    lm = landmarks
    f = get_finger_states(lm)
    thumb, index, middle, ring, pinky = f

    def dist(i, j):
        return np.sqrt((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)

    thumb_index = dist(4, 8)
    thumb_mid   = dist(4, 12)

    if not index and not middle and not ring and not pinky and not thumb:
        if dist(8, 5) < 0.07:   return 'E'
        if dist(8, 4) < 0.05 and dist(12, 4) < 0.05: return 'M'
        if dist(8, 4) < 0.06 and dist(12, 4) < 0.06: return 'N'
        if dist(8, 4) < 0.08:   return 'O'
        if dist(8, 6) < 0.05:   return 'X'
        return 'A'

    if not index and not middle and not ring and not pinky and thumb:
        if dist(8, 4) < 0.15:   return 'C'
        if dist(4, 8) < 0.07:   return 'T'
        if dist(4, 8) < 0.09:   return 'S'
        return 'A'

    if index and middle and ring and pinky and not thumb:
        return 'B'

    if index and not middle and not ring and not pinky:
        if thumb_mid < 0.08:     return 'D'
        if lm[8].y > lm[5].y and thumb: return 'Q'
        if not thumb:            return 'Z'

    if not index and middle and ring and pinky:
        if thumb_index < 0.07:   return 'F'

    if index and not middle and not ring and not pinky and thumb:
        if abs(lm[8].y - lm[5].y) < 0.05: return 'G'
        if lm[8].y > lm[5].y:   return 'Q'
        if dist(4, 12) < 0.1:   return 'K'
        return 'L'

    if index and middle and not ring and not pinky and not thumb:
        if abs(lm[8].y - lm[12].y) < 0.04: return 'H'
        if abs(lm[8].x - lm[12].x) < 0.04: return 'U'
        if abs(lm[8].x - lm[12].x) > 0.04: return 'V'

    if index and middle and not ring and not pinky:
        if lm[8].x < lm[12].x + 0.02: return 'R'

    if not index and not middle and not ring and pinky and not thumb:
        return 'I'

    if index and middle and not ring and not pinky and thumb:
        if lm[8].y > lm[5].y:   return 'P'
        if dist(4, 12) < 0.1:   return 'K'
        return 'L'

    if index and middle and ring and not pinky and not thumb:
        return 'W'

    if not index and not middle and not ring and pinky and thumb:
        return 'Y'

    return None

# ─── SENTENCE BUILDER ─────────────────────────────────────────────────────────

class SentenceBuilder:
    def __init__(self):
        self.sentence = []
        self.current_word = []
        self.prediction_buffer = deque(maxlen=15)
        self.last_letter = None
        self.letter_hold_start = None
        self.letter_hold_duration = 1.2
        self.space_hold_duration = 2.0
        self.no_hand_start = None
        self.last_spoken = ""

    def update(self, letter, hand_detected):
        now = time.time()
        if not hand_detected:
            if self.no_hand_start is None:
                self.no_hand_start = now
            elif now - self.no_hand_start > self.space_hold_duration:
                if self.current_word:
                    self.sentence.append(''.join(self.current_word))
                    self.current_word = []
                self.no_hand_start = None
            return

        self.no_hand_start = None
        if letter is None:
            return

        self.prediction_buffer.append(letter)
        if len(self.prediction_buffer) < 10:
            return

        stable_letter = Counter(self.prediction_buffer).most_common(1)[0][0]
        if stable_letter != self.last_letter:
            self.last_letter = stable_letter
            self.letter_hold_start = now
            return

        if now - self.letter_hold_start > self.letter_hold_duration:
            self.current_word.append(stable_letter)
            self.prediction_buffer.clear()
            self.last_letter = None
            self.letter_hold_start = None

    def get_display(self):
        words = self.sentence + ([''.join(self.current_word)] if self.current_word else [])
        return ' '.join(words)

    def speak_sentence(self):
        text = self.get_display().strip()
        if text and text != self.last_spoken:
            self.last_spoken = text
            speak_text(text)

    def clear(self):
        self.sentence = []
        self.current_word = []
        self.prediction_buffer.clear()
        self.last_letter = None

# ─── UI ───────────────────────────────────────────────────────────────────────

def draw_ui(frame, letter, confidence_ratio, sentence):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 25), -1)
    cv2.putText(frame, "ASL Sign Language to Speech", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 220, 255), 2)

    if letter:
        cv2.putText(frame, letter, (w-120, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 150), 5)
        bar_w = int(200 * min(confidence_ratio, 1.0))
        cv2.rectangle(frame, (w-230, 150), (w-30, 175), (40, 40, 60), -1)
        cv2.rectangle(frame, (w-230, 150), (w-230+bar_w, 175), (0, 255, 150), -1)
        cv2.putText(frame, "Hold...", (w-230, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 170), 1)

    sentence_text = sentence if sentence else "Start signing..."
    cv2.putText(frame, "Sentence:", (15, h-145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 130, 160), 1)

    words = sentence_text.split()
    line, lines = [], []
    for word in words:
        test = ' '.join(line + [word])
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0] < w - 30:
            line.append(word)
        else:
            lines.append(' '.join(line))
            line = [word]
    lines.append(' '.join(line))
    for i, ln in enumerate(lines[-2:]):
        cv2.putText(frame, ln, (15, h - 110 + i * 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, "[SPACE] Speak  |  [C] Clear  |  [Q] Quit", (15, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 120), 1)
    return frame

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    import mediapipe as mp_module
    import urllib.request, os

    BaseOptions           = mp_module.tasks.BaseOptions
    HandLandmarker        = mp_module.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp_module.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult  = mp_module.tasks.vision.HandLandmarkerResult
    VisionRunningMode     = mp_module.tasks.vision.RunningMode

    # Download model if needed
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand landmark model (~30MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")

    latest_result = {"landmarks": None}
    result_lock = threading.Lock()

    def result_callback(result: HandLandmarkerResult, output_image, timestamp_ms):
        with result_lock:
            latest_result["landmarks"] = result.hand_landmarks[0] if result.hand_landmarks else None

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=result_callback
    )

    # ── Auto-detect camera (DroidCam, built-in, USB) ──
    cap = None
    print("Looking for camera...")
    for index in range(6):
        print(f"  Trying camera index {index}...")
        test = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if test.isOpened():
            time.sleep(0.8)  # give DroidCam time to warm up
            ret, frame = test.read()
            if ret and frame is not None and frame.size > 0:
                print(f"✅ Camera found at index {index}")
                cap = test
                break
        test.release()

    if cap is None:
        print("\n❌ No camera found.")
        print("   - Make sure DroidCam desktop app is running")
        print("   - Make sure phone DroidCam app is connected")
        print("   - Try restarting DroidCam on both devices")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    builder = SentenceBuilder()

    print("\n🤟 ASL Sign Language to Speech System")
    print("=" * 40)
    print("Hold a sign steady for ~1.2s → letter added")
    print("Remove hand for 2s           → word space")
    print("SPACE key                    → speak sentence")
    print("C key                        → clear sentence")
    print("Q key                        → quit\n")

    def draw_landmarks_manual(frame, landmarks, h, w):
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
            (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 180), -1)

    class LM:
        def __init__(self, x, y, z):
            self.x = x; self.y = y; self.z = z

    start_time = time.time()
    consecutive_failures = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures > 50:
                    print("❌ Camera lost. Is DroidCam still connected?")
                    break
                time.sleep(0.03)
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)

            try:
                mp_image = mp_module.Image(
                    image_format=mp_module.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )
                landmarker.detect_async(mp_image, timestamp_ms)
            except Exception:
                pass

            with result_lock:
                landmarks = latest_result["landmarks"]

            letter     = None
            hand_detected = False
            hold_ratio = 0.0

            if landmarks:
                hand_detected = True
                draw_landmarks_manual(frame, landmarks, h, w)
                lm_list = [LM(l.x, l.y, l.z) for l in landmarks]
                letter = classify_asl(lm_list)

                if letter and builder.last_letter == letter and builder.letter_hold_start:
                    hold_ratio = (time.time() - builder.letter_hold_start) / builder.letter_hold_duration

            builder.update(letter, hand_detected)
            sentence_text = builder.get_display()
            frame = draw_ui(frame, letter, hold_ratio, sentence_text)

            cv2.imshow("ASL Sign Language to Speech", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                builder.clear()
                print("🗑️  Cleared")
            elif key == ord(' '):
                builder.speak_sentence()
                print(f"🔊 Speaking: {sentence_text}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Session ended.")

if __name__ == "__main__":
    main()