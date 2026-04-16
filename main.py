import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from collections import deque
from collections import Counter
import sys

# Geometric feature extractor shared with train_model.py
from utils import extract_enhanced_features

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def find_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks_list.append(landmarks)
                
        return frame, landmarks_list, hand_detected
    
    def normalize_landmarks(self, landmarks_list):
        if len(landmarks_list) == 0:
            return None
        
        landmarks = np.array(landmarks_list[0])
        wrist = landmarks[0]
        
        normalized = landmarks - wrist
        max_val = np.max(np.abs(normalized))
        if max_val > 0:
            normalized = normalized / max_val
            
        return normalized.flatten()

class GestureStabilizer:
    def __init__(self, buffer_size=15, confidence_threshold=0.8, min_detection_frames=5):
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.min_detection_frames = min_detection_frames
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.hand_detected_buffer = deque(maxlen=buffer_size)
        self.last_stable_gesture = None
        self.stable_count = 0
        
    def add_prediction(self, gesture, hand_detected):
        if hand_detected and gesture:
            self.prediction_buffer.append(gesture)
            self.hand_detected_buffer.append(1)
        else:
            self.prediction_buffer.append("none")
            self.hand_detected_buffer.append(0)
    
    def get_stable_gesture(self):
        hand_detected_sum = sum(self.hand_detected_buffer)
        if hand_detected_sum < self.min_detection_frames:
            return None, 0.0
        
        gesture_counts = Counter(self.prediction_buffer)
        
        if "none" in gesture_counts:
            del gesture_counts["none"]
        
        if not gesture_counts:
            return None, 0.0
        
        most_common = gesture_counts.most_common(1)[0]
        gesture, count = most_common
        
        # Use currently collected history for faster early lock-in.
        effective_window = max(1, len(self.prediction_buffer))
        confidence = count / effective_window
        
        if confidence >= self.confidence_threshold:
            recent_frames = list(self.prediction_buffer)[-min(5, len(self.prediction_buffer)):]
            recent_count = recent_frames.count(gesture)
            
            if recent_count >= 3:
                return gesture, confidence
        
        return None, 0.0
    
    def clear(self):
        self.prediction_buffer.clear()
        self.hand_detected_buffer.clear()
    
    def get_buffer_display(self):
        display = ""
        for pred, hand in zip(self.prediction_buffer, self.hand_detected_buffer):
            if hand == 0:
                display += "-"   # no hand
            elif pred == "none":
                display += "o"   # hand only, no gesture
            else:
                display += "#"   # gesture detected
        return display

class SentenceBuilder:
    def __init__(self, pause_threshold=2.0, min_hold_time=0.5, no_hand_timeout=0.3):
        self.current_sentence = []
        self.pause_threshold = pause_threshold
        self.min_hold_time = min_hold_time
        self.no_hand_timeout = no_hand_timeout
        self.last_gesture_time = time.time()
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_gesture_added = None
        self.last_add_time = 0
        self.hand_present = False
        self.hand_lost_time = None
        
    def update_hand_status(self, hand_detected):
        current_time = time.time()
        
        if hand_detected:
            self.hand_present = True
            self.hand_lost_time = None
        else:
            if self.hand_present:
                if self.hand_lost_time is None:
                    self.hand_lost_time = current_time
                elif current_time - self.hand_lost_time > self.no_hand_timeout:
                    self.hand_present = False
                    self._finalize_current_gesture()
    
    def _finalize_current_gesture(self):
        if self.current_gesture and self.gesture_start_time:
            hold_duration = time.time() - self.gesture_start_time
            if hold_duration >= self.min_hold_time:
                self.current_sentence.append(self.current_gesture)
                self.last_gesture_added = self.current_gesture
                self.last_add_time = time.time()
                print(f"➕ Added (hand lost): {self.current_gesture}")
        
        self.current_gesture = None
        self.gesture_start_time = None
    
    def add_gesture(self, gesture, confidence, hand_detected):
        current_time = time.time()
        
        if not hand_detected:
            self.update_hand_status(False)
            return None
        
        self.update_hand_status(True)
        
        if not gesture:
            return None
        
        if gesture != self.current_gesture:
            if self.current_gesture and self.gesture_start_time:
                hold_duration = current_time - self.gesture_start_time
                if hold_duration >= self.min_hold_time:
                    if (self.current_gesture != self.last_gesture_added or 
                        current_time - self.last_add_time > 1.0):
                        self.current_sentence.append(self.current_gesture)
                        self.last_gesture_added = self.current_gesture
                        self.last_add_time = current_time
                        print(f"➕ Added: {self.current_gesture}")
            
            self.current_gesture = gesture
            self.gesture_start_time = current_time
        
        self.last_gesture_time = current_time
        
        if self._should_speak():
            return self._speak_sentence()
        
        return None
    
    def _should_speak(self):
        if len(self.current_sentence) == 0:
            return False
        return (time.time() - self.last_gesture_time) > self.pause_threshold
    
    def _speak_sentence(self):
        if len(self.current_sentence) == 0:
            return None
        
        sentence = ' '.join(self.current_sentence)
        
        grammar_rules = {
            'how you': 'how are you',
            'thanks you': 'thank you',
            'me fine': 'i am fine',
            'my name': 'my name is',
            'thank': 'thank you',
            'help me': 'please help me',
            'i want': 'i want to',
            'love you': 'i love you',
            'is is': 'is',
            'my my': 'my'
        }
        
        for wrong, correct in grammar_rules.items():
            if wrong in sentence.lower():
                sentence = sentence.lower().replace(wrong, correct)
        
        sentence = sentence.capitalize() + '.'
        
        complete = sentence
        self.current_sentence = []
        self.current_gesture = None
        self.last_gesture_added = None
        
        return complete
    
    def force_speak(self):
        self._finalize_current_gesture()
        return self._speak_sentence()
    
    def get_current(self):
        if self.current_gesture and self.gesture_start_time:
            hold_duration = time.time() - self.gesture_start_time
            if hold_duration > 0.2:
                return self.current_gesture, hold_duration
        return None, 0
    
    def get_sentence_text(self):
        return ' '.join(self.current_sentence)

class SpeechOutput:
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.available = True
        except:
            self.available = False
        
    def speak(self, text):
        print(f"\n🔊 SPEAKING: {text}")
        if self.available:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass

class ASLToSpeech:
    def __init__(self):
        print("\n" + "="*60)
        print("STABILIZED SIGN LANGUAGE TRANSLATOR")
        print("="*60)
        
        self.detector = HandDetector()
        self.stabilizer = GestureStabilizer(
            buffer_size=8,
            confidence_threshold=0.75,
            min_detection_frames=4
        )
        self.sentence_builder = SentenceBuilder(
            pause_threshold=2.0, 
            min_hold_time=0.5,
            no_hand_timeout=0.3
        )
        self.speech = SpeechOutput()
        self.flip_camera = True
        
        # Load custom model
        try:
            with open('custom_gesture_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model      = data['model']
                self.signs      = data['signs']
                feature_type    = data.get('feature_type', 'raw_63')
                feature_dim     = data.get('feature_dim', 63)
                # Auto-detect whether to apply enhanced features
                self.use_enhanced = (feature_type == 'enhanced_83' or feature_dim == 83)
            mode_str = 'enhanced (83-dim)' if self.use_enhanced else 'raw (63-dim)'
            print(f"✅ Loaded custom model: {', '.join(self.signs)}")
            print(f"   Feature mode  : {mode_str}")
            if not self.use_enhanced:
                print(f"   ⚠️  Model was trained on raw 63-dim features.")
                print(f"      Re-train with --augmented for best accuracy:")
                print(f"      python augment_data.py && python train_model.py --augmented")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()
        
        self.running = True
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.1
        self._last_word       = None
        self._last_confidence = 0.0
        self._last_spoken_time = 0          # cooldown between auto-speaks
        self._last_spoken_word = None       # prevent same word repeating instantly
        
    def init_camera(self):
        """Initialize camera — auto-detects Camo Studio across all indices"""
        print("\n🔍 Scanning for Camo Studio camera...")

        # Only MSMF and ANY — DSHOW raises C++ exceptions on this machine
        backends = [
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY,  "ANY"),
        ]

        builtin_cap  = None   # index 0 fallback
        camo_cap     = None   # index >= 1 preferred

        for idx in range(4):   # 0-3 is enough
            for backend, bname in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:
                    continue

                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                time.sleep(0.2)

                bright = 0
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        bright = frame.mean()
                        if bright > 10:
                            break

                if bright > 10:
                    print(f"  ✅ Found camera: index={idx} [{bname}]  brightness={bright:.1f}")
                    if idx == 0:
                        builtin_cap = cap
                    else:
                        camo_cap = cap
                    break   # found working backend for this index, move to next
                else:
                    print(f"  ⚠️  index={idx} [{bname}]: black frame (brightness={bright:.1f})")
                    cap.release()

            # Stop scanning as soon as we found Camo (index >= 1)
            if camo_cap is not None:
                break

        if camo_cap is not None:
            if builtin_cap is not None:
                builtin_cap.release()
            print(f"\n🎥 Using Camo Studio camera")
            return camo_cap

        if builtin_cap is not None:
            print(f"\n🎥 Camo not found — falling back to built-in camera")
            return builtin_cap

        return None
    
    def run(self):
        # Initialize camera
        cap = self.init_camera()
        
        if cap is None:
            print("\n❌ Could not open any camera!")
            print("\n🔧 TROUBLESHOOTING:")
            print("1. Close ALL other apps using camera (Zoom, Teams, Chrome)")
            print("2. Restart Camo Studio")
            print("3. Disconnect and reconnect phone")
            print("4. Run this program again")
            print("5. If still issues, restart computer")
            return
        
        print("\n✅ Camera connected successfully!")
        print("\n🎯 HOW TO USE:")
        print("  • Hold each sign STEADY until bar fills")
        print("  • Remove hand to immediately add last sign")
        print("  • Pause 2 seconds to speak sentence")
        print("\n🎮 Controls:")
        print("  f - Toggle camera flip")
        print("  s - Speak sentence now")
        print("  c - Clear sentence")
        print("  r - Reset stabilizer")
        print("  q - Quit")
        print("="*60)
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        black_frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Lost frame, reconnecting...")
                time.sleep(0.5)
                continue
            
            # Check for black screen
            brightness = frame.mean()
            if brightness < 5:
                black_frame_count += 1
                if black_frame_count > 30:  # 1 second of black frames
                    print("⚠️ Camera showing black screen - Attempting reconnect...")
                    cap.release()
                    time.sleep(1)
                    cap = self.init_camera()
                    if cap is None:
                        print("❌ Could not reconnect camera")
                        break
                    black_frame_count = 0
                continue
            else:
                black_frame_count = 0
            
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Flip camera if enabled
            if self.flip_camera:
                frame = cv2.flip(frame, 1)
            
            # Process frame
            frame, landmarks, hand_detected = self.detector.find_hand_landmarks(frame)
            
            # Predict gesture
            current_time = time.time()
            if current_time - self.last_prediction_time > self.prediction_cooldown:
                features = self.detector.normalize_landmarks(landmarks)

                if features is not None and hand_detected:
                    # ── Apply enhanced geometric features if model was trained with them
                    if self.use_enhanced:
                        features = extract_enhanced_features(features)

                    features = features.reshape(1, -1)
                    pred        = self.model.predict(features)[0]
                    proba       = self.model.predict_proba(features)[0]
                    confidence  = float(np.max(proba))
                    word        = self.signs[pred]

                    # ── Confidence gate: only accept if model is sure enough
                    #    Raised to 0.75 to reduce wrong sign predictions for
                    #    visually-similar gestures (e.g. eat vs drink)
                    if confidence > 0.75:
                        self.stabilizer.add_prediction(word, hand_detected)
                    else:
                        self.stabilizer.add_prediction(None, hand_detected)

                    # Store for display
                    self._last_word       = word
                    self._last_confidence = confidence
                else:
                    self.stabilizer.add_prediction(None, hand_detected)
                    self._last_word       = None
                    self._last_confidence = 0.0

                stable_word, stab_confidence = self.stabilizer.get_stable_gesture()

                if stable_word:
                    sentence = self.sentence_builder.add_gesture(
                        stable_word, stab_confidence, hand_detected
                    )

                    if sentence:
                        self.speech.speak(sentence)
                        self.stabilizer.clear()

                self.last_prediction_time = current_time

            # Update hand status for sentence builder
            self.sentence_builder.update_hand_status(hand_detected)

            # Display
            current, hold_time = self.sentence_builder.get_current()
            sentence_text = self.sentence_builder.get_sentence_text()
            self._draw_display(frame, current, hold_time, sentence_text,
                             self.stabilizer, hand_detected, fps,
                             getattr(self, '_last_word', None),
                             getattr(self, '_last_confidence', 0.0))
            
            cv2.imshow('Sign Language Translator', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sentence = self.sentence_builder.force_speak()
                if sentence:
                    self.speech.speak(sentence)
            elif key == ord('c'):
                self.sentence_builder.current_sentence = []
                print("🧹 Sentence cleared")
            elif key == ord('f'):
                self.flip_camera = not self.flip_camera
                print(f"🔄 Flip: {'ON' if self.flip_camera else 'OFF'}")
            elif key == ord('r'):
                self.stabilizer.clear()
                print("🔄 Stabilizer reset")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_display(self, frame, current, hold_time, sentence_text,
                      stabilizer, hand_detected, fps,
                      raw_word=None, raw_confidence=0.0):
        h, w, _ = frame.shape
        
        # Top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (800, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 40
        
        # Title and hand status
        if hand_detected:
            cv2.putText(frame, "STABILIZED TRANSLATOR - HAND DETECTED", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "STABILIZED TRANSLATOR - NO HAND", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 30
        
        # Current sign with hold progress
        if current:
            hold_progress = min(hold_time / 0.5, 1.0)
            bar_width = 200
            filled = int(bar_width * hold_progress)
            
            cv2.rectangle(frame, (20, y-5), (20+bar_width, y+15), (50,50,50), -1)
            if filled > 0:
                color = (0,255,0) if hold_progress >= 1.0 else (0,255,255)
                cv2.rectangle(frame, (20, y-5), (20+filled, y+15), color, -1)
            
            status = "READY TO ADD" if hold_progress >= 1.0 else "HOLDING..."
            cv2.putText(frame, f"Sign: {current.upper()} [{status}] ({hold_time:.1f}s)", 
                       (20+bar_width+20, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Make a sign (hold for 0.5s)", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 30
        
        # Buffer visualization
        buffer_display = stabilizer.get_buffer_display()
        buffer_info = f"Buffer: [{buffer_display}] {len(stabilizer.prediction_buffer)}/{stabilizer.buffer_size}"
        cv2.putText(frame, buffer_info, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 20
        
        # Buffer legend
        cv2.putText(frame, "-:No Hand  o:Hand Only  #:Gesture", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        
        # Sentence
        if sentence_text:
            cv2.putText(frame, f"Sentence: {sentence_text}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Sentence: (none)", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        # Timer
        if sentence_text:
            remaining = max(0, 2.0 - (time.time() - self.sentence_builder.last_gesture_time))
            cv2.putText(frame, f"Speaking in: {remaining:.1f}s", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Info at bottom
        info_text = f"FPS: {fps} | Signs: {len(self.signs)} | Flip: {'ON' if self.flip_camera else 'OFF'}"
        cv2.putText(frame, info_text, (20, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Live confidence display  (helps debug similar-sign confusion)
        if raw_word and raw_confidence > 0:
            conf_pct  = int(raw_confidence * 100)
            # Colour: green ≥75%, yellow ≥50%, red <50%
            if raw_confidence >= 0.75:
                conf_color = (0, 255, 0)
            elif raw_confidence >= 0.50:
                conf_color = (0, 220, 255)
            else:
                conf_color = (0, 80, 255)
            cv2.putText(frame, f"Raw: {raw_word}  {conf_pct}%",
                       (20, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 2)
        
        # Instructions
        cv2.putText(frame, "f:Flip | s:Speak | c:Clear | r:Reset | q:Quit", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

if __name__ == "__main__":
    app = ASLToSpeech()
    app.run()