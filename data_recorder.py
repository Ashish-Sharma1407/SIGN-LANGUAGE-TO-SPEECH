import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import time

class DataRecorder:
    def __init__(self):
        print("\n" + "="*60)
        print("SIGN LANGUAGE RECORDER - CAMO STUDIO")
        print("="*60)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create data directory
        if not os.path.exists('training_data'):
            os.makedirs('training_data')
        
        # All 15 signs (must match train_model.py)
        self.signs = ['student']
        self.current_sign_index = 0
        self.recorded_samples = {sign: 0 for sign in self.signs}
        self.load_existing_data()
        
    def load_existing_data(self):
        for sign in self.signs:
            filename = f'training_data/{sign}.pkl'
            if os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                        self.recorded_samples[sign] = len(data)
                    print(f"📂 Loaded {len(data)} samples for '{sign}'")
                except:
                    pass
    
    def normalize_landmarks(self, landmarks_list):
        try:
            if len(landmarks_list) == 0:
                return None
            landmarks = np.array(landmarks_list[0])
            wrist = landmarks[0]
            normalized = landmarks - wrist
            max_val = np.max(np.abs(normalized))
            if max_val > 0:
                normalized = normalized / max_val
            return normalized.flatten()
        except:
            return None
    
    def record_sign(self, sign_name, landmarks):
        filename = f'training_data/{sign_name}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            data = []
        data.append(landmarks)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        self.recorded_samples[sign_name] = len(data)
        print(f"✅ Recorded {sign_name} - Total: {len(data)}")
        return len(data)
    
    def init_camo(self):
        """Initialize Camo Studio camera — auto-detects correct index"""
        print("\n🔧 Scanning for Camo Studio camera...")

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
                    break   # found working backend for this index, move to next index
                else:
                    print(f"  ⚠️  index={idx} [{bname}]: black frame (brightness={bright:.1f})")
                    cap.release()

            # Stop scanning as soon as we found Camo (index >= 1)
            if camo_cap is not None:
                break

        if camo_cap is not None:
            if builtin_cap is not None:
                builtin_cap.release()   # release built-in, we don't need it
            print(f"\n🎥 Using Camo Studio camera")
            return camo_cap

        if builtin_cap is not None:
            print(f"\n🎥 Camo not found — falling back to built-in camera")
            return builtin_cap

        return None
    
    def run(self):
        # Initialize Camo
        cap = self.init_camo()
        
        if cap is None:
            print("\n❌ Could not initialize Camo Studio!")
            print("\nPlease check:")
            print("1. Is Camo Studio RUNNING on your PC?")
            print("2. Can you see your phone video in Camo window?")
            print("3. In Camo Settings, set:")
            print("   - Resolution: 640x480")
            print("   - FPS: 30")
            print("   - Disable all filters")
            print("4. Restart Camo Studio")
            print("5. Try disconnecting and reconnecting phone")
            return
        
        print("\n✅ Camo Studio connected successfully!")
        print("\n🎯 HOW TO RECORD:")
        print("  1. Hold your hand in front of phone camera")
        print("  2. Make the sign and HOLD IT")
        print("  3. Press SPACE to save one sample")
        print("  4. Press 'a' to toggle AUTO-RECORD (saves every 1s automatically)")
        print("  5. Record in MULTIPLE lighting conditions for better model!")
        print("     → Record some samples in dim light (home)")
        print("     → Record some samples in bright light (college)")
        print("  6. Press 'n' for next sign, 'p' for previous")
        print("\n🎮 Controls: SPACE=Save, a=AutoRecord, n=Next, p=Prev, d=Delete, q=Quit")
        print("="*60)
        
        last_save = 0
        message = ""
        msg_time = 0
        auto_record = False      # toggle with 'a' key
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Lost frame, reconnecting...")
                time.sleep(0.5)
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # ── Lighting quality indicator ──────────────────────────────────
            brightness = frame.mean()
            if brightness < 50:
                light_label = "VERY LOW"
                light_color = (0, 0, 200)
                light_tip   = "  Tip: Move to brighter area or turn on a light!"
            elif brightness < 100:
                light_label = "LOW"
                light_color = (0, 140, 255)
                light_tip   = "  Tip: OK for home samples - also record in bright light"
            elif brightness < 180:
                light_label = "GOOD"
                light_color = (0, 220, 0)
                light_tip   = ""
            else:
                light_label = "BRIGHT"
                light_color = (255, 255, 0)
                light_tip   = "  Tip: Good for college demo samples!"

            # ── Camera status bar ───────────────────────────────────────────
            cv2.putText(frame, "CAMO STUDIO - WORKING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Light: {light_label} ({brightness:.0f})", (350, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, light_color, 2)
            
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
                        h, w, _ = frame.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        landmarks.append([lm.x, lm.y, lm.z])
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    landmarks_list.append(landmarks)
            
            current_sign = self.signs[self.current_sign_index]
            samples = self.recorded_samples[current_sign]
            
            # Display UI
            y = 70
            cv2.putText(frame, f"Recording: {current_sign.upper()}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += 35
            
            # Progress bar
            progress = min(samples / 30, 1.0)
            bar_width = 300
            filled = int(bar_width * progress)
            cv2.rectangle(frame, (10, y-10), (10+bar_width, y+10), (50,50,50), -1)
            if filled > 0:
                color = (0,255,0) if samples >= 30 else (0,255,255)
                cv2.rectangle(frame, (10, y-10), (10+filled, y+10), color, -1)
            cv2.putText(frame, f"{samples}/30", (10+bar_width+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y += 30
            
            # Hand status
            if hand_detected:
                cv2.putText(frame, "✓ HAND DETECTED - Press SPACE", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(frame, "✗ NO HAND - Show your hand", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            y += 30

            # Auto-record indicator
            if auto_record:
                ar_color = (0, 255, 0) if hand_detected else (0, 100, 200)
                cv2.putText(frame, "AUTO-RECORD ON  (a=off)", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, ar_color, 2)
                y += 25

            # Lighting tip
            if light_tip:
                cv2.putText(frame, light_tip, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, light_color, 1)
                y += 22
            
            if message:
                cv2.putText(frame, message, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Bottom instructions
            cv2.putText(frame, "SPACE:Save | a:AutoRec | n:Next | p:Prev | d:Del | q:Quit", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            cv2.imshow('Camo Studio Recorder', frame)
            
            key = cv2.waitKey(1) & 0xFF

            # ── Auto-record: save automatically when hand detected ──────────
            if auto_record and hand_detected and (time.time() - last_save) > 1.0:
                norm = self.normalize_landmarks(landmarks_list)
                if norm is not None:
                    current_sign = self.signs[self.current_sign_index]
                    new = self.record_sign(current_sign, norm)
                    last_save = time.time()
                    message = f"[AUTO] ✓ Saved! Total: {new}"
                    msg_time = time.time()

            if key == ord(' ') and hand_detected and (time.time() - last_save) > 1:
                norm = self.normalize_landmarks(landmarks_list)
                if norm is not None:
                    new = self.record_sign(current_sign, norm)
                    last_save = time.time()
                    message = f"✓ Saved! Total: {new}"
                    msg_time = time.time()
            
            elif key == ord('a'):
                auto_record = not auto_record
                status = "ON" if auto_record else "OFF"
                print(f"🔄 Auto-record: {status}")
                message = f"Auto-record: {status}"
                msg_time = time.time()

            elif key == ord('n'):
                self.current_sign_index = (self.current_sign_index + 1) % len(self.signs)
                print(f"➡️ Now: {self.signs[self.current_sign_index]}")
            
            elif key == ord('p'):
                self.current_sign_index = (self.current_sign_index - 1) % len(self.signs)
                print(f"⬅️ Now: {self.signs[self.current_sign_index]}")
            
            elif key == ord('d'):
                filename = f'training_data/{current_sign}.pkl'
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                    if data:
                        data.pop()
                        with open(filename, 'wb') as f:
                            pickle.dump(data, f)
                        self.recorded_samples[current_sign] = len(data)
                        message = f"🗑️ Deleted! {len(data)} left"
                        msg_time = time.time()
            
            elif key == ord('q'):
                break
            
            # Clear message after 2 seconds
            if message and time.time() - msg_time > 2:
                message = ""
        
        cap.release()
        cv2.destroyAllWindows()
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "="*60)
        print("RECORDING SUMMARY")
        print("="*60)
        for sign, count in self.recorded_samples.items():
            print(f"{sign}: {count}/30 samples")
        print("="*60)

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()