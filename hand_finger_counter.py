import cv2
import mediapipe as mp
import time
import numpy as np
import threading
from collections import deque

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DESIRED_FPS = 30

MP_MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.6
MP_MIN_TRACKING_CONFIDENCE = 0.6
MP_MODEL_COMPLEXITY = 0


class VideoCaptureThread:
    def __init__(self, src=0, width=640, height=480, name="VideoThread"):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.name = name
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

def fingers_up_count(hand_landmarks, handedness_str):
    lm = [(l.x, l.y, l.z) for l in hand_landmarks.landmark]

    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    scale = dist(lm[0], lm[9]) + 1e-6
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    thumb_tip_x = lm[4][0]
    thumb_mcp_x = lm[2][0]
    x_threshold = 0.25 * scale

    if handedness_str.lower().startswith('right'):
        thumb_is_up = (thumb_mcp_x - thumb_tip_x) > x_threshold
    else:
        thumb_is_up = (thumb_tip_x - thumb_mcp_x) > x_threshold
    fingers.append(bool(thumb_is_up))

    for tip_id, pip_id in zip(tips_ids[1:], pip_ids[1:]):
        tip_y = lm[tip_id][1]
        pip_y = lm[pip_id][1]
        is_up = (pip_y - tip_y) > (0.02 * scale)
        fingers.append(bool(is_up))

    count = sum(fingers)
    return int(count), fingers

def main():
    cap = VideoCaptureThread(src=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MP_MAX_NUM_HANDS,
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE
    )

    prev_time = time.time()
    fps_deque = deque(maxlen=30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            _, orig_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    handedness_str = handedness.classification[0].label
                    count, fingers_bool_list = fingers_up_count(hand_landmarks, handedness_str)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_spec,
                        drawing_spec
                    )

                    cv2.putText(frame, f"{handedness_str} hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Fingers: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    base_x = 10
                    base_y = 90
                    for i, up in enumerate(fingers_bool_list):
                        color = (0, 255, 0) if up else (0, 0, 255)
                        cv2.circle(frame, (base_x + i * 30, base_y), 10, color, -1)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (orig_w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Hand Finger Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
