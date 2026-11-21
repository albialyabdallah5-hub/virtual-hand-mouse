import math
import os
import time
import cv2
import mediapipe as np_mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

PREFERRED_CAM_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MARGIN_X = 90
MARGIN_Y = 70
SMOOTHING = 4.0
DEADZONE_PX = 2.0
CLICK_COOLDOWN = 0.40

CONFIRM_FRAMES = 3

# شروط الإبهام (كليك شمال عند رفعه)
THUMB_LIFT_RATIO = 0.08
THUMB_LENGTH_RATIO = 0.36
MIN_THUMB_ANGLE_DEG = 135.0

# شروط الخنصر (كليك يمين عند فرده)
PINKY_LENGTH_RATIO = 0.62
MIN_PINKY_ANGLE_DEG = 142.0

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ألوان ثيم أسود وذهبي (بصيغة BGR في OpenCV)
GOLD_COLOR = (0, 215, 255)
BRIGHT_GOLD = (80, 235, 255)
DARK_BG = (15, 15, 15)
LEFT_CLICK_COLOR = (0, 255, 140)
RIGHT_CLICK_COLOR = (0, 140, 255)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

screen_w, screen_h = pyautogui.size()


def open_camera(preferred_idx=1):
    candidates = [preferred_idx] + [i for i in range(0, 5) if i != preferred_idx]
    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] Connected to camera index: {idx}", flush=True)
                return cap
        cap.release()
    return None


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def joint_angle_deg(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_theta = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def evaluate_thumb(pts, hand_scale):
    thumb_mcp = pts[2]
    thumb_ip = pts[3]
    thumb_tip = pts[4]

    lifted_up = (thumb_mcp[1] - thumb_tip[1]) >= hand_scale * THUMB_LIFT_RATIO
    thumb_span = distance(thumb_mcp, thumb_tip)
    angle_ip = joint_angle_deg(thumb_mcp, thumb_ip, thumb_tip)

    ok = (
        lifted_up
        and thumb_span >= hand_scale * THUMB_LENGTH_RATIO
        and angle_ip >= MIN_THUMB_ANGLE_DEG
    )
    info = f"angle={angle_ip:.1f}, span={thumb_span / hand_scale:.2f}, lift={(thumb_mcp[1] - thumb_tip[1]) / hand_scale:.2f}"
    return ok, info


def evaluate_pinky(pts, hand_scale):
    pinky_mcp = pts[17]
    pinky_pip = pts[18]
    pinky_dip = pts[19]
    pinky_tip = pts[20]
    wrist = pts[0]

    angle_pip = joint_angle_deg(pinky_mcp, pinky_pip, pinky_dip)
    angle_dip = joint_angle_deg(pinky_pip, pinky_dip, pinky_tip)
    pinky_span = distance(pinky_mcp, pinky_tip)
    extended_outward = distance(pinky_tip, wrist) > distance(pinky_mcp, wrist) * 1.12

    ok = (
        extended_outward
        and angle_pip >= MIN_PINKY_ANGLE_DEG
        and angle_dip >= MIN_PINKY_ANGLE_DEG
        and pinky_span >= hand_scale * PINKY_LENGTH_RATIO
    )
    info = f"pip={angle_pip:.1f}, dip={angle_dip:.1f}, span={pinky_span / hand_scale:.2f}"
    return ok, info


def draw_hand(img, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], GOLD_COLOR, 2)
    for idx, (x, y) in enumerate(pts):
        radius = 6 if idx in (4, 8, 20) else 4
        color = BRIGHT_GOLD if idx in (4, 8, 20) else GOLD_COLOR
        cv2.circle(img, (x, y), radius, color, cv2.FILLED)
    return pts


def draw_hud(img, status_text, left_count, right_count, w):
    cv2.rectangle(img, (0, 0), (w, 52), DARK_BG, cv2.FILLED)
    cv2.line(img, (0, 52), (w, 52), GOLD_COLOR, 1)

    cv2.putText(
        img,
        f"{status_text} | L:{left_count} R:{right_count}",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        BRIGHT_GOLD,
        2,
    )
    cv2.putText(
        img,
        "Quit: Q",
        (w - 110, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        GOLD_COLOR,
        2,
    )


def main():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] الملف hand_landmarker.task غير موجود داخل مجلد المشروع.", flush=True)
        return

    cap = open_camera(PREFERRED_CAM_INDEX)
    if cap is None:
        print("[ERROR] لم يتم العثور على أي كاميرا متاحة.", flush=True)
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.70,
    )

    prev_x, prev_y = None, None
    last_left_click = 0.0
    last_right_click = 0.0
    left_count = 0
    right_count = 0
    thumb_streak = 0
    pinky_streak = 0
    thumb_armed = True
    pinky_armed = True
    last_timestamp_ms = -1
    start_time = time.monotonic()

    print("[INFO] Virtual Mouse started. Press 'q' in the window to exit.", flush=True)

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, img = cap.read()
            if not success or img is None:
                print("[WARNING] فشل قراءة فريم من الكاميرا.", flush=True)
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape

            cv2.rectangle(
                img,
                (MARGIN_X, MARGIN_Y),
                (w - MARGIN_X, h - MARGIN_Y),
                GOLD_COLOR,
                2,
            )

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = np_mp.Image(image_format=np_mp.ImageFormat.SRGB, data=rgb_img)

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            status_text = "Index Move Only"

            if result.hand_landmarks:
                pts = draw_hand(img, result.hand_landmarks[0], w, h)

                wrist = pts[0]
                middle_mcp = pts[9]
                thumb_tip = pts[4]
                index_tip = pts[8]
                pinky_tip = pts[20]

                hand_scale = max(distance(wrist, middle_mcp), 40.0)

                target_x = float(np.interp(index_tip[0], (MARGIN_X, w - MARGIN_X), (0, screen_w)))
                target_y = float(np.interp(index_tip[1], (MARGIN_Y, h - MARGIN_Y), (0, screen_h)))

                if prev_x is None or prev_y is None:
                    curr_x, curr_y = target_x, target_y
                else:
                    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHING

                if prev_x is None or distance((curr_x, curr_y), (prev_x, prev_y)) >= DEADZONE_PX:
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                cv2.circle(img, index_tip, 10, BRIGHT_GOLD, cv2.FILLED)
                cv2.circle(img, thumb_tip, 9, LEFT_CLICK_COLOR, cv2.FILLED)
                cv2.circle(img, pinky_tip, 9, RIGHT_CLICK_COLOR, cv2.FILLED)

                now = time.monotonic()
                thumb_extended, thumb_info = evaluate_thumb(pts, hand_scale)
                pinky_extended, pinky_info = evaluate_pinky(pts, hand_scale)

                if not thumb_extended:
                    thumb_armed = True
                    thumb_streak = 0

                if not pinky_extended:
                    pinky_armed = True
                    pinky_streak = 0

                if thumb_extended and not pinky_extended and thumb_armed:
                    thumb_streak += 1
                    pinky_streak = 0
                elif pinky_extended and not thumb_extended and pinky_armed:
                    pinky_streak += 1
                    thumb_streak = 0
                else:
                    thumb_streak = 0
                    pinky_streak = 0

                if thumb_streak >= CONFIRM_FRAMES:
                    cv2.circle(img, thumb_tip, 15, LEFT_CLICK_COLOR, 3)
                    status_text = "Left Click (Thumb Up)"
                    if now - last_left_click > CLICK_COOLDOWN:
                        pyautogui.click(button="left")
                        left_count += 1
                        print(f"[CLICK #{left_count + right_count}] LEFT (Thumb) | {thumb_info}", flush=True)
                        last_left_click = now
                        thumb_streak = 0
                        thumb_armed = False
                elif pinky_streak >= CONFIRM_FRAMES:
                    cv2.circle(img, pinky_tip, 15, RIGHT_CLICK_COLOR, 3)
                    status_text = "Right Click (Pinky)"
                    if now - last_right_click > CLICK_COOLDOWN:
                        pyautogui.click(button="right")
                        right_count += 1
                        print(f"[CLICK #{left_count + right_count}] RIGHT (Pinky) | {pinky_info}", flush=True)
                        last_right_click = now
                        pinky_streak = 0
                        pinky_armed = False
            else:
                thumb_streak = 0
                pinky_streak = 0
                thumb_armed = True
                pinky_armed = True

            draw_hud(img, status_text, left_count, right_count, w)
            cv2.imshow("Virtual Mouse", img)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()