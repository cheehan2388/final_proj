# lane_following_pi.py
"""
終端指令：
    python lane_following_pi.py --port /dev/ttyAMA0 --baud 115200 --camera 0

Raspberry Pi 讀取相機 → 計算 offset → 模糊控制 → 經 UART 傳給 Arduino
-----------------------------------------------------------------------------  
必要第三方：
    pip install opencv-python numpy pyserial scikit-fuzzy

Arduino 接收訊息範例： <STEER:1500,SPEED:120>\n
"""
import argparse
import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import serial

# ---------------------------------------------------------------------------
# Configuration –– 可直接於命令列覆寫 ––
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    port="/dev/ttyUSB0",   # Pi 世界常為這個；USB 轉 TTL 則可能 /dev/ttyUSB0
    baud=115200,
    camera=0,               # 若為 mp4 檔請填路徑字串
    width=640,
    height=480,
    roi_bottom_ratio=0.33,  # 只用底下那 1/3 區域找車道
    fps_out=30,
    save_video=False,
)

STEER_CENTER = 1500        # 舵機中心 (µs)
STEER_RANGE  = 400         # +-µs 對應 +-最大角度
BASE_SPEED   = 120         # PWM 百分比或自行對應
MAX_OFFSET   = 320         # 畫面半寬，用於正規化 e (-1~1)

# ---------------------------------------------------------------------------
# 視覺處理
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """灰階 + Otsu 二值 + 形態學開運算去雜訊"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


def canny_edge(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(img: np.ndarray, bottom_ratio: float = 0.33) -> np.ndarray:
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0.03 * w), h),
        (int(0.1  * w), int((1 - bottom_ratio) * h)),
        (int(0.9  * w), int((1 - bottom_ratio) * h)),
        (int(0.97 * w), h),
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)


def detect_lane(frame: np.ndarray) -> Tuple[Optional[float], np.ndarray]:
    """回傳 lane_center_x (None 若無) 及 Debug frame"""
    debug = frame.copy()
    h, w = frame.shape[:2]
    processed = region_of_interest(canny_edge(preprocess(frame)))

    lines = cv2.HoughLinesP(
        processed, rho=1, theta=np.pi/180, threshold=50,
        minLineLength=50, maxLineGap=20
    )
    left, right = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            cx = (x1 + x2) / 2
            if slope < 0 and cx < w / 2:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and cx > w / 2:
                right.append((x1, y1, x2, y2))

    lane_center_x = None
    color = (0, 255, 0)
    if left and right:
        lx1, ly1, lx2, ly2 = np.mean(left, axis=0).astype(int)
        rx1, ry1, rx2, ry2 = np.mean(right, axis=0).astype(int)
        cv2.line(debug, (lx1, ly1), (lx2, ly2), color, 3)
        cv2.line(debug, (rx1, ry1), (rx2, ry2), color, 3)
        lx_bot = lx1 + (h - ly1) * (lx2 - lx1) / (ly2 - ly1)
        rx_bot = rx1 + (h - ry1) * (rx2 - rx1) / (ry2 - ry1)
        lane_center_x = (lx_bot + rx_bot) / 2
    elif left:
        x1, y1, x2, y2 = left[0]
        lane_center_x = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        cv2.line(debug, (x1, y1), (x2, y2), color, 3)
    elif right:
        x1, y1, x2, y2 = right[0]
        lane_center_x = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        cv2.line(debug, (x1, y1), (x2, y2), color, 3)

    if lane_center_x is not None:
        offset_px = w / 2 - lane_center_x
        cv2.line(debug, (w // 2, h), (int(lane_center_x), h), (0, 0, 255), 3)
        cv2.putText(debug, f"Offset {offset_px:.1f}px", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return lane_center_x, debug

# ---------------------------------------------------------------------------
# (簡化) 模糊控制器
# ---------------------------------------------------------------------------
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    e_univ = ctrl.Antecedent(np.linspace(-1, 1, 7), "e")
    de_univ = ctrl.Antecedent(np.linspace(-1, 1, 7), "de")
    w_univ = ctrl.Consequent(np.linspace(-1, 1, 7), "w")
    names = ["NB", "NM", "NS", "Z", "PS", "PM", "PB"]
    for ant in (e_univ, de_univ, w_univ):
        ant.automf(names=names)
    rules = []
    for i, en in enumerate(names):
        for j, den in enumerate(names):
            out_idx = 6 - max(min(i + j - 6, 6), 0)
            rules.append(ctrl.Rule(e_univ[en] & de_univ[den], w_univ[names[out_idx]]))
    fuzzy_ctrl = ctrl.ControlSystem(rules)
    fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    def fuzzy_controller(e: float, de: float) -> float:
        fuzzy_sim.input["e"] = e
        fuzzy_sim.input["de"] = de
        fuzzy_sim.compute()
        return float(fuzzy_sim.output["w"])
except ModuleNotFoundError:
    print("[WARN] scikit-fuzzy not installed, fall back to PD controller")
    def fuzzy_controller(e: float, de: float, kp: float = 1.5, kd: float = 0.5):
        return np.tanh(kp * e + kd * de)

# ---------------------------------------------------------------------------
# Serial sending
# ---------------------------------------------------------------------------

def format_msg(omega_norm: float) -> bytes:
    steer_us = int(STEER_CENTER + omega_norm * STEER_RANGE)
    msg = f"<STEER:{steer_us},SPEED:{BASE_SPEED}>\n"
    return msg.encode()

# ---------------------------------------------------------------------------
# Async loops
# ---------------------------------------------------------------------------

async def camera_loop(cap: cv2.VideoCapture, q: asyncio.Queue, cfg: dict):
    w = cfg["width"]
    max_offset = w / 2
    e_prev = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue
        lane_x, dbg = detect_lane(frame)
        if cfg["save_video"]:
            q.writer.write(dbg)
        if lane_x is None:
            await asyncio.sleep(0.01)
            continue
        offset = (w / 2 - lane_x) / max_offset
        de = offset - e_prev
        e_prev = offset
        await q.put((offset, de))

async def control_loop(q_in: asyncio.Queue, q_out: asyncio.Queue):
    while True:
        e, de = await q_in.get()
        omega = fuzzy_controller(e, de)
        await q_out.put(omega)

async def serial_loop(q: asyncio.Queue, ser: serial.Serial):
    while True:
        omega = await q.get()
        ser.write(format_msg(omega))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULTS["port"])
    parser.add_argument("--baud", type=int, default=DEFAULTS["baud"])
    parser.add_argument("--camera", default=DEFAULTS["camera"], help="相機編號或影片路徑")
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()
    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    ser = serial.Serial(cfg["port"], cfg["baud"], timeout=0.1)
    cam_source = int(cfg["camera"]) if Path(str(cfg["camera"])) == Path(str(cfg["camera"])).name else cfg["camera"]
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

    if cfg["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("lane_output.mp4", fourcc, cfg["fps_out"], (cfg["width"], cfg["height"]))
    else:
        writer = None

    q_error = asyncio.Queue(maxsize=1)
    q_omega = asyncio.Queue(maxsize=1)

    try:
        asyncio.run(asyncio.gather(
            camera_loop(cap, q_error, cfg | {"writer": writer}),
            control_loop(q_error, q_omega),
            serial_loop(q_omega, ser),
        ))
    except KeyboardInterrupt:
        print("\n[INFO] Exit by Ctrl‑C")
    finally:
        cap.release()
        if writer:
            writer.release()
        ser.close()


if __name__ == "__main__":
    main()
