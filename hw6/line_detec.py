"""
Lane Finder (simple)
====================
Read frames from a Raspberry Pi camera (or a video file), detect the lane center
with basic computer‑vision operators, draw diagnostics on the frames, and (optionally)
write the annotated video to disk.

Usage examples
--------------
From the Pi camera on /dev/video0 and display only:
    python lane_finder_simple.py --camera 0

From an *.mp4* file and also save the processed frames:
    python lane_finder_simple.py --camera track.mp4 --save_video --output out.mp4

Required third‑party packages
----------------------------
    pip install opencv-python numpy
"""
from pathlib import Path
import argparse
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Gray‑scale + Otsu threshold + morphological opening to clean noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


def canny_edge(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.Canny(blur, 50, 150)


def region_of_interest(img: np.ndarray, bottom_ratio: float = 0.33) -> np.ndarray:
    """Keep only the bottom *bottom_ratio* part of the frame (as a trapezoid)."""
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0.03 * w), h),
        (int(0.10 * w), int((1 - bottom_ratio) * h)),
        (int(0.90 * w), int((1 - bottom_ratio) * h)),
        (int(0.97 * w), h),
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)


def detect_lane(frame: np.ndarray, bottom_ratio: float = 0.33):
    """Return (lane_center_x | None, debug_frame)."""
    debug = frame.copy()
    h, w = frame.shape[:2]

    processed = region_of_interest(canny_edge(preprocess(frame)), bottom_ratio)

    lines = cv2.HoughLinesP(
        processed, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=50, maxLineGap=20
    )

    left, right = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:      # near‑horizontal lines are ignored
                continue
            cx = (x1 + x2) / 2
            if slope < 0 and cx < w / 2:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and cx > w / 2:
                right.append((x1, y1, x2, y2))

    lane_center_x = None
    lane_color = (0, 255, 0)
    if left and right:
        lx1, ly1, lx2, ly2 = np.mean(left, axis=0).astype(int)
        rx1, ry1, rx2, ry2 = np.mean(right, axis=0).astype(int)
        cv2.line(debug, (lx1, ly1), (lx2, ly2), lane_color, 3)
        cv2.line(debug, (rx1, ry1), (rx2, ry2), lane_color, 3)
        lx_bot = lx1 + (h - ly1) * (lx2 - lx1) / (ly2 - ly1)
        rx_bot = rx1 + (h - ry1) * (rx2 - rx1) / (ry2 - ry1)
        lane_center_x = (lx_bot + rx_bot) / 2
    elif left:
        x1, y1, x2, y2 = left[0]
        cv2.line(debug, (x1, y1), (x2, y2), lane_color, 3)
        lane_center_x = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
    elif right:
        x1, y1, x2, y2 = right[0]
        cv2.line(debug, (x1, y1), (x2, y2), lane_color, 3)
        lane_center_x = x1 + (h - y1) * (x2 - x1) / (y2 - y1)

    if lane_center_x is not None:
        offset_px = w / 2 - lane_center_x
        cv2.line(debug, (w // 2, h), (int(lane_center_x), h), (0, 0, 255), 3)
        cv2.putText(debug, f"Offset {offset_px:.1f}px", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return lane_center_x, debug

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=15,
                        help="Camera index (e.g. 0) or video file path.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--save_video", action="store_true",
                        help="Save annotated frames to --output file.")
    parser.add_argument("--output", default="lane_output.mp4",
                        help="Output file when --save_video is given.")
    args = parser.parse_args()

    cam_source = int(args.camera) if Path(str(args.camera)).name == str(args.camera) else args.camera
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (args.width, args.height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, dbg = detect_lane(frame)
            cv2.imshow("Lane debug", dbg)
            if writer is not None:
                writer.write(dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
