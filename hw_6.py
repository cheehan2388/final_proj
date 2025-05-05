import cv2
import numpy as np

# ---------- ① 讀影片 ----------
vid = cv2.VideoCapture('autonomous_vid.mp4')

# ---------- ② 影像前處理 ----------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# ---------- ③ 邊緣 ----------
def canny_edge(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    return cv2.Canny(cv2.convertScaleAbs(mag), 50, 150)

# ---------- ④ ROI ----------
def region_int(img):
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0.03*w), h), (int(0.10*w), int(0.30*h)),
        (int(0.80*w), int(0.30*h)), (int(0.96*w), h)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)

# ======== ⑤ 先在迴圈外「只初始化一次」半車道寬 ========
half_lane_width_px = None        # → 第一幀用 w/2，之後再更新

while True:
    ret, frame = vid.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if half_lane_width_px is None:          # 第一次才進來
        half_lane_width_px = w / 2

    car_center_x = w / 2                    # 車身中心 (每幀可直接算)

    # ---------- 影像處理 ----------
    proc      = preprocess(frame)
    edges     = canny_edge(proc)
    edges_roi = region_int(edges)

    # ---------- 霍夫直線 ----------
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180,
                            threshold=50, minLineLength=50, maxLineGap=20)

    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:            # 垂直線直接跳過
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:    # 過平的線跳過
                continue
            if slope < 0 and x1 < w/2 and x2 < w/2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > w/2 and x2 > w/2:
                right_lines.append((x1, y1, x2, y2))

    # ---------- 畫線 (不變) ----------
    status_text = "No lane"
    if left_lines and right_lines:
        status_text = "Two lanes"
        lx1, ly1, lx2, ly2 = np.mean(left_lines,  axis=0).astype(int)
        rx1, ry1, rx2, ry2 = np.mean(right_lines, axis=0).astype(int)
        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0,255,0), 3)
        cv2.line(frame, (rx1, ry1), (rx2, ry2), (0,255,0), 3)
    elif left_lines:
        status_text = "Only left lane"
        cv2.line(frame, left_lines[0][:2], left_lines[0][2:], (0,255,0), 3)
    elif right_lines:
        status_text = "Only right lane"
        cv2.line(frame, right_lines[0][:2], right_lines[0][2:], (0,255,0), 3)

    cv2.putText(frame, status_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # ---------- ⑥ 計算 lane_center_x ----------
    lane_center_x = None
    if left_lines and right_lines:                       # (雙線)
        lx1, ly1, lx2, ly2 = np.mean(left_lines,  axis=0).astype(int)
        rx1, ry1, rx2, ry2 = np.mean(right_lines, axis=0).astype(int)
        left_x_bot  = lx1 + (h - ly1) * (lx2 - lx1) / (ly2 - ly1)
        right_x_bot = rx1 + (h - ry1) * (rx2 - rx1) / (ry2 - ry1)
        lane_center_x = (left_x_bot + right_x_bot) / 2
        # **更新**半車道寬
        half_lane_width_px = abs(right_x_bot - left_x_bot) / 2

    elif left_lines:                                      # (單左)
        x1, y1, x2, y2 = left_lines[0]
        left_x_bot = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        lane_center_x = left_x_bot + half_lane_width_px

    elif right_lines:                                     # (單右)
        x1, y1, x2, y2 = right_lines[0]
        right_x_bot = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        lane_center_x = right_x_bot - half_lane_width_px

    # ---------- 顯示偏移 ----------
    if lane_center_x is not None:
        offset = car_center_x - lane_center_x
        cv2.line(frame, (int(car_center_x), h),
                 (int(lane_center_x), h), (0,0,255), 3)
        cv2.putText(frame, f"Offset: {offset:.1f}px",
                    (30, h-30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
