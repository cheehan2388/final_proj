import cv2
import numpy as np
import serial

# ---------- Fuzzy Control Functions ----------
def membership_LN(n_offset):
    if n_offset <= -1:
        return 1
    elif -1 < n_offset < -0.5:
        return (-0.5 - n_offset) / 0.5
    else:
        return 0

def membership_SN(n_offset):
    if -1 <= n_offset <= -0.5:
        return (n_offset + 1) / 0.5
    elif -0.5 < n_offset <= 0:
        return -n_offset / 0.5
    else:
        return 0

def membership_Z(n_offset):
    if -0.5 <= n_offset <= 0:
        return (n_offset + 0.5) / 0.5
    elif 0 < n_offset <= 0.5:
        return (0.5 - n_offset) / 0.5
    else:
        return 0

def membership_SP(n_offset):
    if 0 <= n_offset <= 0.5:
        return n_offset / 0.5
    elif 0.5 < n_offset <= 1:
        return (1 - n_offset) / 0.5
    else:
        return 0

def membership_LP(n_offset):
    if n_offset >= 1:
        return 1
    elif 0.5 < n_offset < 1:
        return (n_offset - 0.5) / 0.5
    else:
        return 0

def fuzzy_controller(n_offset):
    fs_LN = membership_LN(n_offset)
    fs_SN = membership_SN(n_offset)
    fs_Z = membership_Z(n_offset)
    fs_SP = membership_SP(n_offset)
    fs_LP = membership_LP(n_offset)
    
    # Steering values: 1.0 (right), 0.0 (straight), -1.0 (left)
    numerator = (fs_LN * 1.0 + fs_SN * 0.5 + fs_Z * 0.0 + fs_SP * (-0.5) + fs_LP * (-1.0))
    denominator = fs_LN + fs_SN + fs_Z + fs_SP + fs_LP
    if denominator == 0:
        return 0.0
    steering = numerator / denominator
    
    # Convert steering to discrete commands
    if steering > 0.3:
        return "right"
    elif steering < -0.3:
        return "left"
    else:
        return "straight"

# ---------- Original Functions ----------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def canny_edge(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    return cv2.Canny(cv2.convertScaleAbs(mag), 50, 150)

def region_int(img):
    h, w = img.shape[:2]
    poly = np.array([[
        (int(0.03*w), h), (int(0.10*w), int(0.30*h)),
        (int(0.80*w), int(0.30*h)), (int(0.96*w), h)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)

# ---------- Main Code ----------
vid = cv2.VideoCapture(14)
ser = serial.Serial('/dev/ttyUSB0', 57600)  # Adjust port as needed

half_lane_width_px = None

while True:
    ret, frame = vid.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if half_lane_width_px is None:
        half_lane_width_px = w / 2

    car_center_x = w / 2

    # Image processing
    proc = preprocess(frame)
    edges = canny_edge(proc)
    edges_roi = region_int(edges)

    # Hough lines
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180,
                            threshold=50, minLineLength=50, maxLineGap=20)

    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            if slope < 0 and x1 < w/2 and x2 < w/2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > w/2 and x2 > w/2:
                right_lines.append((x1, y1, x2, y2))

    # Draw lines
    status_text = "No lane"
    if left_lines and right_lines:
        status_text = "Two lanes"
        lx1, ly1, lx2, ly2 = np.mean(left_lines, axis=0).astype(int)
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

    # Calculate lane_center_x
    lane_center_x = None
    if left_lines and right_lines:
        lx1, ly1, lx2, ly2 = np.mean(left_lines, axis=0).astype(int)
        rx1, ry1, rx2, ry2 = np.mean(right_lines, axis=0).astype(int)
        left_x_bot = lx1 + (h - ly1) * (lx2 - lx1) / (ly2 - ly1)
        right_x_bot = rx1 + (h - ry1) * (rx2 - rx1) / (ry2 - ry1)
        lane_center_x = (left_x_bot + right_x_bot) / 2
        half_lane_width_px = abs(right_x_bot - left_x_bot) / 2
    elif left_lines:
        x1, y1, x2, y2 = left_lines[0]
        left_x_bot = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        lane_center_x = left_x_bot + half_lane_width_px
    elif right_lines:
        x1, y1, x2, y2 = right_lines[0]
        right_x_bot = x1 + (h - y1) * (x2 - x1) / (y2 - y1)
        lane_center_x = right_x_bot - half_lane_width_px

    # Fuzzy control and serial communication
    if lane_center_x is not None:
        offset = car_center_x - lane_center_x
        normalized_offset = offset / half_lane_width_px
        direction = fuzzy_controller(normalized_offset)
        ser.write(f"{direction}\n".encode())
        
        # Display offset
        cv2.line(frame, (int(car_center_x), h),
                 (int(lane_center_x), h), (0,0,255), 3)
        cv2.putText(frame, f"Offset: {offset:.1f}px Direction: {direction}",
                    (30, h-30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)
    else:
        ser.write("straight\n".encode())  # Default when no lane detected

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
ser.close()