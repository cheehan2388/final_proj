import cv2
import numpy as np


vid = cv2.VideoCapture('autonomous_vid.mp4')

fps = vid.get(cv2.CAP_PROP_FPS)             # 幀率
w   = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
writer  = cv2.VideoWriter('lane_output.mp4', fourcc, fps, (w, h))
# 影像預處理
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)#二值化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)#侵蝕再膨脹
    return opened

# 邊緣偵測：
def canny_edge(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)#高斯模糊
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)#Sobel
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)#算mag
    mag_uint8 = cv2.convertScaleAbs(mag)#浮點32 轉u8
    edges = cv2.Canny(mag_uint8, 50, 150)#canny
    return edges

#保留 小區域偵測避免太多噪聲
def region_int(img):

    h, w = img.shape[:2]
    # 以畫面尺寸動態定義一個梯形 ROI
    
    polygons = np.array([[
        (int(0.03*w), h),        # 左下
        (int(0.1*w), int(0.3*h)),  # 左上
        (int(0.8*w), int(0.3*h)),  # 右上
        (int(0.96*w), h)         # 右下
    ]], dtype=np.int32)

    # 根據 img 形狀產生黑色矩陣
    mask = np.zeros_like(img)
    # 塗白 ROI 區域
    cv2.fillPoly(mask, polygons, 255)
    # 與原本對比用AND來取ROI 區域的影像
    return cv2.bitwise_and(img, mask)


while True:
    ret, frame = vid.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    proc = preprocess(frame)
    edges = canny_edge(proc)

    edges_roi = region_int(edges)
    #  霍夫直線偵測 
    lines = cv2.HoughLinesP(edges_roi,
                            rho=1,
                            theta=np.pi/180,
                            threshold=50,
                            minLineLength=50,
                            maxLineGap=20)

    #  分左右線收集
    left_lines  = []
    right_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            # 忽略垂直（無限）和 接近水平的綫
            if x2 == x1: 
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue

            # 以影像中心當分界，在左邊-斜率 就是左邊，右邊正斜率就是右邊
            line_center_x = (x1 + x2) / 2
            if slope < 0 and line_center_x < w/2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and line_center_x > w/2:
                right_lines.append((x1, y1, x2, y2))

    # 畫出車道線：
    status_text = "No lane"
    color = (0,255,0)
    if left_lines and right_lines:
        status_text = "Two lanes"
        # 平均全部綫
        lx1 = np.mean([l[0] for l in left_lines]).astype(int)
        ly1 = np.mean([l[1] for l in left_lines]).astype(int)
        lx2 = np.mean([l[2] for l in left_lines]).astype(int)
        ly2 = np.mean([l[3] for l in left_lines]).astype(int)
        cv2.line(frame, (lx1, ly1), (lx2, ly2), color, 3)

        rx1 = np.mean([l[0] for l in right_lines]).astype(int)
        ry1 = np.mean([l[1] for l in right_lines]).astype(int)
        rx2 = np.mean([l[2] for l in right_lines]).astype(int)
        ry2 = np.mean([l[3] for l in right_lines]).astype(int)
        cv2.line(frame, (rx1, ry1), (rx2, ry2), color, 3)

    elif left_lines:
        status_text = "Only left lane"
        l = left_lines[0]
        cv2.line(frame, (l[0],l[1]), (l[2],l[3]), color, 3)
    elif right_lines:
        status_text = "Only right lane"
        l = right_lines[0]
        cv2.line(frame, (l[0],l[1]), (l[2],l[3]), color, 3)

    cv2.putText(frame,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,0,255),
                3)



    car_center_x = w / 2 # 車身中心
    lane_center_x = None
    # 用線性外插求直線延伸到影像底部 (y = h) 時的 x 座標
    # 公式: x = x1 + (y_target - y1) / (y2 - y1) * (x2 - x1)
    if left_lines and right_lines:
   
        lx1, ly1, lx2, ly2 = np.mean(left_lines,  axis=0).astype(int)
        rx1, ry1, rx2, ry2 = np.mean(right_lines, axis=0).astype(int)
     
        left_x_bot  = lx1 + (h - ly1)*(lx2-lx1)/(ly2-ly1)
        right_x_bot = rx1 + (h - ry1)*(rx2-rx1)/(ry2-ry1)

        lane_center_x = (left_x_bot + right_x_bot) / 2 

    
    elif left_lines:
        x1, y1, x2, y2 = left_lines[0]
        lane_center_x = x1 + (h - y1)*(x2-x1)/(y2-y1)

    elif right_lines:
        x1, y1, x2, y2 = right_lines[0]
        lane_center_x = x1 + (h - y1)*(x2-x1)/(y2-y1)

    # 畫線 還有 顯示 offset
    if lane_center_x is not None:
        offset = car_center_x - lane_center_x
        cv2.line(frame,
                 (int(car_center_x), h),
                 (int(lane_center_x), h),
                 (0,0,255), 3)
        cv2.putText(frame,
                    f"Offset: {offset:.1f}px",
                    (30, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)

    cv2.imshow('Frame', frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
