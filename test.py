import cv2
import numpy as np

# 建立一張黑色圖像
img = np.zeros((400, 600, 3), dtype=np.uint8)

# 模擬 noisy 左邊線段（例如從 Hough Transform 拿到的）
left_lines = [
    [100, 300, 150, 100],
    [105, 310, 155, 110],
    [95, 290, 145, 90],
    [98, 295, 148, 95]
]

# 畫出所有原始 noisy 線段（藍色）
for l in left_lines:
    x1, y1, x2, y2 = l
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

# 計算平均線段端點
lx1 = int(np.mean([l[0] for l in left_lines]))
ly1 = int(np.mean([l[1] for l in left_lines]))
lx2 = int(np.mean([l[2] for l in left_lines]))
ly2 = int(np.mean([l[3] for l in left_lines]))

# 畫出平均線段（綠色）
cv2.line(img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)

# 顯示圖片
cv2.imshow("Noisy lines vs Averaged line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
