import cv2
import numpy as np

# Đọc ảnh
frame_left = cv2.imread(r"C:/Users/ADMIN/OneDrive/Documents/abc/abc/photos/query.jpg")
frame_right = cv2.imread(r"C:/Users/ADMIN/OneDrive/Documents/abc/abc/photos/train.jpg")

if frame_left is None or frame_right is None:
    raise FileNotFoundError("Không đọc được ảnh, kiểm tra lại đường dẫn!")

# Resize về cùng chiều cao
h = min(frame_left.shape[0], frame_right.shape[0])
frame_left = cv2.resize(frame_left, (int(frame_left.shape[1] * h / frame_left.shape[0]), h))
frame_right = cv2.resize(frame_right, (int(frame_right.shape[1] * h / frame_right.shape[0]), h))

# Chuyển ảnh xám
gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

# Template matching để tìm vùng overlap
overlap_width = 200  # bạn có thể chỉnh giá trị này phù hợp với setup
res = cv2.matchTemplate(gray_left, gray_right[:, :overlap_width], cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(res)
x_offset = max_loc[0]

# Ghép ảnh: cắt bỏ phần overlap
stitched = np.hstack((frame_left[:, :x_offset], frame_right))

cv2.imshow("Stitched", stitched)
cv2.waitKey(0)
cv2.destroyAllWindows()
