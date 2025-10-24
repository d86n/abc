import serial
import time
import numpy as np
from testtt.camera_mapping_combined import CameraMappingCombined

cameramappingcombined = CameraMappingCombined()
cameramappingcombined.camera_thread()

WHEEL_BASE = 14.5               # Khoảng cách giữa 2 bánh (cm)
MAX_WHEEL_SPEED = 150.0         # Giới hạn tốc độ bánh (PWM hoặc cm/s)
PORT = 'COM5'                   # Cổng Serial đến ESP32
BAUD = 115200
DIST_THRESH = 1.0               # Ngưỡng đạt đến đích (cm)

# --- Bộ điều khiển PID đơn giản ---
LINEAR_GAIN = 1.2
ANGULAR_GAIN = 2.5

# --- HÀM CẦN BẠN CUNG CẤP ---
def get_current_position():
    # Giả sử bạn dùng marker đầu tiên trong danh sách
    if cameramappingcombined.markers:
        # Lấy id đầu tiên
        marker_id = next(iter(cameramappingcombined.markers))
        marker = cameramappingcombined.markers[marker_id]
        x = marker['x']
        y = marker['y']
        theta = marker['heading']
        return x, y, theta
    else:
        # Nếu chưa có marker, trả về giá trị mặc định
        return 0.0, 0.0, 0.0
# --- CHUẨN HÓA GÓC [-pi, pi] ---
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# --- GIỚI HẠN ---
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

#Di toi diem mục tiêu
def go_to_target(target):
    print(f"Target: {target}")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # Chờ ổn định kết nối

        while True:
            # 1. Lấy vị trí hiện tại
            x, y, theta = get_current_position()
            # 2. Tính sai số vị trí
            dx = target[0] - x
            dy = target[1] - y
            distance = np.hypot(dx, dy)

            if distance < DIST_THRESH:
                ser.write(b"0 0\n")
                break

            # 3. Góc mong muốn & sai số góc
            desired_theta = np.arctan2(dy, dx)
            dtheta = normalize_angle(desired_theta - theta)

            # 4. Tính vận tốc tiến và góc quay
            v = LINEAR_GAIN * distance
            w = ANGULAR_GAIN * dtheta

            vl = v - (w * WHEEL_BASE / 2.0)
            vr = v + (w * WHEEL_BASE / 2.0)

            # 6. Giới hạn vận tốc
            vl = clamp(vl, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
            vr = clamp(vr, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

            # 7. Gửi lệnh
            cmd = f"{vl:.2f} {vr:.2f}\n"
            ser.write(cmd.encode())

            print(f"{x:.1f} {y:.1f} {theta:.2f}")

            time.sleep(0.1)

        ser.close()

    except serial.SerialException:
        print("Conection error")
    except KeyboardInterrupt:
        ser.write(b"0 0\n")
        ser.close()
        print("\nInterrupted.")

