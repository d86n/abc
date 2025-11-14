import ctypes
import cv2
import numpy as np
import time
import serial
import threading

def main():
    camera = Camera()
    camera_matrix, dist_coeffs = camera.load_data()
    aruco_markers = ArUcoMarkers(camera_matrix, dist_coeffs)
    processor = Processor(aruco_markers, camera)

    is_running_event = threading.Event()
    is_running_event.set()

    # ser = Serial(processor, is_running_event)

    # time.sleep(3)

    # ser.reader_thread()

    # ser.writer_thread()

    while is_running_event.is_set():
        ret, frame = camera.read_frame()
        if not ret:
            is_running_event.clear()
            break

        _, ids, rvecs, tvecs = aruco_markers.detect_markers(frame)

        aruco_markers.draw_local_reference_frame(ids, rvecs, tvecs, frame)

        camera.frame = frame
        camera.draw_global_reference_frame(frame)

        marker_positions = processor.get_markers_positions(ids, rvecs, tvecs)
        if marker_positions is not None:
            aruco_markers.show_marker_positions(frame, marker_positions, ids, rvecs, tvecs)

        window_name = "Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)

        # ctypes.windll.user32.ShowWindow(ctypes.windll.user32.FindWindow(None, window_name), 3)

        if cv2.waitKey(1) == ord('q'):
            is_running_event.clear()
            break

    camera.stop()

    # ser.close()

    cv2.destroyAllWindows()

class Serial:
    def __init__(self, processor, is_running_event):
        self.serial_port = 'COM4'
        self.baud_rate = 115200
        self.processor = processor
        self.is_running_event = is_running_event

    def run(self):
        return serial.Serial(self.serial_port, self.baud_rate, timeout=0.1, write_timeout=0, dsrdtr=False)

    def reader_thread(self):
        ser = self.run()
        serial_reader_thread = threading.Thread(target=self.processor.read_serial, args=(ser, self.is_running_event))
        serial_reader_thread.daemon = True
        serial_reader_thread.start()

    def writer_thread(self):
        ser = self.run()
        serial_writer_thread = threading.Thread(target=self.processor.writer_loop, args=(ser, self.is_running_event))
        serial_writer_thread.daemon = True
        serial_writer_thread.start()

class Camera:
    def __init__(self):
        self.calib_data = np.load('calib_data.npz')
        self.camera_matrix = self.calib_data['mtx']
        self.dist_coeffs = self.calib_data['dist']

        self.origin_position = (1800, 100)

        # Mở camera
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

        # Đọc khung hình đầu tiên
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Không thể mở camera.")

        # Cài đặt threading
        self.is_running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                self.is_running = False

    def read_frame(self):
        with self.lock:
            if self.frame is None:
                return False, None
            frame_copy = self.frame.copy()
        return True, frame_copy

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.cap.release()

    def draw_global_reference_frame(self, frame_to_draw_on):
        cv2.circle(frame_to_draw_on, self.origin_position, 3, (0, 0, 255), -1)
        cv2.circle(frame_to_draw_on, self.origin_position, 5, (255, 255, 255), 2)
        axis_length = 150
        # Trục x
        x_end = (self.origin_position[0] - axis_length, self.origin_position[1])
        cv2.arrowedLine(frame_to_draw_on, self.origin_position, x_end, (0, 0, 255), 2)
        cv2.putText(frame_to_draw_on, 'x', (x_end[0] - 20, x_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Trục y
        y_end = (self.origin_position[0], self.origin_position[1] + axis_length)
        cv2.arrowedLine(frame_to_draw_on, self.origin_position, y_end, (0, 255, 0), 2)
        cv2.putText(frame_to_draw_on, 'y', (y_end[0] + 10, y_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def load_data(self):
        return self.camera_matrix, self.dist_coeffs

class ArUcoMarkers:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = 0.13  # = 13 cm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.axis_length = 0.1

    def detect_markers(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        corners, ids, _ = self.detector.detectMarkers(thresh)

        if ids is not None and corners is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix,
                                                                 self.dist_coeffs)
            return corners, ids, rvecs, tvecs

        return None, None, None, None

    def draw_local_reference_frame(self, ids, rvecs, tvecs, frame):
        axis_points = np.array([
            [0,                               0,                 0],
            [self.axis_length,                0,                 0],
            [0,                self.axis_length,                 0],
            [0,                               0, -self.axis_length]
        ], dtype=np.float32)

        if ids is not None and tvecs is not None and len(tvecs) > 0:
            for i in range(len(ids)):
                rvec = rvecs[i]
                tvec = tvecs[i]

                projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

                origin = tuple(projected_points[0][0].astype(int))
                x_axis = tuple(projected_points[1][0].astype(int))
                y_axis = tuple(projected_points[2][0].astype(int))

                cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)
                cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)

    def show_marker_positions(self, frame, marker_positions, ids, rvecs, tvecs):
        for marker_data in marker_positions:
            marker_id = marker_data['id']
            x = marker_data['x']
            y = marker_data['y']
            angle = marker_data['angle']

            if ids is not None:
                for i in range(len(ids)):
                    if ids[i][0] == marker_id:
                        rvec = rvecs[i]
                        tvec = tvecs[i]

                        center_3d = np.array([[0, 0, 0]], dtype=np.float32)
                        center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                        center_pixel = tuple(center_2d[0][0].astype(int))

                        text_pos = (center_pixel[0] + 20, center_pixel[1] - 20)

                        text = f"ID {marker_id}"
                        color = (255, 255, 255)

                        coord_text = f"({x:.2f},{y:.2f})"
                        angle_text = f"{angle:.0f} degrees"

                        cv2.rectangle(frame, (text_pos[0] - 5, text_pos[1] - 15),
                                     (text_pos[0] + 90, text_pos[1] + 35), (0, 0, 0), -1)

                        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(frame, coord_text, (text_pos[0], text_pos[1] + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(frame, angle_text, (text_pos[0], text_pos[1] + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

class Processor:
    def __init__(self, aruco_markers, camera):
        self.angle = None
        self.aruco_markers = aruco_markers
        self.camera = camera
        self.coordinates = None
        self.origin_marker_id = 0

    def get_markers_positions(self, ids, rvecs, tvecs):
        if ids is not None and tvecs is not None and len(tvecs) > 0:
            origin_coordinates = None
            origin_index = None

            for i in range(len(ids)):
                if ids[i][0] == self.origin_marker_id:
                    origin_coordinates = tvecs[i][0]
                    origin_index = i
                    break

            if origin_coordinates is None:
                origin_coordinates = tvecs[0][0]
                origin_index = 0

            results = []
            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec = rvecs[i]
                tvec = tvecs[i]
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])

                angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                angle = 180 - abs(np.degrees(angle_rad))

                if i == origin_index:
                    x, y = 0.0, 0.0
                else:
                    marker_coordinates = tvec[0]
                    relative_coordinates = marker_coordinates - origin_coordinates

                    x = -relative_coordinates[0]
                    y = relative_coordinates[1]

                results.append({
                    'id': marker_id,
                    'x': x,
                    'y': y,
                    'angle': angle
                })

            return results
        return None

    @staticmethod
    def read_serial(ser, is_running_event):
        while is_running_event.is_set():
            try:
                if ser.in_waiting > 0:
                    line_bytes = ser.readline()
                    line_string = line_bytes.decode("utf-8").strip()
                    if line_string:
                        print(f"\n[NHẬN TỪ SLAVE]: {line_string}")
                time.sleep(0.01)
            except serial.SerialException as e:
                print(f"Lỗi Serial: {e}.")
                break

    @staticmethod
    def write_serial(ser):
        try:
            data_input_string = input("Định dạng: 'ID,vx,vy' (hoặc 'exit' để thoát): ")

            if data_input_string.lower() == 'exit':
                return False  # Trả về False để báo hiệu DỪNG

            data_parts_list = data_input_string.split(',')
            if len(data_parts_list) != 3:
                print("Lỗi đầu vào: Phải nhập đúng 3 giá trị.")
                return True  # Trả về True để TIẾP TỤC VÒNG LẶP

            int(data_parts_list[0].strip())
            float(data_parts_list[1].strip())
            float(data_parts_list[2].strip())

            # 4. Gửi dữ liệu (dạng Text + '\n')
            data_to_send = data_input_string + "\n"

            ser.write(data_to_send.encode('utf-8'))

            return True

        except ValueError:
            print("Lỗi: Sai kiểu dữ liệu. Phải là số.")
            return True  # Vẫn tiếp tục vòng lặp
        except serial.SerialException as e:
            print(f"Lỗi Serial: {e}.")
            return False  # Dừng nếu lỗi serial

    def writer_loop(self, ser, is_running_event):
        while is_running_event.is_set():
            try:
                if not self.write_serial(ser):
                    is_running_event.clear()
                    break
            except EOFError:
                is_running_event.clear()
                break

if __name__ == "__main__":
    main()