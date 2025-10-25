import cv2
import numpy as np

# --- 1. Load Calibration Data ---
try:
    calib_data = np.load('calib_data.npz')
    camera_matrix = calib_data['mtx']
    dist_coeffs = calib_data['dist']
    print("Calibration data loaded successfully.")
except FileNotFoundError:
    print("Error: Calibration file 'calib_data_2.npz' not found.")
    print("Please run the calibration script first.")
    exit()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# --- 2. Initialize Camera ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("\nPress 'q' to quit.")

# --- 3. Real-time Undistortion Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # Calculate the new camera matrix and ROI based on the original camera matrix and distortion coefficients.
    # alpha=1 returns an undistorted image with all original pixels, and some black pixels.
    # alpha=0 returns an undistorted image without any black pixels, but some original pixels may be lost.
    # We use alpha=1 here so we can clearly see the full undistorted image before cropping.
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the captured frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # --- Apply ROI to crop the undistorted frame and remove black borders ---
    x, y, crop_w, crop_h = roi
    # Ensure ROI dimensions are valid before cropping
    if crop_w > 0 and crop_h > 0:
        cropped_undistorted_frame = undistorted_frame[y:y + crop_h, x:x + crop_w]
    else:
        # If ROI is invalid (shouldn't happen with alpha=1), fall back to full undistorted
        cropped_undistorted_frame = undistorted_frame

    # --- Display the results ---
    cv2.imshow('Original Video (Distorted)', frame)
    cv2.imshow('Undistorted Video (Cropped)', cropped_undistorted_frame)
    # Optional: Display the full undistorted frame (with black borders) for comparison
    # cv2.imshow('Undistorted Video (Full, with black borders)', undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")