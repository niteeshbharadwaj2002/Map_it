import cv2
import numpy as np
import os

# --- SETTINGS ---
# Number of INNER corners (count where black squares touch)
CHESSBOARD_SIZE = (7, 5) 
# Size of a single square side in your preferred unit (e.g., 25.0 for mm)
SQUARE_SIZE = 1.0  

# Termination criteria for refining corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D points in real world space (0,0,0), (1,0,0), ..., (7,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store 3D points and 2D image points
objpoints = [] 
imgpoints = [] 

# Load the video you recorded
video_path = 'calibration/chessboard.mp4'
cap = cv2.VideoCapture(video_path)

print("Starting corner detection... Press 'q' to stop early.")

frame_count = 0
image_size = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Process every 5th frame to speed up and ensure variety
    frame_count += 1
    if frame_count % 5 != 0: continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        
        # Refine corner locations for sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Calibration in Progress', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if not objpoints or image_size is None:
    print("Error: No chessboard corners were found. Check your video.")
    exit(1)

# --- CALIBRATION ---
print("Computing calibration... this may take a moment.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

print("\n--- RESULTS ---")
print("Retval (Error):", ret)
print("\nCamera Matrix (K):\n", mtx)
print("\nDistortion Coefficients (dist):\n", dist)

# Save the results so you can use them in your mapping software
np.savez('camera_params.npz', mtx=mtx, dist=dist)
print("\nParameters saved to camera_params.npz")