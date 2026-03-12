import cv2
import os
import numpy as np

def extract_best_frames(video_path, output_folder, blur_threshold=100.0, motion_threshold=0.02):
    """
    Extracts high-quality, non-blurry frames from a video for SLAM/3D Reconstruction.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    prev_frame = None
    
    print(f"Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Check for Blur (Laplacian Variance)
        # Higher score = Sharper image.
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < blur_threshold:
            count += 1
            continue  # Skip blurry frames

        # 3. Check for Motion (Difference from last saved frame)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(gray, prev_frame)
            motion_score = np.sum(frame_diff) / (gray.size * 255)
            
            # If motion_score is too low, the camera is stationary (redundant data)
            if motion_score < motion_threshold:
                count += 1
                continue

        # Save the frame
        frame_name = f"frame_{saved_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
        
        prev_frame = gray
        saved_count += 1
        count += 1
        
        if saved_count % 10 == 0:
            print(f"Saved {saved_count} frames... (Current blur score: {blur_score:.2f})")

    cap.release()
    print(f"\nDone! Saved {saved_count} high-quality frames to '{output_folder}'.")

# --- SETTINGS ---
VIDEO_FILE = "input_videos/sample_video_1.mp4" # Path to your video
OUTPUT_DIR = "extracted_frames"
# Adjust blur_threshold: 50 is loose, 150 is very strict.
extract_best_frames(VIDEO_FILE, OUTPUT_DIR, blur_threshold=80.0)