import cv2
import numpy as np
import os
from modules.feature_detection import match_features   # Your matching logic
from modules.pose_extraction import recover_camera_motion  # Your motion logic
from modules.triangulation import triangulate_points        # Your triangulation logic

# --- CONFIGURATION ---
# Replace with your actual calibration data from Step 1
K = np.array([[538.37885281, 0, 238.57756636], 
              [0, 539.02359429, 421.29827541], 
              [0, 0, 1]], dtype=np.float32)

def save_ply(filename, points):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

FRAME_DIR = "extracted_frames"
# Get sorted list of frames: frame_0000.jpg, frame_0001.jpg...
frames = sorted([os.path.join(FRAME_DIR, f) for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])

def run_slam():
    total_trajectory = []
    all_3d_points = []
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))
    
    # Loop through frame pairs
    for i in range(len(frames) - 5): # Skipping 5 frames at a time for better motion
        img1_path = frames[i]
        img2_path = frames[i+5]
        
        print(f"Processing Pair: {os.path.basename(img1_path)} & {os.path.basename(img2_path)}")

        # STEP 2: Get Matches (From your features.py)
        pts1, pts2 = match_features(img1_path, img2_path)
        
        if len(pts1) < 20:
            print("Not enough matches, skipping...")
            continue

        # STEP 3: Recover Motion (From your geometry.py)
        R, t, mask = recover_camera_motion(pts1, pts2, K)

        # Log the movement
        total_trajectory.append(t)
        print(f"Estimated Direction of Move: {t.flatten()}")

        # STEP 4: Triangulate matched points into 3D
        points_3d = triangulate_points(pts1, pts2, K, R, t, mask)

        if len(points_3d) > 0:
            # Transform local 3D points into the global coordinate frame
            global_points = (R_global @ points_3d.T).T + t_global.T
            all_3d_points.append(global_points)
            print(f"Triangulated {len(global_points)} 3D points")

        # Update global pose for the next iteration
        t_global = t_global + R_global @ t
        R_global = R @ R_global
        
    # --- Save point cloud as PLY ---
    if all_3d_points:
        cloud = np.vstack(all_3d_points)
        print(f"\nTotal 3D points: {len(cloud)}")
        save_ply("output_map.ply", cloud)
        print("Saved point cloud to output_map.ply")
    else:
        print("No 3D points were triangulated.")

    print("Done processing all frames.")

if __name__ == "__main__":
    run_slam()