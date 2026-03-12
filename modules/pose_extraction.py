import cv2
import numpy as np

def recover_camera_motion(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    print("--- Camera Motion Recovered ---")
    print("Rotation Matrix (Tilt/Turn):\n", R)
    print("\nTranslation Vector (Direction of movement):\n", t)
    
    valid_points = np.sum(mask_pose > 0)
    print(f"\nUsed {valid_points} valid points to calculate motion.")
    
    return R, t, mask_pose