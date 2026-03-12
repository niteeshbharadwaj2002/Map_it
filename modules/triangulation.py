import cv2
import numpy as np

def triangulate_points(pts1, pts2, K, R, t, mask):
    inliers = mask.ravel() > 0
    inlier_pts1 = pts1[inliers]
    inlier_pts2 = pts2[inliers]

    if len(inlier_pts1) < 8:
        return np.empty((0, 3))

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points_4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    # Filter: keep points in front of both cameras and within reasonable distance
    depths = points_3d[:, 2]
    distances = np.linalg.norm(points_3d, axis=1)
    median_dist = np.median(distances)
    valid = (depths > 0) & (distances < 10 * median_dist)

    return points_3d[valid]
