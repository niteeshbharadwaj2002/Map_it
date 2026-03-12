import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_features(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 1. Initialize SIFT detector
    sift = cv2.SIFT_create()

    # 2. Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Use FLANN Matcher (Faster for SIFT than Brute Force)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Lowe's Ratio Test (Keep only strong matches)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 5. Visualize the matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plt.figure(figsize=(15, 7))
    # plt.imshow(img_matches)
    # plt.title(f"Found {len(good_matches)} valid matches")
    # print("Attempting to open window... (If this fails, check debug_matches.jpg)")
    # plt.show()

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2
# Test it with your first two frames
# match_features('extracted_frames/frame_0045.jpg', 'extracted_frames/frame_0046.jpg')