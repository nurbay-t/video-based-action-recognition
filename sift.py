import cv2
import numpy as np

def extract_sift_features_from_video(video_path):
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    all_keypoints = []
    all_descriptors = []

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Break if the video is over
        if not ret:
            break
        
        # Convert the frame to grayscale for SIFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

    cap.release()

    return all_keypoints, all_descriptors

video_path = '/Users/rakhatm/Desktop/CV_Project/ucf action/Diving-Side/001/2538-5_70133.avi'
keypoints, descriptors = extract_sift_features_from_video(video_path)

# If you want to visualize the keypoints for a specific frame
# frame_number = 10  # example
# img = cv2.drawKeypoints(cv2.imread('path_to_frame_image.jpg'), keypoints[frame_number], None)
# cv2.imshow('SIFT keypoints', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(keypoints, descriptors)
print(len(keypoints), len(descriptors))
