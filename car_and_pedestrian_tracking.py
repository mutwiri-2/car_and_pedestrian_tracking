"""detect and track pedestrians and cars in a video"""

import cv2

#videos
video_file = 'videos/dash_cam.mp4'
video_file = 'videos/pedestrians.mp4'

# read video using opencv
video = cv2.VideoCapture(video_file)

# classifier files
pedestrian_classifier_file = 'classifier_files/haarcascade_fullbody.xml'
car_classifier_file = 'classifier_files/vehicle_detection_haarcascades.xml'