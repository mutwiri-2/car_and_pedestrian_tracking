"""detect and track pedestrians and cars in a video"""

import cv2

#videos
video_file = 'videos/dash_cam.mp4'
video_file = 'videos/pedestrians.mp4'

# get video using opencv
video = cv2.VideoCapture(video_file)

# classifier files
pedestrian_classifier_file = 'classifier_files/haarcascade_fullbody.xml'
car_classifier_file = 'classifier_files/vehicle_detection_haarcascades.xml'

# create classifier objects for pedestrians and cars
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

while True:
    # read the current frame from VideoCapture object
    read_successful, frame = video.read()

    if read_successful:
        # convert to black n white for faster processing
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break



    # Display
    cv2.imshow('Cars and Pedestrians', grayscaled_frame)
    cv2.waitKey(1)