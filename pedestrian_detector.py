"""detect pedestrians in an image"""

import cv2
import random

# image files
img_file = random.choice(
    ['images/pedestrians.jpg', 'images/pedestrians1.jpeg', 'images/pedestrians3.jpg']
)

# read image in opencv format
img = cv2.imread(img_file)

# pre-trained classifier
full_body_classifier = 'classifier_files/haarcascade_fullbody.xml'

# create a classifier object
full_body_detector = cv2.CascadeClassifier(full_body_classifier)

#convert image to gray for faster processing
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect pedestrians
pedestrians = full_body_detector.detectMultiScale(black_n_white)
# print(pedestrians)

# draw rectangles around pedestrians
for x, y, w, h in pedestrians:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,100,100), 2)

# display
cv2.imshow('Pedestrians', img)
cv2.waitKey(10000)

print("Code completed")