"""detect pedestrians in an image"""

import cv2
import random

# image files
img_file = random.choice(
    ['images/pedestrians.jpg', 'images/pedestrians2.jpg', 'images/pedestrians3.jpg']
)

# read image in opencv format
img = cv2.imread(img_file)

# pre-trained classifier
full_body_classifier = 'classifier_files/haarcascade_fullbody.xml'

# create a classifier object
full_body_detector = cv2.CascadeClassifier(full_body_classifier)



# display
cv2.imshow('Pedestrians' ,img)
cv2.waitKey()

print("Code completed")