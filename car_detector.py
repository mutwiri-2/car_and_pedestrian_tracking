"""detect cars in an image"""

import cv2
import random

# image file
img_file = random.choice(['images/cars.jpg', 'images/cars2.jpg', 'images/cars3.jpg'])

# pre-trained car classifier
car_detection_file = 'classifier_files/vehicle_detection_haarcascades.xml'

# create opencv image
img = cv2.imread(img_file)

# create a car classifier object
car_tracker = cv2.CascadeClassifier(car_detection_file)

# convert image to black and white - makes algorith faster
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply classifier object on image to detect cars
cars = car_tracker.detectMultiScale(black_n_white)
# print(cars) 

# draw rectangles around the cars in the image
for x, y, w, h in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,250), 2)

# display image
cv2.imshow('Cars on a highway', img)


# Wait for key press - Don't autoclose
cv2.waitKey()

# print("Code completed")