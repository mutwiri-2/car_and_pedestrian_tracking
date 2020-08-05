import cv2

# image file
img_file = 'images/cars.jpg'

# pre-trained car classifier
car_detection_file = 'classifier_files/vehicle_detection_haarcascades.xml'

# create opencv image
img = cv2.imread(img_file)

# create a car classifier object
car_tracker = cv2.CascadeClassifier(car_detection_file)

# convert image to black and white - makes algorith faster
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# display image
cv2.imshow('Cars on a highway', img)


# Wait for key press - Don't autoclose
cv2.waitKey()

print("Code completed")