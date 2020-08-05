import cv2

# image file
img_file = 'images/cars.jpg'

# pre-trained car classifier
car_detection_file = 'classifier_files/vehicle_detection_haarcascades.xml'

# create opencv image
img = cv2.imread(img_file)


# display image
cv2.imshow('Cars on a highway', img)


# Wait for key press - Don't autoclose
cv2.waitKey()

print("Code completed")