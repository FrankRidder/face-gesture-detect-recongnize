# import OpenCV module
import cv2
import os
import numpy as np

# there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Frank Ridder", "Liza de Graaf", "Vincent Kenbeek", "Robin Vonk", "Jurriaan Mulder", "Martijn Bakker",
            "Bo Sterenborg", "Robin de Jong", "Marijn Stam", "Michel Rummens"]

face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("../training-data/recognizer.xml")


# ### Prediction

# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('../opencv-files/haarcascade_frontalface_alt.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    if len(faces) == 0:
        return None, None

    # for face in faces:
    for rect in faces:
        if rect is not None:
            (x, y, w, h) = rect
            face = gray[y:y + w, x:x + h]

            face = cv2.resize(face, (280, 280))
            # predict the image using our face recognizer
            label, confidence = face_recognizer.predict(face)
            # get name of respective label returned by face recognizer
            if label < 10:
                label_text = subjects[label]
            else:
                label_text = "unkown"

            # draw a rectangle around face detected
            draw_rectangle(img, rect)
            # draw name of predicted person
            draw_text(img, label_text, rect[0], rect[1] - 5)

        elif rect is None:
            print("No face in test")
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

print("Predicting images...")

# load test images
test_img1 = cv2.imread("../test-data/20200106_163538.jpg")
test_img1 = cv2.resize(test_img1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

# perform a prediction
predict(test_img1)
print("Prediction complete")

