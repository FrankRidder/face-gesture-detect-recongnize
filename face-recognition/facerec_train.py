# coding: utf-8

# Face Recognition with OpenCV

# import OpenCV module
import cv2
import os
import numpy as np

# ### Training Data

# ```
# training-data
# |-------------- s1 map name
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# |-------------- s2 map name
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# ```
# Use rename.sh to rename pictures in the folder
# **Note:** As we have not assigned `label 0` to any person so **the mapping for label 0 is empty**.

# there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Frank Ridder", "Liza de Graaf", "Vincent Kenbeek", "Robin Vonk", "Jurriaan Mulder", "Martijn Bakker",
            "Bo Sterenborg", "Robin de Jong", "Marijn Stam", "Michel Rummens"]


# ### Prepare training data

# function to detect face using OpenCV
def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('../opencv-files/haarcascade_frontalface_alt.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system test-data like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)
            image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            if image is not None:
                # display an image window to show the image
                # cv2.imshow("Training on image...", cv2.resize(image, (600, 900)))
                # cv2.waitKey(100)

                # detect face
                face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                face = cv2.resize(face, (280, 280))
                faces.append(face)
                # add label for this face
                labels.append(label)
            else:
                print("failed on " + image_path)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("../training-data")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# or use EigenFaceRecognizer by replacing above line with
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Now that we have initialized our face recognizer and we also have prepared our training data, it's time to train
# the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer.

# train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


save_loc = "../training-data/recognizer.xml"
print("Training succesful, saving to %s" % save_loc)
face_recognizer.save(save_loc)