from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')

import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_v3 import *
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.face_utils import FaceAligner
import glob
import dlib
import time


np.set_printoptions(threshold=np.nan)


FRmodel = InceptionV3()
print("Total Params:", FRmodel.count_params())
detector = dlib.get_frontal_face_detector()
facial_shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
eye_opening_threshold = 0.2

def initialize():
    #load_weights_from_FaceNet(FRmodel)
    #Weights are automatically loaded when using InceptionV3 from keras
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        print(file)
        identity = os.path.splitext(os.path.basename(file))[0]
        print(identity)
        imgpath = glob.glob(file+"/*")[5]
        print(imgpath)
        database[identity] = fr_utils.img_path_to_encoding(imgpath, FRmodel)
    return database


def initialize_with_data_augmentation(verbose_mode = True):
    #Weights are automatically loaded when using InceptionV3 from keras
    database = {}
    print("Loading User info from Database")

    # load all images of individuals from the database
    for user in glob.glob("images/*"):
        username = os.path.splitext(os.path.basename(user))[0]
        if(verbose_mode == True):
            print("Loading Info from all images of ", username)
            database[username] = fr_utils.image_folder_to_encoding(user, FRmodel)
    return database


def eye_aspect_ratio(eye):
    # Eye = array of Point landmarks of eye in clockwise direction
    # Eye[0] = Leftmost point where upper and lower eyelids meet
    # Eye[1] = Left  point on upper eyelid just above iris
    # Eye[2] = Right point on upper eyelid just above iris
    # Eye[3] = Rightmost point where upper and lower eyelids meet
    # Eye[4] = Right point on lower eyelid just below iris
    # Eye[5] = Left  point on lwoer eyelid just below iris

    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


def recognize_face(face_descriptor, database):
    encoding = img_to_encoding(face_descriptor, FRmodel)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' % (name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name

    return str(identity), min_dist

def recognize_face_advanced(face_descriptor, database, min_over_avg = True):
    encoding = img_to_encoding(face_descriptor, FRmodel)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # identity = None
        dist = np.zeros(db_enc.shape[0])
        min_user_dist = 100
        avg_user_dist = 100
        for i in range(db_enc.shape[0]):

            # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist[i] = np.linalg.norm(db_enc[i,:] - encoding)

            # Taking minimum distance from any image of a user as that user's distance
            # Another more secure approach could be to take an average
            if dist[i] < min_user_dist:
                min_user_dist = dist[i]
            avg_user_dist = np.mean(dist)
        if(min_over_avg == True):
            print('Min distance for %s is %s' % (name, min_user_dist))
            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if min_user_dist < min_dist:
                min_dist = min_user_dist
                identity = name
        else:
            print('Avg distance for %s is %s' % (name, avg_user_dist))
            if avg_user_dist < min_dist:
                min_dist = avg_user_dist
                identity = name

    return str(identity), min_dist


def overlay_face_info(img, img_rgb, database,ear):
    faces = detector(img_rgb)
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        faceno = 0
        for face in faces:
            faceno = faceno + 1
            print('Detecting Face no.',faceno)
            (x, y, w, h) = face_utils.rect_to_bb(face)
            image = img_rgb[y:y + h, x:x + w]
            print(image.shape)
            t1 = time.time()
            if image.shape[0] * image.shape[1] != 0 :
                name, min_dist = recognize_face_advanced(image, database)
                print('Time for recognize_face_advcd = {} secs'.format(time.time() - t1))
                t1 = time.time()
                if ear > eye_opening_threshold:
                    if min_dist < 0.4:
                                    # cv2.rectangle(image,(top left corner x,y),(bottom right corner x,y), (color R,G,B), linewidth, lineType, shift)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                        cv2.putText(img, "Dist : " + str(np.round(min_dist,3)), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, 'Eyes Closed', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                print('Time for printing name n box = {} secs'.format(time.time() - t1))

def recognize(update_db = False):



    if update_db == True:
        print("Updating DB")
        gen_database = initialize_with_data_augmentation()
        np.save('database.npy',gen_database)

    # Load the encodings from database
    database = np.load('database.npy').item()
    print(database['Mayank'].shape)
    print(database['Swati'].shape)

    cap = cv2.VideoCapture(0)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame_no = 0
    while True:
        frame_no = frame_no + 1
        frame_time = time.time()
        print("##########################################################")
        print("Analysing Frame no: ", frame_no)
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # t1 = time.time()
        subjects = detector(gray, 0)
        #print('Time to detect Face: {} secs'.format(time.time() - t1)

        for subject in subjects:
            t1 = time.time()
            subject_time = time.time()
            shape = facial_shape_predictor(gray, subject)
            print('Time for Facial Shape Predictor: {} secs'.format(time.time() - t1))
            t1 = time.time()
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            print('Time for shape to np: {} secs'.format(time.time() - t1))
            t1 = time.time()
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            print('Time for face feature calcs: {} secs'.format(time.time() - t1))

            t1 = time.time()
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
            print('Time to draw eye cntrs: {} secs'.format(time.time() - t1))

            overlay_face_info(img, img_rgb, database,ear)
            print('Overall Time for 1 subject: {} secs'.format(time.time() - subject_time))
        print('Overall Time for this frame: {} secs'.format(time.time() - frame_time))
        print("##########################################################")
        print(" ")
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize(update_db = False)
