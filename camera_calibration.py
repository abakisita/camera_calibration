#! /usr/bin/env python
"""
This code assumes that images used for calibration are of the same arUco marker board provided with code

"""

from __future__ import print_function

import time
import cv2
from cv2 import aruco
import yaml
import numpy as np
import os
from tqdm import tqdm

# Assumed arUco marker board
ARUCO_DICT = aruco.DICT_6X6_1000

#Provide length of the marker's side
MARKERLENGTH = 0.039 # metres

# Provide separation between markers
MARKERSEPARATION = 0.0135 # metres

# flag variable
# True --> Calibrate camera
# False --> Validate results in real time
CALIBRATE_CAMERA = False

def get_img_path():
    """ Return the path where the images are stored
    :returns: str
    """
    root_dir = os.path.abspath(os.path.dirname(__file__))
    calib_imgs_path = os.path.join(root_dir, "aruco_data")
    return calib_imgs_path

def validate_results(board, aruco_dict, aruco_params):
    """ Validate calibrations results

    :board: aruco.GridBoard obj
    :aruco_dict: aruco.Dictionary obj
    :aruco_params: aruco.DetectorParameters obj
    :returns: None
    """
    camera = cv2.VideoCapture(0)
    ret, img = camera.read()

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    print(newcameramtx)

    pose_r, pose_t = [], []
    while True:
        time.sleep(0.1)
        for i in range(4):
            camera.grab()
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=aruco_params)
        # print(corners, ids)
        cv2.imshow("dst", dst)
        if corners == None:
            print ("pass")
        else:

            ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist) # For a board
            # print ("Rotation ", rvec, "Translation", tvec)
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break;
            cv2.waitKey(1)
        cv2.imshow("World co-ordinate frame axes", img_aruco)

def calibrate_camera(image_dir, board, aruco_dict, aruco_params):
    """ Calibrate camera based on the images in image_dir

    :image_dir: str
    :board: aruco.GridBoard obj
    :aruco_dict: aruco.Dictionary obj
    :aruco_params: aruco.DetectorParameters obj

    """
    img_list = []
    calib_fnms = [os.path.join(image_dir, i) for i in os.listdir(image_dir) if i[-4:] == '.jpg']
    print('Using', len(calib_fnms), 'images')
    for file_name in calib_fnms:
        img = cv2.imread(file_name)
        img_list.append(img)
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
            corners_list, id_list, counter, board, img_gray.shape, None, None)

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

def main(image_dir):
    """ Main function

    :image_dir: str

    """
    # For validating results, show aruco board to camera.
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)

    # create arUco board
    board = aruco.GridBoard_create(4, 4, MARKERLENGTH, MARKERSEPARATION, aruco_dict)

    '''uncomment following block to draw and show the board'''
    # img = board.draw((864,1080))
    # cv2.imshow("aruco", img)

    aruco_params = aruco.DetectorParameters_create()

    if CALIBRATE_CAMERA:
        calibrate_camera(image_dir, board, aruco_dict, aruco_params)

    else:
        validate_results(board, aruco_dict, aruco_params)


if __name__ == "__main__":
    IMG_DIR = get_img_path()
    main(IMG_DIR)
    cv2.destroyAllWindows()

