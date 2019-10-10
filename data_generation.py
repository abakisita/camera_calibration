#! /usr/bin/env python
'''This script is for generating data
1. Press 'c' to capture image and display it.
2. Press any button to continue.
3. Press 'q' to quit.
'''

import os
import cv2

def get_data_path():
    """ Returns the path of folder in this repository named aruco_data

    :returns: str
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(current_dir, 'aruco_data')
    return data_dir

def main(data_dir):
    """ Save images to a folder

    :data_dir: str
    """
    print("Images will be stored at: " + data_dir)

    camera = cv2.VideoCapture(0)
    ret, img = camera.read()

    counter = 0
    while True:
        name = os.path.join(data_dir, str(counter)+".jpg")
        # for i in range(4):
        #     camera.grab()
        ret, img = camera.read()
        cv2.imshow("img", img)


        if cv2.waitKey(20) & 0xFF == ord('c'):
            cv2.imwrite(name, img)
            cv2.imshow("img", img)
            counter += 1
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    camera.release()

if __name__ == "__main__":
    DATA_DIR = get_data_path()
    main(DATA_DIR)
