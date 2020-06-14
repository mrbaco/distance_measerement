import numpy as np
import glob
import cv2
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

chessboard_size = (9, 6)
images_size = (1920, 1440)

test_file_name = 'images/JZhRq083h38.jpg'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob(ROOT_DIR + '/calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        print('File', fname, 'done!')
    else:
        print('There is a problem with file:', fname)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, images_size[::-1], None, None)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, images_size, 1, images_size)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print('Total error:', mean_error / len(objpoints))

np.savez(ROOT_DIR + '/camera_params.npz', mtx=mtx, dist=dist, newcameramtx=newcameramtx)
print('Prepared data:', 'mtx=', mtx, 'dist=', dist, 'newcameramtx=', newcameramtx, sep="\n")

test_image = cv2.imread(ROOT_DIR + '/' + test_file_name)
new_test_image = cv2.undistort(test_image, mtx, dist, None, newcameramtx)

cv2.imshow('Test image', np.concatenate((
    cv2.resize(test_image, tuple(int(current / 3) for current in images_size)),
    cv2.resize(new_test_image, tuple(int(current / 3) for current in images_size))
), axis=1))

cv2.waitKey(50000)

cv2.destroyAllWindows()

print('Done!')
