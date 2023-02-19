import random

import cv2 as cv
import threading
import numpy as np
import random

campath = 'data/cam%d/intrinsics.xml'
boardpath = 'data/checkerboard.xml'
videopath = 'data/%s/%s.avi'
samplesize = 30


class CameraConfig:
    _instance_lock = threading.Lock()

    mtx: dict = {}
    dist: dict = {}
    video: dict = {}
    cBWidth: int
    cBHeight: int
    cBSquareSize: int
    _rvecs: dict = {}
    _tvecs: dict = {}

    def __new__(cls, *args, **kwargs):
        if not hasattr(CameraConfig, "_instance"):
            with CameraConfig._instance_lock:
                if not hasattr(CameraConfig, "_instance"):
                    CameraConfig._instance = object.__new__(cls)
                    cls.load_parameter(cls)
        return CameraConfig._instance

    # read the checkerboard data from checkerboard.xml
    def load_parameter(self):
        fs = cv.FileStorage(boardpath, cv.FILE_STORAGE_READ)
        self.cBWidth = fs.getNode("CheckerBoardWidth").real()
        self.cBHeight = fs.getNode("CheckerBoardHeight").real()
        self.cBSquareSize = fs.getNode("CheckerBoardSquareSize").real()
        fs.release()
        for i in range(1, 5):
            fs = cv.FileStorage(campath % i, cv.FILE_STORAGE_READ)
            mtx = np.mat(fs.getNode("CameraMatrix").mat())
            self.mtx['cam%d' % i] = mtx
            dist = np.mat(fs.getNode("DistortionCoeffs").mat())
            self.dist['cam%d' % i] = dist
            fs.release()
        print('parameters loaded')

    # update the 'cname' file, if no input, update all
    def __update(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.__update('cam%d' % i)
            return
        # print('update %s' % cname)
        fs = cv.FileStorage('data/%s/intrinsics.xml' % cname, cv.FILE_STORAGE_WRITE)
        fs.write("CameraMatrix", np.matrix(self.mtx[cname]))
        fs.write("DistortionCoeffs", np.array(self.dist[cname]))
        fs.release()

    def mtx_dist_compute(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.mtx_compute(cname='cam%d' % i)
            return
        criteria1 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria0 = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

        size = (int(self.cBHeight), int(self.cBWidth))

        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = (self.cBSquareSize * np.mgrid[0:size[0], 0:size[1]]).T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        cap = cv.VideoCapture(videopath % (cname, 'intrinsics'))
        if not cap.isOpened():
            return
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            h, w = img.shape[:2]
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, size, None, criteria0)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria1)
                imgpoints.append(corners2)
        cap.release()
        if not objpoints:
            return
        print('start calibrating')

        random.seed()
        while True:
            objpoints2 = []
            imgpoints2 = []

            lp = np.arange(len(imgpoints)).tolist()
            lp = random.sample(lp, samplesize)
            for j in range(len(lp)):
                objpoints2.append(objpoints[lp[j]])
                imgpoints2.append(imgpoints[lp[j]])
            lp.clear()

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints2, imgpoints2, (h, w), None, None)
            print(ret)
            if ret < 0.35:
                break
        self.mtx[cname] = mtx
        self.dist[cname] = dist
        self.__update(cname)

    def rt_compute(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.rt_compute(cname='cam%d' % i)
            return

        criteria1 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria0 = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

        size = (int(self.cBHeight), int(self.cBWidth))

        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = (self.cBSquareSize * np.mgrid[0:size[0], 0:size[1]]).T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        cap = cv.VideoCapture(videopath % (cname, 'checkerboard'))
        ret, img = cap.read()
        while ret:
            ret, img = cap.read()
        cap.release()

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, size, None, criteria0)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria1)
            _, self._rvecs[cname], self._tvecs[cname], _ = \
                cv.solvePnPRansac(objp, corners2, self.mtx[cname], self.dist[cname])

# TODO: 计算R矩阵、T矩阵，手动标点找棋盘、计算摄像机位置
# TODO: 背景扣除（超像素&SIFT）
# TODO: 通过视频计算坐标（SIFT、RANSAC）

# for testing
cc = CameraConfig()
cc.mtx_compute()
