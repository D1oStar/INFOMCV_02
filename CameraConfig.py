import cv2 as cv
import threading
import numpy as np
import random

campath = 'data/cam%d/intrinsics.xml'
boardpath = 'data/checkerboard.xml'
videopath = 'data/%s/%s.avi'
samplesize = 30

click = 0
manual_position = np.zeros((4, 2), np.float32)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)


# mouse click event
def click_event(event, x, y, flags, params):
    global click
    global manual_position
    if event == cv.EVENT_LBUTTONDOWN:
        if click < 4:
            manual_position[click] = (x, y)
            # print(manual_position)
            # cv.circle(img, (x, y), 6, (0, 0, 255), -1)
            # cv.imshow('img', img)
            click += 1


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    # imgpoints
    imgpts0 = tuple(imgpts[0].ravel())
    imgpts1 = tuple(imgpts[1].ravel())
    imgpts2 = tuple(imgpts[2].ravel())

    # y axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts0[0]), int(imgpts0[1])), (255, 0, 0), 3)
    # x axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts1[0]), int(imgpts1[1])), (0, 255, 0), 3)
    # z axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts2[0]), int(imgpts2[1])), (0, 0, 255), 3)

    return img


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

    def save_xml(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.save_xml('cam%d' % i)
            return
        fs = cv.FileStorage('data/%s/config.xml' % cname, cv.FILE_STORAGE_WRITE)
        fs.write("CameraMatrix", np.matrix(self.mtx[cname]))
        fs.write("DistortionCoeffs", np.array(self.dist[cname]))
        fs.write("RMatrix", np.matrix(self._rvecs[cname]))
        fs.write("TMatrix", np.matrix(self._tvecs[cname]))
        fs.release()

    def load_xml(self):
        for i in range(1, 5):
            cname = 'cam%d' % i
            fs = cv.FileStorage('data/%s/config.xml' % cname, cv.FILE_STORAGE_READ)
            mtx = np.mat(fs.getNode("CameraMatrix").mat())
            self.mtx[cname] = mtx
            dist = np.mat(fs.getNode("DistortionCoeffs").mat())
            self.dist[cname] = dist
            rvecs = np.mat(fs.getNode("RMatrix").mat())
            self._rvecs[cname] = rvecs
            tvecs = np.mat(fs.getNode("TMatrix").mat())
            self._tvecs[cname] = tvecs
            fs.release()

    def mtx_dist_compute(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.mtx_dist_compute(cname='cam%d' % i)
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
            for i in range(4, 5):
                self.rt_compute(cname='cam%d' % i)
            return

        criteria1 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria0 = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

        size = (int(self.cBHeight), int(self.cBWidth))

        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        cap = cv.VideoCapture(videopath % (cname, 'checkerboard'))
        # cap = cv.VideoCapture(videopath % (cname, 'intrinsics'))
        ret, img = cap.read()
        # while not ret:
        #    ret, img = cap.read()
        # cap.release()

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cv.imshow('img', img)
        cv.setMouseCallback('img', click_event)
        cv.waitKey(0)

        # print(click)
        if click == 4:
            # print(manual_position)
            dst_pts = np.float32(manual_position)
            scr_pts = np.float32([[0, 0], [7, 0], [7, 5], [0, 5]])
            M = cv.getPerspectiveTransform(scr_pts, dst_pts)

            img_pts = np.array([[[j, i]] for i in range(6) for j in range(8)], dtype=np.float32)
            xxx_pts = self.cBSquareSize * np.array([[[i, j, 0]] for i in range(8) for j in range(6)], dtype=np.float32)
            obj_pts = cv.perspectiveTransform(img_pts, M)
            corners = obj_pts

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria1)

            retval, self._rvecs[cname], self._tvecs[cname] = \
                cv.solvePnP(objp, corners2, self.mtx[cname], self.dist[cname])
            imgpts, jac = cv.projectPoints(axis, self._rvecs[cname], self._tvecs[cname], self.mtx[cname],
                                           self.dist[cname])
            self.save_xml(cname)
            img = draw(img, corners2, imgpts)
            cv.imshow('img', img)
            cv.waitKey(0)
        '''
        ret, corners = cv.findChessboardCorners(gray, size, None, criteria0)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria1)
            _, self._rvecs[cname], self._tvecs[cname], _ = \
                cv.solvePnPRansac(objp, corners2, self.mtx[cname], self.dist[cname])
        '''

    def subtract_background(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.subtract_background(cname='cam%d' % i)
            return

        capbg = cv.VideoCapture(videopath % (cname, 'background'))
        capfg = cv.VideoCapture(videopath % (cname, 'video'))
        ret, bg = capbg.read()
        while ret:
            ret, bg = capbg.read()
        ret, fg = capfg.read()
        while ret:
            ret, fg = capfg.read()
        capbg.release()
        capfg.release()

        bgblur = cv.GaussianBlur(bg, (10, 10), 0.5)
        fgblur = cv.GaussianBlur(fg, (10, 10), 0.5)

        mask = np.zeros(bgblur.shape[:2], dtype=np.uint8)
        mask[bgblur != fgblur] = 1
        cv.imshow('mask', mask)

        cv.waitKey(0)

        cv.destroyAllWindows()


# TODO: 计算R矩阵、T矩阵，手动标点找棋盘、计算摄像机位置
# TODO: 背景扣除（超像素&SIFT）
# TODO: 通过视频计算坐标（SIFT、RANSAC）


# for testing
cc = CameraConfig()
# cc.mtx_dist_compute()
cc.rt_compute()
cc.subtract_background()
