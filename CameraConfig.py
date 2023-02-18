import cv2 as cv
import threading
import numpy as np

campath = 'data/cam%d/intrinsics.xml'
boardpath = 'data/checkerboard.xml'
videopath = 'data/%s/%s.avi'


class CameraConfig:
    _instance_lock = threading.Lock()

    mtx: dict = {}
    dist: dict = {}
    video: dict = {}
    cBWidth: int
    cBHeight: int
    cBSquareSize: int

    def __new__(cls, *args, **kwargs):
        if not hasattr(CameraConfig, "_instance"):
            with CameraConfig._instance_lock:
                if not hasattr(CameraConfig, "_instance"):
                    CameraConfig._instance = object.__new__(cls)
                    # read the checkerboard data from checkerboard.xml
                    fs = cv.FileStorage(boardpath, cv.FILE_STORAGE_READ)
                    cls.cBWidth = fs.getNode("CheckerBoardWidth").real()
                    cls.cBHeight = fs.getNode("CheckerBoardHeight").real()
                    cls.cBSquareSize = fs.getNode("CheckerBoardSquareSize").real()
                    fs.release()
                    for i in range(1, 5):
                        fs = cv.FileStorage(campath % i, cv.FILE_STORAGE_READ)
                        mtx = np.mat(fs.getNode("CameraMatrix").mat())
                        cls.mtx['cam%d' % i] = mtx
                        dist = np.mat(fs.getNode("DistortionCoeffs").mat())
                        cls.dist['cam%d' % i] = dist
                        fs.release()
        return CameraConfig._instance

    # update the 'cname' file, if no input, update all
    def update(self, cname=[]):
        if not cname:
            # print('update all')
            for i in range(1, 5):
                self.update('cam%d' % i)
            return
        # print('update %s' % cname)
        fs = cv.FileStorage('data/%s/intrinsics.xml' % cname, cv.FILE_STORAGE_WRITE)
        fs.write("CameraMatrix", np.matrix(self.mtx[cname]))
        fs.write("DistortionCoeffs", np.array(self.dist[cname]))
        fs.release()

    def video_load(self, cname=[]):
        if not cname:
            # print('update all')
            for i in range(1, 5):
                self.video_load('cam%d' % i)
            return
        vtype = ['background', 'checkerboard', 'intrinsics', 'video']
        v: dict = {}
        for i in vtype:
            cap = cv.VideoCapture(videopath % (cname, i))
            v0 = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    v0.append(frame)
                else:
                    print('Read finished: file\t%15s.avi\tof\t%s' % (i, cname))
                    break
            cap.release()
            v[i] = v0
        self.video[cname] = v


cc = CameraConfig()
cc.video_load()
