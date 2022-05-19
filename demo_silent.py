import numpy as np
import cv2
import sys
import threading
import time
from collections import deque
from lib.vis import plot_one_bound_count
from lib.intersect import Line, intersect, clockwise
from lib.detector import Detector
from lib.tracker import Tracker
from lib.camera import Camera

PSTART = ''
PEND = ''
BOUNDS = []
line = Line([701, 321], [339, 257])
BOUNDS.append(line)
MODE = 0
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
        
def background_forward(camera, detector, tracker, BOUNDS):
    while True:
        start_time = time.time()
        img0 = camera.capture.copy()
        img0 = np.ascontiguousarray(img0, dtype=np.float32)
        img0 /= 255.0
        x = np.transpose(img0, (2, 0, 1))[np.newaxis, ...]
        online_tlwhs = []
        online_ids = []
        dets = detector.forward(x)
        # print(dets[0])
        output = tracker.update(*dets)
        for t in output:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            for bound in BOUNDS:
                if len(t.tracklet) > 1:
                    line = Line(*t.tracklet)
                    if intersect(bound,line):
                        if clockwise(bound, line):
                            bound.add()
                            print('down')
                        else:
                            bound.reduce()
                            print('up')
        end_time = time.time()
        # print('FPS %i'%(1/(end_time-start_time)), end="\r", flush=True)

if __name__ == '__main__':
    if len(sys.argv)>1:
        addr = str(sys.argv[1])
        capture = cv2.VideoCapture(addr)
    else:
        capture = cv2.VideoCapture(0)
    camera = Camera()
    # capture = cv2.VideoCapture('sample.mp4')
    ret, camera.img = capture.read()
    
    detector = Detector('post.onnx')
    tracker = Tracker()

    cmsrc = camera.capture
    cmW, cmH, cmC = cmsrc.shape
    musk = np.zeros((608,1088,3),dtype='uint8')
    th = threading.Thread(target=background_forward,args=(camera, detector, tracker, BOUNDS))
    th.setDaemon(True)
    th.start()

    while ret:
        # time.sleep(1/20)
        ret, camera.img = capture.read()
        cmsrc = camera.capture
        
        cmimg = cmsrc.copy()
        if MODE == 1:
            cv2.arrowedLine(cmimg, tuple(PSTART), tuple(PEND), RED, 5)
        for bound in BOUNDS:
            cv2.arrowedLine(cmimg, bound.p1, bound.p2, bound.color, 5)
            cmimg = plot_one_bound_count(cmimg, bound)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('a'):
            MODE = 2
        if key & 0xFF == ord('q'):
            break
        if key == 32:
            for bound in BOUNDS:
                bound.clear()

    sys.exit()
