import numpy as np
import cv2
import sys
import threading
from collections import deque
from lib.vis import plot_one_bound_count
from lib.intersect import Line, intersect, clockwise
from lib.detector import Detector
from lib.tracker import Tracker

PSTART = ''
PEND = ''
BOUNDS = []
MODE = 0
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)

def on_mouse(event, x, y, flags, param):
    global PSTART
    global PEND
    global BOUNDS
    global MODE
    global GREEN
    if event == cv2.EVENT_LBUTTONDOWN:
        if MODE == 1:
            line = Line(PSTART, PEND, GREEN)
            # print(PSTART,PEND)
            BOUNDS.append(line)
            MODE -= 1
        if MODE == 2:
            PSTART = [x, y]
            PEND = [x, y]
            MODE -= 1
    if MODE == 1:
        PEND = [x, y]
        
def background_forward(img, detector, tracker, BOUNDS):
    while True:
        img0 = img.copy()
        img0 = np.ascontiguousarray(img0, dtype=np.float32)
        img0 /= 255.0
        x = np.transpose(img0, (2, 0, 1))[np.newaxis, ...]
        online_tlwhs = []
        online_ids = []
        output = tracker.update(*detector.forward(x))
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

if __name__ == '__main__':
    if len(sys.argv)>1:
        addr = str(sys.argv[1])
        capture = cv2.VideoCapture(addr)
    else:
        capture = cv2.VideoCapture(0)
    ret, cmsrc = capture.read()
    
    detector = Detector('post.onnx')
    tracker = Tracker()
    # tracker = None

    cmsrc = cv2.resize(cmsrc, (1088, 608))
    cmW, cmH, cmC = cmsrc.shape
    musk = np.zeros((608,1088,3),dtype='uint8')
    th = threading.Thread(target=background_forward,args=(cmsrc, detector, tracker, BOUNDS))
    th.setDaemon(True)
    th.start()
    cv2.namedWindow('PeopleCounterDemo')
    cv2.setMouseCallback('PeopleCounterDemo', on_mouse)
    
    while ret:
        ret, cmsrc = capture.read()
        cmsrc = cv2.resize(cmsrc, (1088, 608))
        # cmsrc = cv2.addWeighted(cmsrc,1,musk,1,0)
        
        cmimg = cmsrc.copy()
        if MODE == 1:
            cv2.arrowedLine(cmimg, tuple(PSTART), tuple(PEND), RED, 5)
        for bound in BOUNDS:
            cv2.arrowedLine(cmimg, bound.p1, bound.p2, bound.color, 5)
            cmimg = plot_one_bound_count(cmimg, bound)

        cv2.imshow('PeopleCounterDemo', cmimg)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('a'):
            MODE = 2
        if key & 0xFF == ord('q'):
            break
        if key == 32:
            for bound in BOUNDS:
                bound.clear()

    cv2.destroyAllWindows()
    sys.exit()
