import os
import cv2
import time
import sys
import numpy as np
import sys
sys.path.append('lib')
from detector import Detector
from tracker import Tracker
from intersect import Line, intersect, clockwise, crossEvent
from vis import plot_events, plot_tracking



BOUNDS = []
line = Line([701, 321], [339, 257])
BOUNDS.append(line)

def main(src):
    input_shape = (1088,608)
    detector = Detector('new_people.onnx')
    tracker = Tracker()
    infer_video(src, input_shape, detector, tracker)
    print('Done.')

def infer_video(src, size, detector, tracker):
    up_count = 0
    down_count = 0
    events = []
    FPS = 25
    global BOUNDS
    capture = cv2.VideoCapture(src)
    outname = os.path.basename(src).split('.')[0]+'_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname, fourcc, FPS, size)
    frame_no = 0
    while True:
        frame_no += 1
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        img0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img0, dtype=np.float32)
        img /= 255.0
        x = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

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
                            events.append(crossEvent('down'))
                            down_count += 1
                            # print('%s down'%str(time.strftime("%d_%H_%M_%S", time.localtime())))
                        else:
                            events.append(crossEvent('up'))
                            up_count += 1
                            # print('%s up'%str(time.strftime("%d_%H_%M_%S", time.localtime())))
        draw = plot_tracking(frame, online_tlwhs, online_ids, (up_count, down_count))
        
        for bound in BOUNDS:
            draw = cv2.arrowedLine(draw, bound.p1, bound.p2, bound.color, 5)

        draw, events = plot_events(draw, events)
        out.write(draw)
        end_time = time.time()
        print('%i FPS %i'%(frame_no, 1/(end_time-start_time)), end="\r", flush=True)
    out.release()


if __name__ == "__main__":
    main(sys.argv[1])