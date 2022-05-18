import cv2
import numpy as np

def plot_events(img, events):
    pos = [500, 300]
    color = (0, 0, 255)
    text_thickness = 2
    text_scale = 2
    for i, event in enumerate(events.copy()):
        cv2.putText(img, event.event, (pos[0],pos[1]-(i*50)), cv2.FONT_HERSHEY_PLAIN, text_scale, color, text_thickness)
        event.tick -= 1
        if event.tick == 0:
            events.pop(i)
    return img, events

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, counts, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'M-count: %i' % (counts[0]-counts[1]),
                (150, 257), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
    # cv2.putText(im, 'frame: %d fps: %.2f num: %d up_count: %i down_count: %i' % (frame_id, fps, len(tlwhs), counts[0], counts[1]),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def plot_one_bound_count(img, bound):
    img = cv2.putText(img, 'Count: %i' % (bound.counter), (bound.p1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
    return img