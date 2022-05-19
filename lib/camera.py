import cv2
import numpy as np

class Camera:
    def __init__(self, size=(1088, 608)):
        self.img = np.zeros((608,1088,3),dtype='uint8')
        self.size=size
    @property
    def capture(self):
        return cv2.resize(self.img, self.size)