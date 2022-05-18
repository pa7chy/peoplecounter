import numpy as np

class crossEvent:
    def __init__(self, event) -> None:
        self.event = event
        self.tick = 15

class Line:
    def __init__(self, p1, p2, color = (255, 255, 255)):
        self.p1,self.p2 = tuple(p1),tuple(p2)
        self.color = color
        self.counter = 0
    def add(self):
        self.counter += 1
    def reduce(self):
        self.counter -= 1
    def clear(self):
        self.counter = 0

def onsegment(line, point):
    min_x, min_y = min(line.p1[0],line.p2[0]), min(line.p1[1],line.p2[1])
    max_x, max_y = max(line.p1[0],line.p2[0]), max(line.p1[1],line.p2[1])
    return point[0]<=max_x and point[0]>=min_x and point[1]<=max_y and point[1]>=min_y
    
def intersect(line1, line2):
    i,j,k,l = np.asarray((line1.p1,line1.p2,line2.p1,line2.p2))
    d1 = int(np.cross(i - k, j - k))
    d2 = int(np.cross(i - l, j - l))
    d3 = int(np.cross(k - i, l - i))
    d4 = int(np.cross(k - j, l - j))
    if (d1*d2)<0 and (d3*d4)<0:
        return True
    if d1 == 0 and d2 != 0:
        return onsegment(line1, line2.p1)
    return False

def clockwise(line1, line2):
    i,j,k,l = np.asarray((line1.p1,line1.p2,line2.p1,line2.p2))
    vec1 = j-i
    vec2 = l-k
    if np.cross(vec1, vec2) < 0:
        return True
    else:
        return False
def calc_angle(line1, line2):
    i,j,k,l = np.asarray((line1.p1,line1.p2,line2.p1,line2.p2))
    vec1 = j-i
    vec2 = l-k
    cosangle = vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if np.cross(vec1, vec2) < 0:
        return np.rad2deg(np.arccos(cosangle))
    else:
        return 360 - np.rad2deg(np.arccos(cosangle))