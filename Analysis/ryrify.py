import numpy as np
import pylab
import scipy.optimize

offsets = {'N': np.array([0,1]), 'S': np.array([0,-1]),'E': np.array([1,0]),'E': np.array([-1,0])}

class RyR:
    def __init__(self, parent=None, direction=None):
        self.deltaP = np.array([0,0])
        self.theta = 0

        self.r = 15

        self.parent = parent
        self.dir = direction

        self.N = None
        self.S = None
        self.E = None
        self.W = None

    def getPos(self):
        parentPos = self.parent.getPos()

        return parentPos + 2*self.r*offsets[self.dir] + self.deltaP

    def getCount(self, points):
        x,y = self.getPos()

        return (((points[:,0]-x)**2 + (points[:,1] -y)**2) < r**2).sum()


def goal_f(p, x, y, threshold=15, unitsize=30):
    dx, dy, theta = p;

    x1 = x + dx
    y1 = y + dy
    x2 = x1*np.cos(theta) - y1*np.sin(theta)
    y2 = x1*np.sin(theta) + y1*np.cos(theta)

    h = np.histogram2d(x2,y2, [unitsize*(np.arange(20) - 10), unitsize*(np.arange(20) - 10)])[0]
    return h[h < threshold].sum()


def goal_p(p, x, y, threshold=15, unitsize=30):
    dx, dy, theta = p;

    x1 = x + dx
    y1 = y + dy
    x2 = x1*np.cos(theta) - y1*np.sin(theta)
    y2 = x1*np.sin(theta) + y1*np.cos(theta)

    h = np.histogram2d(x2,y2, [unitsize*(np.arange(20) - 10), unitsize*(np.arange(20) - 10)])[0]
    pylab.imshow(h, interpolation='nearest')

    return h[h < threshold].sum()

def ryrify(obj, threshold=15, unitsize=30):
    x = obj[:,0].ravel()
    y = obj[:,1].ravel()

    x = x - x.mean()
    y = y - y.mean()

    p = scipy.optimize.brute(goal_f, [(0., 30.), (0., 30.), (0, np.pi/2)], (x, y, threshold, unitsize))

    pylab.figure()
    goal_p(p, x,y, threshold, unitsize)
    return p



