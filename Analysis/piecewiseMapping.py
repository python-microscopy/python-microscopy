from numpy import *

class piecewiseMap:
    def __init__(self, y0, xvals, yvals, secsPerFrame = 1, xIsSecs = True):
        self.y0 = y0

        if xIsSecs: #store in frame numbers
            self.xvals = xvals/secsPerFrame
        else:
            self.xvals = xvals
        self.yvals = yvals

        self.secsPerFrame = secsPerFrame
        self.xIsSecs = xIsSecs

    def __call__(self, xp, xpInFrames = True):
        yp = 0*xp

        if not xpInFrames:
            xp = xp/self.secsPerFrame
        
        y0 = self.y0
        x0 = -inf

        for x, y in zip(self.xvals, self.yvals):
            yp += y0 * (xp >= x0) * (xp < x)
            x0, y0 = x, y

        x  = +inf
        yp += y0 * (xp >= x0) * (xp < x)

        return yp

def GeneratePMFromEventList(events, secsPerFrame, x0, y0, eventName = 'ProtocolFocus', dataPos = 1):
    x = []
    y = []

    for e in events:
        if e['EventName'] == eventName:
            x.append(e['Time'] - x0)
            y.append(float(e['EventDescr'].split(', ')[dataPos]))

    return piecewiseMap(y0, array(x), array(y), secsPerFrame)

