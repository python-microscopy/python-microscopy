from pylab import *


class ricsSimulator:
    def __init__(self, nPoints=1000, stepDist = 30., sx =3000., sy = 3000., pixelSize=30., sig = 106.):
        self.nPoints = nPoints
        self.stepDist = stepDist
        self.sx = sx
        self.sy = sy
        self.pixelSize = pixelSize
        self.sig = sig
        
        self.X = arange(0., sx, pixelSize)
        self.Y = arange(0., sy, pixelSize)

        self.x = sx*rand(nPoints)
        self.y = sy*rand(nPoints)

        self.xp = 0
        self.yp = 0

        self.frame = zeros((len(self.X), len(self.Y)))

    def onPixel(self):
        #random walk
        self.x = (self.x + self.stepDist*randn(self.nPoints)) % self.sx
        self.y = (self.y + self.stepDist*randn(self.nPoints)) % self.sy

        x0 = self.xp*self.pixelSize
        y0 = self.yp*self.pixelSize

        ind = ((self.x - x0)**2 + (self.y - y0)**2) < (3*self.sig)**2

        self.frame[self.xp, self.yp] = exp(-(((self.x[ind] - x0)**2 + (self.y[ind] - y0)**2))/(2*self.sig**2)).sum()

        self.xp +=1

        #print (self.x[ind] - x0)**2

        #print exp(-(((self.x[ind] - x0)**2 + (self.y[ind] - y0)**2))/(2*self.sig**2)).sum()
        #print x0

        if self.xp >= self.frame.shape[0]:
            self.xp = 0
            self.yp += 1

            if self.yp >= self.frame.shape[1]:
                self.yp = 0


    def getFrame(self):
        for i in range(self.frame.size):
            self.onPixel()

        return self.frame


def S(Dx, Dy, pixelSize, sig, D,tp, tl):
    return exp(-(((abs(Dx)*pixelSize)**2 + (abs(Dy)*pixelSize)**2)/(2*sig**2))/(1 + (4*D*(tp*Dx + tl*Dy))/(2*sig**2)))

def G(Dx, Dy, gamma, N, sig, sigz, D, tp, tl):
    return (gamma/N)*(1/(1 + (4*D*(tp*Dx + tl*Dy))/(2*sig**2)))*(1/sqrt(1 + (4*D*(tp*Dx + tl*Dy))/(2*sigz**2)))

def Gs(Dx, Dy, N, pixelSize, sig, sigz, D, tp, tl, gamma=0.3535):
    return S(Dx, Dy, pixelSize, sig, D,tp, tl)*G(Dx, Dy, gamma, N, sig, sigz, D, tp, tl)

def GModel(params, Dx, Dy, pixelSize, sig, sigz, tp, tl):
    D, N = params
    return Gs(Dx, Dy, N, pixelSize, sig, sigz, D, tp, tl, gamma=0.3535)
