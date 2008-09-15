import numpy as np
import scipy as sp
import ofind
from PYME.Analysis.FitFactories.LatGaussFitFRTC import FitFactory, FitResultsDType
import MetaData

def fitDep(g,r,ofindThresh, dx, dy):
    rg = r + g #detect objects in sum image
    
    ofd = ofind.ObjectIdentifier(rg)

    ofd.FindObjects(ofindThresh, blurRadius=2)

    res_d = np.empty(len(ofd), FitResultsDType)

    class foo:
        pass

    md = MetaData.TIRFDefault

    md.chroma = foo()
    md.chroma.dx = dx
    md.chroma.dy = dy

    ff = FitFactory(np.concatenate((g.reshape(512, -1, 1), r.reshape(512, -1, 1)),2), md)

    
    for i in range(len(ofd)):    
        p = ofd[i]
        res_d[i] = ff.FromPoint(round(p.x), round(p.y))
        

    return res_d
