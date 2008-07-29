import taskDef
import ofind
import LatGaussFit
import Pyro.core
import matplotlib

#matplotlib.interactive(False)
from pylab import *

class fitResult:
    def __init__(self, task, results):
        self.taskID = task.taskID
        self.results = results

class fitTask(taskDef.Task):
    def __init__(self, data, threshold):
        taskDef.Task.__init__(self)
        self.threshold = threshold

        if len(data.shape) == 2:
            self.data = data.reshape((data.shape[0], data.shape[1], 1))
        else:
            self.data = data

    #def setReturnAdress(self, retAddress):
    #    self.retAddress = retAdress

    def __call__(self, gui=False):
        ofd = ofind.ObjectIdentifier(self.data)
        ofd.FindObjects(self.threshold,0)
        
        #print globals()
        if gui:
            #print 'hello'
            clf()
            imshow(ofd.filteredData, cmap=cm.hot, hold=False)
            plot([p.y for p in ofd], [p.x for p in ofd], 'xg', mew=2)
            axis('image')
            gca().set_ylim([255,0])
            colorbar()
            show()

        fitFac = LatGaussFit.GaussianFitFactory(self.data, None)
        self.res  = [fitFac.FromPoint(round(p.x), round(p.y)) for p in ofd]
        #print self.res
        #def getRes(self):
        return fitResult(self, [(r.fitResults, r.fitErr) for r in self.res] )
