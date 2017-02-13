class Annotater(object):
    def __init__(self, dsviewer):
        self.do = dsviewer.do
    
        self._annotations = []
    
    def AddCurvedLine(self):
        if self.do.selectionMode == self.do.SELECTION_SQUIGGLE:
            self._annotations.append({'type' : 'curve', 'points' : self.do.selection_trace})
        elif self.do.selectionMode == self.do.SELECTION_LNE:
            x0, y0, x1, y1 = self.do.GetSliceSelection()
            self._annotations.append({'type' : 'line', 'points' : [(x0, y0), (x1, y1)]})


def Plug(dsviewer):
    pass