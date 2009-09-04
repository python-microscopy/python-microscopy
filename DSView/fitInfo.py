import wx
#import numpy

class FitInfoPanel(wx.Panel):
    def __init__(self, parent, fitResults, mdh, id=-1):
        wx.Panel.__init__(self, id=id, parent=parent)

        self.fitResults = fitResults
        self.mdh = mdh

        vsizer = wx.BoxSizer(wx.VERTICAL)
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.stSliceNum = wx.StaticText(self, -1, 'No event selected')

        vsizer.Add(self.stSliceNum, 0, wx.LEFT|wx.TOP|wx.BOTTOM, 5)

        sFitRes = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Fit Results'), wx.VERTICAL)

        self.stFitRes = wx.StaticText(self, -1, self.genResultsText(None))
        self.stFitRes.SetFont(wx.Font(10, wx.MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sFitRes.Add(self.stFitRes, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        vsizer.Add(sFitRes, 0, wx.EXPAND|wx.LEFT|wx.TOP|wx.BOTTOM|wx.RIGHT, 5)

        self.SetSizerAndFit(vsizer)

    def genResultsText(self, index):
        s =  u''
        ns = self.fitResults['fitResults'].dtype.names

        nl = max([len(n) for n in ns])

        #print nl

        if not index == None:
            r = self.fitResults[index]



            for n in ns:
                #\u00B1 is the plus-minus sign
                s += u'%s %8.2f \u00B1 %3.2f\n' % ((n + ':').ljust(nl+1), r['fitResults'][n], r['fitError'][n])
                
        else:    
            for n in ns:
                s += u'%s:\n' % (n)
                
        return s



    def UpdateDisp(self, index):
        slN = 'No event selected'

        if not index == None:
            slN = 'Point #: %d    Slice: %d' % (index, self.fitResults['tIndex'][index])

        self.stSliceNum.SetLabel(slN)

        self.stFitRes.SetLabel(self.genResultsText(index))