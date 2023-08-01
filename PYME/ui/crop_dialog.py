import wx
import os

class ExportDialog(wx.Dialog):
    def __init__(self,parent, roi=None):
        wx.Dialog.__init__(self, parent, title='Export Data')

        vsizer = wx.BoxSizer(wx.VERTICAL)

#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        hsizer.Add(wx.StaticText(self, -1, 'Format:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        self.cFormat = wx.Choice(self, -1, choices=exporters.keys())
#        self.cFormat.SetSelection(exporters.keys().index(H5Exporter.descr))
#        hsizer.Add(self.cFormat, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)

        if not roi is None:
            bsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Cropping'), wx.VERTICAL)
            gsizer = wx.FlexGridSizer(5,4,5,5)

            self.tXStart = wx.TextCtrl(self, -1, '%d' % roi[0][0])
            self.tXStop = wx.TextCtrl(self, -1, '%d' % roi[0][1])
            self.tXStep = wx.TextCtrl(self, -1, '1')

            self.tYStart = wx.TextCtrl(self, -1, '%d' % roi[1][0])
            self.tYStop = wx.TextCtrl(self, -1, '%d' % roi[1][1])
            self.tYStep = wx.TextCtrl(self, -1, '1')

            self.tZStart = wx.TextCtrl(self, -1, '%d' % roi[2][0])
            self.tZStop = wx.TextCtrl(self, -1, '%d' % roi[2][1])
            self.tZStep = wx.TextCtrl(self, -1, '1')

            self.tTStart = wx.TextCtrl(self, -1, '%d' % roi[3][0])
            self.tTStop = wx.TextCtrl(self, -1, '%d' % roi[3][1])
            self.tTStep = wx.TextCtrl(self, -1, '1')

            gsizer.AddMany([(10,10),
                            wx.StaticText(self, -1, 'Start'),
                            wx.StaticText(self, -1, 'Stop'),
                            wx.StaticText(self, -1, 'Step'),
                            wx.StaticText(self, -1, 'X:'), self.tXStart, self.tXStop, self.tXStep,
                            wx.StaticText(self, -1, 'Y:'), self.tYStart, self.tYStop, self.tYStep,
                            wx.StaticText(self, -1, 'Z:'), self.tZStart, self.tZStop, self.tZStep,
                            wx.StaticText(self, -1, 'T:'), self.tTStart, self.tTStop, self.tTStep
                            ])

            bsizer.Add(gsizer, 0, wx.ALL, 5)
            vsizer.Add(bsizer, 0, wx.ALL|wx.EXPAND, 5)


        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        vsizer.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizerAndFit(vsizer)

    def GetFormat(self):
        return self.cFormat.GetStringSelection()

    def GetXSlice(self):
        return slice(int(self.tXStart.GetValue()), int(self.tXStop.GetValue()),int(self.tXStep.GetValue()))

    def GetYSlice(self):
        return slice(int(self.tYStart.GetValue()), int(self.tYStop.GetValue()),int(self.tYStep.GetValue()))

    def GetZSlice(self):
        return slice(int(self.tZStart.GetValue()), int(self.tZStop.GetValue()),int(self.tZStep.GetValue()))

    def GetTSlice(self):
        return slice(int(self.tTStart.GetValue()), int(self.tTStop.GetValue()),int(self.tTStep.GetValue()))


def _getFilename(defaultExt = '*.tif'):
    from PYME.IO.dataExporter import exporters
    wcs  = []
    exts = []

    defIndex = 0

    for i, e in enumerate(exporters.values()):
        wcs.append(e.descr + '|' + e.extension)
        exts.append(e.extension[1:])
        if e.extension == defaultExt:
            defIndex = i

    fdialog = wx.FileDialog(None, 'Save file as ...',
            wildcard='|'.join(wcs), style=wx.FD_SAVE)#|wx.HIDE_READONLY)

    fdialog.SetFilterIndex(defIndex)

    succ = fdialog.ShowModal()
    if (succ == wx.ID_OK):
        fname = fdialog.GetPath()

        #we decide which exporter to use based on extension. Ensure that we have one (some platforms do not
        #automatically add to path.
        if os.path.splitext(fname)[1] == '':
            #we didn't get an extension, deduce by looking at which filter was selected
            filtIndex = fdialog.GetFilterIndex()
            fname += exts[filtIndex]
    else:
        fname = None

    fdialog.Destroy()

    return fname

def CropExportData(data, roi=None, mdh=None, events=None, origName = None):
    from PYME.IO.dataExporter import exportersByExtension

    if roi is None:
        roi = [[0,data.shape[0]], [0,data.shape[1]], [0,data.shape[2]]]

    roi = roi + [[0, data.shape[3]],]

    dlg = ExportDialog(None, roi)
    succ = dlg.ShowModal()

    if (succ == wx.ID_OK):
        filename = _getFilename()

        if filename is None:
            dlg.Destroy()
            return

        ext = '*' + os.path.splitext(filename)[1]
        exp = exportersByExtension[ext]()

        exp.Export(data, filename, dlg.GetXSlice(), dlg.GetYSlice(), dlg.GetZSlice(), mdh, events, origName, tslice=dlg.GetTSlice())

    dlg.Destroy()