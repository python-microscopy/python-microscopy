import os.path
import wx
import tables
import numpy
import Image
import os
from PYME.FileUtils import saveTiffStack
from PYME.Acquire import MetaDataHandler

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

#formats = ['PYME HDF - .h5',
#            'TIFF (stack if 3D) - .tif',
#            'TIFF (series if 3D) - .tif',
#            'Pickled numpy ndarray - .npy']

exporters = {}

def exporter(cls):
    exporters[cls.descr] = cls
    return cls

class Exporter:
    def _getFilename(self):
        fdialog = wx.FileDialog(None, 'Save exported file as ...',
                wildcard=self.extension, style=wx.SAVE|wx.HIDE_READONLY)

        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fname = fdialog.GetPath()
        else:
            fname = None

        fdialog.Destroy()

        return fname

    def Export(self, data, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        pass

@exporter
class H5Exporter(Exporter):
    extension = '*.h5'
    descr = 'PYME HDF - .h5'

    def __init__(self, complib='zlib', complevel=5):
        self.complib = complib
        self.complevel = complevel

    def Export(self, data, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        outFile = self._getFilename()

        if outFile == None:
            return

        h5out = tables.openFile(outFile,'w')
        filters=tables.Filters(self.complevel,self.complib,shuffle=True)

        nframes = (zslice.stop - zslice.start)/zslice.step

        xSize, ySize = data[xslice, yslice, 0].squeeze().shape

        ims = h5out.createEArray(h5out.root,'ImageData',tables.UInt16Atom(),(0,xSize,ySize), filters=filters, expectedrows=nframes)

        for frameN in range(zslice.start,zslice.stop, zslice.step):
            im = data[xslice, yslice, frameN].squeeze()[None, :,:]
            for fN in range(frameN+1, frameN+zslice.step):
                im += data[xslice, yslice, fN].squeeze()[None, :,:]
            ims.append(im)
            ims.flush()

        outMDH = MetaDataHandler.HDFMDHandler(h5out)

        if not metadata == None:
            outMDH.copyEntriesFrom(metadata)

            if 'Camera.ADOffset' in metadata.getEntryNames():
                outMDH.setEntry('Camera.ADOffset', zslice.step*metadata.getEntry('Camera.ADOffset'))

        if not origName == None:
            outMDH.setEntry('cropping.originalFile', origName)
            
        outMDH.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
        outMDH.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
        outMDH.setEntry('cropping.zslice', zslice.indices(data.shape[2]))


        outEvents = h5out.createTable(h5out.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))

        if not events == None:
            #copy events to results file
            if len(events) > 0:
                outEvents.append(events)
                
        h5out.flush()
        h5out.close()



@exporter
class TiffStackExporter(Exporter):
    extension = '*.tif'
    descr = 'TIFF (stack if 3D) - .tif'

    def Export(self, data, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        outFile = self._getFilename()
        if outFile == None:
            return

        saveTiffStack.saveTiffMultipage(data[xslice, yslice, zslice], outFile)

        if not metadata == None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)





@exporter
class TiffSeriesExporter(Exporter):
    extension = '*.xml'
    descr = 'TIFF Series - .tif'

    def Export(self, data, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        outFile = self._getFilename()

        if outFile == None:
            return

        #nframes = (zslice.stop - zslice.start)/zslice.step

        outDir = os.path.splitext(outFile)[0]
        os.mkdir(outDir)

        i = 0
        for frameN in range(zslice.start,zslice.stop, zslice.step):
            im = data[xslice, yslice, frameN].squeeze()[None, :,:]
            for fN in range(frameN+1, frameN+zslice.step):
                im += data[xslice, yslice, fN].squeeze()[None, :,:]
            
            Image.fromarray(im.squeeze().astype('uint16'), 'I;16').save(os.path.join(outDir, 'frame_%03d.tif'%i))
            i += 1

        if not metadata == None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)



@exporter
class NumpyExporter(Exporter):
    extension = '*.npy'
    descr = 'Pickled numpy ndarray - .npy'

    def Export(self, data, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        outFile = self._getFilename()

        if outFile == None:
            return

        numpy.save(outFile, data[xslice, yslice, zslice])

        if not metadata == None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)


#exporters = {'PYME HDF - .h5': H5Exporter,
#            'TIFF (stack if 3D) - .tif' : TiffStackExporter,
#            'TIFF (series if 3D) - .tif' : TiffSeriesExporter,
#            'Pickled numpy ndarray - .npy' : NumpyExporter}

#print exporters

class ExportDialog(wx.Dialog):
    def __init__(self,parent, roi=None):
        wx.Dialog.__init__(self, parent, title='Export Data')

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Format:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cFormat = wx.Choice(self, -1, choices=exporters.keys())
        self.cFormat.SetSelection(exporters.keys().index(H5Exporter.descr))
        hsizer.Add(self.cFormat, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)

        if not roi == None:
            bsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Cropping'), wx.VERTICAL)
            gsizer = wx.FlexGridSizer(4,4,5,5)

            self.tXStart = wx.TextCtrl(self, -1, '%d' % roi[0][0])
            self.tXStop = wx.TextCtrl(self, -1, '%d' % roi[0][1])
            self.tXStep = wx.TextCtrl(self, -1, '1')

            self.tYStart = wx.TextCtrl(self, -1, '%d' % roi[1][0])
            self.tYStop = wx.TextCtrl(self, -1, '%d' % roi[1][1])
            self.tYStep = wx.TextCtrl(self, -1, '1')

            self.tZStart = wx.TextCtrl(self, -1, '%d' % roi[2][0])
            self.tZStop = wx.TextCtrl(self, -1, '%d' % roi[2][1])
            self.tZStep = wx.TextCtrl(self, -1, '1')

            gsizer.AddMany([(10,10),
                            wx.StaticText(self, -1, 'Start'),
                            wx.StaticText(self, -1, 'Stop'),
                            wx.StaticText(self, -1, 'Step'),
                            wx.StaticText(self, -1, 'X:'), self.tXStart, self.tXStop, self.tXStep,
                            wx.StaticText(self, -1, 'Y:'), self.tYStart, self.tYStop, self.tYStep,
                            wx.StaticText(self, -1, 'Z:'), self.tZStart, self.tZStop, self.tZStep
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

        vsizer.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizerAndFit(vsizer)

    def GetFormat(self):
        return self.cFormat.GetStringSelection()

    def GetXSlice(self):
        return slice(int(self.tXStart.GetValue()), int(self.tXStop.GetValue()),int(self.tXStep.GetValue()))

    def GetYSlice(self):
        return slice(int(self.tYStart.GetValue()), int(self.tYStop.GetValue()),int(self.tYStep.GetValue()))

    def GetZSlice(self):
        return slice(int(self.tZStart.GetValue()), int(self.tZStop.GetValue()),int(self.tZStep.GetValue()))


def ExportData(vp, mdh=None, events=None, origName = None):
    roi = [[vp.selection_begin_x, vp.selection_end_x],[vp.selection_begin_y, vp.selection_end_y], [0, vp.ds.shape[2]]]
    
    dlg = ExportDialog(None, roi)
    succ = dlg.ShowModal()

    if (succ == wx.ID_OK):
        exp = exporters[dlg.GetFormat()]()

        exp.Export(vp.ds, dlg.GetXSlice(), dlg.GetYSlice(), dlg.GetZSlice(),mdh, events, origName)


    dlg.Destroy()
