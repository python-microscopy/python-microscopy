#!/usr/bin/python

##################
# dataExporter.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import os.path
import wx
import tables
import numpy
try:
    import Image
except ImportError:
    from PIL import Image
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
exportersByExtension = {}

def exporter(cls):
    exporters[cls.descr] = cls
    exportersByExtension[cls.extension] = cls
    return cls

class Exporter:
#    def _getFilename(self):
#        fdialog = wx.FileDialog(None, 'Save exported file as ...',
#                wildcard=self.extension, style=wx.SAVE|wx.HIDE_READONLY)
#
#        succ = fdialog.ShowModal()
#        if (succ == wx.ID_OK):
#            fname = fdialog.GetPath()
#        else:
#            fname = None
#
#        fdialog.Destroy()
#
#        return fname

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        pass

#@exporter
class H5Exporter(Exporter):
    extension = '*.h5'
    descr = 'PYME HDF - .h5'

    def __init__(self, complib='zlib', complevel=6):
        self.complib = complib
        self.complevel = complevel

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        h5out = tables.openFile(outFile,'w')
        filters=tables.Filters(self.complevel,self.complib,shuffle=True)

        nframes = (zslice.stop - zslice.start)/zslice.step

        xSize, ySize = data[xslice, yslice, 0].shape[:2]
        
        print((xSize, ySize))
        
        #atm = tables.UInt16Atom()
        atm = tables.Atom.from_dtype(data[xslice, yslice, 0].dtype)

        ims = h5out.createEArray(h5out.root,'ImageData',atm,(0,xSize,ySize), filters=filters, expectedrows=nframes)#, chunkshape=(1,xSize,ySize))

        for frameN in range(zslice.start,zslice.stop, zslice.step):
            im = data[xslice, yslice, frameN].squeeze()
            
            for fN in range(frameN+1, frameN+zslice.step):
                im += data[xslice, yslice, fN].squeeze()
                
            if im.ndim == 1:
                im = im.reshape((-1, 1))[None, :,:]
            else:
                im = im[None, :,:]
            
            #print im.shape    
            ims.append(im)
            #ims.flush()
            
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

exporter(H5Exporter)


#@exporter
class TiffStackExporter(Exporter):
    extension = '*.tif'
    descr = 'TIFF (stack if 3D) - .tif'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        #xmd = None
        if not metadata == None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            
        if data.shape[3] > 1: #have multiple colour channels
            if data.shape[2] == 1: #2d image -> stack with chans
                #d = numpy.concatenate([numpy.atleast_3d(data[xslice, yslice, 0, i].squeeze()) for i in range(data.shape[3])],2)
                #saveTiffStack.saveTiffMultipage(d, outFile)
                mpt = saveTiffStack.TiffMP(outFile)
                for i in range(data.shape[3]):
                    mpt.AddSlice(data[xslice, yslice, 0, i].squeeze())
                mpt.close()
            else: #save each channel as it's own stack
                if not metadata == None and 'ChannelNames' in metadata.getEntryNames():
                    chanNames = metadata['ChannelNames']    

                else:
                    chanNames = range(data.shape[3])

                chanFiles = [os.path.splitext(os.path.split(outFile)[1])[0] + '__%s.tif' % chanNames[i]  for i in range(data.shape[3])]
                if not metadata == None:
                    xmd['ChannelFiles'] = chanFiles

                for i in range(data.shape[3]):
                    saveTiffStack.saveTiffMultipage(data[xslice, yslice, zslice, i].squeeze(), os.path.splitext(outFile)[0] + '__%s.tif' % chanNames[i])
        else:
            saveTiffStack.saveTiffMultipage(data[xslice, yslice, zslice], outFile)

        if not metadata == None:
            #xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            if not origName == None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))

            print((xslice.indices(data.shape[0])))
            
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)
            # xmd.WriteSimple(xmlFile)

exporter(TiffStackExporter)



#@exporter
class TiffSeriesExporter(Exporter):
    extension = '*.xml'
    descr = 'TIFF Series - .xml'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
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
            if not origName == None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))

            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)
            # xmd.WriteSimple(xmlFile)

exporter(TiffSeriesExporter)


#@exporter
class NumpyExporter(Exporter):
    extension = '*.npy'
    descr = 'Pickled numpy ndarray - .npy'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        numpy.save(outFile, data[xslice, yslice, zslice])

        if not metadata == None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            if not origName == None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))
            
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)
            #xmd.WriteSimple(xmlFile)

exporter(NumpyExporter)

#@exporter
class PSFExporter(Exporter):
    extension = '*.psf'
    descr = 'PYME psf data - .psf'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        #numpy.save(outFile, data[xslice, yslice, zslice])
        import cPickle
        fid = open(outFile, 'wb')
        cPickle.dump((data[xslice, yslice, zslice], metadata.voxelsize), fid, 2)
        fid.close()

exporter(PSFExporter)

#@exporter
class TxtExporter(Exporter):
    extension = '*.txt'
    descr = 'Tab formatted txt  - .txt'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None):
        #numpy.save(outFile, data[xslice, yslice, zslice])
        #import cPickle

        try:
            chanNames = metadata.getEntry('ChannelNames')
        except:
            chanNames = ['Chan %d' % d for d in range(data.shape[3])]

        dat = [data[xslice, yslice, zslice, chan].squeeze() for chan in range(data.shape[3])]

        #print chanNames

        if metadata and 'Profile.XValues' in metadata.getEntryNames():
            dat = [metadata.getEntry('Profile.XValues')] + dat
            chanNames = [metadata.getEntry('Profile.XLabel')] + chanNames

        if metadata and 'Spectrum.Wavelengths' in metadata.getEntryNames():
            dat = [metadata.getEntry('Spectrum.Wavelengths')] + dat
            chanNames = ['Wavelength [nm]'] + chanNames
            

        fid = open(outFile, 'wb')

        #write header
        fid.write('#' + '\t'.join(chanNames))

        for i in range(len(dat[0])):
            fid.write('\n' + '\t'.join(['%f' % d[i] for d in dat]))

        fid.close()

exporter(TxtExporter)
#exporters = {'PYME HDF - .h5': H5Exporter,
#            'TIFF (stack if 3D) - .tif' : TiffStackExporter,
#            'TIFF (series if 3D) - .tif' : TiffSeriesExporter,
#            'Pickled numpy ndarray - .npy' : NumpyExporter}

#print exporters

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


def _getFilename(defaultExt = '*.tif'):
        wcs  = []

        defIndex = 0

        for i, e in enumerate(exporters.values()):
            wcs.append(e.descr + '|' + e.extension)
            if e.extension == defaultExt:
                defIndex = i

        fdialog = wx.FileDialog(None, 'Save file as ...',
                wildcard='|'.join(wcs), style=wx.SAVE)#|wx.HIDE_READONLY)

        fdialog.SetFilterIndex(defIndex)

        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fname = fdialog.GetPath().encode()
        else:
            fname = None

        fdialog.Destroy()

        return fname

def CropExportData(vp, mdh=None, events=None, origName = None):
    #if 'ds' in dir(vp.do):
    ds = vp.do.ds
    #else:
    #    ds= vp.ds

    #if 'selection_begin_x' in dir(vp):
    roi = [[vp.do.selection_begin_x, vp.do.selection_end_x + 1],
              [vp.do.selection_begin_y, vp.do.selection_end_y +1], [0, ds.shape[2]]]
    #else:
    #   roi = [[0, ds.shape[0]],[0, ds.shape[1]],[0, ds.shape[2]]]

    dlg = ExportDialog(None, roi)
    succ = dlg.ShowModal()

    if (succ == wx.ID_OK):
        filename = _getFilename()

        if filename == None:
            dlg.Destroy()
            return

        exp = exportersByExtension['*' + os.path.splitext(filename)[1]]()

        exp.Export(ds, filename, dlg.GetXSlice(), dlg.GetYSlice(), dlg.GetZSlice(),mdh, events, origName)

    dlg.Destroy()

    


def ExportData(ds, mdh=None, events=None, origName = None, defaultExt = '*.tif', filename=None):
    if filename == None:
        #show file selection dialog box
        filename = _getFilename(defaultExt)
        
    if filename == None:
        #we cancelled the dialog - exit
        return

    
    ext = '*' + os.path.splitext(filename)[1]
        
    if not ext in exportersByExtension.keys():
        raise RuntimeError('No exporter found for %s files')
        #wx.MessageBox('No exporter found for %s files\n Try one of the following file types:\n%s' % (ext, ', '.join(exportersByExtension.keys())), "Error saving data", wx.OK|wx.ICON_HAND)
        return

    exp = exportersByExtension[ext]()
    exp.Export(ds, filename, slice(0, ds.shape[0], 1), slice(0, ds.shape[1], 1), slice(0, ds.shape[2], 1),mdh, events, origName)
    return filename
    

