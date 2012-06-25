#!/usr/bin/python
##################
# image.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import os
import numpy
import weakref

from PYME.Acquire import MetaDataHandler
from PYME.Analysis import MetaData
from PYME.DSView import dataWrap
from PYME.Analysis.DataSources import BufferedDataSource
from PYME.Analysis.LMVis.visHelpers import ImageBounds

lastdir = ''

class DefaultDict(dict):
    '''List which returns a default value for items not in the list'''
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self, index):
        try:
            return dict.__getitem__(self, index)
        except KeyError:
            return 0

openImages = weakref.WeakValueDictionary()
nUntitled = DefaultDict()

class ImageStack(object):
    def __init__(self, data = None, mdh = None, filename = None, queueURI = None, events = [], titleStub='Untitled Image'):
        global nUntitled
        self.data = data      #image data
        self.mdh = mdh        #metadata (a MetaDataHandler class)
        self.events = events  #events

        self.queueURI = queueURI
        self.filename = filename

        self.mode = 'LM'

        self.saved = False
        self.volatile = False #is the data likely to change and need refreshing?
        
        if (data == None):
            self.Load(filename)

        self.SetData(self.data)

        if self.filename == None:
            self.filename = '%s %d' % (titleStub, nUntitled[titleStub])
            nUntitled[titleStub] += 1

        if self.mdh == None:
            self.mdh = MetaDataHandler.NestedClassMDHandler()

        if 'Spectrum.Wavelengths' in self.mdh.getEntryNames():
            self.xvals = self.mdh['Spectrum.Wavelengths']
            self.xlabel = 'Wavelength [nm]'

            if self.data.shape[1] == 1:
                self.mode = 'graph'

        openImages[self.filename] = self

    def SetData(self, data):
        #the data does not need to be a numpy array - it could also be, eg., queue data
        #on a remote server - wrap so that is is indexable like an array
        self.data = dataWrap.Wrap(data)
    
    @property
    def pixelSize(self):
        try:
            return 1e3*self.mdh['voxelsize.x']
        except:
            return 1

    @pixelSize.setter
    def pixelSize(self, value):
        self.mdh['voxelsize.x'] = .001*value
        self.mdh['voxelsize.y'] = .001*value

    @property
    def sliceSize(self):
        try:
            return 1e3*self.mdh['voxelsize.z']
        except:
            return 1

    @sliceSize.setter
    def sliceSize(self, value):
        self.mdh['voxelsize.z'] = .001*value


    @property
    def names(self):
        try:
            return self.mdh['ChannelNames']
        except:
            return ['Chan %d'% d for d in range(self.data.shape[3])]

    @names.setter
    def names(self, value):
        self.mdh['ChannelNames'] = value

    @property
    def imgBounds(self):
        try:
            return ImageBounds(self.mdh['ImageBounds.x0'],self.mdh['ImageBounds.y0'],self.mdh['ImageBounds.x1'],self.mdh['ImageBounds.y1'],self.mdh['ImageBounds.z0'],self.mdh['ImageBounds.z1'])
        except:
            return ImageBounds(0, 0, self.pixelSize*self.data.shape[0], self.pixelSize*self.data.shape[1],0, self.sliceSize*self.data.shape[2])

    @imgBounds.setter
    def imgBounds(self, value):
        self.mdh['ImageBounds.x0'] = value.x0
        self.mdh['ImageBounds.y0'] = value.y0
        self.mdh['ImageBounds.x1'] = value.x1
        self.mdh['ImageBounds.y1'] = value.y1
        self.mdh['ImageBounds.z0'] = value.z0
        self.mdh['ImageBounds.z1'] = value.z1


    def LoadQueue(self, filename):
        import Pyro.core
        from PYME.Analysis.DataSources import TQDataSource
        from PYME.misc.computerName import GetComputerName
        compName = GetComputerName()

        if self.queueURI == None:
            #if 'PYME_TASKQUEUENAME' in os.environ.keys():
            #    taskQueueName = os.environ['PYME_TASKQUEUENAME']
            #else:
            #    taskQueueName = 'taskQueue'
            taskQueueName = 'TaskQueues.%s' % compName
            self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)
        else:
            self.tq = Pyro.core.getProxyForURI(self.queueURI)

        self.seriesName = filename[len('QUEUE://'):]

        self.dataSource = TQDataSource.DataSource(self.seriesName, self.tq)
        self.data = self.dataSource #this will get replaced with a wrapped version

        self.mdh = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName)
        MetaData.fillInBlanks(self.mdh, self.dataSource)

        #self.timer.WantNotification.append(self.dsRefresh)

        self.events = self.dataSource.getEvents()

    def Loadh5(self, filename):
        import tables
        from PYME.Analysis.DataSources import HDFDataSource
        from PYME.Analysis.LMVis import inpFilt

        self.dataSource = HDFDataSource.DataSource(filename, None)
        self.data = self.dataSource #this will get replaced with a wrapped version

        if 'MetaData' in self.dataSource.h5File.root: #should be true the whole time
            self.mdh = MetaData.TIRFDefault
            self.mdh.copyEntriesFrom(MetaDataHandler.HDFMDHandler(self.dataSource.h5File))
        else:
            self.mdh = MetaData.TIRFDefault
            wx.MessageBox("Carrying on with defaults - no gaurantees it'll work well", 'ERROR: No metadata found in file ...', wx.OK)
            print "ERROR: No metadata fond in file ... Carrying on with defaults - no gaurantees it'll work well"

        MetaData.fillInBlanks(self.mdh, self.dataSource)

        from PYME.ParallelTasks.relativeFiles import getRelFilename
        self.seriesName = getRelFilename(filename)

        #try and find a previously performed analysis
        fns = filename.split(os.path.sep)
        cand = os.path.sep.join(fns[:-2] + ['analysis',] + fns[-2:]) + 'r'
        print cand
        if os.path.exists(cand):
            h5Results = tables.openFile(cand)

            if 'FitResults' in dir(h5Results.root):
                self.fitResults = h5Results.root.FitResults[:]
                self.resultsSource = inpFilt.h5rSource(h5Results)

                self.resultsMdh = MetaData.TIRFDefault
                self.resultsMdh.copyEntriesFrom(MetaDataHandler.HDFMDHandler(h5Results))

        self.events = self.dataSource.getEvents()

    def LoadKdf(self, filename):
        import PYME.cSMI as cSMI
        self.data = cSMI.CDataStack_AsArray(cSMI.CDataStack(filename), 0).squeeze()
        self.mdh = MetaData.TIRFDefault

        try: #try and get metadata from the .log file
            lf = open(os.path.splitext(filename)[0] + '.log')
            from PYME.DSView import logparser
            lp = logparser.logparser()
            log = lp.parse(lf.read())
            lf.close()

            self.mdh.setEntry('voxelsize.z', log['PIEZOS']['Stepsize'])
        except:
            pass

        from PYME.ParallelTasks.relativeFiles import getRelFilename
        self.seriesName = getRelFilename(filename)

        self.mode = 'psf'

    def LoadPSF(self, filename):
        self.data, vox = numpy.load(filename)
        self.mdh = MetaData.ConfocDefault

        self.mdh.setEntry('voxelsize.x', vox.x)
        self.mdh.setEntry('voxelsize.y', vox.y)
        self.mdh.setEntry('voxelsize.z', vox.z)


        from PYME.ParallelTasks.relativeFiles import getRelFilename
        self.seriesName = getRelFilename(filename)

        self.mode = 'psf'
        

    def FindAndParseMetadata(self, filename):
        mdf = None
        xmlfn = os.path.splitext(filename)[0] + '.xml'
        xmlfnmc = os.path.splitext(filename)[0].split('__')[0] + '.xml'
        if os.path.exists(xmlfn):
            self.mdh = MetaData.TIRFDefault
            self.mdh.copyEntriesFrom(MetaDataHandler.XMLMDHandler(xmlfn))
            mdf = xmlfn
        elif os.path.exists(xmlfnmc): #this is a single colour channel of a pair
            self.mdh = MetaData.TIRFDefault
            self.mdh.copyEntriesFrom(MetaDataHandler.XMLMDHandler(xmlfnmc))
            mdf = xmlfnmc
        else:
            self.mdh = MetaData.BareBones
            
            #check for simple metadata (python code with an .md extension which 
            #fills a dictionary called md)
            mdfn = os.path.splitext(filename)[0] + '.md'
            if os.path.exists(mdfn):
                self.mdh.copyEntriesFrom(MetaDataHandler.SimpleMDHandler(mdfn))
                mdf = mdfn
            elif filename.endswith('.lsm'):
                #read lsm metadata
                from PYME.gohlke.tifffile import TIFFfile
                tf = TIFFfile(filename)
                lsm_info = tf[0].cz_lsm_scan_information
                self.mdh['voxelsize.x'] = lsm_info['line_spacing']
                self.mdh['voxelsize.y'] = lsm_info['line_spacing']
                self.mdh['voxelsize.z'] = lsm_info['plane_spacing']
                
                def lsm_pop(basename, dic):
                    for k, v in dic.items():
                        if isinstance(v, list):
                            #print k, v
                            for i, l_i in enumerate(v):
                                #print i, l_i, basename
                                lsm_pop(basename + k + '.' + k[:-1] + '%i.' %i, l_i)
                                
                        else:
                            self.mdh[basename + k] = v
                
                lsm_pop('LSM.', lsm_info)
                

        if not ('voxelsize.x' in self.mdh.keys() and 'voxelsize.y' in self.mdh.keys()):
            from PYME.DSView.voxSizeDialog import VoxSizeDialog

            dlg = VoxSizeDialog(None)
            dlg.ShowModal()

            self.mdh.setEntry('voxelsize.x', dlg.GetVoxX())
            self.mdh.setEntry('voxelsize.y', dlg.GetVoxY())
            self.mdh.setEntry('voxelsize.z', dlg.GetVoxZ())

        return mdf

    def LoadTiff(self, filename):
        #from PYME.FileUtils import readTiff
        from PYME.Analysis.DataSources import TiffDataSource

        mdfn = self.FindAndParseMetadata(filename)

        self.dataSource = TiffDataSource.DataSource(filename, None)
        self.dataSource = BufferedDataSource.DataSource(self.dataSource, min(self.dataSource.getNumSlices(), 50))
        self.data = self.dataSource #this will get replaced with a wrapped version


        #if we have a multi channel data set, try and pull in all the channels
        if 'ChannelFiles' in self.mdh.getEntryNames():
            try:
                from PYME.DSView.dataWrap import ListWrap
                #pull in all channels

                chans = []

                for cf in self.mdh.getEntry('ChannelFiles'):
                    cfn = os.path.join(os.path.split(filename)[0], cf)

                    ds = TiffDataSource.DataSource(cfn, None)
                    ds = BufferedDataSource.DataSource(ds, min(ds.getNumSlices(), 50))

                    chans.append(ds)

                self.data = ListWrap(chans) #this will get replaced with a wrapped version

                self.filename = mdfn
            except:
                pass
            
        elif 'ChannelNames' in self.mdh.getEntryNames() and len(self.mdh['ChannelNames']) == self.data.getNumSlices():
            from PYME.DSView.dataWrap import ListWrap
            chans = [numpy.atleast_3d(self.data.getSlice(i)) for i in range(len(self.mdh['ChannelNames']))]
            self.data = ListWrap(chans)
        elif filename.endswith('.lsm') and 'LSM.images_number_channels' in self.mdh.keys() and self.mdh['LSM.images_number_channels'] > 1:
            from PYME.DSView.dataWrap import ListWrap
            nChans = self.mdh['LSM.images_number_channels']
            
            chans = []
            
            for n in range(nChans):
                ds = TiffDataSource.DataSource(filename, None, n)
                ds = BufferedDataSource.DataSource(ds, min(ds.getNumSlices(), 50))

                chans.append(ds)

            self.data = ListWrap(chans)


        
        #self.data = readTiff.read3DTiff(filename)

        

        from PYME.ParallelTasks.relativeFiles import getRelFilename
        self.seriesName = getRelFilename(filename)

        self.mode = 'default'

    def LoadImageSeries(self, filename):
        #from PYME.FileUtils import readTiff
        from PYME.Analysis.DataSources import ImageSeriesDataSource

        self.dataSource = ImageSeriesDataSource.DataSource(filename, None)
        self.dataSource = BufferedDataSource.DataSource(self.dataSource, min(self.dataSource.getNumSlices(), 50))
        self.data = self.dataSource #this will get replaced with a wrapped version
        #self.data = readTiff.read3DTiff(filename)

        self.FindAndParseMetadata(filename)

        from PYME.ParallelTasks.relativeFiles import getRelFilename
        self.seriesName = getRelFilename(filename)

        self.mode = 'default'

    def Load(self, filename=None):
        print filename
        if (filename == None):
            import wx #only introduce wx dependency here - so can be used non-interactively
            global lastdir
            
            fdialog = wx.FileDialog(None, 'Please select Data Stack to open ...',
                wildcard='Image Data|*.h5;*.tif;*.lsm;*.kdf;*.md;*.psf|All files|*.*', style=wx.OPEN, defaultDir = lastdir)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                filename = fdialog.GetPath()
                lastdir = fdialog.GetDirectory()

        if not filename == None:
            if filename.startswith('QUEUE://'):
                self.LoadQueue(filename)
            elif filename.endswith('.h5'):
                self.Loadh5(filename)
            elif filename.endswith('.kdf'):
                self.LoadKdf(filename)
            elif filename.endswith('.psf'): #psf
                self.LoadPSF(filename)
            elif filename.endswith('.md'): #treat this as being an image series
                self.LoadImageSeries(filename)
            else: #try tiff
                self.LoadTiff(filename)


            #self.SetTitle(filename)
            self.filename = filename
            self.saved = True

    def Save(self, filename=None, crop=False, view=None):
        import dataExporter

        ofn = self.filename

        if crop:
            dataExporter.CropExportData(view, self.mdh, self.events, self.seriesName)
        else:
            if 'defaultExt' in dir(self):
                self.filename = dataExporter.ExportData(self.data, self.mdh, self.events, defaultExt=self.defaultExt)
            else:
                self.filename = dataExporter.ExportData(self.data, self.mdh, self.events)
            #self.SetTitle(fn)

            if not (self.filename == None):
                self.saved = True

                openImages.pop(ofn)
                openImages[self.filename] = self

            else:
                self.filename = ofn


def GeneratedImage(img, imgBounds, pixelSize, sliceSize, channelNames, mdh=None):
    image = ImageStack(img, mdh=mdh)
    image.pixelSize = pixelSize
    image.sliceSize = sliceSize
    image.imgBounds = imgBounds
    image.names = [c if c else 'Image' for c in channelNames]

    return image



