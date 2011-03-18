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
from PYME.Acquire import MetaDataHandler
from PYME.Analysis import MetaData
from PYME.DSView import dataWrap

lastdir = ''

class ImageStack:
    def __init__(self, data = None, mdh = None, filename = None, queueURI = None, events = []):
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

        #the data does not need to be a numpy array - it could also be, eg., queue data
        #on a remote server - wrap so that is is indexable like an array
        self.data = dataWrap.Wrap(self.data)


    def LoadQueue(self, filename):
        import Pyro.core
        from PYME.Analysis.DataSources import TQDataSource

        if self.queueURI == None:
            if 'PYME_TASKQUEUENAME' in os.environ.keys():
                taskQueueName = os.environ['PYME_TASKQUEUENAME']
            else:
                taskQueueName = 'taskQueue'
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
        xmlfn = os.path.splitext(filename)[0] + '.xml'
        if os.path.exists(xmlfn):
            self.mdh = MetaData.TIRFDefault
            self.mdh.copyEntriesFrom(MetaDataHandler.XMLMDHandler(xmlfn))
        else:
            self.mdh = MetaData.BareBones
            
            #check for simple metadata (python code with an .md extension which 
            #fills a dictionary called md)
            mdfn = os.path.splitext(filename)[0] + '.md'
            if os.path.exists(mdfn):
                self.mdh.copyEntriesFrom(MetaDataHandler.SimpleMDHandler(mdfn))

        if not ('voxelsize.x' in self.mdh.keys() and 'voxelsize.y' in self.mdh.keys()):
            from PYME.DSView.voxSizeDialog import VoxSizeDialog

            dlg = VoxSizeDialog(None)
            dlg.ShowModal()

            self.mdh.setEntry('voxelsize.x', dlg.GetVoxX())
            self.mdh.setEntry('voxelsize.y', dlg.GetVoxY())
            self.mdh.setEntry('voxelsize.z', dlg.GetVoxZ())

    def LoadTiff(self, filename):
        #from PYME.FileUtils import readTiff
        from PYME.Analysis.DataSources import TiffDataSource

        self.dataSource = TiffDataSource.DataSource(filename, None)
        self.data = self.dataSource #this will get replaced with a wrapped version
        #self.data = readTiff.read3DTiff(filename)

        self.FindAndParseMetadata(filename)

        from PYME.ParallelTasks.relativeFiles import getRelFilename
        self.seriesName = getRelFilename(filename)

        self.mode = 'blob'

    def Load(self, filename=None):
        print filename
        if (filename == None):
            import wx #only introduce wx dependency here - so can be used non-interactively
            global lastdir
            
            fdialog = wx.FileDialog(None, 'Please select Data Stack to open ...',
                wildcard='PYME Data|*.h5|TIFF files|*.tif|KDF files|*.kdf|All files|*.*', style=wx.OPEN, defaultDir = lastdir)
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
            else: #try tiff
                self.LoadTiff(filename)


            #self.SetTitle(filename)
            self.filename = filename
            self.saved = True

    def Save(self, filename=None, crop=False, view=None):
        import dataExporter

        if crop:
            dataExporter.CropExportData(view, self.mdh, self.events, self.seriesName)
        else:
            self.filename = dataExporter.ExportData(self.data, self.mdh, self.events)
            #self.SetTitle(fn)

            if not (filename == None):
                self.saved = True






