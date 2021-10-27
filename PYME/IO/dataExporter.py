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
import tables
import numpy

import warnings

try:
    from PIL import Image
except ImportError:
    import Image
    
import os
from PYME.IO.FileUtils import saveTiffStack
from PYME.IO import MetaDataHandler

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

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, tslice=None):
        pass

    def _prepare(self, data, xslice, yslice, zslice, tslice):
        from PYME.IO.dataWrap import Wrap
        from PYME.IO.DataSources import BaseDataSource, CropDataSource
        data = Wrap(data) #make sure we can index along a colour dimension
    
        if data.ndim != 5:
            # promote to xyztc
            data = BaseDataSource.XYZTCWrapper.auto_promote(data)
        
        cropped = CropDataSource.DataSource(data, xrange=xslice, yrange=yslice, zrange=zslice, trange=tslice)
        
        #if tslice is None:
        #    tslice = slice(0, data.shape[3], 1)
        
        #nChans = data.shape[4]
        #nZ = (zslice.stop - zslice.start) // zslice.step
        #nT = (tslice.stop - tslice.start) // tslice.step

        xSize, ySize, nZ, nT, nChans = cropped.shape
        
        nframes = nZ * nT

        #slice0 = data[xslice, yslice, 0, 0, 0]
        #xSize, ySize = slice0.shape[:2]
        
        return cropped, tslice, xSize, ySize, nZ, nT, nChans, nframes
        

#@exporter
class H5Exporter(Exporter):
    extension = '*.h5'
    descr = 'PYME HDF - .h5'

    def __init__(self, complib='zlib', complevel=1):
        self.complib = complib
        self.complevel = complevel

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None, tslice=None):
        h5out = tables.open_file(outFile,'w', chunk_cache_size=2**23)
        filters=tables.Filters(self.complevel,self.complib,shuffle=True)
        

        # prepare now does cropping, no need to handle in individual exporters
        # TODO - allow averaging as an alternative to sub-sampling
        data, tslice, xSize, ySize, nZ, nT, nChans, nframes = self._prepare(data, xslice, yslice, zslice, tslice)
        
        print((xSize, ySize))
        dtype = data.dtype
        
        if not dtype in ['uint8', 'uint16', 'float32']:
            warnings.warn('Attempting to save an unsupported data-type (%s) - data should be one of uint8, uint16, or float32' % dtype,
                           stacklevel=2)
        
        atm = tables.Atom.from_dtype(dtype)

        ims = h5out.create_earray(h5out.root,'ImageData',atm,(0,xSize,ySize), filters=filters, expectedrows=nframes, chunkshape=(1,xSize,ySize))

        curFrame = 0
        for ch_num in range(nChans):
            for t in range(nT):
                for z in range(nZ):
                    curFrame += 1
                    im = data[:, :, z, t, ch_num].squeeze()
                        
                    if im.ndim == 1:
                        im = im.reshape((-1, 1))[None, :,:]
                    else:
                        im = im[None, :,:]
                    
                    #print im.shape
                    ims.append(im)
                    if ((curFrame % 10) == 0)  and progressCallback:
                        try:
                            progressCallback(curFrame, nframes)
                        except:
                            pass
            #ims.flush()
        
        ims.attrs.DimOrder = 'XYZTC'
        ims.attrs.SizeC = nChans
        ims.attrs.SizeZ = nZ
        ims.attrs.SizeT = nT
            
        ims.flush()

        outMDH = MetaDataHandler.HDFMDHandler(h5out)

        if not metadata is None:
            outMDH.copyEntriesFrom(metadata)

            if 'Camera.ADOffset' in metadata.getEntryNames():
                outMDH.setEntry('Camera.ADOffset', zslice.step*metadata.getEntry('Camera.ADOffset'))

        if not origName is None:
            outMDH.setEntry('cropping.originalFile', origName)
            
        outMDH.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
        outMDH.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
        outMDH.setEntry('cropping.zslice', zslice.indices(data.shape[2]))
        if tslice is not None:
            outMDH.setEntry('cropping.tslice', tslice.indices(data.shape[3]))

        if not events is None and len(events) > 0:
            assert isinstance(events, numpy.ndarray), "expected type of events object to be numpy array, but was {}".format(type(events))
            # this should get the sorting done automatically
            outEvents = h5out.create_table(h5out.root, 'Events', events, filters=tables.Filters(complevel=5, shuffle=True))
        else:
            outEvents = h5out.create_table(h5out.root, 'Events', SpoolEvent, filters=tables.Filters(complevel=5, shuffle=True))

        h5out.flush()
        h5out.close()

        if progressCallback:
            try:
                progressCallback(nframes, nframes)
            except:
                pass

exporter(H5Exporter)


#@exporter
class TiffStackExporter(Exporter):
    extension = '*.tiff'
    descr = 'TIFF [old style] - .tiff'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None):
        warnings.warn('export of old style tiffs should only be used in exceptional circumstances')
        #xmd = None
        if not metadata is None:
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
                if not metadata is None and 'ChannelNames' in metadata.getEntryNames():
                    chanNames = metadata['ChannelNames']    

                else:
                    chanNames = range(data.shape[3])

                chanFiles = [os.path.splitext(os.path.split(outFile)[1])[0] + '__%s.tif' % chanNames[i]  for i in range(data.shape[3])]
                if not metadata is None:
                    xmd['ChannelFiles'] = chanFiles

                for i in range(data.shape[3]):
                    saveTiffStack.saveTiffMultipage(data[xslice, yslice, zslice, i].squeeze(), os.path.splitext(outFile)[0] + '__%s.tif' % chanNames[i])
        else:
            saveTiffStack.saveTiffMultipage(data[xslice, yslice, zslice], outFile)

        if not metadata is None:
            #xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            if not origName is None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))

            print((xslice.indices(data.shape[0])))
            
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)
            # xmd.WriteSimple(xmlFile)

        if progressCallback:
            try:
                progressCallback(100, 100)
            except:
                pass

#exporter(TiffStackExporter)

import numpy as np
class _CastWrapper(object):
    def __init__(self, data, dtype=np.uint8):
        self._data = data
        self._dtype = dtype
        
    @property
    def dtype(self):
        return self._dtype
        
    def getSlice(self, ind):
        return self._data.getSlice(ind).astype(self._dtype)
    
    def __getattr__(self, item):
        return getattr(self._data, item)
    
    def __getitem__(self, item):
        return self._data.__getitem__(item).astype(self._dtype)
        

class OMETiffExporter(Exporter):
    extension = '*.tif'
    descr = 'OME TIFF - .tif'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None, tslice=None):
        from PYME.contrib.gohlke import tifffile

        data, _, _, _, _, _, _, _ = self._prepare(data, xslice, yslice, zslice, tslice)
             
        if data.dtype == 'bool':
            data = _CastWrapper(data, 'uint8')
        
        if data.nbytes > 2e9:
            warnings.warn('Data is larger than 2GB, generated TIFF may not read in all software')
            
        if data.nbytes > 4e9:
            raise RuntimeError('TIFF has a maximum file size of 4GB, crop data or save as HDF')
        
        #dw = dataWrap.ListWrapper([numpy.atleast_3d(_bool_to_uint8(data[xslice, yslice, zslice, i].squeeze())) for i in range(data.shape[3])])
        #xmd = None
        
        
        if not metadata is None:
            xmd = MetaDataHandler.OMEXMLMDHandler(mdToCopy=metadata)
            if not origName is None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))
            xmd.setEntry('cropping.tslice', tslice.indices(data.shape[3]))
            
            description=xmd.getXML(data)
        else:
            description = None
            
            
            
        tifffile.imsave_f(outFile, data, description = description)

        if progressCallback:
            try:
                progressCallback(100, 100)
            except:
                pass


exporter(OMETiffExporter)

#@exporter
class TiffSeriesExporter(Exporter):
    extension = '*.xml'
    descr = 'TIFF Series - .xml'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None):
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

        if not metadata is None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            if not origName is None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))

            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)
            # xmd.WriteSimple(xmlFile)

        if progressCallback:
            try:
                progressCallback(100, 100)
            except:
                pass

exporter(TiffSeriesExporter)


#@exporter
class NumpyExporter(Exporter):
    extension = '*.npy'
    descr = 'Pickled numpy ndarray - .npy'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None):
        numpy.save(outFile, data[xslice, yslice, zslice])

        if not metadata is None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            if not origName is None:
                xmd.setEntry('cropping.originalFile', origName)

            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))
            
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)
            #xmd.WriteSimple(xmlFile)

        if progressCallback:
            try:
                progressCallback(100, 100)
            except:
                pass

exporter(NumpyExporter)

#@exporter
class PSFExporter(Exporter):
    extension = '*.psf'
    descr = 'PYME psf data - .psf'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None):
        #numpy.save(outFile, data[xslice, yslice, zslice])
        warnings.warn('The .psf format is deprecated. Save PSFs as .tif instead')
        
        from six.moves import cPickle
            
        fid = open(outFile, 'wb')
        cPickle.dump((data[xslice, yslice, zslice], metadata.voxelsize), fid, 2)
        fid.close()

        if progressCallback:
            try:
                progressCallback(100, 100)
            except:
                pass

exporter(PSFExporter)

#@exporter
class TxtExporter(Exporter):
    extension = '*.txt'
    descr = 'Tab formatted txt  - .txt'

    def Export(self, data, outFile, xslice, yslice, zslice, metadata=None, events = None, origName=None, progressCallback=None):
        #numpy.save(outFile, data[xslice, yslice, zslice])
        #import cPickle

        try:
            chanNames = metadata.getEntry('ChannelNames')
        except:
            chanNames = ['Chan %d' % d for d in range(data.shape[4])]

        dat = [data[xslice, yslice, zslice, :, chan].squeeze() for chan in range(data.shape[4])]

        #print chanNames

        if metadata and 'Profile.XValues' in metadata.getEntryNames():
            dat = [metadata.getEntry('Profile.XValues')] + dat
            chanNames = [metadata.getEntry('Profile.XLabel')] + chanNames

        if metadata and 'Spectrum.Wavelengths' in metadata.getEntryNames():
            dat = [metadata.getEntry('Spectrum.Wavelengths')] + dat
            chanNames = ['Wavelength [nm]'] + chanNames
            

        fid = open(outFile, 'wb')

        #write header
        fid.write(('#' + '\t'.join(chanNames)).encode('utf-8'))

        for i in range(len(dat[0])):
            fid.write(('\n' + '\t'.join(['%f' % d[i] for d in dat])).encode('utf-8'))

        fid.close()
        
        # write metadata in xml file
        if not metadata is None:
            xmd = MetaDataHandler.XMLMDHandler(mdToCopy=metadata)
            if not origName is None:
                xmd.setEntry('cropping.originalFile', origName)
    
            xmd.setEntry('cropping.xslice', xslice.indices(data.shape[0]))
            xmd.setEntry('cropping.yslice', yslice.indices(data.shape[1]))
            xmd.setEntry('cropping.zslice', zslice.indices(data.shape[2]))
    
            xmlFile = os.path.splitext(outFile)[0] + '.xml'
            xmd.writeXML(xmlFile)

        if progressCallback:
            try:
                progressCallback(100, 100)
            except:
                pass

exporter(TxtExporter)
#exporters = {'PYME HDF - .h5': H5Exporter,
#            'TIFF (stack if 3D) - .tif' : TiffStackExporter,
#            'TIFF (series if 3D) - .tif' : TiffSeriesExporter,
#            'Pickled numpy ndarray - .npy' : NumpyExporter}

#print exporters

def ExportData(ds, mdh=None, events=None, origName = None, defaultExt = '*.tif', filename=None, progressCallback=None):
    if filename is None:
        #show file selection dialog box
        from PYME.ui.crop_dialog import _getFilename
        filename = _getFilename(defaultExt)
        
    if filename is None:
        #we cancelled the dialog - exit
        return

        
    ext = '*' + os.path.splitext(filename)[1]
        
    if not ext in exportersByExtension.keys():
        raise RuntimeError('No exporter found for %s files' % ext)
        #wx.MessageBox('No exporter found for %s files\n Try one of the following file types:\n%s' % (ext, ', '.join(exportersByExtension.keys())), "Error saving data", wx.OK|wx.ICON_HAND)
        return

    exp = exportersByExtension[ext]()
    exp.Export(ds, filename, slice(0, ds.shape[0], 1), slice(0, ds.shape[1], 1), slice(0, ds.shape[2], 1),mdh, events, origName, progressCallback=progressCallback)
    return filename
    

