#!/usr/bin/python
##################
# objectMeasurements.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# c.soeller@exeter.ac.uk
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

import wx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import StringIO

def cumuexpfit(t,tau):
    return 1-np.exp(-t/tau)

def Warn(parent, message, caption = 'Warning!'):
    dlg = wx.MessageDialog(parent, message, caption, wx.OK | wx.ICON_WARNING)
    dlg.ShowModal()
    dlg.Destroy()

class QPObjectSegmenter:
    def __init__(self, visFr):
        self.visFr = visFr
        self.pipeline = self.visFr.pipeline

        visFr.AddMenuItem('Extras', itemType='separator')
        visFr.AddMenuItem('Extras', "qPAINT - Get object IDs from image",self.OnGetIDs)
        visFr.AddMenuItem('Extras', "qPAINT - Measure nonzero object ID dark times",self.OnMeasure)
        visFr.AddMenuItem('Extras', "qPAINT - Plot Dark Time Histogram",self.OnDarkT)
        visFr.AddMenuItem('Extras', itemType='separator')

    def OnGetIDs(self, event):
        from PYME.DSView import dsviewer

        visFr = self.visFr
        pipeline = visFr.pipeline

        dlg = wx.SingleChoiceDialog(
                None, 'choose the image which contains labels', 'Use Segmentation',
                list(dsviewer.openViewers.keys()),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            img = dsviewer.openViewers[dlg.GetStringSelection()].image
            
            #account for ROIs
            #dRx = pipeline.mdh['Camera.ROIPosX']*pipeline.mdh['voxelsize.x']*1e3 - img.mdh['Camera.ROIPosX']*img.mdh['voxelsize.x']*1e3
            #dRy = pipeline.mdh['Camera.ROIPosY']*pipeline.mdh['voxelsize.y']*1e3 - img.mdh['Camera.ROIPosY']*img.mdh['voxelsize.y']*1e3

            pixX = np.round((pipeline.filter['x'] - img.imgBounds.x0 )/img.pixelSize).astype('i')
            pixY = np.round((pipeline.filter['y'] - img.imgBounds.y0 )/img.pixelSize).astype('i')

            ind = (pixX < img.data.shape[0])*(pixY < img.data.shape[1])*(pixX >= 0)*(pixY >= 0)

            ids = np.zeros_like(pixX)
            #assume there is only one channel
            ids[ind] = img.data[:,:,:,0].squeeze()[pixX[ind], pixY[ind]].astype('i')

            numPerObject, b = np.histogram(ids, np.arange(ids.max() + 1.5) + .5)

            pipeline.addColumn('objectID', ids)
            pipeline.addColumn('NEvents', numPerObject[ids-1])

            pipeline.Rebuild()

        dlg.Destroy()

    def setMapping(self, mapname, mapattrname, var):
        pipeline = self.pipeline
        setattr(pipeline.selectedDataSource, mapattrname, -1*np.ones_like(pipeline.selectedDataSource['x']))
        mapattr = getattr(pipeline.selectedDataSource,mapattrname)
        mapattr[pipeline.filter.Index] = var
        pipeline.selectedDataSource.setMapping(mapname, mapattrname)

    def OnMeasure(self, event):
        from PYME.LMVis import objectDarkMeasure

        chans = self.pipeline.colourFilter.getColourChans()

        ids = set(self.pipeline.mapping['objectID'].astype('i'))
        self.pipeline.objectMeasures = {}

        pipeline = self.pipeline
        
        if len(chans) == 0:
            pipeline.objectMeasures['Everything'], tau1, qidx, ndt = objectDarkMeasure.measureObjectsByID(self.pipeline.colourFilter, ids)
            pipeline.addColumn('taudark',tau1)
            pipeline.addColumn('NDarktimes',ndt)
            pipeline.addColumn('QIndex',qidx)
            pipeline.Rebuild()
        else:
            curChan = self.pipeline.colourFilter.currentColour

            chanNames = chans[:]

#            if 'Sample.Labelling' in metadata.getEntryNames():
#                lab = metadata.getEntry('Sample.Labelling')
#
#                for i in range(len(lab)):
#                    if lab[i][0] in chanNames:
#                        chanNames[chanNames.index(lab[i][0])] = lab[i][1]

            for ch, i in zip(chans, range(len(chans))):
                self.pipeline.colourFilter.setColour(ch)
                #fitDecayChan(colourFilter, metadata, chanNames[i], i)
                self.pipeline.objectMeasures[chanNames[i]], tau1, qidx, ndt = objectDarkMeasure.measureObjectsByID(self.visFr.colourFilter, ids)
            self.pipeline.colourFilter.setColour(curChan)

    def OnDarkT(self,event):
        visFr = self.visFr
        pipeline = visFr.pipeline
        mdh = pipeline.mdh

        NTMIN = 5
        maxPts = 1e4
        t = pipeline['t']
        if len(t) > maxPts:
            Warn(None,'aborting darktime analysis: too many events, current max is %d' % maxPts)
            return
        x = pipeline['x']
        y = pipeline['y']

        # determine darktime from gaps and reject zeros (no real gaps) 
        dts = t[1:]-t[0:-1]-1
        dtg = dts[dts>0]
        nts = dtg.shape[0]

        if nts > NTMIN:
            # now make a cumulative histogram from these
            cumux = np.sort(dtg+0.01*np.random.random(nts)) # hack: adding random noise helps us ensure uniqueness of x values
            cumuy = (1.0+np.arange(nts))/np.float(nts)
            bbx = (x.min(),x.max())
            bby = (y.min(),y.max())
            voxx, voxy, _ = mdh.voxelsize_nm
            
            bbszx = bbx[1]-bbx[0]
            bbszy = bby[1]-bby[0]
            maxtd = dtg.max()

            # generate histograms 2nd way
            binedges = 0.5+np.arange(0,maxtd)
            binctrs = 0.5*(binedges[0:-1]+binedges[1:])
            h,be2 = np.histogram(dtg,bins=binedges)
            hc = np.cumsum(h)
            hcg = hc[h>0]/float(nts) # only nonzero bins and normalise
            binctrsg = binctrs[h>0]

        # fit theoretical distributions
        popth,pcovh,popt,pcov = (None,None,None,None)
        if nts > NTMIN:
            popth,pcovh,infodicth,errmsgh,ierrh = curve_fit(cumuexpfit,binctrsg,hcg, p0=(300.0),full_output=True)
            chisqredh = ((hcg - infodicth['fvec'])**2).sum()/(hcg.shape[0]-1)
            popt,pcov,infodict,errmsg,ierr = curve_fit(cumuexpfit,cumux,cumuy, p0=(300.0),full_output=True)
            chisqred = ((cumuy - infodict['fvec'])**2).sum()/(nts-1)


            # plot data and fitted curves
            plt.figure()
            plt.subplot(211)
            plt.plot(cumux,cumuy,'o')
            plt.plot(cumux,cumuexpfit(cumux,popt[0]))
            plt.plot(binctrs,hc/float(nts),'o')
            plt.plot(binctrs,cumuexpfit(binctrs,popth[0]))
            plt.ylim(-0.2,1.2)
            plt.subplot(212)
            plt.semilogx(cumux,cumuy,'o')
            plt.semilogx(cumux,cumuexpfit(cumux,popt[0]))
            plt.semilogx(binctrs,hc/float(nts),'o')
            plt.semilogx(binctrs,cumuexpfit(binctrs,popth[0]))
            plt.ylim(-0.2,1.2)
            plt.show()
            
            outstr = StringIO()

            analysis = {
                'Nevents' : t.shape[0],
                'Ndarktimes' : nts,
                'filterKeys' : pipeline.filterKeys.copy(),
                'darktimes' : (popt[0],popth[0]),
                'darktimeErrors' : (np.sqrt(pcov[0][0]),np.sqrt(pcovh[0][0]))
            }

            if not hasattr(self.visFr,'analysisrecord'):
                self.visFr.analysisrecord = []
                self.visFr.analysisrecord.append(analysis)

            outstr.write(u"events: %d \n" % t.shape[0])
            outstr.write(u"dark times: %d \n" % nts)
            outstr.write(u"region: %d x %d nm (%d x %d pixel) \n" % (bbszx,bbszy,bbszx/voxx,bbszy/voxy))
            outstr.write(u"centered at %d,%d (%d,%d pixels) \n" % (x.mean(),y.mean(),x.mean()/voxx,y.mean()/voxy))
            outstr.write(u"darktime: %.1f (%.1f) frames - chisqr %.2f (%.2f) \n" % (popt[0],popth[0],chisqred,chisqredh))
            outstr.write(u"qunits: %.2f (%.2f), eunits: %.2f \n" % (100.0/popt[0], 100.0/popth[0],t.shape[0]/500.0))

            labelstr = str(outstr.getvalue())
            plt.annotate(labelstr, xy=(0.5, 0.1), xycoords='axes fraction',
                         fontsize=10)
        else:
            Warn(None, 'not enough data points (<%d)' % NTMIN, 'Error')

def Plug(visFr):
    '''Plugs this module into the gui'''
    QPObjectSegmenter(visFr)
