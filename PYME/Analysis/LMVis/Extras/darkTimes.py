import wx
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import StringIO

def cumuexpfit(t,tau):
    return 1-np.exp(-t/tau)

def Warn(parent, message, caption = 'Warning!'):
    dlg = wx.MessageDialog(parent, message, caption, wx.OK | wx.ICON_WARNING)
    dlg.ShowModal()
    dlg.Destroy()

class DarkTimes:
    def __init__(self, visFr):
        self.visFr = visFr

        ID_DARKT = wx.NewId()
        visFr.extras_menu.Append(ID_DARKT, "Plot Dark Time Histogram")
        visFr.Bind(wx.EVT_MENU, self.OnDarkT, id=ID_DARKT)

    def OnDarkT(self,event):
        visFr = self.visFr
        pipeline = visFr.pipeline
        mdh = pipeline.mdh

        NTMIN = 5
        t = pipeline['t']
        if len(t) > 1e4:
            Warn(None,'aborting darktime analysis: too many events, current max is 1e4')
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
            voxx = 1e3*mdh['voxelsize.x']
            voxy = 1e3*mdh['voxelsize.y']
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
            
            outstr = StringIO.StringIO()

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

            print >>outstr, "events: %d" % t.shape[0]
            print >>outstr, "dark times: %d" % nts
            print >>outstr, "region: %d x %d nm (%d x %d pixel)" % (bbszx,bbszy,bbszx/voxx,bbszy/voxy)
            print >>outstr, "centered at %d,%d (%d,%d pixels)" % (x.mean(),y.mean(),x.mean()/voxx,y.mean()/voxy)
            print >>outstr, "darktime: %.1f (%.1f) frames - chisqr %.2f (%.2f)" % (popt[0],popth[0],chisqred,chisqredh)
            print >>outstr, "qunits: %.2f (%.2f), eunits: %.2f" % (100.0/popt[0], 100.0/popth[0],t.shape[0]/500.0)

            labelstr = outstr.getvalue()
            plt.annotate(labelstr, xy=(0.5, 0.1), xycoords='axes fraction',
                         fontsize=10)
        else:
            Warn(None, 'not enough data points (<%d)' % NTMIN, 'Error')
def Plug(visFr):
    '''Plugs this module into the gui'''
    DarkTimes(visFr)
