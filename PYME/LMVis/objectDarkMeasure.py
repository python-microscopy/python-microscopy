import numpy as np

from scipy.optimize import curve_fit
def cumuexpfit(t,tau):
    return 1-np.exp(-t/tau)

def notimes(ndarktimes):
    analysis = {
        'NDarktimes' : ndarktimes,
        'tau1' : [None,None,None],
        'tau2' : [None,None,None]
    }
    return analysis

def fitDarktimes(t):
    # determine darktime from gaps and reject zeros (no real gaps) 
    nts = 0 # initialise to safe default
    NTMIN = 5
    #print t
    #print type(t)
    
    if t.size > NTMIN:
        dts = t[1:]-t[0:-1]-1
        dtg = dts[dts>0]
        nts = dtg.shape[0]

    if nts > NTMIN:
        # now make a cumulative histogram from these
        cumux = np.sort(dtg+0.01*np.random.random(nts)) # hack: adding random noise helps us ensure uniqueness of x values
        cumuy = (1.0+np.arange(nts))/np.float(nts)
        maxtd = dtg.max()
        
        # generate histograms 2nd way
        binedges = 0.5+np.arange(0,maxtd)
        binctrs = 0.5*(binedges[0:-1]+binedges[1:])
        h,be2 = np.histogram(dtg,bins=binedges)
        hc = np.cumsum(h)
        hcg = hc[h>0]/float(nts) # only nonzero bins and normalise
        binctrsg = binctrs[h>0]
        
        success = True
        # fit theoretical distributions
        try:
            popth,pcovh,infodicth,errmsgh,ierrh  = curve_fit(cumuexpfit,binctrsg,hcg, p0=(300.0),full_output=True)
        except:
            success = False
        else:
            chisqredh = ((hcg - infodicth['fvec'])**2).sum()/(hcg.shape[0]-1)
        try:
            popt,pcov,infodict,errmsg,ierr = curve_fit(cumuexpfit,cumux,cumuy, p0=(300.0),full_output=True)
        except:
            success = False
        else:
            chisqred = ((cumuy - infodict['fvec'])**2).sum()/(nts-1)

        if success:
            analysis = {
                'NDarktimes' : nts,
                'tau1' : [popt[0],np.sqrt(pcov[0][0]),chisqredh],
                'tau2' : [popth[0],np.sqrt(pcovh[0][0]),chisqred]
            }
        else:
            analysis = notimes(nts)
    else:
        analysis = notimes(nts)

    return analysis

measureDType = [('objID', 'i4'), ('xPos', 'f4'), ('yPos', 'f4'), ('NEvents', 'i4'), ('NDarktimes', 'i4'), ('tau1', 'f4'),
                ('tau2', 'f4'), ('tau1err', 'f4'),
                ('tau2err', 'f4'), ('chisqr1', 'f4'), ('chisqr2', 'f4')]


def measure(object, measurements = np.zeros(1, dtype=measureDType)):
    #measurements = {}

    measurements['NEvents'] = object.shape[0]
    measurements['xPos'] = object[:,0].mean()
    measurements['yPos'] = object[:,1].mean()

    t = object[:,2].squeeze()

    darkanalysis = fitDarktimes(t)
    measurements['tau1'] = darkanalysis['tau1'][0]
    measurements['tau2'] = darkanalysis['tau2'][0]
    measurements['tau1err'] = darkanalysis['tau1'][1]
    measurements['tau2err'] = darkanalysis['tau2'][1]
    measurements['chisqr1'] = darkanalysis['tau1'][2]
    measurements['chisqr2'] = darkanalysis['tau2'][2]
    
    measurements['NDarktimes'] = darkanalysis['NDarktimes']
    
    return measurements

def measureObjectsByID(filter, ids):
    x = filter['x'] #+ 0.1*random.randn(filter['x'].size)
    y = filter['y'] #+ 0.1*random.randn(x.size)
    id = filter['objectID'].astype('i')
    t = filter['t']
    floattype = 'float64'
    tau1 = -1*np.ones_like(t, dtype = floattype)
    ndt = -1*np.ones_like(t, dtype = floattype)
    qus = -1*np.ones_like(t, dtype = floattype)
    #ids = set(ids)

    measurements = np.zeros(len(ids), dtype=measureDType)

    for j,i in enumerate(ids):
        if not i == 0:
            ind = id == i
            obj = np.vstack([x[ind],y[ind],t[ind]]).T
            #print obj.shape
            measure(obj, measurements[j])
            measurements[j]['objID'] = i
            if not np.isnan(measurements[j]['tau1']):
                tau1[ind] = measurements[j]['tau1']
                qus[ind] = 100.0/measurements[j]['tau1']
            ndt[ind] = measurements[j]['NDarktimes']

    return (measurements, tau1, qus, ndt)
