
import wx
import numpy as np
from PYME.Analysis.points import twoColour
from PYME.Analysis.points.DeClump import pyDeClump
import os
from PYME.IO.FileUtils import nameUtils
import json
import importlib

def foldX(pipeline):
    """
    At this point the origin of x should be the corner of the concatenated frame
    """
    xOG = pipeline['x']
    roiSizeNM = (pipeline.mdh['Multiview.ROISize'][1]*pipeline.mdh['voxelsize.x']*1000)  # voxelsize is in um
    xfold = xOG % roiSizeNM
    mvQuad = np.floor(xOG / roiSizeNM)

    pipeline.mapping.setMapping('xFolded', xfold)
    pipeline.mapping.setMapping('whichFOV', mvQuad)
    return

def plotFolded(regX, regY, multiviewChannels, title=''):
    import matplotlib.pyplot as plt
    nChan = multiviewChannels.max()

    c = iter(plt.cm.rainbow(np.linspace(0, 1, nChan)))
    for ii in range(nChan):
        mask = (ii == multiviewChannels)
        plt.scatter(regX[mask], regY[mask], c=next(c))
    plt.title(title)
    return

def plotRegistered(regX, regY, numFOV, title=''):
    import matplotlib.pyplot as plt
    plt.figure()
    c = iter(plt.cm.rainbow(np.linspace(0, 1, numFOV)))
    for ii in range(numFOV):
        plt.scatter(regX[ii], regY[ii], c=next(c), label='FOV #%i' % ii)
    plt.title(title)
    return

def pairMolecules_old(xFold, y, whichFOV, combineDist, FOV1=0, FOV2=1):
    """
    This function will be depreciated, as it is not C-accelerated. Additionally,
    this function would need to be called nROI-1 times, as opposed to once.
    """
    interestingMolecules = np.logical_or(whichFOV == FOV1, whichFOV == FOV2)
    xx = xFold[interestingMolecules]
    yy = y[interestingMolecules]
    chan = whichFOV[interestingMolecules]
    numMol = len(xx)
    dist = np.zeros((numMol, numMol))
    for ind in range(numMol):
        for ii in range(numMol):
            dist[ind, ii] = np.sqrt((xx[ind] - xx[ii])**2 + (yy[ind] - yy[ii])**2)
    dist = dist + (dist == 0)*999999
    minDist = dist.min(axis=0)
    minLoc = dist.argmin(axis=0)

    # keepList = np.where(np.logical_and(FOV[minLoc] != FOV, minLoc[range(numMol)] == minLoc[minLoc[range(numMol)]]))

    # only keep molecule pairs that are mutually nearest neighbors, within a certain distance, and heterozygous in planes
    keep = np.logical_and(minLoc[range(numMol)] == minLoc[minLoc[range(numMol)]],
                          np.logical_and(minDist <= combineDist, chan != chan[minLoc]))
    keepList = np.where(keep)

    minLocKept = minLoc[keep]

    # numKept = len(keepList)
    chan1Keep = np.logical_and(keep, whichFOV == FOV1)
    pairs = minLocKept[chan1Keep]
    # chan2Keep = minLocKept[~chan1Keep] #  np.logical_and(keep, whichFOV == FOV2)
    x1 = xFold[chan1Keep]
    y1 = y[chan1Keep]
    x2 = xFold[pairs]
    y2 = y[pairs]


    return x1, y1, x2, y2

def pairMolecules(tIndex, x, y, deltaX, whichFOV, numFOV):
    # sort everything in frame order
    I = tIndex.argsort()
    tIndex = tIndex[I]
    x = x[I]
    y = y[I]
    whichFOV = whichFOV[I]

    # group within 100nm
    dX = 100.*np.ones_like(deltaX)
    # group localizations
    assigned = pyDeClump.findClumps(tIndex, x, y, dX)

    # only look at clumps with N=numFOVs, where each FOV is represented
    # clumps, nInClump = np.unique(assigned, return_counts=True)
    clumps = np.unique(assigned)
    keptClumps = [np.array_equal(np.unique(whichFOV[assigned == clumps[ii]]), np.arange(numFOV)) for ii in range(len(clumps))]

    keptMoles = []# assigned in clumps[np.where(keptClumps)]
    # np.array_equal(clumps, np.arange(1, np.max(assigned) + 1)) equaluates to True
    # assigned == np.any(clumps[keptClumps])
    for elem in assigned:
        keptMoles.append(elem in clumps[np.where(keptClumps)])
    keep = np.where(keptMoles)

    #return x[keptMoles], y[keptMoles], whichFOV[keptMoles], clumps[keptClumps]
    return x[keep], y[keep], whichFOV[keep], assigned[keep] #  , clumps[np.where(keptClumps)]

class biplaneMapper:
    def __init__(self, visFr):
        self.visFr = visFr

        ID_REGISTER_BIPLANE = wx.NewId()
        visFr.extras_menu.Append(ID_REGISTER_BIPLANE, "Biplane - Register Channels")
        visFr.Bind(wx.EVT_MENU, self.OnRegisterBiplane, id=ID_REGISTER_BIPLANE)

        ID_MAP_BIPLANE = wx.NewId()
        visFr.extras_menu.Append(ID_MAP_BIPLANE, "Biplane - Map Z")
        visFr.Bind(wx.EVT_MENU, self.OnFoldAndMap, id=ID_MAP_BIPLANE)
        return

    def applyShiftmaps(self, x, y, numFOV):
        pipeline = self.visFr.pipeline
        # __dict__.__setitem__('Shiftmap.FOV0%s' % ii, [spx, spy])
        #xReg = [x[ii] + pipeline.mdh.__dict__['Shiftmap.FOV0%s' % ii][0].ev(x[ii], y[ii]) for ii in range(1, numFOV)]
        #yReg = [y[ii] + pipeline.mdh.__dict__['Shiftmap.FOV0%s' % ii][1].ev(x[ii], y[ii]) for ii in range(1, num
        model = self.shiftWallet['shiftModel'].split('.')[-1]
        shiftModule = importlib.import_module(self.shiftWallet['shiftModel'].split('.' + model)[0])
        shiftModel = getattr(shiftModule, model)

        xReg, yReg = [x[0]], [y[0]]
        for ii in range(1, numFOV):
            #xReg.append(x[ii] + pipeline.mdh.__dict__['Shiftmap']['FOV0%s' % ii][0].ev(x[ii], y[ii]))
            #yReg.append(y[ii] + pipeline.mdh.__dict__['Shiftmap']['FOV0%s' % ii][1].ev(x[ii], y[ii]))
            xReg.append(x[ii] + shiftModel(dict=self.shiftWallet['FOV0%s.X' % ii]).ev(x[ii], y[ii]))
            yReg.append(y[ii] + shiftModel(dict=self.shiftWallet['FOV0%s.Y' % ii]).ev(x[ii], y[ii]))

        #xReg = x + [(whichFOV == ii)*shiftMapsX[ii](x) for ii in range(numFOV)]
        #yReg = y + [(whichFOV == ii)*shiftMapsY[ii](y) for ii in range(numFOV)]

        pipeline.mapping.setMapping('xReg', xReg)
        pipeline.mapping.setMapping('yReg', yReg)
        return

    def OnFoldAndMap(self, event):
        pipeline = self.visFr.pipeline

        try:  # load shiftmaps from metadata, if present
            # numFOV = pipeline.mdh['Multiview.NumROIs']
            self.shiftWallet = pipeline.mdh.__dict__['Shiftmap']
        except KeyError:
            try:
                defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + 'MultiView.sf'
                fdialog = wx.FileDialog(None, 'Save shift field as ...', wildcard='Shift Field file (*.sf)|*.sf',
                                        style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
                succ = fdialog.ShowModal()
                if (succ == wx.ID_OK):
                    fpath = fdialog.GetPath()
                    # load json
                    fid = open(fpath, 'r')
                    self.shiftWallet = json.load(fid)
                    fid.close()
            except:
                raise IOError('Shiftmaps not found in metadata and could not be loaded from file')

        numFOV = pipeline.mdh['Multiview.NumROIs']
        foldX(pipeline)

        print('length is %i, max is %d' % (len(pipeline.mapping.__dict__['xFolded']), np.max(pipeline.mapping.__dict__['xFolded'])))
        print('Number of ROIs: %f' % np.max(pipeline.mapping.__dict__['whichFOV']))
        plotFolded(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], 'Raw')

        #self.applyShiftmaps(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
        #                    pipeline.mapping.__dict__['whichFOV'], numFOV)
        xfold, yfold = [], []
        for ii in range(numFOV):
            xfold.append(pipeline.mapping.__dict__['xFolded'][np.where(pipeline.mapping.__dict__['whichFOV'] == ii)])
            yfold.append(pipeline['y'][np.where(pipeline.mapping.__dict__['whichFOV'] == ii)])
        self.applyShiftmaps(xfold, yfold, numFOV)

        plotRegistered(pipeline.mapping.__dict__['xReg'], pipeline.mapping.__dict__['yReg'],
                            numFOV, 'After Registration')

    def OnRegisterBiplane(self, event):
        from PYME.Analysis.points import twoColour
        pipeline = self.visFr.pipeline

        try:
            numFOV = pipeline.mdh['Multiview.NumROIs']
        except KeyError:
            raise UserWarning('You are either not looking at Biplane Data, or your metadata is incomplete')
            return

        foldX(pipeline)

        print('length is %i, max is %d' % (len(pipeline.mapping.__dict__['xFolded']), np.max(pipeline.mapping.__dict__['xFolded'])))
        print('Number of ROIs: %f' % np.max(pipeline.mapping.__dict__['whichFOV']))

        # Now we need to match up molecules
        #combineDist = 20
        x, y, FOV, clumpID, = pairMolecules(pipeline['tIndex'], pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                      pipeline['error_x'], pipeline.mapping.__dict__['whichFOV'], numFOV)

        # merge clumps within channels
        molList = np.unique(clumpID)
        numMoles = len(molList)
        #xClumped = [[np.mean(x[np.logical_and(clumpID == ii, FOV == ind)]) for ii in range(numMoles)] for ind in range(numFOV)]
        #yClumped = [[np.mean(y[np.logical_and(clumpID == ii, FOV == ind)]) for ii in range(numMoles)] for ind in range(numFOV)]

        # Generate raw shift vectors (map of displacements between channels) for each FOV
        #xRawShiftVec = [twoColour.genShiftVectors(x[FOV == 0], x[FOV == ii]) for ii in range(1, numFOV)]
        #yRawShiftVec = [twoColour.genShiftVectors(y[FOV == 0], y[FOV == ii]) for ii in range(1, numFOV)]
        #chan1 = (FOV == 0)
        dx = np.zeros((numFOV - 1, numMoles))
        dy = np.zeros_like(dx)
        dxErr = np.zeros_like(dx)
        dyErr = np.zeros_like(dx)
        xClump, yClump, xStd, yStd = [], [], [], []
        self.shiftWallet = {}
        dxWallet, dyWallet = {}, {}
        for ii in range(numFOV):
            chan = (FOV == ii)
            #numInChan = len(np.unique(clumpID[chan]))
            xChan = np.zeros(numMoles)
            yChan = np.zeros(numMoles)
            xChanStd = np.zeros(numMoles)
            yChanStd = np.zeros(numMoles)


            for ind in range(numMoles):
                clumpMask = np.where(np.logical_and(chan, clumpID == molList[ind]))
                xChan[ind] = x[clumpMask].mean()
                yChan[ind] = y[clumpMask].mean()
                xChanStd[ind] = x[clumpMask].std()
                yChanStd[ind] = y[clumpMask].std()

            xClump.append(xChan)
            yClump.append(yChan)
            xStd.append(xChanStd)
            yStd.append(yChanStd)

            if ii > 0:
                dx[ii - 1, :] = xClump[0] - xClump[ii]
                dy[ii - 1, :] = yClump[0] - yClump[ii]
                dxErr[ii - 1, :] = np.sqrt(xStd[ii]**2 + xStd[0]**2)
                dyErr[ii - 1, :] = np.sqrt(yStd[ii]**2 + yStd[0]**2)
                dxx, dyy, spx, spy, good = twoColour.genShiftVectorFieldQ(xClump[0], yClump[0], dx[ii-1, :], dy[ii-1, :], dxErr[ii-1, :], dyErr[ii-1, :])
                self.shiftWallet['FOV0%s.X' % ii], self.shiftWallet['FOV0%s.Y' % ii] = spx.__dict__, spy.__dict__
                dxWallet['FOV0%s' % ii], dyWallet['FOV0%s' % ii] = dxx, dyy


        self.shiftWallet['shiftModel'] = '.'.join([spx.__class__.__module__, spx.__class__.__name__])
        # store shiftmaps in metadata
        pipeline.mdh.__dict__.__setitem__('Shiftmap', self.shiftWallet)
        # store shiftvectors in metadata
        pipeline.mdh.__dict__.__setitem__('chroma.dx', dxWallet)
        pipeline.mdh.__dict__.__setitem__('chroma.dy', dyWallet)
        # save shiftmaps (spx and spy)
        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + 'MultiView.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            #save as a pickle containing the data and voxelsize

            fid = open(fpath, 'wb')
            #cPickle.dump(shiftWallet, fid, 2)
            json.dump(self.shiftWallet, fid)
            fid.close()

        # apply shiftmaps
        self.applyShiftmaps(xClump, yClump, numFOV)

        xfold, yfold = [], []
        for ii in range(numFOV):
            xfold.append(pipeline.mapping.__dict__['xFolded'][np.where(pipeline.mapping.__dict__['whichFOV'] == ii)])
            yfold.append(pipeline['y'][np.where(pipeline.mapping.__dict__['whichFOV'] == ii)])

        plotRegistered(xfold, yfold, numFOV, 'Raw')

        plotRegistered(xClump, yClump, numFOV, 'Clumped')

        plotRegistered(pipeline.mapping.__dict__['xReg'], pipeline.mapping.__dict__['yReg'],
                            numFOV, 'After Registration')




def Plug(visFr):
    """Plugs this module into the gui"""
    biplaneMapper(visFr)