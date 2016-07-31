
import wx
import numpy as np
from PYME.Analysis.points import twoColour

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

def plotRegistered(regX, regY, multiviewChannels, title=''):
    import matplotlib.pyplot as plt
    nChan = multiviewChannels.max()

    c = iter(plt.cm.rainbow(np.linspace(0, 1, nChan)))
    for ii in range(nChan):
        mask = (ii == multiviewChannels)
        plt.scatter(regX[mask], regY[mask], c=next(c))
    plt.title(title)
    return

def pairMolecules(xFold, y, whichFOV, FOV1=0, FOV2=1, combineDist):
    """

    Args:
        xFold:
        y:
        whichFOV:
        FOV1:
        FOV2:
        combineDist:

    Returns:

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

    numKept = len(keepList)
    chan1Keep = np.logical_and(keep, whichFOV == FOV1)
    pairs = minLocKept[chan1Keep]
    # chan2Keep = minLocKept[~chan1Keep] #  np.logical_and(keep, whichFOV == FOV2)
    x1 = xFold[chan1Keep]
    y1 = y[chan1Keep]
    x2 = xFold[pairs]
    y2 = y[pairs]


    return x1, y1, x2, y2

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

    def applyShiftmaps(self, xFolded, y, whichFOV, numFOV):
        pipeline = self.visFr.pipeline

        xReg = xFolded + [(whichFOV == ii)*self.shiftMapsX[ii](xFolded) for ii in range(numFOV)]
        yReg = y + [(whichFOV == ii)*self.shiftMapsY[ii](y) for ii in range(numFOV)]

        pipeline.mapping.setMapping('xReg', xReg)
        pipeline.mapping.setMapping('yReg', yReg)
        return

    def OnFoldAndMap(self, event):
        pipeline = self.visFr.pipeline

        try:  # load shiftmaps, if present
            numFOV = pipeline.mdh['Multiview.NumROIs']
            self.shiftMapsX = [pipeline.mdh['shiftMapsX.FOV%d' % ii] for ii in range(numFOV)]
            self.shiftMapsY = [pipeline.mdh['shiftMapsY.FOV%d' % ii] for ii in range(numFOV)]
        except KeyError:
            raise UserWarning('Shiftmaps or number of FOVs not found in metadata')
            return

        foldX(pipeline)

        print('length is %i, max is %d' % (len(pipeline.mapping.__dict__['xFolded']), np.max(pipeline.mapping.__dict__['xFolded'])))
        print('Number of ROIs: %f' % np.max(pipeline.mapping.__dict__['whichFOV']))
        plotRegistered(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], 'Raw')

        self.applyShiftmaps(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], numFOV)

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
        x1, y1, x2, y2 = pairMolecules(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                                      pipeline.mapping.__dict__['whichFOV'], 20)

        # Generate raw shift vectors (map of displacements between channels) for each FOV
        xRawShifts = [twoColour.genShiftVectors(xFold[FOV == 0], xFold[FOV == ii]) for ii in range(1, numFOV)]
        yRawShifts = [twoColour.genShiftVectors(y[FOV == 0], y[FOV == ii]) for ii in range(1, numFOV)]

        # Generate shiftmaps

        # apply shiftmaps

        # plot unshifted and shifted
        plotRegistered(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], 'Raw Folding')

        plotRegistered(pipeline.mapping.__dict__['xReg'], pipeline['yReg'],
                            pipeline.mapping.__dict__['whichFOV'], 'After Registration')



def Plug(visFr):
    """Plugs this module into the gui"""
    biplaneMapper(visFr)
