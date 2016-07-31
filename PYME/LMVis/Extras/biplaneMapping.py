
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

def pairMolecules(xFold, y, FOV):
    numMol = len(xFold)
    dist = np.zeros((numMol, numMol))
    for ind in range(numMol):
        for ii in range(numMol):
            dist[ind, ii] = np.sqrt((xFold[ind] - xFold[ii])**2 + (y[ind] - y[ii])**2)
    dist = dist + (dist == 0)*999999
    minDist = dist.min(axis=0)
    minLoc = dist.argmin(axis=0)

    # only keep molecule pairs heterozygous in planes and who are mutually nearest neighbors
    keepList = np.where(np.logical_and(FOV[minLoc] != FOV, minLoc[range(numMol)] == minLoc[minLoc[range(numMol)]]))
    '''
    keepList = []  # np.zeros(numMol, dtype=bool)
    for ind in range(numMol):
        # ignore molecules whose nearest neighbors are in the same FOV
        # keepList[ind] = (FOV[minLoc[ind]] == FOV[ind])
        if (FOV[minLoc[ind]] == FOV[ind]):
            keepList.append(ind)
    '''
    numKept = len(keepList)

    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.scatter(xFold, y)

    for ii in range(numKept):
        plt.plot([xFold[keepList[ii]], xFold[minLoc[keepList[ii]]]], [y[keepList[ii]], y[minLoc[keepList[ii]]]], color='red')
    #plt.show()

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
            #self.viewOrigins = [croppingInfo['Multiview.ROI%dOrigin' % i] for i in range(self.numROIs)]

        foldX(pipeline)

        print('length is %i, max is %d' % (len(pipeline.mapping.__dict__['xFolded']), np.max(pipeline.mapping.__dict__['xFolded'])))
        print('Number of ROIs: %f' % np.max(pipeline.mapping.__dict__['whichFOV']))
        plotRegistered(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], 'Raw')

        self.applyShiftmaps(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], numFOV)

    def OnRegisterBiplane(self, event):
        pipeline = self.visFr.pipeline

        try:
            numFOV = pipeline.mdh['Multiview.NumROIs']
        except KeyError:
            raise UserWarning('You are either not looking at Biplane Data, or your metadata is incomplete')
            return

        foldX(pipeline)

        print('length is %i, max is %d' % (len(pipeline.mapping.__dict__['xFolded']), np.max(pipeline.mapping.__dict__['xFolded'])))
        print('Number of ROIs: %f' % np.max(pipeline.mapping.__dict__['whichFOV']))
        plotRegistered(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichFOV'], 'Raw')

        # Now we need to match up molecules
        pairMolecules(pipeline.mapping.__dict__['xFolded'], pipeline['y'], pipeline.mapping.__dict__['whichFOV'])





def Plug(visFr):
    """Plugs this module into the gui"""
    biplaneMapper(visFr)
