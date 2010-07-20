import numpy as np
from PYME.Acquire.Hardware import splitter


def tile(ds, xm, ym, mdh, split=True, skipMoveFrames=True, shiftfield=None, mixmatrix=[[1.,0.],[0.,1.]]):
    frameSizeX, frameSizeY, numFrames = ds.shape[:3]

    if split:
        frameSizeY /=2
        nchans = 2
        unmux = splitter.Unmixer(shiftfield, 1e3*mdh.getEntry('voxelsize.x'))
    else:
        nchans = 1

    #x & y positions of each frame
    xps = xm(np.arange(numFrames))
    yps = ym(np.arange(numFrames))

    #print xps
    
    #convert to pixels
    xdp = ((xps - xps.min()) / (1e-3*mdh.getEntry('voxelsize.x'))).round()
    ydp = ((yps - yps.min()) / (1e-3*mdh.getEntry('voxelsize.x'))).round()

    #print xdp

    #work out how big our tiled image is going to be
    imageSizeX = np.ceil(xdp.max() + frameSizeX)
    imageSizeY = np.ceil(ydp.max() + frameSizeY)

    #allocate an empty array for the image
    im = np.zeros([imageSizeX, imageSizeY, nchans])

    # and to record occupancy (to normalise overlapping tiles)
    occupancy = np.zeros([imageSizeX, imageSizeY, nchans])

    #calculate a weighting matrix (to allow feathering at the edges - TODO)
    weights = np.ones((frameSizeX, frameSizeY, nchans))
    weights[:, :10, :] = 0 #avoid splitter edge artefacts
    weights[:, -10:, :] = 0

    ROIX1 = mdh.getEntry('Camera.ROIPosX')
    ROIY1 = mdh.getEntry('Camera.ROIPosY')

    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

    offset = mdh.getEntry('Camera.ADOffset')

#    #get a sorted list of x and y values
#    xvs = list(set(xdp))
#    xvs.sort()
#
#    yvs = list(set(ydp))
#    yvs.sort()

    for i in range(mdh.getEntry('Protocol.DataStartsAt'), numFrames):
        if xdp[i - 1] == xdp[i] or not skipMoveFrames:
            d = ds[:,:,i]
            if split:
                d = np.concatenate(unmux.Unmix(d, mixmatrix, offset, [ROIX1, ROIY1, ROIX2, ROIY2]), 2)
            im[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :] += weights*d
            occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :] += weights

    ret =  (im/occupancy).squeeze()
    ret[occupancy == 0] = 0 #fix up /0s

    return ret