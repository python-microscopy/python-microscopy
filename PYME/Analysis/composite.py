import os
import numpy as np

def common_prefix(strings):
    """ Find the longest string that is a prefix of all the strings.
    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix

def remap_data_2d(image, chan, shape, voxelsize, origin, shiftField=None, ignoreZ=True, order=3):
    """apply a vectorial correction for chromatic shift to an image - this
    is a generic vectorial shift compensation, rather than the secial case
    correction used with the splitter.
    """
    from scipy import ndimage
    
    #data = image.data[:, :, :, chan]
    
    X, Y, Z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    vx, vy, vz = image.voxelsize
    vxm, vym, vzm = voxelsize
    
    if vz == 0:
        vz = 1
    if vzm == 0:
        vzm = 1
    
    x0, y0, z0 = image.origin
    xm0, ym0, zm0 = origin
    
    if ignoreZ:
        z0 = 0
        zm0 = 0
    
    #desired coordinates in nm from camera origin
    Xnm = X * vxm + xm0
    Ynm = Y * vym + ym0
    Znm = Z * vzm + zm0
    
    if shiftField and os.path.exists(shiftField):
        spx, spy, dz = np.load(shiftField, allow_pickle=True)
        
        dx = spx.ev(Xnm, Ynm)
        dy = spy.ev(Xnm, Ynm)
        
        if ignoreZ:
            dz = 0
        
        Xnm += dx
        Ynm += dy
        Znm += dz
    
    print((vx, vy, vz, image.data_xyztc.shape))
    
    return [ndimage.map_coordinates(np.atleast_3d(image.data_xyztc[:,:,:,t,chan]), [(Xnm - x0) / vx, (Ynm - y0) / vy, (Znm - z0) / vz],
                                   mode='nearest', order=order) for t in range(image.data_xyztc.shape[3])]


def make_composite(images, ignoreZ=True, interp=True, shape=None, origin=None, voxelsize=None):
    """Make a composite image from multiple ImageStacks, interpolating if necessary

    Parameters
    ----------
    images : list-like
        ImageStack objects to combine, potentially as a list of tuples with the second element being the channel to use,
        and the third element being a chromatic shift field to apply
    ignoreZ : bool, optional
        ignore z origin when combining, by default True
    interp : bool, optional
        whether to interpolate with a cubic spline (True) or linear (False), by default True
    shape : list-like, optional
        xyztc shape of the composite, if different than the shape of the first image (default, None)
    origin : list-like, optional
        XYZ origin of the composite image, if different than the origin of the first image (default, None)
    voxelsize : MetaDataHandler.VoxelSize, optional
        voxelsize to construct the composite image, if different than that of the first image (default, None)

    Returns
    -------
    PYME.IO.image.ImageStack
        composite image
    """
    from PYME.IO import MetaDataHandler
    from PYME.IO.image import ImageStack
    
    assert(len(images) > 0)

    newNames = []
    newData = []
    
    if isinstance(images[0], tuple):
        im0 = images[0][0]
    else:
        im0 = images[0]
    
    if voxelsize is None:
        voxelsize = im0.voxelsize
    
    if origin is None:
        origin = im0.origin
        
    if shape is None:
        shape = im0.data_xyztc.shape

    if interp:
        order = 3
    else:
        order = 0
    
    for im in images:
        if isinstance(im, tuple):
            im, chan, shiftField = im
        else:
            chan = 0
            shiftField = None
            
        data = im.data_xyztc
        
        try:
            cn = im.mdh.getEntry('ChannelNames')[chan]
            otherName = '%s - %s' % (os.path.split(other.filename)[1], cn)
        except:
            if data.shape[4] == 1:
                otherName = os.path.split(im.filename)[1]
            else:
                otherName = '%s -  %d' % (os.path.split(im.filename)[1], chan)
        
        newNames.append(otherName)
        
        if ignoreZ:
            originsEqual = np.allclose(im.origin[:2], origin[:2], atol=1)
        else:
            originsEqual = np.allclose(im.origin, origin, atol=1)
        
        print(shape[:3], data.shape[:3])
        
        if (not np.allclose(im.pixelSize, voxelsize[0], rtol=.001)) or (not (tuple(data.shape[:3]) == tuple(shape[:3]))) or (not originsEqual) or shiftField:
            #need to rescale ...
            print(('Remapping ', im.filename, originsEqual, im.origin,
                   np.allclose(im.pixelSize, voxelsize[0], rtol=.001), (not (tuple(data.shape[:3]) == tuple(shape[:3]))),
                   shiftField, im.pixelSize, ignoreZ))
            #print origin, voxelsize
            od = remap_data_2d(im, chan, shape, voxelsize, origin, shiftField=shiftField, ignoreZ=ignoreZ, order=order)
        else:
            od = data[:,:,:,:,chan]
        
        #print('od.shape:', od.shape)
        
        newData += [od]
        
    pre = common_prefix(newNames)
    print(pre)
    lPre = len(pre)
    newNames = [n[lPre:] for n in newNames]
    
    mdh = MetaDataHandler.NestedClassMDHandler(im0.mdh)
    mdh.setEntry('ChannelNames', newNames)
    
    mdh['voxelsize.x'] = voxelsize[0] / 1e3
    mdh['voxelsize.y'] = voxelsize[1] / 1e3
    mdh['voxelsize.z'] = voxelsize[2] / 1e3
    
    mdh['Origin.x'] = origin[0]
    mdh['Origin.y'] = origin[1]
    mdh['Origin.z'] = origin[2]
    
    return ImageStack(data=newData, mdh=mdh)
        
        
    
    