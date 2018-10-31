import numpy as np
from PYME.IO.MetaDataHandler import get_camera_roi_origin

def get_splitter_rois(md, data_shape):
    x0, y0 = get_camera_roi_origin(md)
    
    if 'Splitter.Channel0ROI' in md.getEntryNames():
        xg, yg, wg, hg = md['Splitter.Channel0ROI']
        xr, yr, wr, hr = md['Splitter.Channel1ROI']
        #print 'Have splitter ROIs'
    else:
        xg = 0
        yg = 0
        wg = data_shape[0]
        hg = data_shape[1] / 2
        
        xr = 0
        yr = hg
        wr = data_shape[0]
        hr = data_shape[1] / 2
    
    def _bdsClip(x, w, x0, iw):
        x -= x0
        if (x < 0):
            w += x
            x = 0
        if ((x + w) > iw):
            w -= (x + w) - iw
        
        return x, w
    
    #print yr, hr
    
    xg, wg = _bdsClip(xg, wg, x0, data_shape[0])
    xr, wr = _bdsClip(xr, wr, 0, data_shape[0])
    yg, hg = _bdsClip(yg, hg, y0, data_shape[1])
    yr, hr = _bdsClip(yr, hr, 0, data_shape[1])
    
    w = min(wg, wr)
    h = min(hg, hr)
    
    #print yr, hr
    
    if ('Splitter.Flip' in md.getEntryNames() and not md.getEntry('Splitter.Flip')):
        step = 1
        return (slice(xg, xg + w, 1), slice(xr, xr + w, 1), slice(yg, yg + h, 1), slice(yr, yr + h, step))
    else:
        step = -1
        return (slice(xg, xg + w, 1), slice(xr, xr + w, 1), slice(yg, yg + h, 1), slice(yr + h, yr - 1, step))

def map_splitter_coords(md, data_shape, x, y):
    vx = md['voxelsize.x'] * 1e3
    vy = md['voxelsize.y'] * 1e3

    x0, y0 = get_camera_roi_origin(md)

    if 'Splitter.Channel0ROI' in md.getEntryNames():
        xg, yg, w, h = md['Splitter.Channel0ROI']
        xr, yr, w, h = md['Splitter.Channel1ROI']
    
        w2 = w - x0
        h2 = h - y0
    else:
        xg, yg, w, h = 0, 0, data_shape[0], data_shape[1]
        xr, yr = w, h

    ch1 = (x >= (xr - x0)) & (y >= (yr - y0))

    xn = x - ch1 * (xr - xg)
    yn = y - ch1 * (yr - yg)

    if not (('Splitter.Flip' in md.getEntryNames() and not md.getEntry('Splitter.Flip'))):
        yn += ch1 * (h - y0 - 2 * yn)

    #chromatic shift
    if 'chroma.dx' in md.getEntryNames():
        dx = md['chroma.dx'].ev((xn + x0) * vx, (yn + y0) * vy) / vx
        dy = md['chroma.dy'].ev((xn + x0) * vy, (yn + y0) * vy) / vy
    
        xn += dx * ch1
        yn += dy * ch1

    return np.clip(xn, 0, w2 - 1), np.clip(yn, 0, h2 - 1)

def remap_splitter_coords(md, data_shape, x, y):
    vx = md['voxelsize.x'] * 1e3
    vy = md['voxelsize.y'] * 1e3

    x0, y0 = get_camera_roi_origin(md)

    if 'Splitter.Channel0ROI' in md.getEntryNames():
        xg, yg, w, h = md['Splitter.Channel0ROI']
        xr, yr, w, h = md['Splitter.Channel1ROI']
    else:
        xg, yg, w, h = 0, 0, data_shape[0], data_shape[1]
        xr, yr = w, h

    xn = x + (xr - xg)
    yn = y + (yr - yg)

    if not (('Splitter.Flip' in md.getEntryNames() and not md.getEntry('Splitter.Flip'))):
        yn = (h - y0 - y) + yr - yg

    #chromatic shift
    if 'chroma.dx' in md.getEntryNames():
        dx = md['chroma.dx'].ev((x + x0) * vx, (y + y0) * vy) / vx
        dy = md['chroma.dy'].ev((x + x0) * vx, (y + y0) * vy) / vy
    
        xn -= dx
        yn -= dy

    return xn, yn