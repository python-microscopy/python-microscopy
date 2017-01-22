import sys
import os
import argparse
import numpy as np
from PYME.IO.image import ImageStack
from PYME.IO.MetaDataHandler import NestedClassMDHandler
from PYME.IO.FileUtils import nameUtils

import logging
logger = logging.getLogger(__name__)

def _meanvards(dataSource, start=0, end=-1):
    """
    Calculate the mean and variance of a data source

    Parameters
    ----------
    dataSource
    start
    end

    Returns
    -------

    """
    nslices = dataSource.getNumSlices()
    if end < 0:
        end = nslices + end

    nframes = end - start
    xSize, ySize = dataSource.getSliceShape()

    m = np.zeros((xSize,ySize),dtype='float64')
    for frameN in range(start,end):
        m += dataSource.getSlice(frameN)
    m = m / nframes

    v = np.zeros((xSize,ySize),dtype='float64')
    for frameN in range(start,end):
        v += (dataSource.getSlice(frameN)-m)**2
    v = v / (nframes-1)

    return (m,v)


def makePathUnlessExists(path):
    import errno

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def mkDestPath(destdir,stem,mdh):
    if not os.path.isdir(destdir):
        raise ValueError('directory %s does not exist; please create' % destdir)
    itime = int(1000*mdh['Camera.IntegrationTime'])
    return os.path.join(destdir,'%s_%dms.tif' % (stem,itime))


def mkDefaultPath(stem,mdh):
    caldir = nameUtils.getCalibrationDir(mdh['Camera.SerialNumber'])
    makePathUnlessExists(caldir)
    return mkDestPath(caldir,stem,mdh)


def listCalibrationDirs():
    from glob import glob
    rootdir = nameUtils.getCalibrationDir('')
    result = [y for x in os.walk(rootdir) for y in glob(os.path.join(x[0], '*.tif'))]
    if result is not None:
        print('List of installed maps:')
        for m in result:
            print(m)


def insertIntoFullMap(m, ve, smdh, chipsize=(2048,2048)):
    """
    this function embeds the calculated maps into a full chipsize array
    that is padded with the camera default readnoise and offset as extracted from the
    metadata
    if m, ve are None nothing is copied into the map and just the uniform array is returned

    Parameters
    ----------
    m
    ve
    smdh
    chipsize

    Returns
    -------

    """
    validROI = {
        'PosX' : smdh['Camera.ROIPosX'],
        'PosY' : smdh['Camera.ROIPosY'],
        'Width' : smdh['Camera.ROIWidth'],
        'Height' : smdh['Camera.ROIHeight']
        }
    
    bmdh = NestedClassMDHandler()
    bmdh.copyEntriesFrom(smdh)
    bmdh.setEntry('Analysis.name', 'mean-variance')
    bmdh.setEntry('Analysis.valid.ROIPosX', validROI['PosX'])
    bmdh.setEntry('Analysis.valid.ROIPosY', validROI['PosY'])
    bmdh.setEntry('Analysis.valid.ROIWidth', validROI['Width'])
    bmdh.setEntry('Analysis.valid.ROIHeight', validROI['Height'])

    bmdh['Camera.ROIPosX'] = 1
    bmdh['Camera.ROIPosY'] = 1
    bmdh['Camera.ROIWidth'] = chipsize[0]
    bmdh['Camera.ROIHeight'] = chipsize[1]
    bmdh['Camera.ROI'] = (1,1,chipsize[0]+1,chipsize[1]+1)

    if m is None:
        mfull = np.zeros(chipsize, dtype='float64')
        vefull = np.zeros(chipsize, dtype='float64')
    else:
        mfull = np.zeros(chipsize, dtype=m.dtype)
        vefull = np.zeros(chipsize, dtype=ve.dtype)
    mfull.fill(smdh['Camera.ADOffset'])
    vefull.fill(smdh['Camera.ReadNoise']**2)
    
    if m is not None:
        mfull[validROI['PosX']-1:validROI['PosX']-1+validROI['Width'],
              validROI['PosY']-1:validROI['PosY']-1+validROI['Height']] = m
        vefull[validROI['PosX']-1:validROI['PosX']-1+validROI['Width'],
               validROI['PosY']-1:validROI['PosY']-1+validROI['Height']] = ve      

    return mfull, vefull, bmdh

def install_map(filename):
    """Installs a map file to its default location"""

    source = ImageStack(filename=filename)
    if source.mdh.getOrDefault('Analysis.name', '') != 'mean-variance':
        logger.error('Analysis.name is not equal to "mean-variance" - probably not a map')
        sys.exit('aborting...')

    if not (source.mdh['Analysis.valid.ROIHeight'] == source.mdh['Camera.ROIHeight']
            and source.mdh['Analysis.valid.ROIHeight'] == source.mdh['Camera.ROIHeight']):
        logger.error('Partial (ROI based) maps cannot be installed to the default location')
        sys.exit(-1)

    if source.mdh.getOrDefault('Analysis.isuniform', False):
        logger.error('Uniform maps cannot be installed to the default location')
        sys.exit(-1)

    if source.mdh['Analysis.resultname'] == 'mean':
        maptype = 'dark'
    else:
        maptype = 'variance'

    mapname = mkDefaultPath(maptype, source.mdh)

    source.Save(filename=mapname)

def main():

    default_chipsize = (2048,2048) # we currently assume this is correct but could be chosen based
                            # on camera model in meta data TODO - add CCD size to camera metadata
    darkthreshold = 1e4    # this really should depend on the gain mode (12bit vs 16 bit etc)
    variancethreshold = 300**2  # again this is currently picked fairly arbitrarily
    blemishvariance = 1e8

    # options parsing
    op = argparse.ArgumentParser(description='generate offset and variance maps from darkseries.')
    op.add_argument('filename', metavar='filename', nargs='?', default=None,
                    help='filename of the darkframe series')
    op.add_argument('-s', '--start', type=int, default=0, 
                    help='start frame to use')
    op.add_argument('-e', '--end', type=int, default=-1, 
                    help='end frame to use')
    op.add_argument('-u','--uniform', action='store_true',
                help='make uniform map using metadata info')
    op.add_argument('-i','--install', action='store_true',
                help='install map in default location - the filename argument is a map')
    op.add_argument('-d', '--dir', metavar='destdir', default=None,
                    help='destination directory (default is PYME calibration path)')
    op.add_argument('-l','--list', action='store_true',
                help='list all maps in default location')
    args = op.parse_args()

    if args.list:
        listCalibrationDirs()
        sys.exit(0)

    # body of script
    filename = args.filename

    if filename is None:
        op.error('need a file name if -l or --list not requested')

    if args.install:
        #copy the map to the default maps directory
        install_map(filename)
        sys.exit(0)



    logger.info('Opening image series...')
    source = ImageStack(filename=filename)

    start = args.start
    end = args.end
    if end < 0:
        end = int(source.dataSource.getNumSlices() + end)

    # pre-checks before calculations to minimise the pain
    chipsize = list(default_chipsize)
    try:
        chipsize[0] = source.mdh['Camera.SensorSizeX']
    except AttributeError:
        logger.warn('no valid x sensor size in metadata - using default %d' % chipsize[0])
    try:
        chipsize[1] = source.mdh['Camera.SensorSizeY']
    except AttributeError:
        logger.warn('no valid y sensor size in metadata - using default %d' % chipsize[1])
    
    if not ((source.mdh['Camera.ROIWidth'] == chipsize[0]) and (source.mdh['Camera.ROIHeight'] == chipsize[1])):
        logger.warn('Generating a map from data with ROI set. Use with EXTREME caution.\nMaps should be calculated from the whole chip.')

        if args.dir is None:
            logger.error('Maps with an ROI set cannot be stored to the default map directory\nPlease specify an output directory.')
            sys.exit(-1)

    logger.info('Calculating mean and variance...')

    m, ve = (None,None)
    if not args.uniform:
        m, v = _meanvards(source.dataSource, start = start, end=end)
        eperADU = source.mdh['Camera.ElectronsPerCount']
        ve = v*eperADU*eperADU
        # occasionally the cameras seem to have completely unusable pixels
        # one example was dark being 65535 (i.e. max value for 16 bit)
        if m.max() > darkthreshold:
            ve[m > darkthreshold] = blemishvariance
        if ve.max() > variancethreshold:
            ve[ve > variancethreshold] = blemishvariance
        nbad = np.sum((m > darkthreshold)*(ve > variancethreshold))
    else:
        logger.warn('Simulating uniform maps - use with care')
        nbad = 0

        if args.dir is None:
            logger.error('Uniform maps cannot be stored to the default map directory\nPlease specify an output directory.')
            sys.exit(-1)

    # if the uniform flag is set, then m and ve are passed as None
    # which makes sure that just the uniform defaults from meta data are used
    mfull, vefull, basemdh = insertIntoFullMap(m, ve, source.mdh, chipsize=chipsize)

    logger.info('Saving results...')

    if args.dir is None:
        logger.info('installing in standard location...')
        mname = mkDefaultPath('dark',source.mdh)
        vname = mkDefaultPath('variance',source.mdh)
    else:
        mname = mkDestPath(args.dir,'dark',source.mdh)
        vname = mkDestPath(args.dir,'variance',source.mdh)

    logger.info('dark map -> %s...' % mname)
    logger.info('var  map -> %s...' % vname)

    commonMD = NestedClassMDHandler()
    commonMD.setEntry('Analysis.name', 'mean-variance')
    commonMD.setEntry('Analysis.start', start)
    commonMD.setEntry('Analysis.end', end)
    commonMD.setEntry('Analysis.SourceFilename', filename)
    commonMD.setEntry('Analysis.darkThreshold', darkthreshold)
    commonMD.setEntry('Analysis.varianceThreshold', variancethreshold)
    commonMD.setEntry('Analysis.blemishVariance', blemishvariance)
    commonMD.setEntry('Analysis.NBadPixels', nbad)

    if args.uniform:
        commonMD.setEntry('Analysis.isuniform', True)
    
    mmd = NestedClassMDHandler(basemdh)
    mmd.copyEntriesFrom(commonMD)
    mmd.setEntry('Analysis.resultname', 'mean')
    mmd.setEntry('Analysis.units', 'ADU')

    vmd = NestedClassMDHandler(basemdh)
    vmd.copyEntriesFrom(commonMD)
    vmd.setEntry('Analysis.resultname', 'variance')
    vmd.setEntry('Analysis.units', 'electrons^2')

    ImageStack(mfull, mdh=mmd).Save(filename=mname)
    ImageStack(vefull, mdh=vmd).Save(filename=vname)



if __name__ == "__main__":
    main()
