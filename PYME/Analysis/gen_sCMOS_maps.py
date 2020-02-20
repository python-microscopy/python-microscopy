import sys
import os
import argparse
import numpy as np
from PYME.IO.image import ImageStack
from PYME.IO.MetaDataHandler import NestedClassMDHandler, get_camera_roi_origin
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

def map_filename(mdh, type):
    if type != 'flatfield':
        itime = int(1000*mdh['Camera.IntegrationTime'])
        return '%s_%dms.tif' % (type,itime)
    else:
        return '%s.tif' % (type)


def makePathUnlessExists(path):
    import errno

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def mkDestPath(destdir,stem,mdh,create=True):
    if create and not os.path.isdir(destdir):
        raise ValueError('directory %s does not exist; please create' % destdir)
    
    return os.path.join(destdir, map_filename(mdh, stem))


def mkDefaultPath(stem,mdh,create=True,calibrationDir=None):
    if calibrationDir is None:
        camDir = nameUtils.getCalibrationDir(mdh['Camera.SerialNumber'],create=create)
    else:
        camDir = os.path.join(calibrationDir,mdh['Camera.SerialNumber'])
    if create:
        makePathUnlessExists(camDir)
    return mkDestPath(camDir,stem,mdh,create=create)


def listCalibrationDirs():
    from glob import glob
    rootdir = nameUtils.getCalibrationDir('',create=False)
    result = [y for x in os.walk(rootdir) for y in glob(os.path.join(x[0], '*.tif'))]
    if result is not None:
        print('List of installed maps:')
        for m in result:
            print(m)

def insert_into_full_map(dark, variance, metadata, sensor_size=(2048, 2048)):
    """

    Embeds partial-sensor camera maps into full-sized camera map by padding with basic values in metadata. Alternatively
    can be used to create boring maps to use in place of metadata scalars.

    Parameters
    ----------
    dark: ndarray or None
        darkmap for valid ROI, or None to generate a uniform, ~useless metadata map
    variance: ndarray
        variance for valid ROI, or None to generate a uniform, ~useless metadata map
    metadata: MetaDataHandler instance
        ROI informatrion and camera noise parameters to use when padding maps
    sensor_size: 2-int tuple
        x and y camera sensor size

    Returns
    -------
    full_dark: ndarray
        padded dark map
    full_var: ndarray
        padded variance map
    mdh: PYME.IO.MetadataHandler.NestedClassMDHandler
        metadata handler to be associated with full maps while maintaining information about the original/valid ROI.
    """

    mdh = NestedClassMDHandler(metadata)
    x_origin, y_origin = get_camera_roi_origin(mdh)
    mdh.setEntry('CameraMap.ValidROI.ROIOriginX', x_origin)
    mdh.setEntry('CameraMap.ValidROI.ROIOriginY', y_origin)
    mdh.setEntry('CameraMap.ValidROI.ROIWidth', mdh['Camera.ROIWidth'])
    mdh.setEntry('CameraMap.ValidROI.ROIHeight', mdh['Camera.ROIHeight'])
    mdh['Camera.ROIOriginX'], mdh['Camera.ROIOriginY'] = 0, 0
    mdh['Camera.ROIWidth'], mdh['Camera.ROIHeight'] = sensor_size
    mdh['Camera.ROI'] = (0, 0, sensor_size[0], sensor_size[1])


    if dark is not None and variance is not None:
        full_dark = mdh['Camera.ADOffset'] * np.ones(sensor_size, dtype=dark.dtype)
        full_var = (mdh['Camera.ReadNoise'] ** 2) * np.ones(sensor_size, dtype=variance.dtype)

        xslice = slice(x_origin, x_origin + metadata['Camera.ROIWidth'])
        yslice = slice(y_origin, y_origin + metadata['Camera.ROIHeight'])

        full_dark[xslice, yslice] = dark
        full_var[xslice, yslice] = variance
    else:
        logger.warning('Generating uniform maps')
        full_dark = mdh['Camera.ADOffset'] * np.ones(sensor_size)
        full_var = (mdh['Camera.ReadNoise'] ** 2) * np.ones(sensor_size)
        
        mdh['CamerMap.Uniform'] = True

    return full_dark, full_var, mdh

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
    
def generate_maps(filename, start_frame, end_frame):

def main():
    logging.basicConfig() # without it got 'No handlers could be found for logger...'

    defaultSensorSize = (2048,2048) # we currently assume this is correct but could be chosen based
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
    op.add_argument('-p','--prefix', metavar='prefix', default='',
                    help='prefix for dark/variance map filenames')

    args = op.parse_args()

    if args.list:
        listCalibrationDirs()
        sys.exit(0)

    # body of script
    filename = args.filename
    prefix = args.prefix

    if filename is None:
        op.error('need a file name if -l or --list not requested')

    if args.install:
        #copy the map to the default maps directory
        install_map(filename)
        sys.exit(0)

    start = args.start
    end = args.end

    logger.info('Opening image series...')
    source = ImageStack(filename=filename)
    
    if end < 0:
        end = int(source.dataSource.getNumSlices() + end)

    # pre-checks before calculations to minimise the pain
    sensorSize = list(defaultSensorSize)
    try:
        sensorSize[0] = source.mdh['Camera.SensorWidth']
    except (KeyError, AttributeError):
        logger.warning('no valid sensor width in metadata - using default %d' % sensorSize[0])
    try:
        sensorSize[1] = source.mdh['Camera.SensorHeight']
    except (KeyError, AttributeError):
        logger.warning('no valid sensor height in metadata - using default %d' % sensorSize[1])
    
    if not ((source.mdh['Camera.ROIWidth'] == sensorSize[0]) and (source.mdh['Camera.ROIHeight'] == sensorSize[1])):
        logger.warning('Generating a map from data with ROI set. Use with EXTREME caution.\nMaps should be calculated from the whole chip.')


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
        logger.warning('Simulating uniform maps - use with care')
        nbad = 0

        if args.dir is None:
            logger.error('Uniform maps cannot be stored to the default map directory\nPlease specify an output directory.')
            sys.exit(-1)

    # if the uniform flag is set, then m and ve are passed as None
    # which makes sure that just the uniform defaults from meta data are used
    mfull, vefull, mapmdh = insert_into_full_map(m, ve, source.mdh, sensor_size=sensorSize)

    mapmdh['CameraMap.StartFrame'] = start
    mapmdh['CameraMap.EndFrame'] = end
    mapmdh['CameraMap.SourceFilename'] = filename
    mapmdh['CameraMap.DarkThreshold'] = darkthreshold
    mapmdh['CameraMap.VarianceThreshold'] = variancethreshold
    mapmdh['CameraMap.BlemishVariance'] = blemishvariance
    mapmdh['CameraMap.NBadPixels'] = nbad

    if args.uniform:
        mapmdh['CameraMap.Uniform'] = True
    
    mmd = NestedClassMDHandler(mapmdh)
    mmd['CameraMap.Type'] = 'mean'
    mmd['CameraMap.Units'] = 'ADU'

    vmd = NestedClassMDHandler(mapmdh)
    vmd['CameraMap.Type'] = 'variance'
    vmd['CameraMap.Units'] = 'electrons^2'

    im_dark = ImageStack(mfull, mdh=mmd)
    im_variance = ImageStack(vefull, mdh=vmd)

    
    logger.info('Saving results...')

    if args.dir is None:
        logger.info('installing in standard location...')
        
        if not ((source.mdh['Camera.ROIWidth'] == sensorSize[0]) and (source.mdh['Camera.ROIHeight'] == sensorSize[1])):
            logger.error('Maps with an ROI set cannot be stored to the default map directory\nPlease specify an output directory.')
            sys.exit(-1)
            
        if im_dark.mdh.get('CameraMap.Uniform', False) or im_variance.mdh.get('CameraMap.Uniform', False):
            logger.error('Uniform maps cannot be stored to the default map directory\nPlease specify an output directory.')
            sys.exit(-1)
            
        mname = mkDefaultPath('dark', source.mdh)
        vname = mkDefaultPath('variance', source.mdh)
    else:
        mname = mkDestPath(args.dir, prefix + 'dark', source.mdh)
        vname = mkDestPath(args.dir, prefix + 'variance', source.mdh)

    logger.info('dark map -> %s...' % mname)
    logger.info('var  map -> %s...' % vname)
    
    im_dark.Save(filename=mname)
    im_variance.Save(filename=vname)



if __name__ == "__main__":
    main()
