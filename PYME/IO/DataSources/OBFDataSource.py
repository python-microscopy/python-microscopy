
from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from PYME.IO.DataSources.BaseDataSource import XYTCDataSource
from PYME.IO.FileUtils import obf_support
import logging

logger = logging.getLogger(__name__)

class DataSource(XYTCDataSource):
    moduleName = 'OBFDataSource'
    def __init__(self, filename, stack_number=None):
        self.filename = getFullExistingFilename(filename)#convert relative path to full path
        
        self.obf = obf_support.File(self.filename)

        logger.debug('file: {}'.format(self.filename))
        # logger.debug('format version: {}'.format(self.obf.format_version))
        logger.debug('description: "{}"'.format(self.obf.description))
        logger.debug('contains {} stacks'.format(len(self.obf.stacks)))

        if stack_number is None:
            stack_number = 0
            # TODO - would be nice to have an 'all' flag somewhere in image.py to open all found
            if len(self.obf.stacks) > 1:
                logger.error('More than one stack present, add ?stack=X to filename to select one:')
                for index, stack in enumerate(self.obf.stacks):
                    logger.debug('\nstack {}'.format(index))
                    logger.debug(' format version: {}'.format(stack.format_version))
                    logger.debug(' name: "{}"'.format(stack.name))
                    logger.debug(' description: "{}"'.format(stack.description))
                raise RuntimeError(logger.error('More than one stack present, add ?stack=X to filename to select one'))
        
        self.stack_number = stack_number
        stack = self.obf.stacks[self.stack_number]

        # for index, stack in enumerate(self.obf.stacks[-1:]):
        # logger.debug('\nstack {}'.format(index))
        logger.debug(' format version: {}'.format(stack.format_version))
        logger.debug(' name: "{}"'.format(stack.name))
        logger.debug(' description: "{}"'.format(stack.description))
        logger.debug(' shape: {}'.format(stack.shape))
        logger.debug(' dimensionality: {}'.format(stack.dimensionality))
        logger.debug(' lengths: {}'.format(stack.lengths))
        logger.debug(' pixel sizes: {}'.format(stack.pixel_sizes))
        logger.debug(' offsets: {}'.format(stack.offsets))
        logger.debug(' data type: {}'.format(stack.data_type.__name__))

        self.stack = stack
        self.slice2d = tuple([slice(None), slice(None)] + [0] * (len(self.stack.data.shape) - 2))
        # # load stack data and show first 2D image
        # data = stack.data
        # if data.size > 0:  # don't display empty stacks
        #     fig, ax = plt.subplots()
        #     idx = [slice(None), slice(None)] + [0] * (len(data.shape) - 2)
        #     im = ax.imshow(data[tuple(idx)].reshape(data.shape[:2]), cmap=cm.hot)
        #     ax.set_title(stack.name)

        # plt.show()
        
    def getSlice(self, ind):
        # handle different dimensionality
        
        #     im = ax.imshow(data[tuple(idx)].reshape(data.shape[:2])
        return self.stack.data[self.slice2d].reshape(self.getSliceShape())

    def getSliceShape(self):
        return self.stack.data.shape[:2]

    def getNumSlices(self):
        return self.stack.data.shape[2]

    def getEvents(self):
        return []

    def release(self):
        self.obf.close()

    def reloadData(self):
        self.close()
        self.obf = obf_support.File(self.filename)
        self.stack = self.obf.stacks[self.stack_number]
