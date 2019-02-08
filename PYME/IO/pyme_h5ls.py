import logging
logger = logging.getLogger(__name__)
import sys

def list_h5(filename):
    import tables
    from PYME.IO import MetaDataHandler
    from PYME.IO import tabular
    from PYME.IO import unifiedIO
    import json
    
    with unifiedIO.local_or_temp_filename(filename) as fn:
        with tables.open_file(fn, mode='r') as h5f:
            #make sure our hdf file gets closed
            
            try:
                mdh = MetaDataHandler.NestedClassMDHandler(MetaDataHandler.HDFMDHandler(h5f))
                print('Metadata:\n____________')
                print(repr(mdh))
            except tables.FileModeError:  # Occurs if no metadata is found, since we opened the table in read-mode
                logger.warning('No metadata found, proceeding with empty metadata')
                mdh = MetaDataHandler.NestedClassMDHandler()
                
            print('\n\n')
            
            for t in h5f.list_nodes('/'):
                # FIXME - The following isinstance tests are not very safe (and badly broken in some cases e.g.
                # PZF formatted image data, Image data which is not in an EArray, etc ...)
                # Note that EArray is only used for streaming data!
                # They should ideally be replaced with more comprehensive tests (potentially based on array or dataset
                # dimensionality and/or data type) - i.e. duck typing. Our strategy for images in HDF should probably
                # also be improved / clarified - can we use hdf attributes to hint at the data intent? How do we support
                # > 3D data?
                
                if not isinstance(t, tables.Group):
                    print(t.name)
                    print('______________')
                    
                    if isinstance(t, tables.VLArray):
                        data = h5f.get_node(h5f.root, t.name)
                        print('Ragged (VLArray) with %d rows' % len(data))
                        print('Row 0: %s' % data)
                    
                    elif isinstance(t, tables.table.Table):
                        #  pipe our table into h5r or hdf source depending on the extension
                        data = h5f.get_node(h5f.root, t.name)
                        
                        print('Table with %d rows\n dtype = %s' % (len(data), data[0].dtype))
                    
                    elif isinstance(t, tables.EArray):
                        data = h5f.get_node(h5f.root, t.name)
                        
                        print('Image, shape = %s' % data.shape)
                        
                    print('\n\n')
                
if __name__ == '__main__':
    list_h5(sys.argv[1])