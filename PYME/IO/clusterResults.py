from . import clusterIO
from . import image
#from . import PZFFormat
from . import MetaDataHandler
import requests
import numpy as np
#import cStringIO
from io import BytesIO

try:
    # noinspection PyCompatibility
    import cPickle
except ImportError:
    #py3
    import pickle as cPickle
    
import os
import logging
import socket


def _gzip_compress(data):
    import gzip
    from io import BytesIO
    zbuf = BytesIO()
    zfile = gzip.GzipFile(mode='wb', fileobj=zbuf)#, compresslevel=9)
    zfile.write(data)
    zfile.close()
    
    return zbuf.getvalue()



def fileFormattedResults(URI, data, mimetype=None, compression=True):
    #TODO - send the mimetype
    #handle non-http requests
    if URI.startswith('PYME-CLUSTER') or URI.startswith('pyme-cluster'):
        clusterfilter = URI.split('://')[1].split('/')[0]
        sequenceName = URI.split('://%s/' % clusterfilter)[1]
        #print sequenceName, clusterfilter
        if not sequenceName.startswith('/'):
            sequenceName = '/' + sequenceName

        clusterIO.put_file(sequenceName, data, clusterfilter)
    elif URI.startswith('HTTP') or URI.startswith('http'):
        logging.debug('fileFormattedResults - URI: ' + URI)
        #logging.debug('data: ' + data)
        #logging.debug('type(data) = %s, len(data) = %d' % (type(data), len(data)))
        if not isinstance(URI, bytes):
            URI = URI.encode()
        # if not isinstance(data, bytes):
        #     data = data.encode()
            
        s = clusterIO._getSession(URI)
        
        if compression:
            r = s.put(URI, _gzip_compress(data), timeout=5, headers={'Content-Encoding': 'gzip'})
        else:
            r = s.put(URI, data=data, timeout=5)
        #print r.status_code
        if not r.status_code == 200:
            raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))
    else:
        raise RuntimeError('Unknown protocol %s' % URI.split(':')[0])

_loc_cache = {}
def pickResultsServer(filename, serverfilter=clusterIO.local_serverfilter):
    #logging.debug('pickResultsServer - input: ' + filename)
    if filename.startswith('__aggregate_txt/'):
        fn = filename[len('__aggregate_txt/'):]
        stub = ''
        prefix = '__aggregate_txt'
    elif filename.startswith('__aggregate_h5r/'):
        fn = filename[len('__aggregate_h5r/'):]
        if '.h5r' in filename:
            fn, stub = fn.split('.h5r')
            fn = fn  + '.h5r'
        else:
            fn, stub = fn.split('.hdf')
            fn = fn  + '.hdf'
        prefix = '__aggregate_h5r'
    else:
        fn = filename
        stub = ''
        prefix = ''

    cache_key = serverfilter + '::' + fn
    try:
        loc = _loc_cache[cache_key]
        #logging.debug('pickResultsServer - output[cached]: ' + loc + stub)
        return loc + stub
    except KeyError:
        locs_ = clusterIO.locate_file(fn, serverfilter)
        locs = []
        for l, dt in locs_:
            parts = l.split('/')
            nl = '/'.join(parts[:3] +[prefix,] + parts[3:])
            locs.append((nl, dt))


        if len(locs) > 1:
            raise RuntimeError('More than one copy of the file found. This should never happen, '
                               'and is probably the result of a race condition. Data corruption is likely.')

        if len(locs) == 1:
            loc =  locs[0][0]
        else:
            name, info = clusterIO._chooseServer(serverfilter)
            loc = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, '/'.join([prefix, fn]))

        _loc_cache[cache_key] = loc
        #logging.debug('pickResultsServer - output: ' + loc + stub)
        return loc + stub



def format_results(data_raw, URI=''):
    """
    translate data into wire format
    """
    output_format = None
    
    if URI.endswith('.csv') or URI.endswith('.txt') or URI.endswith('.log'):
        output_format = 'text/csv'
        
        if isinstance(data_raw, bytes):
            data = data_raw
        elif isinstance(data_raw, str):
            data = data_raw.encode()
        else:
            import pandas as pd
            df = pd.DataFrame(data_raw)
            data = df.to_csv()
    
    elif URI.endswith('.json'):
        output_format = 'text/json'
        if isinstance(data_raw, bytes):
            data = data_raw
        elif isinstance(data_raw, str):
            data = data_raw.encode()
        elif hasattr(data_raw, 'to_JSON'):
            data = data_raw.to_JSON().encode()
        else:
            import pandas as pd
            if isinstance(data_raw, np.ndarray):
                from PYME.IO.tabular import unnest_dtype
                data_raw = data_raw.view(unnest_dtype(data_raw.dtype))
            
            df = pd.DataFrame(data_raw)
            data = df.to_json().encode()
    
    elif URI.endswith('.pzf'):
        raise RuntimeError('PZF format needs parameters, pack data yourself and call fileFormattedResults')
    
    elif isinstance(data_raw, image.ImageStack):
        #output_format = 'image'
        raise NotImplementedError('Need to add code for saving images')
        #TODO - easy solution is to save locally and then copy. Better solution would be to change exporters to write into a file object
    
    elif URI.endswith('.npy'): # or isinstance(data_raw, np.ndarray):
        #output_format = 'numpy'
        data = BytesIO()
        np.save(data, np.array(data_raw))
        data = data.getvalue()
    
    elif isinstance(data_raw, np.ndarray):
        #very reluctantly use pickle to serialize numpy arrays rather than the better .npy format as reading .npy is really slow.
        data = data_raw.dumps()
    
    elif isinstance(data_raw, MetaDataHandler.MDHandlerBase):
        # NB - this may be redundant as we usually request metadata.json which will get caught and handled by the .endswith('json') cas above
        # keeping for now in case we want to change default metadata handling and/or also support .xml or .md formats
        output_format = 'text/json'
        data = data_raw.to_JSON().encode()
    
    else:
        logging.warning('No handler for data type found, using pickle')
        #output_format = 'pickle'
        data = cPickle.dumps(data_raw)
        
    return data, output_format

def fileResults(URI, data_raw):
    data, output_format = format_results(data_raw, URI)

    #now do URI translation
    if '__aggregate' in URI and URI.startswith('PYME-CLUSTER') or URI.startswith('pyme-cluster'):
        # pick a server to  send the results to. This should be the same server which already has the results file, if it
        # already exists. NOTE, this is not advised as there is a possible race condition here if multiple workers try to pick a
        # a location. This can be worked around by e.g. setting the metadata and creating the file from the node which initiates
        # the analysis, or ideally, encoding the correct HTTP:// url in the task specification.
        clusterfilter = URI.split('://')[1].split('/')[0]
        sequenceName = URI.split('://%s/' % clusterfilter)[1]

        #logging.debug('URI: ' + URI)
        #logging.debug('clusterfilter: ' + clusterfilter)
        #logging.debug('sequencename: ' + sequenceName)

        URI = pickResultsServer(sequenceName, clusterfilter)

        #logging.debug('URI: ' + URI)




    fileFormattedResults(URI, data)

