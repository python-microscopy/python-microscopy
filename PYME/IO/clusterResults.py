from . import clusterIO
from . import image
#from . import PZFFormat
from . import MetaDataHandler
import requests
import numpy as np
import cStringIO
import cPickle
import pandas as pd
import os
import logging
import socket


def fileFormattedResults(URI, data, mimetype=None):
    #TODO - send the mimetype
    #handle non-http requests
    if URI.startswith('PYME-CLUSTER') or URI.startswith('pyme-cluster'):
        clusterfilter = URI.split('://')[1].split('/')[0]
        sequenceName = URI.split('://%s/' % clusterfilter)[1]
        #print sequenceName, clusterfilter
        if not sequenceName.startswith('/'):
            sequenceName = '/' + sequenceName

        clusterIO.putFile(sequenceName, data, clusterfilter)
    elif URI.startswith('HTTP') or URI.startswith('http'):
        print URI
        r = requests.put(URI, data=data, timeout=5)
        #print r.status_code
        if not r.status_code == 200:
            raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))
    else:
        raise RuntimeError('Unknown protocol %s' % URI.split(':')[0])

_loc_cache = {}
def pickResultsServer(filename, serverfilter=''):
    if filename.startswith('__aggregate_txt/'):
        fn = filename[len('__aggregate_txt/'):]
        stub = ''
    elif filename.startswith('__aggregate_h5r/'):
        fn = filename[len('__aggregate_h5r/'):]
        fn, stub = fn.split('.h5r')
        fn = fn  + '.h5r'
    else:
        fn = filename
        stub = ''

    cache_key = serverfilter + '::' + fn
    try:
        return _loc_cache[cache_key]
    except KeyError:
        locs = clusterIO.locateFile(fn, serverfilter)
        locs = [(l + stub, dt) for l, dt in locs]


        if len(locs) > 1:
            raise RuntimeError('More than one copy of the file found. This should never happen, '
                               'and is probably the result of a race condition. Data corruption is likely.')

        if len(locs) == 1:
            loc =  locs[0][0]
        else:
            name, info = clusterIO._chooseServer(serverfilter)
            loc = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, filename)

        _loc_cache[cache_key] = loc
        return loc


def fileResults(URI, data_raw):
    # translate data into wire format
    output_format = None

    if URI.endswith('.csv'):
        output_format = 'text/csv'

        if isinstance(data_raw, str):
            data = data_raw
        else:
            df = pd.DataFrame(data_raw)
            data = df.to_csv()

    elif URI.endswith('.json'):
        output_format = 'text/json'
        if isinstance(data_raw, str):
            data = data_raw
        else:
            df = pd.DataFrame(data_raw)
            data = df.to_json()

    elif URI.endswith('.pzf'):
        raise RuntimeError('PZF format needs parameters, pack data yourself and call fileFormattedResults')

    elif isinstance(data_raw, image.ImageStack):
        #output_format = 'image'
        raise NotImplementedError('Need to add code for saving images')
        #TODO - easy solution is to save locally and then copy. Better solution would be to change exporters to write into a file object

    elif isinstance(data_raw, np.ndarray) or URI.endswith('.npy'):
        #output_format = 'numpy'
        data = cStringIO.StringIO()
        np.save(data, np.array(data_raw))

    elif isinstance(data_raw, MetaDataHandler.MDHandlerBase):
        output_format = 'text/json'
        data = data_raw.to_JSON()

    else:
        logging.warn('No handler for data type found, using pickle')
        #output_format = 'pickle'
        data = cPickle.dumps(data_raw)


    #now do URI translation
    if '__aggregate' in URI and URI.startswith('PYME-CLUSTER') or URI.startswith('pyme-cluster'):
        # pick a server to  send the results to. This should be the same server which already has the results file, if it
        # already exists. NOTE, this is not advised as there is a possible race condition here if multiple workers try to pick a
        # a location. This can be worked around by e.g. setting the metadata and creating the file from the node which initiates
        # the analysis, or ideally, encoding the correct HTTP:// url in the task specification.
        clusterfilter = URI.split('://')[1].split('/')[0]
        sequenceName = URI.split('://%s/' % clusterfilter)[1]

        URI = pickResultsServer(sequenceName, clusterfilter)




    fileFormattedResults(URI, data)

