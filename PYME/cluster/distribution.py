import requests
import json
import socket
from PYME.misc import pyme_zeroconf
from PYME.misc import hybrid_ns
import time

import logging
logger = logging.getLogger(__name__)

def getNodeInfo(ns=None):
    if ns is None:
        ns = hybrid_ns.getNS('_pyme-taskdist')

    queueURLs = {}

    for name, info in ns.get_advertised_services():
        if name.startswith('PYMENodeServer'):
            try:
                queueURLs[name] = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)
            except TypeError:
                if info.port is None:
                    logger.debug('Service info from %s has no port info' % name)
                else:
                    logger.debug('ValueError: %s %s, %s' % (name, repr(info), info.port))

    return queueURLs

def getDistributorInfo(ns=None):
    if ns is None:
        ns = hybrid_ns.getNS('_pyme-taskdist')

    queueURLs = {}

    for name, info in ns.get_advertised_services():
        if name.startswith('PYMEDistributor') or name.startswith('PYMERuleServer'):
            queueURLs[name] = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)

    return queueURLs


def getQueueInfo(distributorURL):
    if not distributorURL.endswith('/'):
        distributorURL += '/'

    session = requests.Session()
    session.trust_env = False
    r = session.get(distributorURL + 'distributor/queues')
    if r.status_code == 200:
        resp = r.json()
        if resp['ok']:
            return resp['result']
        else:
            raise RuntimeError('distributor/queues query did not return ok')
    else:
        raise RuntimeError('Unexpected status code: %d from distributor/queues query' % r.status_code)


QUEUE_INFO_TIMEOUT = 1 # seconds to hold queue info
_queue_info_cache = {}
def get_cached_queue_info(distributorURL):
    try:
        expiry, info = _queue_info_cache[distributorURL]
        if expiry < time.time():
            raise TimeoutError('Cache timeout expired')
    except (KeyError, TimeoutError):
        info = getQueueInfo(distributorURL)
        _queue_info_cache[distributorURL] = (time.time() + QUEUE_INFO_TIMEOUT, info)

    return info

