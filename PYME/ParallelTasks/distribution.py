import requests
import json
import socket
from PYME.misc import pyme_zeroconf
import logging
logger = logging.getLogger(__name__)

def getNodeInfo():
    ns = pyme_zeroconf.getNS('_pyme-taskdist')

    queueURLs = {}

    for name, info in ns.advertised_services.items():
        if name.startswith('PYMENodeServer'):
            try:
                queueURLs[name] = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)
            except TypeError:
                logger.debug('ValueError: %s %s, %s' % (name, repr(info), info.port))

    return queueURLs

def getDistributorInfo():
    ns = pyme_zeroconf.getNS('_pyme-taskdist')

    queueURLs = {}

    for name, info in ns.advertised_services.items():
        if name.startswith('PYMEDistributor'):
            queueURLs[name] = 'http://%s:%d/' % (socket.inet_ntoa(info.address), info.port)

    return queueURLs


def getQueueInfo(distributorURL):
    r = requests.get(distributorURL + 'distributor/queues')
    if r.status_code == 200:
        resp = r.json()
        if resp['ok']:
            return resp['result']
        else:
            raise RuntimeError('distributor/queues query did not return ok')
    else:
        raise RuntimeError('Unexpected status code: %d from distributor/queues query' % r.status_code)
