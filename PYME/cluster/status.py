
import threading
from PYME.IO.clusterIO import get_status, local_serverfilter
import logging

logger = logging.getLogger(__name__)

class StatusInfo(dict):
    def __del__(self):
        for k in self.keys():
            try:
                self[k]['polling'] = False
            except Exception as e:
                logger.error(str(e))
        
        for k in self.keys():
            try:
                self[k]['thread'].join()
            except Exception as e:
                logger.error(str(e))

_INFO = StatusInfo()

def get_polled_status(serverfilter=local_serverfilter, poll_wait=1):
    """ Polled version of PYME.IO.clusterIO.get_status
    Get status of cluster servers (currently only used in the clusterIO web 
    service)
    
    Parameters
    ----------
    serverfilter: str
        the cluster name (optional), to select a specific cluster
    poll_wait: float
        number of seconds for polling thread to wait before updating the
        status again. Only has an effect if this is the first call for
        a given `serverfilter`.

    Returns
    -------
    status_list: list
        a status dictionary for each node. See 
        PYME.cluster.HTTPDataServer.updateStatus
            Disk: dict
                total: int
                    storage on the node [bytes]
                used: int
                    used storage on the node [bytes]
                free: int
                    available storage on the node [bytes]
            CPUUsage: float
                cpu usage as a percentile
            MemUsage: dict
                total: int
                    total RAM [bytes]
                available: int
                    free RAM [bytes]
                percent: float
                    percent usage
                used: int
                    used RAM [bytes], calculated differently depending on platform
                free: int
                    RAM which is zero'd and ready to go [bytes]
                [other]:
                    more platform-specific fields
            Network: dict
                send: int
                    bytes sent per second since the last status update
                recv: int
                    bytes received per second since the last status update
            GPUUsage: list of float
                [optional] returned for NVIDIA GPUs only. Should be compute usage per gpu as percent?
            GPUMem: list of float
                [optional] returned for NVIDIA GPUs only. Should be memory usage per gpu as percent?


    """
    global _INFO

    if serverfilter not in _INFO.keys():
        _INFO[serverfilter] = {
            'lock': threading.Lock(),
            'status': get_status(serverfilter),
            'thread': threading.Thread(target=_poll_status, 
                                       args=(serverfilter, poll_wait)),
            'polling': True 
        }
        _INFO[serverfilter]['thread'].start()
    
    with _INFO[serverfilter]['lock']:
        return _INFO[serverfilter]['status']

def _poll_status(serverfilter, poll_wait):
    import time

    global _INFO

    while (serverfilter in _INFO.keys()) and _INFO[serverfilter]['polling']:
        status = get_status(serverfilter)
        with _INFO[serverfilter]['lock']:
            _INFO[serverfilter]['status'] = status
        time.sleep(poll_wait)
