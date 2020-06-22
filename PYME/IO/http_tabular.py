
from PYME.IO.tabular import FitResultsSource, MappingFilter
from PYME.IO import clusterIO, unifiedIO
import threading
import time
import numpy as np
import requests
import dispatch
from PYME.IO.MetaDataHandler import NestedClassMDHandler
import logging
import numpy as np
from io import BytesIO
import json



class HTTPFitResultsSource(FitResultsSource):
    def __init__(self, filename, server_filter='', update_lock=None, update_interval=10, keep_alive_time=60):
        """
        FitResults source which loads through the cluster and updates live until the remote file stops being added to.

        Parameters
        ----------
        filename : str
            relative path to the file from the PYMEDataServer root directory
        server_filter : str, optional
            cluster name to make sure we load the right file, by default '' which will load from any cluster on the network
        update_lock : threading.Lock, optional
            any context manager we should use when adding new fit results, by default None
        update_interval : int, optional
            time in seconds to delay between polling the PYMEDataServer for updates to the results file, by default 10
        keep_alive_time : int, optional
            time in seconds after last results file modification to keep checking for new fit results, by default 60
        """
        self.filename = filename
        self.update_interval = update_interval
        self.keep_alive_time = keep_alive_time
        self.fitResults = ()
        self.mdh = NestedClassMDHandler()

        start_time = time.time()
        while time.time() - start_time < keep_alive_time:
            try:
                logging.debug('trying to locate results file')
                self.uri = clusterIO.locate_file(self.filename, server_filter, return_first_hit=True)[0][0]
                logging.debug('series uri: %s' % self.uri)
                break
            except IndexError:
                time.sleep(1)
        
        self.updating, self.last_update_time = True, time.time()
        self.update_lock = threading.Lock() if update_lock is None else update_lock
        self.updated = dispatch.Signal()
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.start()
        while self.updating and (len(self.fitResults) == 0):
            time.sleep(1)  # don't let the init exit until fitResults/transkeys are set
            logging.debug('waiting for localization results')
    
    def _add_results(self, new_results):
        if len(new_results) > 0:
            with self.update_lock:
                if len(self.fitResults) == 0:
                    logging.debug('setting localization results (n: %d)' % len(new_results))
                    self.setResults(new_results)
                    self.mdh.update(json.load(BytesIO(requests.get(self.uri + '/MetaData.json').content)))
                else:
                    logging.debug('adding %d new localization results' % len(new_results))
                    self.fitResults = np.concatenate((self.fitResults, new_results))
                self.last_update_time = time.time()
                logging.debug('releaseing update lock')
            self.updated.send(self)
    
    def _update_loop(self):
        while self.updating:
            if time.time() - self.last_update_time > self.keep_alive_time:
                self.updating = False
                logging.debug('ending update loop')
                break
            
            # note, will receive np.array(0, dtype=fit_results_dtype) if there are no new results
            self._add_results(np.load(BytesIO(requests.get(self.uri + '/FitResults.npy?from=%d' % len(self.fitResults)).content)))
            # TODO - update events too
            
            time.sleep(self.update_interval)
        


