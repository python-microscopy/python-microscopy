
from PYME.IO.tabular import FitResultsSource
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
    def __init__(self, filename, server_filter='', update_interval=3, keep_alive_time=60):
        self.filename = filename
        self.update_interval = update_interval
        self.keep_alive_time = keep_alive_time
        self.fitResults = ()
        self.updated = dispatch.Signal()

        self.mdh = NestedClassMDHandler()

        start_time = time.time()
        while time.time() - start_time < keep_alive_time:
            try:
                logging.debug('trying to locate results file')
                self.uri = clusterIO.locate_file(self.filename, server_filter, return_first_hit=True)[0][0]
                print(self.uri)
                break
            except IndexError:
                time.sleep(1)
        
        self.updating, self.last_update_time = True, time.time()
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.start()
        while self.updating and (len(self.fitResults) == 0):
            time.sleep(1)  # don't let the init exit until fitResults/transkeys are set
            logging.debug('waiting for localization results')
    
    def _add_results(self, new_results):
        if len(new_results) > 0:
            if len(self.fitResults) == 0:
                logging.debug('setting fit results')
                self.setResults(new_results)
                # print(json.load(BytesIO(requests.get(self.uri + '/Metadata.json').content)))
                self.mdh.update(json.load(BytesIO(requests.get(self.uri + '/MetaData.json').content)))
                print(self.mdh)
            else:
                self.fitResults = np.concatenate((self.fitResults, new_results))
            self.last_update_time = time.time()
            self.updated.send(self)
    
    def _update_loop(self):
        while self.updating:
            if time.time() - self.last_update_time > self.keep_alive_time:
                self.updating = False
                break
            
            # FIXME - not sure what actually happens if there are no new -? zero len array or None?
            self._add_results(np.load(BytesIO(requests.get(self.uri + '/FitResults.npy?from=%d' % len(self.fitResults)).content)))
            # TODO - update events too
            
            time.sleep(self.update_interval)
        


