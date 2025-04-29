"""Watches a directory for new files and re-saves as a compressed .pcs image stack. Used for interfacing with 3rd party acquisition software."""


import os
import sys
import time
import threading
import logging
logger = logging.getLogger(__name__)

from PYME.IO import acquisition_backends
from PYME.IO import image



class Watcher(object):
    def __init__(self, indir, outdir, overwrite=False, input_ext = '.tif', write_time=5, ad_offset=95., electrons_per_adu=0.25, shape=[-1, -1, 1, -1, 1]):
        self.dir = indir
        self.outdir = outdir
        self._input_ext = input_ext
        self._write_time = write_time # time to wait for acquisition software to finish writing the file (so we don't delete it mid-write)
        self.overwrite = overwrite

        backend_kwargs = {
            'shape': shape,
            'compression_settings': {'quantizationOffset': ad_offset, 'quantizationScale': 1./(2*electrons_per_adu)}
        }

        if outdir.upper().startswith('PYMECLUSTER://'):
            if outdir.endswith('.h5'):
                self.backend = acquisition_backends.ClusterBackend(outdir, cluster_h5=True, **backend_kwargs)
            else:
                self.backend = acquisition_backends.ClusterBackend(outdir, **backend_kwargs)
        elif outdir.endswith('.h5'):
            self.backend = acquisition_backends.HDFBackend(outdir, **backend_kwargs)
        else:
            raise ValueError('Unknown output format for outdir: %s' % outdir)
            # TODO - additional backends - e.g. TIFF

        
        self._n = 0
        
        self._t_poll = threading.Thread(target=self.run)
        self._t_poll.start()


    def run(self):
        # polling loop to run in a separate thread
        self._running = True
        self.backend.initialise()
        while self._running:
            for f in os.listdir(self.dir):
                fn = os.path.join(self.dir, f)
                if f.endswith(self._input_ext) and (os.path.getmtime(fn) < (time.time() - self._write_time)):
                    # only process if the file is older than write_time
                    self.process_file(fn)
            time.sleep(1)
        self.backend.finalise()
        
    def process_file(self, fn):
        '''store the file in the backend and delete the original'''
        logger.debug('Processing file %s' % fn)

        img = image.ImageStack(filename=fn)
        # support multi-frame input files TODO - is this needed?
        for i in range(img._data.getNumSlices()):
            self.backend.store_frame(self._n, img._data.getSlice(i))
            self._n += 1

        os.remove(fn)
        logger.debug('Stored %s' % fn)

    def stop(self):
        self._running = False
        self._t_poll.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    w = Watcher(sys.argv[1], sys.argv[2])
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()
        print('Exiting')