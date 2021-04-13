'''
Drive a galvo scanner from a raspberry pi and expose a web interface
'''

import logging
import sys

import numpy as np
import pygame
from PYME.util import webframework

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ScannerController(object):
    def __init__(self, scan_frequency=100, scan_amplitude=1.0):
        self._scan_frequency = scan_frequency
        self._scan_amplitude = scan_amplitude
        
        self._sound = None
        
        self._fullscale = 2**15
        
        pygame.mixer.pre_init(44100, size=-16, channels=1)
        pygame.init()
        
    @webframework.register_endpoint('/start_scanning')
    def start_scanning(self, frequency=None, amplitude=None, waveform='sine'):
        if not self._sound is None:
            self._sound.stop()
            self._sound = None
            
        if frequency is None:
            frequency = self._scan_frequency
        else:
            frequency = float(frequency)
        
        if amplitude is None:
            amplitude = self._scan_amplitude
        else:
            amplitude = float(amplitude)
            
        
        num_samples_per_period = (44100.0/frequency)
        
        #take at least 2 periods
        num_periods = max(np.ceil(frequency), 2.0)
        
        t = np.linspace(0, num_periods/frequency, int(num_periods*num_samples_per_period))
        
        if waveform == 'sine':
            y = amplitude*self._fullscale*np.sin(2*np.pi*frequency*t)
        elif waveform == 'tri':
            t1 = (frequency*t)%2
            y = 2*(t1 - 2*(t1>1)*(t1-1)) - 1
            y = amplitude*self._fullscale*y
        else:
            raise RuntimeError('waveform %s not supported' % waveform)
        
        
        if pygame.mixer.get_init()[2]  == 2:
            #hack for when we can't get a mono mixer
            print(y.shape)
            y = np.tile(y.astype('int16')[:,None], (1,2))
            print(y.shape)
            self._sound = pygame.sndarray.make_sound(y)
        else:
            self._sound = pygame.sndarray.make_sound(y.astype('int16'))
        self._sound.play(-1)
        
        return ''

    @webframework.register_endpoint('/stop_scanning')
    def stop_scanning(self):
        if not self._sound is None:
            self._sound.stop()
            self._sound = None
            
        return ''
        
    
    
class ScanServer(webframework.APIHTTPServer, ScannerController):
    def __init__(self, port):
        ScannerController.__init__(self)
        
        server_address = ('', port)
        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True

def run(port):
    import socket
    
    externalAddr = socket.gethostbyname(socket.gethostname())
    server = ScanServer(port=port)
    
    try:
        logger.info('Starting nodeserver on %s:%d' % (externalAddr, port))
        server.serve_forever()
    finally:
        logger.info('Shutting down ...')
        server.shutdown()
        server.server_close()

if __name__ == '__main__':
    port = sys.argv[1]
    run(int(port))