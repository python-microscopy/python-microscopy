import logging
import logging.handlers
import os
import socket
import time

import yaml
from PYME import config as conf
from PYME.cluster import ruleserver
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#confFile = os.path.join(config.user_config_dir, 'distributor.yaml')

LOG_STREAMS = True

def log_stream(stream, logger):
    while LOG_STREAMS:
        line = stream.readline()
        logger.debug(line.strip())

def main():
    global LOG_STREAMS
    confFile = os.path.join(conf.user_config_dir, 'distributor.yaml')
    with open(confFile) as f:
        config = yaml.load(f)

    serverAddr, serverPort = config['distributor']['http_endpoint'].split(':')
    externalAddr = socket.gethostbyname(socket.gethostname())
    
    #set up logging
    data_root = conf.get('dataserver-root')
    if data_root:
        distr_log_dir = '%s/LOGS' % data_root

        dist_log_err_file = os.path.join(distr_log_dir, 'distributor.log')
        if os.path.exists(dist_log_err_file):
            os.remove(dist_log_err_file)

        dist_err_handler = logging.handlers.RotatingFileHandler(filename=dist_log_err_file, mode='w', maxBytes=1e6, backupCount=1)
        #dist_err_handler.setFormatter(logging.Formatter('%(message)s'))
        distLogErr = logging.getLogger('distributor')
        distLogErr.setLevel(logging.DEBUG)
        distLogErr.addHandler(dist_err_handler)
    
    
    proc = ruleserver.ServerThread(serverPort, profile=False)
    proc.start()
    #proc = subprocess.Popen('python -m PYME.ParallelTasks.distributor 1234', shell=True)

    ns = pyme_zeroconf.getNS('_pyme-taskdist')
    ns.register_service('PYMERuleServer: ' + GetComputerName(), externalAddr, int(serverPort))

    try:
        while proc.is_alive():
            time.sleep(1)

    finally:
        logger.debug('trying to shut down server')
        proc.shutdown()
        ns.unregister('PYMERuleServer: ' + GetComputerName())
        #try and shut down the distributor cleanly
        
        #time.sleep(2)
        #proc.kill()


if __name__ == '__main__':
    main()





