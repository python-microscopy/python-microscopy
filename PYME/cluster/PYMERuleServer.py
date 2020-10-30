import logging
import logging.handlers
import os
import socket
import time

import yaml
from PYME import config
from PYME.cluster import ruleserver
from PYME.misc import pyme_zeroconf, sqlite_ns
from PYME.misc.computerName import GetComputerName

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#confFile = os.path.join(config.user_config_dir, 'distributor.yaml')

from argparse import ArgumentParser

LOG_STREAMS = True



def log_stream(stream, logger):
    while LOG_STREAMS:
        line = stream.readline()
        logger.debug(line.strip())

def main():
    global LOG_STREAMS
    
    op = ArgumentParser(description="PYME rule server for task distribution. This should run once per cluster.")

    #NOTE - currently squatting on port 15346 for testing - TODO can we use an ephemeral port
    op.add_argument('-p', '--port', dest='port', default=config.get('ruleserver-port', 15346), type=int,
                  help="port number to serve on (default: 15346, see also 'ruleserver-port' config entry)")
    
    op.add_argument('-a','--advertisements', dest='advertisements', choices=['zeroconf', 'local'], default='zeroconf',
                  help='Optionally restrict advertisements to local machine')
    
    args = op.parse_args()

    serverPort = args.port
    
    if args.advertisements == 'local':
        #bind on localhost
        bind_addr = '127.0.0.1'
    else:
        bind_addr = '' #bind all interfaces
    
    #set up logging
    data_root = config.get('dataserver-root')
    if data_root:
        distr_log_dir = '%s/LOGS' % data_root
        try:  # make sure the directory exists
            os.makedirs(distr_log_dir)  # exist_ok flag not present on py2
        except OSError as e:
            import errno
            if e.errno != errno.EEXIST:
                raise e

        dist_log_err_file = os.path.join(distr_log_dir, 'distributor.log')
        if os.path.exists(dist_log_err_file):
            os.remove(dist_log_err_file)

        dist_err_handler = logging.handlers.RotatingFileHandler(filename=dist_log_err_file, mode='w', maxBytes=1e6, backupCount=1)
        #dist_err_handler.setFormatter(logging.Formatter('%(message)s'))
        distLogErr = logging.getLogger('distributor')
        distLogErr.setLevel(logging.DEBUG)
        distLogErr.addHandler(dist_err_handler)
    
    
    proc = ruleserver.ServerThread(serverPort, bind_addr=bind_addr, profile=False)
    proc.start()
    #proc = subprocess.Popen('python -m PYME.ParallelTasks.distributor 1234', shell=True)

    if args.advertisements == 'zeroconf':
        ns = pyme_zeroconf.getNS('_pyme-taskdist')
    else:
        #assume 'local'
        ns = sqlite_ns.getNS('_pyme-taskdist')

    time.sleep(0.5)
    #get the actual adress (port) we bound to
    sa = proc.distributor.socket.getsockname()
    ns.register_service('PYMERuleServer: ' + GetComputerName(), proc.externalAddr, int(sa[1]))

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





