from PYME import config as conf
import os
import yaml
from PYME.misc import pyme_zeroconf
from PYME.IO.FileUtils.nameUtils import get_service_name
import subprocess
import time
import socket
import sys

import logging, logging.handlers
logging.basicConfig()
import threading
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
    #logfile_error = None
    #logfile_debug = None

    data_root = conf.get('dataserver-root')
    if data_root:
        #logfile_error = open('%s/LOGS/distributor_error.log' % data_root, 'w')
        #logfile_debug = open('%s/LOGS/distributor_debug.log' % data_root, 'w')

        distr_log_dir = '%s/LOGS' % data_root

        dist_log_err_file = os.path.join(distr_log_dir, 'distributor_error.log')
        if os.path.exists(dist_log_err_file):
            os.remove(dist_log_err_file)

        dist_err_handler = logging.handlers.RotatingFileHandler(dist_log_err_file, 'w', maxBytes=1e6, backupCount=1)
        dist_err_handler.setFormatter(logging.Formatter('%(message)s'))
        distLogErr = logging.getLogger('dist_err')
        distLogErr.addHandler(dist_err_handler)
        distLogErr.setLevel(logging.DEBUG)
        distLogErr.propagate = False

        dist_log_dbg_file = os.path.join(distr_log_dir, 'distributor_debug.log')
        if os.path.exists(dist_log_dbg_file):
            os.remove(dist_log_dbg_file)

        dist_dbg_handler = logging.handlers.RotatingFileHandler(dist_log_dbg_file, 'w', maxBytes=1e6, backupCount=1)
        dist_dbg_handler.setFormatter(logging.Formatter('%(message)s'))
        distLogDbg = logging.getLogger('dist_debug')
        distLogDbg.addHandler(dist_dbg_handler)
        distLogDbg.setLevel(logging.DEBUG)
        distLogDbg.propagate = False

        if not (len(sys.argv) == 2 and sys.argv[1] == '-n'):
            proc = subprocess.Popen('distributor -c %s' % confFile, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen('python -m PYME.cluster.distributor 1234', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        t_log_stderr = threading.Thread(target=log_stream, args=(proc.stderr, distLogErr))
        t_log_stderr.setDaemon(False)
        t_log_stderr.start()

        t_log_stdout = threading.Thread(target=log_stream, args=(proc.stdout, distLogDbg))
        t_log_stdout.setDaemon(False)
        t_log_stdout.start()
    else:
        if not (len(sys.argv) == 2 and sys.argv[1] == '-n'):
            proc = subprocess.Popen('distributor -c %s' % confFile, shell=True)
        else:
            proc = subprocess.Popen('python -m PYME.cluster.distributor 1234', shell=True)

    ns = pyme_zeroconf.getNS('_pyme-taskdist')
    service_name = get_service_name('PYMEDistributor')
    ns.register_service(service_name, externalAddr, int(serverPort))

    try:
        while not proc.poll():
            time.sleep(1)

            # if logfile_error:
            #     #do crude log rotation
            #     if logfile_error.tell() > 1e6:
            #         logfile_error.seek(0)
            #
            #     if logfile_debug.tell() > 1e6:
            #         logfile_debug.seek(0)

    finally:
        ns.unregister(service_name)
        #try and shut down the distributor cleanly
        proc.send_signal(1)
        time.sleep(2)
        proc.kill()

        LOG_STREAMS = False

        
    #logfile_error.close()
    #logfile_debug.close()

if __name__ == '__main__':
    main()





