#!/usr/bin/python
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import tempfile
import time

import yaml
from PYME import config as conf
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


from PYME.cluster import distribution
from PYME.cluster import rulenodeserver
from multiprocessing import cpu_count
import sys
import threading

def main():
    confFile = os.path.join(conf.user_config_dir, 'nodeserver.yaml')
    with open(confFile) as f:
        config = yaml.load(f)

    serverAddr, serverPort = config['nodeserver']['http_endpoint'].split(':')
    externalAddr = socket.gethostbyname(socket.gethostname())
    
    print(distribution.getDistributorInfo().values())

    distributors = [u.lstrip('http://').rstrip('/') for u in distribution.getDistributorInfo().values()]
    
    #set up nodeserver logging
    cluster_root = conf.get('dataserver-root')
    if cluster_root:
        nodeserver_log_dir = os.path.join(cluster_root, 'LOGS', GetComputerName())
        
        #remove old log files
        try:
            os.remove(os.path.join(nodeserver_log_dir, 'nodeserver.log'))
        except OSError:  # if we cant clear out old log files, we might not have a log directory set up
            try:
                if not os.path.exists(os.path.join(nodeserver_log_dir)):
                    os.makedirs(os.path.join(nodeserver_log_dir))  # NB - this will create all intermediate directories as well
            except:  # throw error because the RotatingFileHandler will fail to initialize
                raise IOError('Unable to initialize log files at %s' % nodeserver_log_dir)
        
        try:
            shutil.rmtree(os.path.join(nodeserver_log_dir, 'taskWorkerHTTP'))
        except:
            pass
        

        nodeserver_log_handler = logging.handlers.RotatingFileHandler(os.path.join(nodeserver_log_dir, 'nodeserver.log'), 'w', maxBytes=1e6,backupCount=0)
        nodeserverLog = logging.getLogger('nodeserver')
        nodeserverLog.setLevel(logging.DEBUG)
        nodeserver_log_handler.setLevel(logging.DEBUG)
        nodeserver_log_handler.setFormatter(formatter)
        nodeserverLog.addHandler(nodeserver_log_handler)
        nodeserverLog.addHandler(stream_handler)
        
        #nodeserverLog.propagate=False
        
    else:
        nodeserver_log_dir = os.path.join(os.curdir, 'LOGS', GetComputerName())

    #proc = subprocess.Popen('python -m PYME.cluster.rulenodeserver %s %s' % (distributors[0], serverPort), shell=True,
    #                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    proc = rulenodeserver.ServerThread(distributors[0], serverPort, profile=False)
    proc.start()
    

    ns = pyme_zeroconf.getNS('_pyme-taskdist')
    ns.register_service('PYMENodeServer: ' + GetComputerName(), externalAddr, int(serverPort))

    time.sleep(2)
    logger.debug('Launching worker processors')
    numWorkers = config.get('nodeserver-num_workers', cpu_count())

    workerProcs = [subprocess.Popen('python -m PYME.cluster.taskWorkerHTTP', shell=True, stdin=subprocess.PIPE)
                   for i in range(numWorkers -1)]

    #last worker has profiling enabled
    profiledir = os.path.join(nodeserver_log_dir, 'mProf')      
    workerProcs.append(subprocess.Popen('python -m PYME.cluster.taskWorkerHTTP -p %s' % profiledir, shell=True,
                                        stdin=subprocess.PIPE))

    try:
        while proc.is_alive():
            time.sleep(1)

    finally:
        logger.info('Shutting down workers')
        
        try:
            ns.unregister('PYMENodeServer: ' + GetComputerName())
        except:
            pass

        
        for p in workerProcs:
            #ask the workers to quit (nicely)
            try:
                p.send_signal(1)
            except:
                pass
            
        time.sleep(2)

        for p in workerProcs:
            #now kill them off
            try:
                p.kill()
            except:
                pass

        logger.info('Shutting down nodeserver')
        #try:
        proc.shutdown()
        proc.join()
        #except:
        #    pass

        logger.info('Workers and nodeserver are shut down')

        sys.exit()
            
    #nodeserverLog.close()


if __name__ == '__main__':
    main()



