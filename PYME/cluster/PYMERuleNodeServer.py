#!/usr/bin/python
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import tempfile
import time
import sys
import yaml
from PYME import config as conf
from PYME.misc import pyme_zeroconf, sqlite_ns
from PYME.misc.computerName import GetComputerName
from PYME.IO.FileUtils.nameUtils import get_service_name

logging.basicConfig(level=logging.INFO)
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
from argparse import ArgumentParser
import threading

from PYME.util import fProfile, mProfile

def main():
    op = ArgumentParser(description="PYME node server for task distribution. This should run on every node of the cluster")


    #NOTE - currently squatting on port 15347 for testing - TODO can we use an ephemeral port?
    op.add_argument('-p', '--port', dest='port', default=conf.get('nodeserver-port', 15347), type=int,
                    help="port number to serve on (default: 15347, see also 'nodeserver-port' config entry)")

    op.add_argument('-a', '--advertisements', dest='advertisements', choices=['zeroconf', 'local'], default='zeroconf',
                    help='Optionally restrict advertisements to local machine')

    args = op.parse_args()

    serverPort = args.port
    externalAddr = socket.gethostbyname(socket.gethostname())
    
    if args.advertisements == 'zeroconf':
        ns = pyme_zeroconf.getNS('_pyme-taskdist')
    else:
        #assume local
        ns = sqlite_ns.getNS('_pyme-taskdist')
        externalAddr = '127.0.0.1' #bind to localhost
    
    #TODO - move this into the nodeserver proper so that the ruleserver doesn't need to be up before we start
    print(distribution.getDistributorInfo(ns).values())
    distributors = [u.lstrip('http://').rstrip('/') for u in distribution.getDistributorInfo(ns).values()]
    
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
        nodeserverLog = logger


    proc = rulenodeserver.ServerThread(distributors[0], serverPort, externalAddr=externalAddr, profile=False)
    proc.start()
        
    # TODO - do we need this advertisement
    #get the actual adress (port) we bound to
    time.sleep(0.5)
    sa = proc.nodeserver.socket.getsockname()
    serverPort = int(sa[1])
    service_name = get_service_name('PYMENodeServer')
    ns.register_service(service_name, externalAddr, serverPort)

    time.sleep(2)
    nodeserverLog.debug('Launching worker processors')
    numWorkers = conf.get('nodeserver-num_workers', cpu_count())

    workerProcs = [subprocess.Popen('"%s" -m PYME.cluster.taskWorkerHTTP -s %d' % (sys.executable, serverPort), shell=True, stdin=subprocess.PIPE)
                   for i in range(numWorkers -1)]

    #last worker has profiling enabled
    profiledir = os.path.join(nodeserver_log_dir, 'mProf')
    workerProcs.append(subprocess.Popen('"%s" -m PYME.cluster.taskWorkerHTTP -s % d -p --profile-dir="%s"' % (sys.executable, serverPort, profiledir), shell=True,
                                        stdin=subprocess.PIPE))

    try:
        while proc.is_alive():
            time.sleep(1)

    finally:
        logger.info('Shutting down workers')
        
        try:
            ns.unregister(service_name)
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

        proc.shutdown()
        proc.join()

        logger.info('Workers and nodeserver are shut down')

        sys.exit()
            
    #nodeserverLog.close()


if __name__ == '__main__':
    #prof = fProfile.thread_profiler()
    #prof.profileOn('.*PYME.*|.*zeroconf.*', 'ruleserver_prof.txt')
    #mProfile.profileOn(['rulenodeserver.py', 'zeroconf.py'])
    main()

    #mProfile.report()
    #prof.profileOff()



