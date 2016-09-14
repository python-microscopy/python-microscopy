#!/usr/bin/python
from PYME import config as conf
import os
import shutil
import yaml
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName
import subprocess
import time
import socket
import tempfile
import logging
logging.basicConfig(level=logging.DEBUG)

from PYME.ParallelTasks import distribution
from multiprocessing import cpu_count




def main():
    cluster_root = conf.get('dataserver-root', '/home/ubuntu/PYME/test01')
    
    confFile = os.path.join(conf.user_config_dir, 'nodeserver.yaml')
    with open(confFile) as f:
        config = yaml.load(f)

    serverAddr, serverPort = config['nodeserver']['http_endpoint'].split(':')
    externalAddr = socket.gethostbyname(socket.gethostname())

    ns = pyme_zeroconf.getNS('_pyme-taskdist')
    #
    # #find distributor(s)
    # distributors = []
    # for name, info in ns.advertised_services.items():
    #     if name.startswith('PYMEDistributor'):
    #         distributors.append('%s:%d' % (socket.inet_ntoa(info.address), info.port))

    distributors = [u.lstrip('http://').rstrip('/') for u in distribution.getDistributorInfo().values()]


    #modify the configuration to reflect the discovered distributor(s)
    config['nodeserver']['distributors'] = distributors

    #write a new config file for the nodeserver
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_conf_file:
        temp_conf_file_name = temp_conf_file.name
        temp_conf_file.write(yaml.dump(config))

    logging.debug('Config file: ' + temp_conf_file_name)
    
    #set up nodeserver logging
    nodeserver_log_dir = os.path.join(cluster_root, 'LOGS', GetComputerName())
    
    #remove old log files
    try:
        os.remove(os.path.join(nodeserver_log_dir, 'nodeserver.log'))
    except:
        pass
    
    try:
        shutil.rmtree(os.path.join(nodeserver_log_dir, 'taskWorkerHTTP'))
    except:
        pass
    
    nodeserverLog = open(os.path.join(nodeserver_log_dir, 'nodeserver.log'), 'w')

    proc = subprocess.Popen('nodeserver -c %s' % temp_conf_file_name, shell=True, stdout=nodeserverLog, stderr=nodeserverLog)
    #proc = subprocess.Popen('python -m PYME.ParallelTasks.nodeserver %s %s' % (distributors[0], serverPort), shell=True,
    #                        stdout=nodeserverLog, stderr=nodeserverLog)

    ns.register_service('PYMENodeServer: ' + GetComputerName(), externalAddr, int(serverPort))

    time.sleep(2)
    logging.debug('Launching worker processors')
    numWorkers = config.get('numWorkers', cpu_count())

    workerProcs = [subprocess.Popen('python -m PYME.ParallelTasks.taskWorkerHTTP', shell=True, stdin=subprocess.PIPE)
                   for i in range(numWorkers -1)]

    #last worker has profiling enabled
    profiledir = os.path.join(nodeserver_log_dir, 'mProf')      
    workerProcs.append(subprocess.Popen('python -m PYME.ParallelTasks.taskWorkerHTTP -p %s' % profiledir, shell=True,
                                        stdin=subprocess.PIPE))

    try:
        while not proc.poll():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info('Shutting down workers')
        try:
            ns.unregister('PYMENodeServer: ' + GetComputerName())
        except:
            pass

        os.unlink(temp_conf_file_name)

        
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

        logging.info('Shutting down nodeserver')
        try:
            proc.kill()
        except:
            pass

        logging.info('Workers and nodeserver are shut down')
            
    nodeserverLog.close()


if __name__ == '__main__':
    main()



