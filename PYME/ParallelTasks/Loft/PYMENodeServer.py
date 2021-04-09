#!/usr/bin/python
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import tempfile
import time
from PYME.IO.FileUtils.nameUtils import get_service_name
import yaml
from PYME import config as conf
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName

logging.basicConfig(level=logging.DEBUG)

from PYME.cluster import distribution
from multiprocessing import cpu_count
import sys
import threading


LOG_STREAMS = True

def log_stream(stream, logger):
    while LOG_STREAMS:
        line = stream.readline()
        logger.debug(line.strip())

def main():
    global LOG_STREAMS
    cluster_root = conf.get('dataserver-root', conf.user_config_dir)
    
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
    except OSError:  # if we cant clear out old log files, we might not have a log directory set up
        try:
            os.makedirs(os.path.join(nodeserver_log_dir))  # NB - this will create all intermediate directories as well
        except:  # throw error because the RotatingFileHandler will fail to initialize
            raise IOError('Unable to initialize log files at %s' % nodeserver_log_dir)
        pass
    
    try:
        shutil.rmtree(os.path.join(nodeserver_log_dir, 'taskWorkerHTTP'))
    except:
        pass
    
    #nodeserverLog = open(os.path.join(nodeserver_log_dir, 'nodeserver.log'), 'w')
    nodeserver_log_handler = logging.handlers.RotatingFileHandler(os.path.join(nodeserver_log_dir, 'nodeserver.log'), 'w', maxBytes=1e6,backupCount=0)
    nodeserver_log_handler.setFormatter(logging.Formatter('%(message)s'))
    nodeserverLog = logging.getLogger('nodeserver')
    nodeserverLog.addHandler(nodeserver_log_handler)
    nodeserverLog.setLevel(logging.DEBUG)
    nodeserverLog.propagate=False

    if not (len(sys.argv) == 2 and sys.argv[1] == '-n'):
        proc = subprocess.Popen('nodeserver -c %s' % temp_conf_file_name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        proc = subprocess.Popen('python -m PYME.cluster.nodeserver %s %s' % (distributors[0], serverPort), shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    t_log_stderr = threading.Thread(target=log_stream, args=(proc.stderr, nodeserverLog))
    t_log_stderr.setDaemon(False)
    t_log_stderr.start()

    t_log_stdout = threading.Thread(target=log_stream, args=(proc.stdout, nodeserverLog))
    t_log_stdout.setDaemon(False)
    t_log_stdout.start()

    service_name = get_service_name('PYMENodeServer')
    ns.register_service(service_name, externalAddr, int(serverPort))

    time.sleep(2)
    logging.debug('Launching worker processors')
    numWorkers = config.get('numWorkers', cpu_count())
    subprocess.Popen('"%s" -m PYME.cluster.PYMERuleNodeServer -a local -p 0' % sys.executable, shell=True)
    workerProcs = [subprocess.Popen('"%s" -m PYME.cluster.taskWorkerHTTP' % sys.executable, shell=True, stdin=subprocess.PIPE)
                   for i in range(numWorkers -1)]

    #last worker has profiling enabled
    profiledir = os.path.join(nodeserver_log_dir, 'mProf')      
    workerProcs.append(subprocess.Popen('"%s" -m PYME.cluster.taskWorkerHTTP -p "%s"' % (sys.executable, profiledir),
                                        shell=True, stdin=subprocess.PIPE))

    try:
        while not proc.poll():
            time.sleep(1)

            #try to keep log size under control by doing crude rotation
            #if nodeserverLog.tell() > 1e6:
            #    nodeserverLog.seek(0)
    except KeyboardInterrupt:
        pass
    finally:
        LOG_STREAMS = False
        logging.info('Shutting down workers')
        try:
            ns.unregister(service_name)
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

        sys.exit()
            
    #nodeserverLog.close()


if __name__ == '__main__':
    main()



