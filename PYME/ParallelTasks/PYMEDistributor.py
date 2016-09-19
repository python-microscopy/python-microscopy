from PYME import config as conf
import os
import yaml
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName
import subprocess
import time
import socket
import sys
#confFile = os.path.join(config.user_config_dir, 'distributor.yaml')

def main():
    confFile = os.path.join(conf.user_config_dir, 'distributor.yaml')
    with open(confFile) as f:
        config = yaml.load(f)

    serverAddr, serverPort = config['distributor']['http_endpoint'].split(':')
    externalAddr = socket.gethostbyname(socket.gethostname())
    
    #set up logging
    logfile_error = None
    logfile_debug = None

    data_root = conf.get('dataserver-root')
    if data_root:
        logfile_error = open('%s/LOGS/distributor_error.log' % data_root, 'w')
        logfile_debug = open('%s/LOGS/distributor_debug.log' % data_root, 'w')

        if not (len(sys.argv) == 2 and sys.argv[1] == '-n'):
            proc = subprocess.Popen('distributor -c %s' % confFile, shell=True, stdout=logfile_debug, stderr=logfile_error)
        else:
            proc = subprocess.Popen('python -m PYME.ParallelTasks.distributor 1234 -k', shell=True, stdout=logfile_debug, stderr=logfile_error)
    else:
        if not (len(sys.argv) == 2 and sys.argv[1] == '-n'):
            proc = subprocess.Popen('distributor -c %s' % confFile, shell=True)
        else:
            proc = subprocess.Popen('python -m PYME.ParallelTasks.distributor 1234', shell=True)

    ns = pyme_zeroconf.getNS('_pyme-taskdist')
    ns.register_service('PYMEDistributor: ' + GetComputerName(), externalAddr, int(serverPort))

    try:
        while not proc.poll():
            time.sleep(1)

            if logfile_error:
                #do crude log rotation
                if logfile_error.tell() > 1e6:
                    logfile_error.seek(0)

                if logfile_debug.tell() > 1e6:
                    logfile_debug.seek(0)

    finally:
        ns.unregister('PYMEDistributor: ' + GetComputerName())
        #try and shut down the distributor cleanly
        proc.send_signal(1)
        time.sleep(2)
        proc.kill()
        
    logfile_error.close()
    logfile_debug.close()

if __name__ == '__main__':
    main()





