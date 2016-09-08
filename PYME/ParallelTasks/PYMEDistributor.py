from PYME import config as conf
import os
import yaml
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName
import subprocess
import time
import socket

#confFile = os.path.join(config.user_config_dir, 'distributor.yaml')

def main():
    confFile = os.path.join(conf.user_config_dir, 'distributor.yaml')
    with open(confFile) as f:
        config = yaml.load(f)

    serverAddr, serverPort = config['distributor']['http_endpoint'].split(':')
    externalAddr = socket.gethostbyname(socket.gethostname())
    
    #set up logging
    data_root = conf.get('dataserver-root')
    if data_root:
        logfile_error = open('%s/LOGS/distributor_error.log' % data_root, 'w')
        logfile_debug = open('%s/LOGS/distributor_debug.log' % data_root, 'w')

    proc = subprocess.Popen('distributor -c %s' % confFile, shell=True, stdout=logfile_debug, stderr=logfile_error)

    ns = pyme_zeroconf.getNS('_pyme-taskdist')
    ns.register_service('PYMEDistributor: ' + GetComputerName(), externalAddr, int(serverPort))

    try:
        while not proc.poll():
            time.sleep(1)

    finally:
        ns.unregister('PYMEDistributor: ' + GetComputerName())
        proc.kill()
        
    logfile_error.close()
    logfile_debug.close()

if __name__ == '__main__':
    main()





