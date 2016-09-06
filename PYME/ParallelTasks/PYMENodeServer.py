from PYME import config
import os
import yaml
from PYME.misc import pyme_zeroconf
from PYME.misc.computerName import GetComputerName
import subprocess
import time
import socket

if __name__ == '__main__':
    confFile = os.path.join(config.user_config_dir, 'nodeserver.yaml')
    with open(confFile) as f:
        config = yaml.load(f)

    serverAddr, serverPort = config['nodeserver']['http_endpoint'].split(':')
    externalAddr = socket.gethostbyname(socket.gethostname())

    proc = subprocess.Popen('nodeserver -c %s' % confFile, shell=True)

    ns = pyme_zeroconf.getNS('_pyme-distrib')
    ns.register_service('PYMENodeServer: ' + GetComputerName(), externalAddr, int(serverPort))

    try:
        while not proc.poll():
            time.sleep(1)

    finally:
        ns.unregister('PYMENodeServer: ' + GetComputerName())
        proc.kill()





