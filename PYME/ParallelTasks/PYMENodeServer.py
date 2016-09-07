from PYME import config
import os
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


def main():
    confFile = os.path.join(config.user_config_dir, 'nodeserver.yaml')
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

    distributors = distribution.getDistributorInfo().values()


    #modify the configuration to reflect the discovered distributor(s)
    config['nodeserver']['distributors'] = distributors

    #write a new config file for the nodeserver
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_conf_file:
        temp_conf_file_name = temp_conf_file.name
        temp_conf_file.write(yaml.dump(config))

    logging.debug('Config file: ' + temp_conf_file_name)

    proc = subprocess.Popen('nodeserver -c %s' % temp_conf_file_name, shell=True)
    ns.register_service('PYMENodeServer: ' + GetComputerName(), externalAddr, int(serverPort))

    try:
        while not proc.poll():
            time.sleep(1)

    finally:
        ns.unregister('PYMENodeServer: ' + GetComputerName())
        proc.kill()
        os.unlink(temp_conf_file_name)

if __name__ == '__main__':
    main()



