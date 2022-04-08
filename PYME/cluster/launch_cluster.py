# from PYME.misc import big_sur_fix
import subprocess
import time
import sys
import os
import webbrowser

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ClusterNode(object):
    def __init__(self, root_dir):
        self._data_server = None
        self._rule_server = None
        self._node_server = None
        self._cluster_ui = None
        
        self._root_dir = root_dir
        
    def _kill_procs(self, procs):
        #ask nicely
        for p in procs:
            if not p is None:
                p.send_signal(1)
        
        #give the processes a chance to close
        time.sleep(2)
        
        #kill off stragglers
        for p in procs:
            if not p is None:
                p.kill()
            
        time.sleep(1)
        
        
    def _launch_data_server(self):
        if not self._data_server is None:
            self._kill_procs([self._data_server,])
            
        logger.info('Launching data server: root=%s' % self._root_dir)
        self._data_server = subprocess.Popen('"%s" -m PYME.cluster.HTTPDataServer -a local -p 0 -r "%s"' % (sys.executable, self._root_dir), shell=True)
        
    def _launch_rule_server(self):
        if not self._rule_server is None:
            self._kill_procs([self._rule_server, ])

        logger.info('Launching rule server')
        self._rule_server = subprocess.Popen('"%s" -m PYME.cluster.PYMERuleServer -a local -p 0'
                                             '' % sys.executable, shell=True)
        
    def _launch_node_server(self):
        if not self._node_server is None:
            self._kill_procs([self._node_server, ])

        logger.info('Launching node server')
        self._node_server = subprocess.Popen('"%s" -m PYME.cluster.PYMERuleNodeServer -a local -p 0' % sys.executable, shell=True)
        
    def _launch_cluster_ui(self, gui=False):
        try:
            import django
        except ImportError:
            logger.error('django is not installed, to use clusterUI install django (2.0.x, 2.1.x)')
            
        if not self._cluster_ui is None:
            self._kill_procs([self._cluster_ui, ])

        logger.info('Launching clusterUI')
        self._cluster_ui_stderr = open('clusterui.log', 'w')
        self._cluster_ui = subprocess.Popen('"%s" %s runserver 9999' % (sys.executable, os.path.join(os.path.split(__file__)[0], 'clusterUI', 'manage.py')), stderr=self._cluster_ui_stderr, shell=True)
        
        if gui:
            #launch a web-browser to view clusterUI
            time.sleep(5)
            webbrowser.open_new_tab('http://127.0.0.1:9999/')
            
    def _launch_ruleserver_ui(self):
        from PYME.misc import sqlite_ns
        from . import distribution
        ns = sqlite_ns.getNS('_pyme-taskdist')
        ruleservers = distribution.getDistributorInfo(ns)
        webbrowser.open_new_tab(list(ruleservers.values())[0])
            
            
    def shutdown(self):
        logger.info('Shutting down cluster')
        print('Shutting down cluster')
        self._kill_procs([self._node_server, self._rule_server, self._cluster_ui, self._data_server])
        try:
            self._cluster_ui_stderr.close()
        except:
            pass


    def launch(self, gui=False, clusterUI=True, main_node=False):
        self._launch_data_server()
        if main_node:
            self._launch_rule_server()
        
            #wait for the rule server to come up before launching the node server
            time.sleep(5)
        self._launch_node_server()
        if clusterUI:
            self._launch_cluster_ui(gui=gui)
        elif gui:
            self._launch_ruleserver_ui()
            
        
        
    def run(self):
        # runs a busy loop monitoring status
        self.launch()
        
        try:
            while True:
                time.sleep(30)
                
                #TODO - poll the processes to see if they are still running
        finally:
            self.shutdown()
            
            
def main():
    import PYME.resources
    from PYME import config
    from optparse import OptionParser
    from PYME.IO.FileUtils import nameUtils

    op = OptionParser(usage='usage: %s [options]' % sys.argv[0])
    default_root = config.get('dataserver-root')
    op.add_option('-r', '--root', dest='root',
                  help="Root directory of virtual filesystem (default %s, see also 'dataserver-root' config entry)" % default_root,
                  default=default_root)
    op.add_option('--ui', dest='ui', help='launch web based ui', default=True)
    op.add_option('--clusterUI', dest='clusterui', help='launch the full django-based cluster UI', 
                  action='store_true', default=False)
    op.add_option('--main', dest='main_node', help='Launch this as the main node for a cluster.'.
                  default=False)

    options, args = op.parse_args()

    cluster = ClusterNode(root_dir=options.root)
    
    try:
        cluster.launch(gui=options.ui, clusterUI=options.clusterui, main_node=options.main_node)
    finally:
        cluster.shutdown()
    
    
if __name__ == '__main__':
    main()
    
        