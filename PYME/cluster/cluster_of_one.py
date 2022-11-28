from PYME.misc import big_sur_fix
import subprocess
import time
import sys
import os
import webbrowser

import logging
logger = logging.getLogger(__name__)

class ClusterOfOne(object):
    def __init__(self, root_dir, num_workers):
        self._data_server = None
        self._rule_server = None
        self._node_server = None
        self._cluster_ui = None
        
        self._num_workers = num_workers
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
        self._node_server = subprocess.Popen('"%s" -m PYME.cluster.PYMERuleNodeServer -a local -p 0 --num-workers=%d' % (sys.executable, self._num_workers), shell=True)
        
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


    def launch(self, gui=False, clusterUI=True):
        self._launch_data_server()
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
    logging.basicConfig(level=logging.DEBUG)
    import wx
    import PYME.resources
    from PYME import config
    #from optparse import OptionParser
    from argparse import ArgumentParser
    from PYME.IO.FileUtils import nameUtils
    from multiprocessing import cpu_count

    op = ArgumentParser(description='Launch all aspects of the PYME cluster on a single node')
    default_root = config.get('dataserver-root')
    #op.add_option('-r', '--root', dest='root',
    #              help="Root directory of virtual filesystem (default %s, see also 'dataserver-root' config entry)" % default_root,
    #              default=default_root)
    op.add_argument('--ui', dest='ui', help='launch web based ui', default=True)
    op.add_argument('--clusterUI', dest='clusterui', help='launch the full django-based cluster UI', 
                  action='store_true', default=False)
    op.add_argument('-n', '--num-workers', dest='num_workers', default=config.get('nodeserver-num_workers', cpu_count()), type=int,
                    help="number of worker processes to run - default: num cpu cores")


    args = op.parse_args()
    
    if wx.__version__ > '4':
        from wx.adv import TaskBarIcon, TBI_DOCK
    else:
        from wx import TaskBarIcon, TBI_DOCK
    
    cluster = ClusterOfOne(root_dir=default_root, num_workers=args.num_workers)
    
    app = wx.App()

    ico = wx.Icon(PYME.resources.getIconPath('pymeLogo.png'))
    
    class ClusterIcon(TaskBarIcon):
        TBMENU_CLUSTERUI = wx.NewId()
        TBMENU_CLOSE = wx.NewId()
        
        def __init__(self, frame):
            TaskBarIcon.__init__(self, TBI_DOCK)
            self.frame = frame
            
            
            print('Setting icon')
            self.SetIcon(ico, 'PYMECluster')

        def CreatePopupMenu(self, evt=None):
            """
            This method is called by the base class when it needs to popup
            the menu for the default EVT_RIGHT_DOWN event.  Just create
            the menu how you want it and return it from this function,
            the base class takes care of the rest.
            """
            menu = wx.Menu()
            menu.Append(self.TBMENU_CLUSTERUI, 'ClusterUI')
            
            return menu
            
    
    frame = wx.Frame(None)
    frame.SetIcon(ico)
    
    tb_icon = ClusterIcon(frame)
    
    #frame.Show()
    
    try:
        cluster.launch(gui=args.ui, clusterUI=args.clusterui)
        app.MainLoop()
    finally:
        cluster.shutdown()
    
    
if __name__ == '__main__':
    main()
    
        