import subprocess
import time
import sys
import os
import webbrowser

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ClusterOfOne(object):
    def __init__(self):
        self._data_server = None
        self._rule_server = None
        self._node_server = None
        self._cluster_ui = None
        
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
            
        logger.info('Launching data server')
        self._data_server = subprocess.Popen('%s -m PYME.cluster.HTTPDataServer' % sys.executable, shell=True)
        
    def _launch_rule_server(self):
        if not self._rule_server is None:
            self._kill_procs([self._rule_server, ])

        logger.info('Launching rule server')
        self._rule_server = subprocess.Popen('%s -m PYME.cluster.PYMERuleServer' % sys.executable, shell=True)
        
    def _launch_node_server(self):
        if not self._node_server is None:
            self._kill_procs([self._node_server, ])

        logger.info('Launching node server')
        self._node_server = subprocess.Popen('%s -m PYME.cluster.PYMERuleNodeServer' % sys.executable, shell=True)
        
    def _launch_cluster_ui(self, gui=False):
        if not self._cluster_ui is None:
            self._kill_procs([self._cluster_ui, ])

        logger.info('Launching clusterUI')
        self._cluster_ui = subprocess.Popen('%s %s runserver 9999' % (sys.executable, os.path.join(os.path.split(__file__)[0], 'clusterUI', 'manage.py')), shell=True)
        
        if gui:
            #launch a web-browser to view clusterUI
            time.sleep(5)
            webbrowser.open_new_tab('http://127.0.0.1:9999/')
            
            
    def shutdown(self):
        logger.info('Shutting down cluster')
        print('Shutting down cluster')
        self._kill_procs([self._node_server, self._rule_server, self._cluster_ui, self._data_server])


    def launch(self, gui=False):
        self._launch_data_server()
        self._launch_rule_server()
        
        #wait for the rule server to come up before launching the node server
        time.sleep(5)
        self._launch_node_server()
        #self._launch_cluster_ui(gui=gui)
        
        
    def run(self):
        # runs a busy loop monitoring status
        self.launch()
        
        try:
            while True:
                time.sleep(30)
                
                #TODO - poll the processes to see if they are still running
        finally:
            self.shutdown()
            
            
def gui_main():
    import wx
    import PYME.resources
    
    if wx.__version__ > '4':
        from wx.adv import TaskBarIcon, TBI_DOCK
    else:
        from wx import TaskBarIcon, TBI_DOCK
    
    cluster = ClusterOfOne()
    
    app = wx.App()

    ico = wx.Icon(PYME.resources.getIconPath('PYMELogo.png'))
    
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
        cluster.launch(True)
        app.MainLoop()
    finally:
        cluster.shutdown()
    
    
if __name__ == '__main__':
    gui_main()
    
        