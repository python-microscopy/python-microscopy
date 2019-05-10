# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:23:43 2015

@author: david
"""
import cherrypy

from distutils.version import LooseVersion

try:
    if LooseVersion(cherrypy.__version__) < LooseVersion('8.7.0'):
        ##########
        # Monkey Patch cherrypy to allow us to use default automatically selected ports
        # The unpatched version will wait until it gets a timeout, and then kill the program
        # if we try and open with port=0
        # here, we replace force the wait look to return immediately if we're using automatic
        # port selection
        #####
        from cherrypy.process import servers
        
        f1 = servers.wait_for_occupied_port
        
        def f2(host, port, timeout=None):
            if port == 0:
                return
            else:
                return f1(host, port, timeout)
           
        servers.wait_for_occupied_port = f2
        
        ## End Monkey patch
except:
    print('Failed to patch cherrypy, hoping for the best')


#open on localhost, on whichever port the OS gives us (by calling with socket_port = 0)

cherrypy.config.update({'server.socket_port': 0,
                        #'server.socket_host': '0.0.0.0',
                        #'log.screen' : False,
                        'engine.autoreload.on': False})

#cherrypy.server.socket_host = '127.0.0.1'
#cherrypy.server.socket_port = 0

def _serve():
    cherrypy.engine.start()
    cherrypy.engine.block()
    
    
isServing = False


def StartServing():
    global isServing
    if not isServing:
        try: 
            import threading
            serveThread = threading.Thread(target=_serve)
            serveThread.start()
            isServing = True
        except ImportError:
            pass
    
#StartServing()

def mount(object, location):
    cherrypy.tree.mount(object, location)

def getURL():
    return 'http://%s:%d/' % cherrypy.server.httpserver.socket.getsockname()