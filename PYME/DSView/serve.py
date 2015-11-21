# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:49:52 2015

@author: david
"""

import cherrypy

cherrypy.config.update({'server.socket_port': 0,
                        #'server.socket_host': '0.0.0.0',
                        'engine.autoreload.on': False})

def _serve():
    cherrypy.engine.start()
    cherrypy.engine.block()
    
    
def StartServing():
    try: 
        import threading
        serveThread = threading.Thread(target=_serve)
        serveThread.start()
    except ImportError:
        pass
    
StartServing()