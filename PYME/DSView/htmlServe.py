# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:23:43 2015

@author: david
"""

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
        return f2(host, port, timeout)
   
servers.wait_for_occupied_port = f2

## End Monkey patch

import cherrypy

#open on localhost, on whichever socket the OS gives us

cherrypy.server.socket_host = '127.0.0.1'
cherrypy.server.socket_port = 0