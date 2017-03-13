#!/usr/bin/python
##################
# view3d.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
import wx
import sys
import socket
from optparse import OptionParser
# noinspection PyCompatibility
import socketserver

op = OptionParser(usage = 'usage: %s [options] [filename]' % sys.argv[0])

op.add_option('-m', '--mode', dest='mode', help="mode (or personality), as defined in PYME/DSView/modules/__init__.py")
op.add_option('-q', '--queueURI', dest='queueURI', help="the Pyro URI of the task queue - to avoid having to use the nameserver lookup")

#options, args = op.parse_args()



def OpenFile(argv, show=False):
    from PYME.DSView.dsviewer import ImageStack, DSViewFrame
    options, args = op.parse_args(argv)

    if len (args) > 0:
        im = ImageStack(filename=args[0], queueURI=options.queueURI)
    else:
        im = ImageStack(queueURI=options.queueURI)

    if options.mode is None:
        mode = im.mode
    else:
        mode = options.mode

    vframe = DSViewFrame(im, None, im.filename, mode = mode)

    if show:
        vframe.Show()

    return vframe


class MyApp(wx.App):
    def OnInit(self):
        vframe = OpenFile(sys.argv[1:])

        self.SetTopWindow(vframe)
        vframe.Show(1)

        return 1

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        data = self.request.recv(1024).strip()
        print(("%s wrote:" % self.client_address[0]))
        print(data)
        
        wx.CallAfter(OpenFile, data.split('\t'), True)


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect(('localhost',9898))
        sock.send('\t'.join(sys.argv[1:]))

    except:
        #server not running
        import threading
        
        app = MyApp(0)

        sockServ = socketserver.TCPServer(('localhost',9898), MyTCPHandler)

        thrd = threading.Thread(target=sockServ.serve_forever)
        thrd.start()

        app.MainLoop()

        sockServ.shutdown()
        thrd.join()

    finally:
        sock.close()


    
