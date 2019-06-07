import os
#import SimpleHTTPServer
# noinspection PyCompatibility
import http.server

# noinspection PyCompatibility
import socketserver
import webbrowser
import threading
import time

PORT = 9000

def launch_browser():
    #wait for the server to come up
    time.sleep(.5)
    print('Opening in browser')
    webbrowser.open("http://127.0.0.1:%d/" %PORT)


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.split(__file__)[0], 'html'))

    Handler = http.server.SimpleHTTPRequestHandler

    httpd = socketserver.TCPServer(("127.0.0.1", PORT), Handler)

    print("Serving at http://127.0.0.1:%d/" %PORT)
    threading.Thread(target=launch_browser).start()
    httpd.serve_forever()