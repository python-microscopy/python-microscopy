from PYME.util import webframework
import jinja2
from PYME.Analysis import deTile
import numpy as np
import cherrypy
from io import BytesIO

#try:
#    import Image
#except ImportError:
from PIL import Image

env = jinja2.Environment(
    loader=jinja2.PackageLoader('PYME.tileviewer', 'templates'),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)


class TileServer(object):
    def __init__(self, tile_dir, tile_size=256):
        self._pyramid =deTile.ImagePyramid(tile_dir, pyramid_tile_size=tile_size)

    @cherrypy.expose
    def get_tile(self, layer, x, y, vmin=0, vmax=255):
        cherrypy.response.headers["Content-Type"] = "image/png"
        
        im = self._pyramid.get_tile(int(layer), int(x), int(y))
        
        #scale data
        im = np.clip((255*(im - float(vmin)))/(float(vmax)-float(vmin)), 0, 255).astype('uint8')

        out = BytesIO()
        Image.fromarray(im.T).save(out, 'PNG')
        s = out.getvalue()
        out.close()
        return s
    
    @cherrypy.expose
    def index(self):
        return env.get_template('tileviewer.html').render()
    
    
    
if __name__ == '__main__':
    import sys
    tile_server = TileServer(sys.argv[1], tile_size=128)

    cherrypy.config.update({'server.socket_port': 8979,
                            'server.socket_host': '0.0.0.0',
                            'log.screen': False,
                            'log.access_file': '',
                            'log.error_file': '',
                            'server.thread_pool': 50,
                            })

    #logging.getLogger('cherrypy.access').setLevel(logging.ERROR)

    #externalAddr = socket.gethostbyname(socket.gethostname())


    #app = cherrypy.tree.mount(tile_server, '/')
    #app.log.access_log.setLevel(logging.ERROR)

    cherrypy.quickstart(tile_server)