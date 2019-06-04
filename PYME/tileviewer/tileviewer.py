from PYME.util import webframework
import jinja2
from PYME.Analysis import tile_pyramid
import numpy as np
import cherrypy
from io import BytesIO
import os

from PYME.IO import MetaDataHandler

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
        self._set_tile_source(tile_dir)
        self.roi_locations = []
        
    def _set_tile_source(self, tile_dir):
        self.tile_dir = tile_dir
        self.mdh = MetaDataHandler.load_json(os.path.join(tile_dir, 'metadata.json'))
        self._pyramid = tile_pyramid.ImagePyramid(tile_dir, pyramid_tile_size=self.mdh['Pyramid.TileSize'])
        
    @cherrypy.expose
    def set_tile_source(self, tile_dir):
        self._set_tile_source(tile_dir)
        
        raise cherrypy.HTTPRedirect('/')

    @cherrypy.expose
    def set_roi_locations(self, locations_file, tablename='Locations'):
        from PYME.IO import tabular
        
        if locations_file.endswith('.h5'):
            self.roi_locations = tabular.hdfSource(locations_file, tablename=tablename)
        # elif locations_file.endswith('.csv'):
        #     self.roi_locations = tabular.textfileSource(locations_file)
        raise cherrypy.HTTPRedirect('/')
    
    @cherrypy.expose
    def get_roi_locations(self):
        if not self.roi_locations is None:
            return self.roi_locations.toDataFrame().to_json()
        
    @cherrypy.expose
    def add_roi(self, x, y):
        self.roi_locations.append({'x': float(x), 'y': float(y)})
        
        raise cherrypy.HTTPRedirect('/roi_list')
        
    @cherrypy.expose
    def roi_list(self):
        return env.get_template('roi_list.html').render(roi_list=self.roi_locations)

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
        return env.get_template('tileviewer.html').render(tile_size=self.mdh['Pyramid.TileSize'],
                                                          pyramid_width_px=self.mdh['Pyramid.PixelsX'],
                                                          pyramid_height_px=self.mdh['Pyramid.PixelsY'],
                                                          pyramid_depth=self.mdh['Pyramid.Depth'],
                                                          tile_dir = self.tile_dir,
                                                          )
    
    
    
if __name__ == '__main__':
    import sys
    tile_server = TileServer(sys.argv[1], tile_size=128)
    
    from PYME import cluster
    from PYME import resources
    
    static_dir = os.path.join(os.path.split(cluster.__file__)[0], 'clusterUI','static')
    
    
    cherrypy.config.update({'server.socket_port': 8979,
                            'server.socket_host': '127.0.0.1',
                            #'log.screen': False,
                            'log.access_file': '',
                            'log.error_file': '',
                            'server.thread_pool': 50,
                            #'tools.staticdir.on': True,'tools.staticdir.root': static_dir
                 })

    conf = {#'/':{
                            #'tools.staticdir.on': True,'tools.staticdir.root': static_dir
                 #},
            '/static':{'tools.staticdir.on': True,
                       'tools.staticdir.dir': static_dir},
            '/favicon.ico':
            {
                'tools.staticfile.on': True,
                'tools.staticfile.filename': resources.getIconPath('pymeLogo.png')
            }
            }

    #logging.getLogger('cherrypy.access').setLevel(logging.ERROR)

    #externalAddr = socket.gethostbyname(socket.gethostname())


    #app = cherrypy.tree.mount(tile_server, '/')
    #app.log.access_log.setLevel(logging.ERROR)

    cherrypy.quickstart(tile_server, '/', conf)