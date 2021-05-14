from PYME.util import webframework
import jinja2
from PYME.Analysis import tile_pyramid
import numpy as np
import cherrypy
from io import BytesIO
import os
from collections import namedtuple

Location = namedtuple('Location', 'x, y')

from PYME.IO import MetaDataHandler
import logging
logger = logging.getLogger(__name__)

#try:
#    import Image
#except ImportError:
from PIL import Image
import time

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
    def set_roi_locations(self, locations_file, tablename='roi_locations'):
        from PYME.IO import tabular
        
        print(locations_file)
        
        if locations_file.endswith('.hdf'):
            locs = tabular.HDFSource(locations_file, tablename=tablename)
            self.roi_locations = [Location(x, y) for x, y in zip(locs['x_um'], locs['y_um'])]
            locs.close()
            del(locs)
        # elif locations_file.endswith('.csv'):
        #     self.roi_locations = tabular.textfileSource(locations_file)
        raise cherrypy.HTTPRedirect('/roi_list')
    
    @cherrypy.expose
    def get_roi_locations(self):
        if not self.roi_locations is None:
            import json
            return json.dumps(self.roi_locations)
            #return DataFrame(self.roi_locations).to_json() #FIXME - this is broken for tabular objects
        
    @cherrypy.expose
    def add_roi(self, x, y):
        #self.roi_locations.append({'x': float(x), 'y': float(y)}) #FIXME - broken for tabular objects
        self.roi_locations.append(Location(x=float(x), y=float(y)))
        
        raise cherrypy.HTTPRedirect('/roi_list')
    
    @cherrypy.expose
    def toggle_roi(self, x, y, hit_radius=10):
        """
        Add an ROI if no ROI exists within hit_radius of the supplied position, otherwise delete the first ROI that is
        found within hit_radius.
        
        Parameters
        ----------
        x : str
            x position (in um)
        y : str
            y position (in um)
        hit_radius : float
            radius around current point to use when looking for existing ROIS (um)
        
        Notes
        -----
        We remove the first ROI found, not the closest. This is easier to code and will give better performance, but
        might be a little more unwieldy in the UI (if you want to remove one of two overlapping ROIs then you might
        remove the wrong one on the first attempt, meaning that you have to remove both and then add one back in).

        """
        
        # Hit-testing is currently a bit crude and requires iterating the roi_locations list.
        # TODO - check performance
        
        hr2 = hit_radius**2
        
        for l in self.roi_locations:
            if ((l.x - float(x))**2 + (l.y - float(y))**2) < hr2:
                #found a hit, remove and redirect
                self.roi_locations.remove(l)
                raise cherrypy.HTTPRedirect('/roi_list')
            
        # didn't find a hit, add ROI
        self.add_roi(x, y)
        
    @cherrypy.expose
    def roi_list(self):
        return env.get_template('roi_list.html').render(roi_list=self.roi_locations)
    
    @cherrypy.expose
    def clear_rois(self):
        self.roi_locations = []
        raise cherrypy.HTTPRedirect('/roi_list')
    
    @cherrypy.expose
    def run_recipe_for_locations(self, recipe_path, tile_level=2):
        from PYME.recipes import batchProcess
        
        output_dir = os.path.join(self.tile_dir, 'detections')
        
        print('launching recipe: %s' % recipe_path)
        batchProcess.bake_recipe(recipe_path,
                          inputGlobs={'input': ['SUPERTILE:%s?level=%d' % (self.tile_dir, tile_level),]},
                          output_dir=output_dir, num_procs=1)
        
        print('Recipe complete')
        time.sleep(2) #wait for output file to flush
        return self.set_roi_locations(os.path.join(output_dir, 'roi_locations.hdf'))

    @cherrypy.expose
    def get_tile(self, layer, x, y, vmin=0, vmax=255):
        cherrypy.response.headers["Content-Type"] = "image/png"
        
        im = self._pyramid.get_tile(int(layer), int(x), int(y))
        
        if im is None:
            im = np.zeros((self._pyramid.tile_size, self._pyramid.tile_size), dtype='uint8')
        else:
            #scale data
            #im = np.clip((255*(im - float(vmin)))/(float(vmax)-float(vmin)), 0, 255).astype('uint8')
        
            im = np.sqrt(im).astype('uint8')

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
                                                          pyramid_x0=self.mdh['Pyramid.x0'],
                                                          pyramid_y0=self.mdh['Pyramid.y0'],
                                                          pyramid_pixel_size_um=self.mdh['Pyramid.PixelSize'],
                                                          )
    
    
    
if __name__ == '__main__':
    import sys
    tile_server = TileServer(sys.argv[1], tile_size=128)
    
    from PYME import cluster
    from PYME import resources
    
    static_dir = resources.get_web_static_dir()
    
    
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