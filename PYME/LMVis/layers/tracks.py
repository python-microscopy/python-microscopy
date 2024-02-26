import logging
logger = logging.getLogger(__name__)

from OpenGL.GL import *

from .base import BaseEngine, EngineLayer
from PYME.LMVis.shader_programs.DefaultShaderProgram import DefaultShaderProgram

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List
# from pylab import cm
from PYME.misc.colormaps import cm
import numpy as np
from PYME.contrib import dispatch

from PYME.Analysis.Tracking.trackUtils import ClumpManager


class Track3DEngine(BaseEngine):
    def __init__(self):
        BaseEngine.__init__(self)
        self.set_shader_program(DefaultShaderProgram)
    
    def render(self, gl_canvas, layer):
        self._set_shader_clipping(gl_canvas)
        
        with self.get_shader_program(gl_canvas):
            glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_DEPTH_TEST)
            
            vertices = layer.get_vertices()
            #n_vertices = vertices.shape[0]
            
            glVertexPointerf(vertices)
            glNormalPointerf(layer.get_normals())
            glColorPointerf(layer.get_colors())

            glLineWidth(layer.line_width)

            for i, cl in enumerate(layer.clumpSizes):
                if cl > 0:
                    glDrawArrays(GL_LINE_STRIP, layer.clumpStarts[i], cl)


ENGINES = {
    'tracks': Track3DEngine,
}


class TrackRenderLayer(EngineLayer):
    """
    A layer for viewing tracking data

    """
    
    # properties to show in the GUI. Note that we also inherit 'visible' from BaseLayer
    vertexColour = CStr('', desc='Name of variable used to colour our points')
    cmap = Enum(*cm.cmapnames, default='gist_rainbow', desc='Name of colourmap used to colour points')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')
    alpha = Float(1.0, desc='Tranparency')
    line_width = Float(1.0, desc='Track line width')
    method = Enum(*ENGINES.keys(), desc='Method used to display tracks')
    clump_key = CStr('clumpIndex', desc="Name of column containing the track identifier")
    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as a source of points')
    _datasource_keys = List()
    _datasource_choices = List()
    
    def __init__(self, pipeline, method='tracks', dsname='', **kwargs):
        EngineLayer.__init__(self, **kwargs)
        self._pipeline = pipeline
        self.engine = None
        self.cmap = 'gist_rainbow'
        
        self.x_key = 'x' #TODO - make these traits?
        self.y_key = 'y'
        self.z_key = 'z'
        
        self._bbox = None
        
        # define a signal so that people can be notified when we are updated (currently used to force a redraw when
        # parameters change)
        self.on_update = dispatch.Signal()
        
        # define responses to changes in various traits
        self.on_trait_change(self._update, 'vertexColour')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname, clump_key')
        self.on_trait_change(self._set_method, 'method')
        
        # update any of our traits which were passed as command line arguments
        self.set(**kwargs)
        
        # update datasource name and method
        #logger.debug('Setting dsname and method')
        self.dsname = dsname
        self.method = method
        
        self._set_method()
        
        # if we were given a pipeline, connect ourselves to the onRebuild signal so that we can automatically update
        # ourselves
        if not self._pipeline is None:
            self._pipeline.onRebuild.connect(self.update)
    
    @property
    def datasource(self):
        """
        Return the datasource we are connected to (through our dsname property).
        """
        return self._pipeline.get_layer_data(self.dsname)
    
    def _set_method(self):
        #logger.debug('Setting layer method to %s' % self.method)
        self.engine = ENGINES[self.method]()
        self.update()
    
    def _get_cdata(self):
        try:
            if isinstance(self.datasource, ClumpManager):
                cdata = []
                for track in self.datasource.all:
                    cdata.extend(track[self.vertexColour])
                cdata = np.array(cdata)
            else:
                # Assume tabular dataset
                cdata = self.datasource[self.vertexColour]
        except KeyError:
            cdata = np.array([0, 1])
        
        return cdata
    
    def _update(self, *args, **kwargs):
        cdata = self._get_cdata()
        self.clim = [float(np.nanmin(cdata)), float(np.nanmax(cdata))]
        #self.update(*args, **kwargs)
    
    def update(self, *args, **kwargs):
        print('lw update')
        self._datasource_choices = self._pipeline.layer_data_source_names
        if not self.datasource is None:
            if isinstance(self.datasource, ClumpManager):
                # Grab the keys from the first Track in the ClumpManager
                self._datasource_keys = sorted(self.datasource[0].keys())
            else:
                # Assume we have a tabular data source
                self._datasource_keys = sorted(self.datasource.keys())
        
        if not (self.engine is None or self.datasource is None):
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)
    
    @property
    def bbox(self):
        return self._bbox
    
    def update_from_datasource(self, ds):
        if isinstance(ds, ClumpManager):
            x = []
            y = []
            z = []
            c = []
            self.clumpSizes = []

            # Copy data from tracks. This is already in clump order
            # thanks to ClumpManager
            for track in ds.all:
                x.extend(track['x'])
                y.extend(track['y'])
                z.extend(track['z'])
                self.clumpSizes.append(track.nEvents)

                if not self.vertexColour == '':
                    c.extend(track[self.vertexColour])
                else:
                    c.extend([0 for i in track['x']])

            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            c = np.array(c)

            # print(x,y,z,c)
            # print(x.shape,y.shape,z.shape,c.shape)

        else:
            # Assume tabular data source        
            x, y = ds[self.x_key], ds[self.y_key]
            
            if not self.z_key is None:
                try:
                    z = ds[self.z_key]
                except KeyError:
                    z = 0 * x
            else:
                z = 0 * x
            
            if not self.vertexColour == '':
                c = ds[self.vertexColour]
            else:
                c = 0 * x

            # Work out clump start and finish indices
            # TODO - optimize / precompute????
            ci = ds[self.clump_key]

            NClumps = int(ci.max())

            clist = [[] for i in range(NClumps)]
            for i, cl_i in enumerate(ci):
                clist[int(cl_i - 1)].append(i)

            # This and self.clumpStarts are class attributes for 
            # compatibility with the old Track rendering layer, 
            # PYME.LMVis.gl_render3D.TrackLayer
            self.clumpSizes = [len(cl_i) for cl_i in clist]

            #reorder x, y, z, c in clump order  
            I = np.hstack([np.array(cl) for cl in clist]).astype(int)

            x = x[I]
            y = y[I]
            z = z[I]
            c = c[I]

        self.clumpStarts = np.cumsum([0, ] + self.clumpSizes)
        
        #do normal vertex stuff
        vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
        vertices = vertices.T.ravel().reshape(len(x.ravel()), 3)
        
        self._vertices = vertices
        self._normals = -0.69 * np.ones(vertices.shape)
        self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        
        clim = self.clim
        cmap = cm[self.cmap]
        
        if clim is not None:
            cs_ = ((c - clim[0]) / (clim[1] - clim[0]))
            cs = cmap(cs_)
            cs[:, 3] = float(self.alpha)
            
            self._colors = cs.ravel().reshape(len(c), 4)
        else:
            if not vertices is None:
                self._colors = np.ones((vertices.shape[0], 4), 'f')
        
        self._color_map = cmap
        self._color_limit = clim
        self._alpha = float(self.alpha)
            
    
    def get_vertices(self):
        return self._vertices
    
    def get_normals(self):
        return self._normals
    
    def get_colors(self):
        return self._colors
    
    def get_color_map(self):
        return self._color_map
    
    @property
    def colour_map(self):
        return self._color_map
    
    def get_color_limit(self):
        return self._color_limit
    
    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor, TextEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        vis_when = 'cmap not in %s' % cm.solid_cmaps
        
        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('method'),
                     Item('vertexColour', editor=EnumEditor(name='_datasource_keys'), label='Colour',
                          visible_when=vis_when),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata, update_signal=self.on_update),
                                 show_label=False), ], visible_when=vis_when),
                     Group(Item('cmap', label='LUT'),
                           Item('alpha', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                           Item('line_width')
                           )])
        #buttons=['OK', 'Cancel'])
    
    def default_traits_view(self):
        return self.default_view

                    
