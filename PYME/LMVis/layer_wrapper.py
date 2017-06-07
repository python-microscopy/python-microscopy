from . import layers
from PYME.IO import tabular
import dispatch

from PYME.recipes.traits import HasTraits, Enum, ListFloat, Float, Bool

from pylab import cm

LAYERS = {
    'points' : layers.Point3DRenderLayer,
    'pointsprites' : layers.PointSpritesRenderLayer,
    'triangles' : layers.TriangleRenderLayer,
}

class LayerWrapper(HasTraits):
    cmap = Enum(cm.cmapnames)
    def __init__(self, pipeline, method='points', ds_name='', cmap='gist_rainbow', clim=[0,1], alpha=1.0):
        self._pipeline = pipeline
        self._namespace=getattr(pipeline, 'namespace', {})
        self._dsname = None
        self._engine = None
        
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha
        
        self._visible = True

        self.on_update = dispatch.Signal()
        
        self.set_datasource(ds_name)
        self.set_method(method)
        
        self._pipeline.onRebuild.connect(self.update)
        
    @property
    def data_source_names(self):
        names = []
        for k, v in self._namespace.items():
            names.append(k)
            if isinstance(v, tabular.colourFilter):
                for c in v.getColourChans():
                    names.append('.'.join([k, c]))
                    
        return names
        
    @property
    def datasource(self):
        if self._dsname == '':
            return self._pipeline
        
        parts = self._dsname.split('.')
        if len(parts) == 2:
            # special case - permit access to channels using dot notation
            # NB: only works if our underlying datasource is a ColourFilter
            ds, channel = parts
            return self._namespace.get(ds, None).get_channel_ds(channel)
        else:
            return self._namespace.get(self._dsname, None)
        
    @property
    def visible(self):
        return self._visible
    
    @visible.setter
    def visible(self, value):
        self._visible = value
        self.on_update.send(self) #force a redraw
        
        
    def set_method(self, method):
        self.method = method
        self._engine = LAYERS[method]()
        
        self.update()
        
    def set_datasource(self, ds_name):
        self._dsname = ds_name
        
        self.update()
    
    def update(self, *args, **kwargs):
        if not (self._engine is None or self.datasource is None):
            self._engine.update_from_datasource(self.datasource, getattr(cm, self.cmap), self.clim, self.alpha)
            self.on_update.send(self)
        
    def render(self, gl_canvas):
        if self.visible:
            self._engine.render(gl_canvas)
        