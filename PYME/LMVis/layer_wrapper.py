from . import layers
from PYME.IO import tabular

LAYERS = {
    'points' : layers.Point3DRenderLayer,
    'pointsprites' : layers.PointSpritesRenderLayer,
    'triangles' : layers.TriangleRenderLayer,
}

class LayerWrapper(object):
    def __init__(self, method='points', namespace={}, ds_name=''):
        self._namespace=namespace
        self._dsname = None
        self._engine = None
        
        self.set_datasource(ds_name)
        self.set_method(method)
        
        
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
        parts = self._dsname.split('.')
        if len(parts) == 2:
            # special case - permit access to channels using dot notation
            # NB: only works if our underlying datasource is a ColourFilter
            ds, channel = parts
            return self._namespace.get(ds, None).get_channel_ds(channel)
        else:
            return self._namespace.get(self._dsname, None)
        
    def set_method(self, method):
        self.method = method
        self._engine = LAYERS[method]()
        if self.datasource:
            self._engine.update_from_datasource(self.datasource)
        
    def set_datasource(self, ds_name):
        self._dsname = ds_name
        if self._engine:
            self._engine.update_from_datasource(self.datasource)
        
        
    def render(self, gl_canvas):
        self._engine.render(self, gl_canvas)