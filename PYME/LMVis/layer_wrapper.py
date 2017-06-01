from . import layers

LAYERS = {
    'points' : layers.Point3DRenderLayer,
    'pointsprites' : layers.PointSpritesRenderLayer,
    'triangles' : layers.TriangleRenderLayer,
}

class LayerWrapper(object):
    def __init__(self, method='points', namespace=[], dsname=''):
        self._namespace=namespace
        self._dsname=dsname
        
        self.set_method(method)
        
    @property
    def datasource(self):
        return self._namespace[self._dsname]
        
    def set_method(self, method):
        self.method = method
        self.renderer = LAYERS[method]()
        self.renderer.update_from_datasource(self.datasource)
        